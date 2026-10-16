"""Быстрые команды не читают пиксели; явная проверка замечает изменения."""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from document_vos import data, validation_cache
from document_vos.adapters import prepare
from document_vos.cli import dispatch, parser
from document_vos.export import export_training
from document_vos.io import PipelineError, write_jsonl
from document_vos.split import split_manifest


def forbidden(*args, **kwargs):
    raise AssertionError("Дорогая повторная проверка не должна запускаться")


def test_structural_does_not_read_pixels_hash_or_build_gt(monkeypatch, sequences):
    monkeypatch.setattr(data, "sha256", forbidden)
    monkeypatch.setattr(data.Image, "open", forbidden)
    monkeypatch.setattr(data, "ground_truth", forbidden)
    stats = data.validate_data(sequences)
    assert stats["frames"] == 24 and stats["missing_gt"] == 0
    assert stats["validation"] == "structural" and not stats["content_checked"]
    assert stats["empty_gt"] is None and stats["repeated_frames_within_video"] is None
    assert stats["checked_frames"] == stats["cached_frames"] == 0


def test_prepare_split_export_have_no_full_validation(tmp_path, sequences, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir()
    write_jsonl(
        root / "index.jsonl",
        [
            {k: v for k, v in s.to_dict().items() if k not in {"dataset", "schema_version"}}
            for s in sequences
        ],
    )
    monkeypatch.setattr(data, "sha256", forbidden)
    monkeypatch.setattr(data, "ground_truth", forbidden)
    manifest = tmp_path / "prepared.jsonl"
    result = prepare(
        {
            "selected": ["midv2020"],
            "datasets": {
                "midv2020": {"root": str(root), "index": "index.jsonl"},
            },
        },
        tmp_path,
        manifest,
    )
    assert result["validation"] == "structural"
    split = tmp_path / "split"
    split_manifest(manifest, split)
    # Запись новых масок нужна; повторный validate_export после записи не нужен.
    monkeypatch.setattr("document_vos.export.validate_export", forbidden)
    assert export_training(split / "train.jsonl", split, tmp_path / "export")["frames"] > 0


def test_full_cache_reuses_frames_between_manifests(tmp_path, sequences, monkeypatch):
    cache = tmp_path / "cache.sqlite3"
    first = data.validate_data(sequences[:3], full=True, cache_path=cache)
    assert first["checked_frames"] == 12 and first["cached_frames"] == 0
    extended = data.validate_data(sequences, full=True, cache_path=cache)
    assert extended["cached_frames"] == extended["checked_frames"] == 12
    monkeypatch.setattr(data, "sha256", forbidden)
    monkeypatch.setattr(data.Image, "open", forbidden)
    monkeypatch.setattr(data, "ground_truth", forbidden)
    subset = data.validate_data(sequences[1:4], full=True, cache_path=cache)
    assert subset["cached_frames"] == 12 and subset["checked_frames"] == 0
    assert subset["empty_gt"] == 0 and subset["content_checked"]


def test_cache_invalidates_file_even_with_restored_mtime(tmp_path, sequences):
    cache = tmp_path / "cache.sqlite3"
    data.validate_data(sequences, full=True, cache_path=cache)
    image = Path(sequences[0].frames[0].path)
    stat = image.stat()
    # Перезаписаны те же байты: размер и mtime прежние, ctime новый.
    image.write_bytes(image.read_bytes())
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    stats = data.validate_data(sequences, full=True, cache_path=cache)
    assert stats["checked_frames"] == 1 and stats["cached_frames"] == 23


def test_cache_invalidates_annotations_masks_and_validation_rules(tmp_path, sequences, monkeypatch):
    mask = tmp_path / "gt.png"
    Image.fromarray(np.ones((32, 48), np.uint8)).save(mask, compress_level=0)
    frame = sequences[0].frames[0]
    frame.polygon, frame.mask = None, str(mask)
    cache = tmp_path / "cache.sqlite3"
    data.validate_data(sequences, full=True, cache_path=cache)
    Image.fromarray(np.zeros((32, 48), np.uint8)).save(mask, compress_level=0)
    sequences[0].frames[1].polygon = [[80, 60], [90, 60], [90, 80], [80, 80]]
    stats = data.validate_data(sequences, full=True, cache_path=cache)
    assert stats["checked_frames"] == 2 and stats["cached_frames"] == 22
    assert stats["empty_gt"] == 2
    monkeypatch.setattr(validation_cache, "VALIDATION_VERSION", 2)
    assert data.validate_data(sequences, full=True, cache_path=cache)["checked_frames"] == 24


def test_cached_hashes_still_detect_cross_video_duplicates(tmp_path, sequences, monkeypatch):
    cache = tmp_path / "cache.sqlite3"
    data.validate_data(sequences, full=True, cache_path=cache)
    duplicate = deepcopy(sequences[0])
    duplicate.video = "duplicate"
    duplicate.frames = [deepcopy(sequences[1].frames[0])]
    copy = tmp_path / "copy.png"
    copy.write_bytes(Path(duplicate.frames[0].path).read_bytes())
    duplicate.frames[0].path = str(copy)
    data.validate_data([duplicate], full=True, cache_path=cache)
    monkeypatch.setattr(data, "sha256", forbidden)
    monkeypatch.setattr(data.Image, "open", forbidden)
    with pytest.raises(PipelineError, match="Дублирующийся"):
        data.validate_data([sequences[1], duplicate], full=True, cache_path=cache)


def test_missing_paths_bad_geometry_and_corruption_are_not_cached(tmp_path, sequences):
    cache = tmp_path / "cache.sqlite3"
    data.validate_data(sequences, full=True, cache_path=cache)
    bad = deepcopy(sequences)
    bad[0].frames[0].polygon = [[0, 0]] * 4
    with pytest.raises(PipelineError, match="Вырожденный"):
        data.validate_data(bad)
    image = Path(sequences[0].frames[0].path)
    original = image.read_bytes()
    image.write_bytes(b"not an image")
    # Структурная проверка намеренно не подтверждает целостность пикселей.
    assert data.validate_data(sequences)["validation"] == "structural"
    with pytest.raises(OSError):
        data.validate_data(sequences, full=True, cache_path=cache)
    image.write_bytes(original)
    stats = data.validate_data(sequences, full=True, cache_path=cache)
    assert stats["checked_frames"] == 1 and stats["cached_frames"] == 23
    image.unlink()
    with pytest.raises(PipelineError, match="кадр не найден"):
        data.validate_data(sequences, full=True, cache_path=cache)


def test_cache_keeps_completed_frames_after_failed_pass(tmp_path, sequences):
    cache = tmp_path / "cache.sqlite3"
    image = Path(sequences[0].frames[2].path)
    original = image.read_bytes()
    image.write_bytes(b"broken")
    with pytest.raises(OSError):
        data.validate_data(sequences, full=True, cache_path=cache)
    image.write_bytes(original)
    stats = data.validate_data(sequences, full=True, cache_path=cache)
    assert stats["cached_frames"] == 2 and stats["checked_frames"] == 22


def test_cli_default_full_cached_and_refresh(tmp_path, manifest):
    def command(*flags):
        return dispatch(parser().parse_args(["validate-data", "--manifest", str(manifest), *flags]))

    cache = tmp_path / ".validation-cache.sqlite3"
    assert command()["validation"] == "structural" and not cache.exists()
    assert command("--full")["checked_frames"] == 24
    assert command("--full")["cached_frames"] == 24
    assert command("--full", "--refresh")["checked_frames"] == 24
    with pytest.raises(PipelineError, match="--full"):
        command("--refresh")
