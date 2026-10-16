from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from document_vos.data import load_manifest, save_manifest, validate_data
from document_vos.export import export_training, validate_export
from document_vos.io import PipelineError, read_json
from document_vos.split import assert_disjoint, components, split_manifest


def test_split_reproducible_and_cross_dataset_connected(tmp_path, sequences):
    sequences[1].dataset = "midv2019"
    sequences[1].template = sequences[0].template
    path = tmp_path / "m.jsonl"
    save_manifest(path, sequences)
    assert len(components(sequences)) == 5
    a = split_manifest(path, tmp_path / "s1")
    b = split_manifest(path, tmp_path / "s2")
    assert a == b
    train = load_manifest(tmp_path / "s1/train.jsonl")
    val = load_manifest(tmp_path / "s1/validation.jsonl")
    assert_disjoint(train, val)
    assert any(
        {sequences[0].key, sequences[1].key} <= {s.key for s in group} for group in [train, val]
    )


def test_previous_split_preserved_and_conflicting_relationship_rejected(
    tmp_path, manifest, sequences
):
    old = tmp_path / "old"
    split_manifest(manifest, old)
    # Дополнительный документ связан с существующим; назначение обязано сохраниться.
    extra = deepcopy(sequences[0])
    extra.video = "additional"
    for i, f in enumerate(extra.frames):
        path = tmp_path / f"extra-{i}.png"
        Image.fromarray(np.full((32, 48, 3), 60 + i, np.uint8)).save(path)
        f.path = str(path)
    extended = tmp_path / "extended.jsonl"
    save_manifest(extended, sequences + [extra])
    split_manifest(extended, tmp_path / "new", previous=old)
    for label in ["train", "validation"]:
        assert {s.key for s in load_manifest(old / f"{label}.jsonl")} <= {
            s.key for s in load_manifest(tmp_path / "new" / f"{label}.jsonl")
        }
    train, val = [load_manifest(old / f"{label}.jsonl") for label in ["train", "validation"]]
    bridge = deepcopy(extra)
    bridge.group = train[0].group
    bridge.template = val[0].template
    conflicting = tmp_path / "conflict.jsonl"
    save_manifest(conflicting, sequences + [bridge])
    with pytest.raises(PipelineError, match="объединила"):
        split_manifest(conflicting, tmp_path / "rejected", previous=old)


def test_export_only_gt_and_train(tmp_path, manifest, sequences):
    sequences[0].frames[1].polygon = None
    path = tmp_path / "missing.jsonl"
    save_manifest(path, sequences)
    split = tmp_path / "split"
    split_manifest(path, split)
    out = tmp_path / "export"
    stats = export_training(split / "train.jsonl", split, out)
    assert stats["frames"] >= 3
    assert validate_export(out) == stats
    meta = read_json(out / "export.json")
    for video in meta["sequences"]:
        for f in video["frames"]:
            assert f["name"].isdigit() and len(f["name"]) == 6
            im = Image.open(out / "images" / video["export_video"] / (f["name"] + ".jpg"))
            gt = Image.open(out / "masks" / video["export_video"] / (f["name"] + ".png"))
            assert im.size == gt.size and set(np.unique(gt)) <= {0, 1}
    with pytest.raises(PipelineError, match="train-манифеста"):
        export_training(split / "validation.jsonl", split, tmp_path / "invalid")
    mask = next((out / "masks").rglob("*.png"))
    mask.unlink()
    with pytest.raises(PipelineError):
        validate_export(out)


def test_duplicate_missing_annotation_and_masks(tmp_path, sequences):
    assert validate_data(sequences, full=True)["frames"] == 24
    bad = deepcopy(sequences)
    bad[1].frames[0].path = bad[0].frames[0].path
    with pytest.raises(PipelineError, match="повторно"):
        validate_data(bad, full=True)
    bad = deepcopy(sequences)
    bad[1].frames[0].path = str(tmp_path / "copy.png")
    (tmp_path / "copy.png").write_bytes(open(bad[0].frames[0].path, "rb").read())
    with pytest.raises(PipelineError, match="Дублирующийся"):
        validate_data(bad, full=True)
    bad = deepcopy(sequences)
    for f in bad[0].frames:
        f.polygon = None
    with pytest.raises(PipelineError, match="нет необходимой"):
        validate_data(bad, full=True)
    bad = deepcopy(sequences)
    bad[0].frames[1].time_s = bad[0].frames[0].time_s
    with pytest.raises(PipelineError, match="возрастать"):
        validate_data(bad, full=True)
