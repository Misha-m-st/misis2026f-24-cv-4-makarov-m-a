"""Public workflow boundaries: portable config, tiny assets, video timing, resume safety."""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from document_vos.assets import fetch
from document_vos.cli import dispatch, parser
from document_vos.config import load
from document_vos.data.samples import catalog, download, manifest
from document_vos.io import PipelineError, read_json, sha256, write_jsonl
from document_vos.runs import Run
from document_vos.video import encode, extract
from document_vos.videos import prepare


@pytest.fixture
def sample(tmp_path):
    stage = tmp_path / "source"
    stage.mkdir()
    frames = []
    for n in range(3):
        rgb = np.full((33, 65, 3), 50 + n, dtype=np.uint8)
        Image.fromarray(rgb).save(stage / f"{n}.png")
        frames.append({"id": str(n), "path": f"{n}.png", "mask": "initial.png" if n == 0 else None})
    mask = np.zeros((33, 65), dtype=np.uint8)
    mask[3:20, 4:40] = 1
    Image.fromarray(mask).save(stage / "initial.png")
    write_jsonl(
        stage / "manifest.jsonl",
        [
            {
                "dataset": "smartdoc2015",
                "video": "fixture",
                "document": "doc",
                "template": "type",
                "group": "doc",
                "frames": frames,
                "provenance": {"validation_sample": {"id": "smartdoc2015-01"}},
                "time_source": "unknown",
            }
        ],
    )
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for file in stage.iterdir():
            out.write(file, file.name)
    entry = {
        "id": "smartdoc2015-01",
        "dataset": "smartdoc2015",
        "sequence": "smartdoc2015/fixture",
        "url": archive.as_uri(),
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "unpacked_bytes": sum(p.stat().st_size for p in stage.iterdir()),
    }
    cat = {"schema": "document-vos/samples-v1", "samples": [entry]}
    file = tmp_path / "catalog.json"
    file.write_text(json.dumps(cat))
    return file, entry, stage


def test_config_yaml_and_cli_have_different_path_bases(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    config = folder / "settings.yaml"
    config.write_text("paths:\n  results: outputs\n  videos: вход\nmodel: baseline\n")
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(
        "document_vos.workflows.inference", lambda cfg, **kw: seen.update(cfg=cfg, kw=kw) or {}
    )
    dispatch(
        parser().parse_args(
            [
                "infer",
                "--config",
                str(config),
                "--input-dir",
                "CLI вход",
                "--output",
                "CLI result",
                "--model",
                "finetuned",
                "--seed",
                "0",
            ]
        )
    )
    assert seen["cfg"]["paths"]["videos"] == str(folder / "вход")
    assert seen["cfg"]["paths"]["results"] == str(folder / "outputs")
    assert seen["cfg"]["model"] == "finetuned"
    assert seen["kw"]["output"] == tmp_path / "CLI result"
    assert seen["kw"]["input_dir"] == tmp_path / "CLI вход"
    assert seen["kw"]["seed"] == 0


@pytest.mark.parametrize(
    "text",
    [
        "unknown: true",
        "training:\n  typo: 2",
        "inference:\n  long_term:\n    typo: 1",
        "training:\n  batch_size: true",
        "training:\n  crop_size: [480, 320]",
        "runtime:\n  workers: 0",
        "samples:\n  count: 3",
        "model: old",
        "sam:\n  step: false",
        "visualization:\n  fps: .nan",
        'paths:\n  raw: ""',
    ],
)
def test_invalid_settings_fail_before_any_output(tmp_path, text):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(text)
    before = list(tmp_path.iterdir())
    with pytest.raises((PipelineError, TypeError)):
        load(cfg)
    assert list(tmp_path.iterdir()) == before


def test_sample_plan_select_reuse_and_no_training_fallback(tmp_path, sample, monkeypatch):
    file, entry, stage = sample
    monkeypatch.setattr(
        "document_vos.data.download.download_archive", lambda *a: pytest.fail("training download")
    )
    output = tmp_path / "samples"
    cat = catalog(file)
    plan = download(output, cat, "smartdoc2015", 1, plan=True)
    assert plan["download_bytes"] == entry["bytes"] and not output.exists()
    assert download(output, cat, "smartdoc2015", 1)["status"] == "COMPLETE"
    assert download(output, cat, "smartdoc2015", 1)["status"] == "COMPLETE"
    assert (output / entry["id"] / "0.png").read_bytes() == (stage / "0.png").read_bytes()
    with pytest.raises(PipelineError):
        download(output, cat, "smartdoc2015", 2)
    with pytest.raises(PipelineError):
        download(output, cat, "midv2020", 1)
    from document_vos.data import load_manifest

    rows = load_manifest(manifest(output, tmp_path / "manifest.jsonl", cat, "smartdoc2015", 1))
    assert all(f.time_s is None for f in rows[0].frames)
    assert sum(f.has_gt for f in rows[0].frames) == 1
    (output / entry["id"] / "1.png").write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="Повреждён"):
        download(output, cat, "smartdoc2015", 1)


def test_asset_resume_and_corruption_preserve_evidence(tmp_path, sample):
    _, entry, _ = sample
    target = tmp_path / "downloaded.zip"
    data = Path(entry["url"].removeprefix("file://")).read_bytes()
    target.with_suffix(".zip.part").write_bytes(data[:31])
    fetch(entry, target)
    assert target.read_bytes() == data
    target.write_bytes(b"bad")
    with pytest.raises(PipelineError, match="Повреждён"):
        fetch(entry, target)
    assert target.read_bytes() == b"bad"


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "C:/outside", "link"])
def test_bad_sample_archive_is_not_admitted(tmp_path, sample, name):
    _, entry, _ = sample
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        info = zipfile.ZipInfo(name)
        if name == "link":
            info.external_attr = 0o120777 << 16
        out.writestr(info, b"bad")
    entry = {
        **entry,
        "url": archive.as_uri(),
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "unpacked_bytes": 3,
    }
    from document_vos.data.samples import unpack

    with pytest.raises(PipelineError):
        unpack(archive, tmp_path / "extracted", entry)
    assert not (tmp_path / "extracted").exists()


def test_pyav_vfr_unicode_odd_dimensions_no_external_binaries(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    video = tmp_path / "видео с пробелом.mp4"
    images = [Image.fromarray(np.full((33, 65, 3), n * 30, dtype=np.uint8)) for n in range(5)]
    report = encode(video, iter(images), [2.0, 2.1, 2.2, 2.5, 2.7])
    frames, timing = extract(video, tmp_path / "кадры")
    assert len(frames) == 5 and report["verified"]
    assert [f["time_s"] for f in frames] == pytest.approx([0, 0.1, 0.2, 0.5, 0.7], abs=1e-6)
    assert report["original_size"] == [65, 33] and timing["oriented_size"] == [66, 34]


@pytest.mark.parametrize("kind", ["png1", "png255", "corners"])
def test_video_initial_sources_reuse_and_deleted_review(tmp_path, kind, monkeypatch):
    directory = tmp_path / "мои видео"
    directory.mkdir()
    video = directory / "лист.mp4"
    encode(video, iter([Image.new("RGB", (64, 32), "gray")] * 3), [0, 0.1, 0.3])
    if kind.startswith("png"):
        mask = np.zeros((32, 64), np.uint8)
        mask[3:20, 4:40] = 255 if kind == "png255" else 1
        Image.fromarray(mask).save(video.with_suffix(".mask.png"))
    else:
        video.with_suffix(".corners.json").write_text("[[4,3],[39,3],[39,19],[4,19]]")
    cfg = load()
    prepared = tmp_path / "prepared"
    result = prepare(directory, prepared, cfg)
    output = Path(result["output"])
    from document_vos.data.input_folder import select_input_folder

    assert len(select_input_folder(output)[0]) == 1
    preview = next((output / "review_initial").glob("*.png"))
    preview.unlink()
    assert prepare(directory, prepared, cfg)["status"] == "EXISTING"
    assert not preview.exists() and len(select_input_folder(output)[0]) == 0
    (output / "frames" / next((output / "frames").iterdir()).name / "000001.png").write_bytes(
        b"bad"
    )
    with pytest.raises(PipelineError, match="изменился"):
        prepare(directory, prepared, cfg)


@pytest.mark.parametrize("problem", ["both", "empty", "wrong_size", "rgb", "gray128"])
def test_bad_initial_mask_blocks_video(tmp_path, problem):
    directory = tmp_path / "videos"
    directory.mkdir()
    video = directory / "input.mp4"
    encode(video, iter([Image.new("RGB", (64, 32), "white")] * 2), [0, 0.1])
    mask = np.zeros((32, 64), np.uint8)
    if problem != "empty":
        mask[3:10, 3:10] = 1
    if problem == "wrong_size":
        mask = mask[:10]
    if problem == "rgb":
        mask = np.stack([mask] * 3, axis=-1)
    if problem == "gray128":
        mask[3:10, 3:10] = 128
    Image.fromarray(mask).save(video.with_suffix(".mask.png"))
    if problem == "both":
        video.with_suffix(".corners.json").write_text("[[1,1],[20,1],[20,20],[1,20]]")
    with pytest.raises(PipelineError):
        prepare(directory, tmp_path / "prepared", load())


def test_resume_verifies_config_inputs_artifacts_and_logs(tmp_path):
    cfg = {"settings": 42}
    inputs = {"source": "hash"}
    run = Run(tmp_path / "results", "test", cfg, inputs)

    def stage(output):
        output.mkdir()
        (output / "proof.json").write_text("success")

    target = run.stage("a", stage)
    run.finish()
    resumed = Run(tmp_path / "results", "test", cfg, inputs, resume=run.output)
    assert resumed.stage("a", lambda p: pytest.fail("reexecuted")) == target
    for bad_cfg, bad_inputs in [({"settings": 1}, inputs), (cfg, {"source": "changed"})]:
        with pytest.raises(PipelineError, match="изменились"):
            Run(tmp_path / "results", "test", bad_cfg, bad_inputs, resume=run.output)
    (target / "proof.json").write_text("changed")
    with pytest.raises(PipelineError, match="изменились"):
        Run(tmp_path / "results", "test", cfg, inputs, resume=run.output)
    (target / "proof.json").write_text("success")
    (run.output / "a-1.log").write_text("changed")
    with pytest.raises(PipelineError, match="Журнал"):
        Run(tmp_path / "results", "test", cfg, inputs, resume=run.output)


def test_failed_stage_retries_into_new_directory(tmp_path):
    run = Run(tmp_path / "results", "test", {}, {})

    def broken(output):
        output.mkdir()
        (output / "partial").write_text("keep me")
        raise RuntimeError("interrupt")

    with pytest.raises(RuntimeError):
        run.stage("a", broken)
    old = run.output / "a-1" / "partial"
    resumed = Run(tmp_path / "results", "test", {}, {}, resume=run.output)

    def succeed(output):
        output.mkdir()
        (output / "proof").write_text("ok")

    assert resumed.stage("a", succeed).name == "a-2"
    assert old.read_text() == "keep me"
    assert read_json(run.output / "status.json")["stages"]["a"]["status"] == "COMPLETE"


def test_reproduce_rejects_changed_science_before_setup(tmp_path, monkeypatch):
    from document_vos.workflows import reproduce

    cfg = load()
    cfg["training"]["num_iterations"] = 2
    monkeypatch.setattr(
        "document_vos.cutie.check_checkout", lambda *a: pytest.fail("setup reached")
    )
    with pytest.raises(PipelineError, match="отличаются"):
        reproduce(cfg, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_recipe_assignment_is_portable_and_checks_source_pixels(tmp_path, sequences, monkeypatch):
    from document_vos.data import save_manifest
    from document_vos.data.recipe import identity, source_identity, split

    entries = [
        {
            "key": s.key,
            "split": "validation" if n == 0 else "train",
            "sha256": identity(s),
            "source_sha256": source_identity(s),
        }
        for n, s in enumerate(sequences)
    ]
    recipe = {"sequences": entries}
    monkeypatch.setattr("document_vos.data.recipe.published", lambda: recipe)
    moved = tmp_path / "relocated"
    moved.mkdir()
    import shutil
    from dataclasses import replace

    copies = []
    for seq in sequences:
        frames = []
        for index, frame in enumerate(seq.frames):
            target = moved / f"{seq.video}-{index}.png"
            shutil.copyfile(frame.path, target)
            frames.append(replace(frame, path=str(target)))
        copies.append(replace(seq, frames=frames))
    manifest = moved / "manifest.jsonl"
    save_manifest(manifest, copies)
    assert split(manifest, moved / "split")["counts"] == {"train": 5, "validation": 1}
    Image.new("RGB", (48, 32), "red").save(copies[1].frames[0].path)
    with pytest.raises(PipelineError, match="Изменились"):
        split(manifest, moved / "changed")
    assert not (moved / "changed").exists()


def test_http_range_requires_server_confirmation(tmp_path, sample):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _, entry, _ = sample
    payload = Path(entry["url"].removeprefix("file://")).read_bytes()
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            observed.append(self.headers.get("Range"))
            offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
            self.send_response(206 if offset else 200)
            if offset:
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            self.wfile.write(payload[offset:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = tmp_path / "http.zip"
        target.with_suffix(".zip.part").write_bytes(payload[:73])
        entry = {**entry, "url": f"http://127.0.0.1:{server.server_port}/sample.zip"}
        fetch(entry, target)
        assert target.read_bytes() == payload and observed == ["bytes=73-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_normalization_respects_custom_raw_directory(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    from document_vos.data.download import Dataset
    from document_vos.data.preparation import ensure_local

    archives = tmp_path / "archives"
    archives.mkdir()
    archive = archives / "small.tar"
    with tarfile.open(archive, "w") as out:
        for name in ["images/a.png", "annotations/a.json"]:
            member = tarfile.TarInfo(name)
            member.size = 3
            out.addfile(member, io.BytesIO(b"raw"))
    entry = Dataset(
        "midv2020",
        "unused",
        "small.tar",
        archive.stat().st_size,
        "sha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "raw/midv2020/clips",
    )
    monkeypatch.setattr("document_vos.data.download.dataset_archives", lambda _: [entry])
    cfg = load()
    cfg["datasets"]["selected"] = ["midv2020"]
    cfg["paths"]["archives"] = str(archives)
    cfg["paths"]["raw"] = str(tmp_path / "custom source")
    result = ensure_local(cfg, normalize=True)
    assert result["midv2020"] == tmp_path / "custom source/midv2020/clips"
    assert (result["midv2020"] / "images/a.png").read_bytes() == b"raw"
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("presentation", ["original", "color_copy", "gray_copy"])
def test_release_sample_roundtrip_retains_dlc_admission_metadata(tmp_path, sequences, presentation):
    from dataclasses import replace

    from document_vos.data import load_manifest
    from document_vos.release import sample_asset

    seq = replace(sequences[0], dataset="dlc2021", provenance={"presentation": presentation})
    release = tmp_path / "release"
    release.mkdir()
    entry = sample_asset(seq, release, "dlc2021-01", {"test": "synthetic recipe"})
    entry["url"] = (release / "dlc2021-01.zip").as_uri()
    cat = {"schema": "document-vos/samples-v1", "samples": [entry]}
    download(tmp_path / "downloaded", cat, "dlc2021", 1)
    target = tmp_path / "downloaded/dlc2021-01"
    rows = load_manifest(target / "manifest.jsonl")
    assert rows[0].provenance["presentation"] == presentation
    assert len(rows[0].frames) == len(seq.frames)
    assert sum(f.has_gt for f in rows[0].frames) == 1
    assert Path(rows[0].frames[0].path).read_bytes() == Path(seq.frames[0].path).read_bytes()
    assert not list(target.rglob("training-export"))
