"""Synthetic FFmpeg and prepared-folder protocol checks."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from document_vos.cutie.inference import infer
from document_vos.data import Frame, Sequence, save_manifest
from document_vos.data.input_folder import select_input_folder
from document_vos.diagnostics import dispatch, parser
from document_vos.evaluation import evaluate
from document_vos.io import PipelineError, read_json, read_jsonl, sha256
from document_vos.video import extract


class Propagator:
    last_model_seconds = 0.01

    def __init__(self, *args):
        self.calls = []

    def reset(self):
        self.mask = None

    def sync(self):
        pass

    def step(self, image, mask=None, end=False):
        self.calls.append(mask is not None)
        if mask is not None:
            self.mask = mask.copy()
        return self.mask.copy()

    def memory_stats(self):
        return {"status": "NOT_MEASURED"}


@pytest.fixture
def videos(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/ffprobe unavailable; real extraction checks NOT_RUN")
    paths = []
    for n in range(2):
        path = tmp_path / f"phone-{n}.mkv"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x32:rate=10",
                "-frames:v",
                "23",
                "-vf",
                "setpts=PTS+if(gte(N\\,10)\\,0.2/TB\\,0)",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-output_ts_offset",
                "2",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        paths.append(path)
    return paths


@pytest.fixture
def phone_set(tmp_path, videos):
    folder = tmp_path / "prepared"
    review = folder / "review_initial"
    review.mkdir(parents=True)
    sequences = []
    for n, video in enumerate(videos):
        name = f"video-{n}"
        frames, timing = extract(video, folder / "frames" / name)
        masks = folder / "masks" / name
        masks.mkdir(parents=True)
        for i in [0, 10, 20]:
            mask = np.zeros((32, 64), dtype=np.uint8)
            mask[4:20, 8:30] = 1
            path = masks / f"{i:06d}.png"
            Image.fromarray(mask).save(path)
            frames[i]["mask"] = str(path)
        with Image.open(frames[0]["path"]) as image:
            image.save(review / f"{name}.png")
        sequences.append(
            Sequence(
                "custom",
                name,
                name,
                name,
                name,
                [Frame(**f) for f in frames],
                provenance={
                    "annotation_source": "synthetic fixture",
                    "phone_testset": {
                        "status": "eligible",
                        **timing,
                    },
                },
                time_source="ffprobe source PTS",
            )
        )
    save_manifest(folder / "manifest.jsonl", sequences)
    return folder


def fake_cutie(monkeypatch):
    import document_vos.cutie.inference as module

    monkeypatch.setattr(module, "check_checkout", lambda p: p)
    monkeypatch.setattr(module, "local_weights", lambda p: p)
    monkeypatch.setattr(module, "inference_config", lambda *a: {"mem_every": 5})
    monkeypatch.setattr(module, "versions", lambda: {"test": "synthetic"})
    model = Propagator()
    monkeypatch.setattr(module, "CutiePredictor", lambda *a: model)
    return model


def run_folder(tmp_path, folder, output, monkeypatch):
    fake_cutie(monkeypatch)
    weights = tmp_path / "test-weights"
    weights.write_bytes(b"synthetic")
    return infer(
        None, output, tmp_path, weights, {}, "independent", None, device="cpu", input_dir=folder
    )


def test_pts_orientation_and_sparse_masks(tmp_path, videos, phone_set):
    rows = read_jsonl(phone_set / "manifest.jsonl")
    for row, video in zip(rows, videos):
        frames = row["frames"]
        assert len(frames) == 23
        assert [i for i, f in enumerate(frames) if f.get("mask")] == [0, 10, 20]
        pts = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=pts_time",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )["frames"]
        assert [f["time_s"] for f in frames] == [float(f["pts_time"]) for f in pts]
        assert frames[0]["time_s"] == 2.0
        assert frames[10]["time_s"] - frames[9]["time_s"] == pytest.approx(0.3)
        for f in frames:
            if f.get("mask"):
                with Image.open(f["mask"]) as image:
                    mask = np.asarray(image)
                    assert image.size == (64, 32) and mask.ndim == 2
                    assert set(np.unique(mask)) == {0, 1}
    rotated = tmp_path / "rotated.mp4"
    unrotated = tmp_path / "unrotated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(videos[0]), "-c", "copy", str(unrotated)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(unrotated),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
        capture_output=True,
    )
    frames, meta = extract(rotated, tmp_path / "oriented")
    assert meta["oriented_size"] == [32, 64]
    with Image.open(rows[0]["frames"][0]["path"]) as im:
        original = np.asarray(im)
    with Image.open(frames[0]["path"]) as im:
        assert np.array_equal(np.asarray(im), np.rot90(original))


def test_deleted_png_excludes_whole_video_and_snapshot_survives(
    tmp_path,
    phone_set,
    videos,
    monkeypatch,
    capsys,
):
    rows = read_jsonl(phone_set / "manifest.jsonl")
    review = phone_set / "review_initial" / f"{rows[0]['video']}.png"
    saved = review.read_bytes()
    frame_hashes = {
        p: sha256(p) for sub in ["frames", "masks"] for p in (phone_set / sub).rglob("*.png")
    }
    review.unlink()
    spy = fake_cutie(monkeypatch)
    weights = tmp_path / "weights"
    weights.write_bytes(b"synthetic")
    cfg = tmp_path / "infer.yaml"
    cfg.write_text(
        f"manifest: missing.jsonl\nscope: validation\ncutie_root: .\nweights: {weights}\n"
    )
    run = tmp_path / "run"
    result = dispatch(
        parser().parse_args(
            [
                "infer",
                "--config",
                str(cfg),
                "--input-dir",
                str(phone_set),
                "--output",
                str(run),
                "--device",
                "cpu",
            ]
        )
    )
    assert result["videos"] == 1 and len(spy.calls) == 23 and sum(spy.calls) == 1
    assert "включено: 1; исключено: 1" in capsys.readouterr().err
    report = read_json(run / "selection.json")
    assert report["decisions"][0]["reason"] == "review_png_missing"
    snapshot = run / "selected-manifest.jsonl"
    assert [r["video"] for r in read_jsonl(snapshot)] == [rows[1]["video"]]
    assert len(select_input_folder(phone_set)[0]) == 1
    assert not review.exists()
    assert frame_hashes == {p: sha256(p) for p in frame_hashes}
    # Deleting the last review PNG and even changing the source index cannot change a run.
    next((phone_set / "review_initial").glob("*.png")).unlink()
    (phone_set / "manifest.jsonl").write_text("broken index")
    assert evaluate(run, tmp_path / "evaluation", workers=1)["evaluated_frames"] == 2
    run_info = read_json(run / "run.json")
    assert run_info["manifest"] == str(snapshot)
    assert run_info["signature"]["initializations"][0]["index"] == 0
    assert "annotation" not in run_info["selection"]
    # Restoring a PNG restores admission only on a new run.
    (phone_set / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    review.write_bytes(saved)
    assert len(select_input_folder(phone_set)[0]) == 1


def test_empty_selection_never_loads_cutie_and_bad_initial_excluded(
    tmp_path,
    phone_set,
    monkeypatch,
):
    rows = read_jsonl(phone_set / "manifest.jsonl")
    initial = Path(rows[0]["frames"][0]["mask"])
    Image.fromarray(np.zeros((32, 64), np.uint8)).save(initial)
    review = phone_set / "review_initial" / f"{rows[1]['video']}.png"
    review.unlink()

    def forbidden(*args):
        pytest.fail("Cutie must not be checked/loaded for an empty selection")

    monkeypatch.setattr("document_vos.cutie.inference.check_checkout", forbidden)
    result = infer(
        None,
        tmp_path / "empty",
        tmp_path / "absent",
        tmp_path / "absent",
        {},
        "independent",
        None,
        input_dir=phone_set,
    )
    assert result["status"] == "NO_ELIGIBLE_VIDEOS" and not result["model_loaded"]
    reasons = {r["reason"] for r in result["selection"]["decisions"]}
    assert reasons == {"initial_mask_empty", "review_png_missing"}


def test_cli_input_resolution_and_scope(tmp_path, monkeypatch):
    seen = {}

    def capture(*args, **kwargs):
        seen.update(manifest=args[0], scope=args[5], **kwargs)
        return {}

    monkeypatch.setattr("document_vos.cutie.inference.infer", capture)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "manifest: old.jsonl\ncutie_root: .\nweights: weights\noutput: result\ninput_dir: data/phone\n"
    )
    dispatch(parser().parse_args(["infer", "--config", str(cfg)]))
    assert seen["manifest"] is None and seen["scope"] == "independent"
    assert seen["input_dir"] == tmp_path / "data/phone"
    dispatch(parser().parse_args(["infer", "--config", str(cfg), "--manifest", "legacy.jsonl"]))
    assert seen["input_dir"] is None and seen["scope"] == "validation"
    for extra in [
        ["--input-dir", "folder", "--manifest", "file"],
        ["--input-dir", "folder", "--scope", "validation"],
    ]:
        with pytest.raises(PipelineError):
            dispatch(parser().parse_args(["infer", "--config", str(cfg), *extra]))
    cfg.write_text(cfg.read_text() + "scope: validation\n")
    with pytest.raises(PipelineError, match="validation"):
        dispatch(parser().parse_args(["infer", "--config", str(cfg)]))


@pytest.mark.parametrize("change", ["review", "mask", "frame"])
def test_compatibility_rejects_changed_folder_inputs(tmp_path, phone_set, monkeypatch, change):
    from document_vos.evaluation.comparison import compatible

    run_folder(tmp_path, phone_set, tmp_path / "baseline", monkeypatch)
    evaluate(tmp_path / "baseline", tmp_path / "baseline-eval", workers=1)
    rows = read_jsonl(phone_set / "manifest.jsonl")
    if change == "review":
        (phone_set / "review_initial" / f"{rows[0]['video']}.png").unlink()
    else:
        path = Path(rows[0]["frames"][10][change if change == "mask" else "path"])
        with Image.open(path) as image:
            pixels = np.asarray(image).copy()
        pixels[0, 0] = 1 - pixels[0, 0] if change == "mask" else 0
        Image.fromarray(pixels).save(path)
    run_folder(tmp_path, phone_set, tmp_path / "finetuned", monkeypatch)
    evaluate(tmp_path / "finetuned", tmp_path / "finetuned-eval", workers=1)
    a = read_json(tmp_path / "baseline-eval/evaluation.json")
    b = read_json(tmp_path / "finetuned-eval/evaluation.json")
    with pytest.raises(PipelineError, match="Несовместимые"):
        compatible(a, b)
    if change != "review":
        with pytest.raises(PipelineError, match="изменены"):
            evaluate(tmp_path / "baseline", tmp_path / "stale-eval", workers=1)


@pytest.mark.parametrize("pts", [[None, 1], [0, 0], [1, 0]])
def test_missing_or_nonmonotonic_timestamps_never_fall_back_to_fps(tmp_path, monkeypatch, pts):
    from fractions import Fraction
    from types import SimpleNamespace

    class Container:
        streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            return iter(
                [
                    SimpleNamespace(
                        pts=p,
                        time_base=Fraction(1, 1000),
                        rotation=0,
                        width=64,
                        height=32,
                        to_ndarray=lambda **kw: np.zeros((32, 64, 3), np.uint8),
                    )
                    for p in pts
                ]
            )

    monkeypatch.setattr("document_vos.video.av.open", lambda *a, **kw: Container())
    with pytest.raises(ValueError, match="PTS|timestamps"):
        extract(tmp_path / "video", tmp_path / "frames")
