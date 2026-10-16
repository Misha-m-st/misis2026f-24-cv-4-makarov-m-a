"""Common overlays for ordinary inference and side-by-side comparisons; no GUI."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from document_vos.data import load_binary, load_manifest
from document_vos.io import PipelineError, read_json, read_jsonl, sha256, write_json
from document_vos.video import encode


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> Image.Image:
    result = rgb.copy()
    selected = mask.astype(bool)
    result[selected] = ((1 - alpha) * result[selected] + alpha * np.array([0, 255, 80])).astype(
        "uint8"
    )
    return Image.fromarray(result)


def render(run: Path, *, fps: float = 30, alpha: float = 0.35, other: Path | None = None) -> dict:
    if other:
        a, b = read_json(run / "run.json"), read_json(other / "run.json")
        if (
            a.get("status") != "COMPLETE"
            or b.get("status") != "COMPLETE"
            or a["signature"] != b["signature"]
        ):
            raise PipelineError("Парный рендер требует совместимые COMPLETE runs")
    sequences = load_manifest(run / "selected-manifest.jsonl")
    predictions = {(r["video"], r["frame"]): r for r in read_jsonl(run / "predictions.jsonl")}
    second = (
        {(r["video"], r["frame"]): r for r in read_jsonl(other / "predictions.jsonl")}
        if other
        else None
    )
    reports = []
    for seq in sequences:
        frames = [f for f in seq.frames if (seq.key, f.id) in predictions]
        if not frames:
            raise PipelineError(f"Нет предсказаний: {seq.key}")
        unknown = all(f.time_s is None for f in frames)
        if not unknown and any(f.time_s is None for f in frames):
            raise PipelineError("Частично отсутствующие timestamps")
        times = [n / fps for n in range(len(frames))] if unknown else [f.time_s for f in frames]

        def images():
            for f in frames:
                with Image.open(f.path) as source:
                    rgb = np.asarray(source.convert("RGB"))
                panels = [Image.fromarray(rgb)] if other else []
                for folder, rows, label in [
                    (run, predictions, "baseline" if other else "Cutie"),
                    (other, second, "finetuned"),
                ]:
                    if folder is None:
                        continue
                    row = rows.get((seq.key, f.id))
                    if row is None:
                        raise PipelineError("Парные предсказания имеют разный состав")
                    path = folder / row["path"]
                    if sha256(path) != row["sha256"]:
                        raise PipelineError(f"Маска изменилась: {path}")
                    panel = overlay(rgb, load_binary(path, (rgb.shape[1], rgb.shape[0])), alpha)
                    if other:
                        ImageDraw.Draw(panel).text(
                            (8, 8), label, fill="white", stroke_fill="black", stroke_width=1
                        )
                    panels.append(panel)
                canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height))
                offset = 0
                for panel in panels:
                    canvas.paste(panel, (offset, 0))
                    offset += panel.width
                yield canvas

        path = run / "videos" / f"{seq.export_id}.mp4"
        report = encode(path, images(), times, fps=fps)
        reports.append(
            {
                "video": seq.key,
                "path": str(path),
                "sha256": sha256(path),
                "timing": report,
                "time_source": seq.time_source,
                "visualization_fps": fps if unknown else None,
                "source_timestamps_available": not unknown,
            }
        )
    result = {"status": "COMPLETE", "videos": reports}
    write_json(run / "videos" / "render.json", result)
    return result


def bundle(
    run: Path, output: Path, *, other: Path | None = None, fps: float = 30, alpha: float = 0.35
) -> dict:
    import shutil

    from document_vos.io import new_directory

    new_directory(output)
    for name in ["selected-manifest.jsonl", "predictions.jsonl", "run.json"]:
        shutil.copy2(run / name, output / name)
    shutil.copytree(run / "predictions", output / "predictions")
    return render(output, other=other, fps=fps, alpha=alpha)
