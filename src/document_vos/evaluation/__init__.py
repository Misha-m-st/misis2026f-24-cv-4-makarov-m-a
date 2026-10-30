"""Покадровая оценка и три явно названные политики усреднения."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from document_vos.cutie.inference import snapshot_inputs
from document_vos.data import Sequence, first_mask, ground_truth, load_binary, load_manifest
from document_vos.evaluation.metrics import METRIC_PROTOCOL, METRICS, decay, mask_metrics
from document_vos.io import (
    PipelineError,
    digest,
    mask_digest,
    new_directory,
    read_json,
    read_jsonl,
    sha256,
    write_json,
)
from document_vos.parallel import DEFAULT_WORKERS, Progress, check_workers, ordered_map
from document_vos.timing import speed_compatibility


def means(rows: list[dict], fields=METRICS) -> dict:
    return {
        k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        if any(r[k] is not None for r in rows)
        else None
        for k in fields
    }


def aggregate(rows: list[dict], videos: list[dict]) -> dict:
    datasets = defaultdict(list)
    for v in videos:
        if v["evaluated_frames"]:
            datasets[v["dataset"]].append(v)
    by_dataset = {d: means(v) for d, v in sorted(datasets.items())}
    nonempty = [r for r in rows if r["nonempty_gt"]]
    n_videos = []
    for v in videos:
        selected = [r for r in nonempty if r["video"] == v["video"]]
        if selected:
            n_videos.append({"dataset": v["dataset"], **means(selected)})
    n_datasets = defaultdict(list)
    for v in n_videos:
        n_datasets[v["dataset"]].append(v)
    return {
        "primary_equal_datasets": means(list(by_dataset.values())),
        "by_dataset": by_dataset,
        "equal_all_videos": means([v for v in videos if v["evaluated_frames"]]),
        "all_frames": means(rows),
        "nonempty_gt_primary": means([means(v) for v in n_datasets.values()]),
        "nonempty_gt_frames": len(nonempty),
        "decay_equal_datasets": means(
            [means(v, ["J_decay", "F_decay", "BIoU_decay"]) for v in datasets.values()],
            ["J_decay", "F_decay", "BIoU_decay"],
        ),
    }


def temporal_curve(rows: list[dict], bins: int = 10) -> list[dict]:
    """В каждом интервале сначала усредняем кадры видео, затем видео датасета и датасеты."""
    result = []
    for index in range(bins):
        selected = [r for r in rows if min(bins - 1, int(r["normalized_time"] * bins)) == index]
        per_video = defaultdict(list)
        for r in selected:
            per_video[(r["dataset"], r["video"])].append(r["BIoU"])
        per_dataset = defaultdict(list)
        for (dataset, video), values in per_video.items():
            per_dataset[dataset].append(float(np.mean(values)))
        result.append(
            {
                "bin": index,
                "start": index / bins,
                "end": (index + 1) / bins,
                "BIoU": float(np.mean([np.mean(v) for v in per_dataset.values()]))
                if selected
                else None,
                "videos": len(per_video),
                "frames": len(selected),
                "dataset_video_counts": {d: len(v) for d, v in per_dataset.items()},
            }
        )
    return result


def _evaluate_video(task: tuple[Sequence, dict, dict | None, Path]) -> tuple:
    seq, indexed, initialization, run_dir = task
    expected = set()
    initial_index, initial_mask = first_mask(seq)
    init = initialization
    if (
        not init
        or init["index"] != initial_index
        or init["frame"] != seq.frames[initial_index].id
        or init["mask_sha256"] != mask_digest(initial_mask)
    ):
        raise PipelineError(f"{seq.key}: первая маска не соответствует протоколу")
    if seq.frames[0].time_s is not None:
        start_t, end_t = seq.frames[initial_index].time_s, seq.frames[-1].time_s

        def coordinate(i):
            return (seq.frames[i].time_s - start_t) / (end_t - start_t) if end_t > start_t else 0.0

        time_basis = seq.time_source
    else:

        def coordinate(i):
            return (i - initial_index) / max(1, len(seq.frames) - 1 - initial_index)

        time_basis = "frame_index (timestamps unavailable)"
    video_rows, missing_gt = [], 0
    for i in range(initial_index, len(seq.frames)):
        f = seq.frames[i]
        key = (seq.key, f.id)
        expected.add(key)
        if key not in indexed:
            raise PipelineError(f"Нет предсказания {seq.key}/{f.id}, включая кадры без GT")
        p = indexed[key]
        if p["index"] != i or p["initialization"] != (i == initial_index):
            raise PipelineError("Индекс/инициализация предсказания не соответствует кадру")
        pred_path = (run_dir / p["path"]).resolve()
        if not pred_path.is_relative_to(run_dir.resolve()):
            raise PipelineError("Предсказание должно находиться внутри run-каталога")
        if not pred_path.is_file() or sha256(pred_path) != p["sha256"]:
            raise PipelineError(f"Предсказание отсутствует или изменено: {pred_path}")
        with Image.open(f.path) as im:
            pred = load_binary(pred_path, im.size)
            gt = ground_truth(f, im.size)
        if i == initial_index:
            continue
        if gt is None:
            missing_gt += 1
            continue
        r = {
            "dataset": seq.dataset,
            "video": seq.key,
            "document": seq.document,
            "frame": f.id,
            "index": i,
            "time_s": f.time_s,
            "time_basis": time_basis,
            "normalized_time": coordinate(i),
            "nonempty_gt": bool(gt.any()),
            **mask_metrics(pred, gt),
        }
        previous = video_rows[-1] if video_rows else None
        r["JF_adjacent_change"] = (
            r["JF"] - previous["JF"] if previous and previous["index"] == i - 1 else None
        )
        r["BIoU_adjacent_change"] = (
            r["BIoU"] - previous["BIoU"] if previous and previous["index"] == i - 1 else None
        )
        video_rows.append(r)
    adjacent = [
        b["JF"] - a["JF"]
        for a, b in zip(video_rows, video_rows[1:])
        if b["index"] == a["index"] + 1
    ]
    v = {
        "dataset": seq.dataset,
        "video": seq.key,
        "document": seq.document,
        "evaluated_frames": len(video_rows),
        "missing_gt_after_init": missing_gt,
        "skipped_before_init": initial_index,
        "initialized_frame": seq.frames[initial_index].id,
        **means(video_rows),
        "JF_std": float(np.std([r["JF"] for r in video_rows])) if video_rows else None,
        "JF_adjacent_change_mean": float(np.mean(adjacent)) if adjacent else None,
        "JF_adjacent_abs_change_mean": float(np.mean(np.abs(adjacent))) if adjacent else None,
        "adjacent_pairs": len(adjacent),
    }
    v.update({k + "_decay": decay([r[k] for r in video_rows]) for k in ["J", "F", "BIoU"]})
    return v, video_rows, expected


def evaluate(
    run_dir: Path, output: Path, *, workers: int = DEFAULT_WORKERS, progress: bool = False
) -> dict:
    check_workers(workers)
    if output.exists():
        raise PipelineError(f"Каталог уже существует: {output}. Выберите новую версию результата.")
    run = read_json(run_dir / "run.json")
    if run.get("status") != "COMPLETE":
        raise PipelineError("Оценка принимает только COMPLETE inference-run")
    manifest = Path(run["manifest"])
    if sha256(manifest) != run["signature"]["manifest_sha256"]:
        raise PipelineError("Манифест изменён после инференса")
    sequences = load_manifest(manifest)
    if (
        snapshot_inputs(sequences) != run["inputs"]
        or digest(run["inputs"]) != run["signature"]["inputs_sha256"]
    ):
        raise PipelineError("Исходные кадры/GT изменены после инференса")
    if sha256(run_dir / "predictions.jsonl") != run["predictions_sha256"]:
        raise PipelineError("Индекс предсказаний изменён")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    indexed = {(p["video"], p["frame"]): p for p in predictions}
    if len(indexed) != len(predictions):
        raise PipelineError("Повтор предсказания")
    rows, videos, expected = [], [], set()
    initializations = {r["video"]: r for r in run["signature"]["initializations"]}
    tasks = (
        (
            seq,
            {
                (seq.key, f.id): indexed[(seq.key, f.id)]
                for f in seq.frames
                if (seq.key, f.id) in indexed
            },
            initializations.get(seq.key),
            run_dir,
        )
        for seq in sequences
    )
    bar = Progress("Оценка видео", len(sequences), workers, progress)
    for video, video_rows, video_expected in ordered_map(_evaluate_video, tasks, workers):
        videos.append(video)
        rows.extend(video_rows)
        expected.update(video_expected)
        bar.advance()
    if expected != indexed.keys():
        raise PipelineError("Есть лишние предсказания вне выбранных кадров")
    unmeasured = sorted({s.dataset for s in sequences} - {r["dataset"] for r in rows})
    if unmeasured:
        raise PipelineError(
            f"Нет оценённых кадров после инициализации в выбранных датасетах: {unmeasured}"
        )
    result = {
        "status": "COMPLETE",
        "scope": run["scope"],
        "run_sha256": sha256(run_dir / "run.json"),
        "run_dir": str(run_dir.resolve()),
        "run": run,
        "speed_compatibility": speed_compatibility(
            {run.get("model", run.get("backend", "official-cutie")): run}
        ),
        "metric_protocol": METRIC_PROTOCOL,
        "selection": {
            "datasets": sorted({s.dataset for s in sequences}),
            "documents": sorted({s.document for s in sequences}),
            "videos": [s.key for s in sequences],
            "evaluated_frames": len(rows),
            "videos_without_evaluated_gt": [
                v["video"] for v in videos if not v["evaluated_frames"]
            ],
            "missing_gt": sum(v["missing_gt_after_init"] for v in videos),
            "annotation_sources": run.get("selection", {}).get("annotation_sources", []),
            "annotation_coverage": run.get("selection", {}).get("annotation_coverage", []),
        },
        "summary": aggregate(rows, videos),
        "videos": videos,
        "frames": rows,
        "temporal_curve": temporal_curve(rows),
        "interpretation": "Падение качества во времени не доказывает накопление ошибки без контрольного эксперимента",
    }
    output = new_directory(output)
    write_json(output / "evaluation.json", result)
    from document_vos.evaluation.reports import render_report

    render_report(result, output)
    return {"status": "COMPLETE", "evaluated_frames": len(rows), "output": str(output)}
