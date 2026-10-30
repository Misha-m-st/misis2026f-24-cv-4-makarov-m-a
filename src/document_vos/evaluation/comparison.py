"""Только парные сравнения одинаковых кадров и совместимых завершённых запусков."""

from pathlib import Path

from document_vos.cutie.inference import snapshot_inputs
from document_vos.data import first_mask, load_manifest
from document_vos.evaluation import aggregate
from document_vos.evaluation.metrics import METRIC_PROTOCOL, METRICS
from document_vos.io import (
    PipelineError,
    digest,
    new_directory,
    read_json,
    read_jsonl,
    sha256,
    write_json,
)
from document_vos.timing import speed_compatibility


def checked_evaluation(path: Path) -> dict:
    data = read_json(path)
    if data.get("status") != "COMPLETE" or data.get("metric_protocol") != METRIC_PROTOCOL:
        raise PipelineError("Нужна завершённая оценка с текущим протоколом метрик")
    run = data["run"]
    original_path = Path(data["run_dir"]) / "run.json"
    if sha256(original_path) != data["run_sha256"] or read_json(original_path) != run:
        raise PipelineError("Inference-run изменён после оценки")
    run_dir = Path(data["run_dir"]).resolve()
    index_path = run_dir / "predictions.jsonl"
    if sha256(index_path) != run["predictions_sha256"]:
        raise PipelineError("Индекс предсказаний изменён после оценки")
    for p in read_jsonl(index_path):
        prediction = (run_dir / p["path"]).resolve()
        if not prediction.is_relative_to(run_dir) or not prediction.is_file():
            raise PipelineError("Сохранённое предсказание отсутствует или имеет неверный путь")
        if sha256(prediction) != p["sha256"]:
            raise PipelineError("Сохранённое предсказание изменено после оценки")
    manifest = Path(run["manifest"])
    if sha256(manifest) != run["signature"]["manifest_sha256"]:
        raise PipelineError("Манифест изменён после оценки")
    sequences = load_manifest(manifest)
    if snapshot_inputs(sequences) != run["inputs"]:
        raise PipelineError("Данные изменены после оценки")
    expected = {
        (s.key, f.id) for s in sequences for f in s.frames[first_mask(s)[0] + 1 :] if f.has_gt
    }
    actual = {(r["video"], r["frame"]) for r in data["frames"]}
    if not actual or actual != expected or len(actual) != len(data["frames"]):
        raise PipelineError(
            "Оценка неполна: набор оценённых кадров не соответствует GT после инициализации"
        )
    if data["selection"]["evaluated_frames"] != len(actual):
        raise PipelineError("Число оценённых кадров противоречит покадровым результатам")
    return data


def compatible(a: dict, b: dict) -> None:
    if a["run"]["signature"] != b["run"]["signature"]:
        raise PipelineError(
            "Несовместимые runs: различаются входы, первые маски, seed, Cutie, scope, версии или параметры"
        )
    if a["metric_protocol"] != b["metric_protocol"] or a["selection"] != b["selection"]:
        raise PipelineError("Несовместимые протоколы/подмножества оценки")
    if a["run"].get("backend") != "official-cutie" or b["run"].get("backend") != "official-cutie":
        raise PipelineError("Исследовательское сравнение требует официальный Cutie backend")
    lineage = b["run"].get("training", {})
    reference_training = a["run"].get("training", {})
    reference_initial = reference_training.get("initial_weights_sha256", a["run"]["weights_sha256"])
    if lineage.get("initial_weights_sha256") != reference_initial:
        raise PipelineError(
            "Не подтверждено происхождение finetuned от baseline; infer требует --training-run"
        )
    # A previously finetuned model can be a reference when both runs descend
    # from the same base checkpoint and all paired signatures still match.


def compare(baseline: Path, finetuned: Path, output: Path) -> dict:
    a, b = checked_evaluation(baseline), checked_evaluation(finetuned)
    compatible(a, b)
    indexed = {(r["video"], r["frame"]): r for r in b["frames"]}
    paired = []
    for old in a["frames"]:
        new = indexed[(old["video"], old["frame"])]
        if any(old[k] != new[k] for k in ["index", "time_s", "nonempty_gt", "normalized_time"]):
            raise PipelineError("Покадровое сравнение имеет разные индексы/время/GT")
        row = {
            k: old[k]
            for k in [
                "dataset",
                "video",
                "document",
                "frame",
                "index",
                "time_s",
                "nonempty_gt",
                "normalized_time",
            ]
        }
        row.update({m: new[m] - old[m] for m in METRICS})
        paired.append(row)
    old_videos = {v["video"]: v for v in a["videos"]}
    delta_videos = []
    for new in b["videos"]:
        old = old_videos[new["video"]]
        row = {k: new[k] for k in ["dataset", "video", "document", "evaluated_frames"]}
        for k in [
            *METRICS,
            "J_decay",
            "F_decay",
            "BIoU_decay",
            "JF_std",
            "JF_adjacent_abs_change_mean",
        ]:
            row[k] = new[k] - old[k] if new[k] is not None and old[k] is not None else None
        delta_videos.append(row)
    result = {
        "status": "COMPLETE",
        "speed_compatibility": speed_compatibility({"baseline": a["run"], "finetuned": b["run"]}),
        "kind": "comparison",
        "scope": a["scope"],
        "selection": a["selection"],
        "metric_protocol": a["metric_protocol"],
        "baseline": {
            "path": str(baseline.resolve()),
            "sha256": sha256(baseline),
            "summary": a["summary"],
        },
        "finetuned": {
            "path": str(finetuned.resolve()),
            "sha256": sha256(finetuned),
            "summary": b["summary"],
        },
        "summary": aggregate(paired, delta_videos),
        "videos": delta_videos,
        "frames": paired,
        "baseline_curve": a["temporal_curve"],
        "finetuned_curve": b["temporal_curve"],
        "interpretation": "Все delta = finetuned − baseline. Положительный decay означает ухудшение внутри видео; уменьшение decay не доказывает накопление ошибки.",
    }
    # Контроль согласованности парных и независимо посчитанных dataset means.
    expected = {
        m: b["summary"]["primary_equal_datasets"][m] - a["summary"]["primary_equal_datasets"][m]
        for m in METRICS
    }
    if any(
        abs(result["summary"]["primary_equal_datasets"][m] - expected[m]) > 1e-12 for m in METRICS
    ):
        raise PipelineError("Нарушена согласованность парного усреднения")
    result["paired_delta_sha256"] = digest(paired)
    output = new_directory(output)
    write_json(output / "comparison.json", result)
    from document_vos.evaluation.reports import render_report

    render_report(result, output)
    return {
        "status": "COMPLETE",
        "delta": result["summary"]["primary_equal_datasets"],
        "output": str(output),
    }
