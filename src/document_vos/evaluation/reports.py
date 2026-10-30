"""CSV, PNG и русскоязычный Markdown; графики не скрывают отсутствие данных."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from document_vos.io import PipelineError, new_directory, read_json, write_json


def csv_file(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "NOT_MEASURED"


def render_report(data: dict, output: Path) -> None:
    comparison = data.get("kind") == "comparison"
    run_info = data.get("run", {})
    model_name = run_info.get("model") or (
        "Cutie"
        if run_info.get("backend", "official-cutie") == "official-cutie"
        else run_info["backend"]
    )
    csv_file(output / "frames.csv", data["frames"])
    csv_file(output / "videos.csv", data["videos"])
    per_video = defaultdict(list)
    for r in data["frames"]:
        per_video[r["video"]].append(r)
    (output / "video_curves").mkdir()
    from document_vos.io import digest

    for name, rows in per_video.items():
        csv_file(output / "video_curves" / f"{digest(name)[:16]}.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    curves = (
        [("baseline", data["baseline_curve"]), ("finetuned", data["finetuned_curve"])]
        if comparison
        else [("BIoU", data["temporal_curve"])]
    )
    for label, curve in curves:
        x = [(r["start"] + r["end"]) / 2 for r in curve]
        ax.plot(
            x,
            [r["BIoU"] if r["BIoU"] is not None else float("nan") for r in curve],
            marker="o",
            label=label,
        )
        for r, pos in zip(curve, x):
            if r["BIoU"] is not None:
                ax.annotate(
                    f"n={r['videos']}",
                    (pos, r["BIoU"]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7,
                )
        csv_file(output / (label + "_temporal.csv"), curve)
    ax.set(
        xlabel="Normalized duration (timestamps or declared frame index)",
        ylabel="Boundary IoU",
        xlim=(0, 1),
        ylim=(0, 1.12),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "temporal.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    values = [v["JF"] for v in data["videos"] if v["JF"] is not None]
    ax.bar(range(len(values)), values)
    ax.set(xlabel="Video index (see videos.csv)", ylabel="Delta J&F" if comparison else "J&F")
    if comparison:
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output / "videos.png", dpi=160)
    plt.close(fig)
    selected = data["selection"]
    geometric_pseudo_labels = any(
        item.get("kind") == "geometric_pseudo_labels"
        for item in selected.get("annotation_sources", [])
    )
    scope = (
        "независимых собственных видео"
        if data["scope"] == "independent"
        else "выбранного validation-подмножества"
    )
    lines = [
        "# " + ("Парное сравнение Cutie" if comparison else f"Оценка {model_name}"),
        "",
        f"Результат {scope}. Датасеты: {', '.join(selected['datasets'])}.",
        *(
            [
                "",
                "Оценка согласия с автоматической геометрической псевдоразметкой. Контур под окклюзией восстановлен по предположению о плоском листе; ручная проверка масок не проводилась.",
            ]
            if geometric_pseudo_labels
            else []
        ),
        "",
        f"Документов: {len(selected['documents'])}; видео: {len(selected['videos'])}; оценённых кадров: {selected['evaluated_frames']}; кадров без GT после инициализации: {selected['missing_gt']}.",
        *(
            [
                "",
                "Покрытие запланированной автоматической разметки после первого кадра:",
                "",
                *[
                    f"- {item['video']}: {item['accepted_after_initial']}/{item['sampled_after_initial']}; неудачных разметок: {item['failed_after_initial']}."
                    for item in selected["annotation_coverage"]
                ],
                "",
                "Кадры с неудачной разметкой переданы Cutie, но исключены из метрик. Пропуск сложных кадров ограничивает выводы об устойчивости модели.",
            ]
            if selected.get("annotation_coverage")
            else []
        ),
        "",
        "Основное усреднение: кадры → равное среднее видео датасета → равное среднее выбранных датасетов. Значения в [0,1]; первый кадр с заданной маской исключён.",
        "",
        "| Политика | J | F | J&F | BIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("Основная" + (" delta" if comparison else ""), "primary_equal_datasets"),
        ("Все видео поровну", "equal_all_videos"),
        ("Все кадры", "all_frames"),
        ("Непустая GT, основная", "nonempty_gt_primary"),
    ]:
        values = data["summary"][key]
        lines.append(
            f"| {label} | " + " | ".join(fmt(values[m]) for m in ["J", "F", "JF", "BIoU"]) + " |"
        )
    lines.extend(
        [
            "",
            "Полные списки документов и видео: JSON; покадровые значения, время, std J&F, соседние изменения и decay: CSV.",
            "",
            "Decay считается по четырём интервалам DAVIS: начало минус конец. Положительное значение означает ухудшение. Для кривой BIoU используются 10 интервалов нормализованной длительности; n — число видео в интервале. При отсутствии timestamps используется индекс кадра; секунды не восстанавливаются по догадке.",
            "",
            data["interpretation"],
            "",
            "![BIoU по времени](temporal.png)",
            "",
            "![J&F по видео](videos.png)",
        ]
    )
    if comparison:
        speed = data.get("speed_compatibility", {"status": "INCOMPATIBLE", "participants": {}})
        csv_file(
            output / "speed-compatibility.csv",
            [
                {
                    "model": label,
                    "status": speed["participants"].get(label, {}).get("status", "INCOMPATIBLE"),
                    "reasons": "; ".join(
                        speed["participants"]
                        .get(label, {})
                        .get("reasons", ["missing speed evidence"])
                    ),
                }
                for label in ["baseline", "finetuned"]
            ],
        )
        lines.extend(
            [
                "",
                f"Совместимость скорости: {speed['status']}. Несовместимые FPS не используются для сравнительных выводов; совместимость качества проверяется отдельно.",
            ]
        )
        for label in ["baseline", "finetuned"]:
            values = data[label]["summary"]["primary_equal_datasets"]
            lines.extend(
                ["", f"{label}: " + ", ".join(f"{m}={fmt(values[m])}" for m in values) + "."]
            )
    else:
        run = data["run"]
        timing = run["timing"]
        from document_vos.timing import speed_compatibility

        speed = speed_compatibility({model_name: run})
        csv_file(
            output / "timing.csv",
            [
                {
                    **timing,
                    "speed_compatibility": speed["status"],
                    "speed_incompatibility_reasons": "; ".join(
                        speed["participants"][model_name]["reasons"]
                    ),
                }
            ],
        )
        lines.extend(
            [
                "",
                f"Измеренные FPS модели: {fmt(timing['model_fps'])}; полного прохода: {fmt(timing['full_fps'])}. Совместимость скорости: {speed['status']}. Несовместимые FPS не используются для сравнительных выводов; метрики качества сохраняют силу.",
                "",
                f"Таймер: {timing.get('protocol_version', 'UNKNOWN')}; границы: {timing.get('definition', 'UNKNOWN')}. Инициализация включена, дополнительного warmup нет.",
                "",
                f"VRAM: {run['vram']}. Модель: `{model_name}`; backend revision: "
                f"`{run.get('backend_revision', run.get('cutie_revision'))}`. Seed: {run['seed']}.",
                "",
                f"SHA-256 весов: `{run['weights_sha256']}`; манифеста: `{run['signature']['manifest_sha256']}`.",
            ]
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(source: Path, output: Path) -> dict:
    data = read_json(source)
    if data.get("status") != "COMPLETE":
        raise PipelineError(
            "Нельзя формировать исследовательский отчёт из незавершённого результата"
        )
    output = new_directory(output)
    write_json(output / "source.json", data)
    render_report(data, output)
    return {"output": str(output), "status": "COMPLETE"}
