"""Адаптеры опубликованных CSV/JSON. Не угадываем связи документов по похожим именам."""

import csv
import gzip
from collections import defaultdict
from pathlib import Path

from document_vos.data import DATASETS, Frame, Sequence, natural_key, resolve_path, save_manifest, validate_data
from document_vos.io import PipelineError, read_json, read_jsonl, sha256, write_json

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def identity(dataset: str, video: str, document: str, template: str, mappings: dict) -> dict:
    result = {"document": document, "template": template, "group": document}
    if f"{dataset}/{video}" in mappings:
        result.update(mappings[f"{dataset}/{video}"])
    return result


def smartdoc(root: Path, spec: dict, mappings: dict) -> list[Sequence]:
    metadata = root / spec.get("metadata", "metadata.csv.gz")
    opener = gzip.open if metadata.suffix == ".gz" else open
    rows = defaultdict(list)
    with opener(metadata, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["bg_name"], r["model_name"], r["modeltype_name"])].append(r)
    sequences = []
    for (bg, doc, kind), items in sorted(rows.items()):
        video = f"{bg}/{doc}"
        frames = [
            Frame(
                id=str(r["frame_index"]),
                path=str((root / r["image_path"]).resolve()),
                polygon=[
                    [float(r[f"{c}_x"]), float(r[f"{c}_y"])] for c in ["tl", "bl", "br", "tr"]
                ],
            )
            for r in sorted(items, key=lambda r: int(r["frame_index"]))
        ]
        sequences.append(
            Sequence(
                "smartdoc2015",
                video,
                frames=frames,
                **identity(
                    "smartdoc2015", video, f"smartdoc:{doc}", f"smartdoc-type:{kind}", mappings
                ),
                provenance={
                    "annotation_source": str(metadata),
                    "presentation": "physical",
                    "origin": "SmartDoc2015",
                },
            )
        )
    return sequences


def via_polygons(path: Path, *, allow_empty: bool = False) -> dict[str, list | None]:
    """VIA 2: имя файла обязательно; текстовые регионы не считаются документом."""
    data = read_json(path)
    if (
        allow_empty
        and data.get("_via_img_metadata") == {}
        and data.get("_via_image_id_list", []) == []
    ):
        return {}
    data = data.get("_via_img_metadata", data)
    result = {}
    for r in data.values():
        if not isinstance(r, dict) or "filename" not in r:
            continue
        polygons = []
        regions = r.get("regions", [])
        regions = regions.values() if isinstance(regions, dict) else regions
        for region in regions:
            shape = region["shape_attributes"]
            if shape.get("name") == "polygon":
                x, y = shape["all_points_x"], shape["all_points_y"]
                if len(x) == len(y) == 4:
                    polygons.append([[float(a), float(b)] for a, b in zip(x, y)])
        if len(polygons) != 1:
            raise PipelineError(
                f"{path}/{r['filename']}: нужна ровно одна документная quad-разметка"
            )
        name = Path(r["filename"]).name
        if name in result:
            raise PipelineError(f"{path}: повтор разметки {name}")
        result[name] = polygons[0]
    if not result:
        raise PipelineError(f"{path}: не найдено VIA-разметки")
    return result


def frame_list(
    directory: Path, annotation: Path | None, gt_dir: Path | None, spec: dict
) -> list[Frame]:
    images = sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXT],
        key=lambda p: natural_key(p.name),
    )
    by_name = (
        via_polygons(annotation, allow_empty=spec.get("skip_empty_via") is True)
        if annotation
        else {}
    )
    if by_name.keys() - {p.name for p in images}:
        raise PipelineError(f"{annotation}: разметка ссылается на отсутствующие кадры")
    if gt_dir and gt_dir.is_dir():
        allowed = {name for p in images for name in [p.stem + ".json", p.name + ".json"]}
        if {p.name for p in gt_dir.glob("*.json")} - allowed:
            raise PipelineError(f"{gt_dir}: разметка ссылается на отсутствующие кадры")
    frames = []
    for image in images:
        polygon = by_name.get(image.name)
        if gt_dir:
            candidates = [gt_dir / (image.stem + ".json"), gt_dir / (image.name + ".json")]
            files = [p for p in candidates if p.is_file()]
            if len(files) > 1:
                raise PipelineError(f"Неоднозначная разметка {image}")
            if files:
                gt = read_json(files[0])
                if "quad" in gt:
                    polygon = gt["quad"]
                else:
                    templates = gt.get("document", {}).get("templates", {})
                    if len(templates) != 1:
                        raise PipelineError(f"{files[0]}: нужен один документный quad/template")
                    polygon = next(iter(templates.values())).get("template_quad")
                    if polygon is None:
                        raise PipelineError(f"{files[0]}: отсутствует документный template_quad")
        frames.append(Frame(image.stem, str(image.resolve()), polygon=polygon))
    # fps применяется только при явном указании; номера отражают пропуски кадров.
    fps = spec.get("fps")
    if fps is not None:
        if float(fps) <= 0:
            raise PipelineError("fps должен быть положительным")
        import re

        indices = [int(re.search(r"(\d+)$", p.stem).group(1)) for p in images]
        for f, i in zip(frames, indices):
            f.time_s = (i - indices[0]) / float(fps)
    return frames


def indexed(dataset: str, root: Path, spec: dict, mappings: dict) -> list[Sequence]:
    """Явный индекс для SIDTD/собственных данных и изменённых раскладок всех наборов."""
    result = []
    for row in read_jsonl(root / spec["index"]):
        row = dict(row)
        video = row.pop("video")
        provenance = row.pop("provenance", {})
        presentation = provenance.get("presentation")
        if dataset == "dlc2021" and presentation in {"screen", "re"}:
            continue
        source = str(provenance.get("source_dataset", "")).lower().replace("-", "")
        if dataset == "sidtd" and (presentation == "screen" or source == "midv2020"):
            continue
        ids = {k: row.pop(k) for k in ["document", "template", "group"]}
        ids.update(mappings.get(f"{dataset}/{video}", {}))
        if "frames" in row:
            frames = [Frame(**r) for r in row.pop("frames")]
            for f in frames:
                f.path = str(resolve_path(root, f.path))
                if f.mask:
                    f.mask = str(resolve_path(root, f.mask))
        else:
            directory = resolve_path(root, row.pop("images"))
            annotation = row.pop("annotation", None)
            gt_dir = row.pop("ground_truth", None)
            frames = frame_list(
                directory,
                resolve_path(root, annotation) if annotation else None,
                resolve_path(root, gt_dir) if gt_dir else None,
                {**spec, **row},
            )
        time_source = row.pop("time_source", "unknown")
        row.pop("fps", None)
        if row:
            raise PipelineError(f"{video}: неизвестные поля индекса: {sorted(row)}")
        result.append(
            Sequence(
                dataset, video, frames=frames, provenance=provenance, time_source=time_source, **ids
            )
        )
    return result


def _frame_list_task(task: tuple) -> list[Frame]:
    return frame_list(*task)


def midv(
    dataset: str, root: Path, spec: dict, mappings: dict, *, exclusions: list[dict] | None = None
) -> list[Sequence]:
    sequences = []
    frame_tasks = []
    if dataset == "midvholo":
        metadata = root / spec.get("metadata", "metadata")
        metadata_files = sorted(metadata.rglob("*.json")) if metadata.is_dir() else []
        if not metadata_files:
            raise PipelineError("MIDV-Holo требует готовые images, markup и metadata/*.json")
        metadata_hashes = {str(p): sha256(p) for p in metadata_files}
    # MIDV-500/2019: document/images/clip; LAIT: document/images.
    if dataset in {"midv500", "midv2019", "midvlait"}:
        image_roots = (
            sorted(root.rglob("*_images"))
            if dataset == "midvlait"
            else sorted(root.rglob("images"))
        )
        for image_root in image_roots:
            doc = image_root.parent.name
            if dataset != "midvlait":
                directories = sorted(p for p in image_root.iterdir() if p.is_dir())
                template = doc.split("_", 1)[-1]
            else:
                directories = [image_root]
                template = doc.rsplit("_", 1)[0]
            for directory in directories:
                video = str(directory.relative_to(root))
                gt_dir = (
                    image_root.parent / f"{doc}_groundtruth"
                    if dataset == "midvlait"
                    else image_root.parent / "ground_truth" / directory.relative_to(image_root)
                )
                family = "midv500" if dataset in {"midv500", "midv2019"} else "midvlait"
                frames = []
                frame_tasks.append((directory, None, gt_dir, spec))
                sequences.append(
                    Sequence(
                        dataset,
                        video,
                        frames=frames,
                        **identity(
                            dataset, video, f"{family}:{doc}", f"midv-template:{template}", mappings
                        ),
                        time_source=spec.get("time_source", "unknown"),
                        provenance={
                            "origin": dataset,
                            "presentation": "physical",
                            "annotation_source": str(gt_dir),
                        },
                    )
                )
    else:
        image_root = root / spec.get("images", "images")
        markup_root = root / spec.get(
            "annotations", "markup" if dataset == "midvholo" else "annotations"
        )
        directories = sorted(
            {
                p.parent
                for p in image_root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXT
            }
        )
        for directory in directories:
            rel = directory.relative_to(image_root)
            video = rel.as_posix()
            doc_type, clip = rel.parts[-2:]
            provenance = {"origin": dataset, "annotation_source": str(markup_root)}
            if dataset == "dlc2021":
                kind = clip.split(".")[-1][:2]
                if kind == "re":
                    continue
                if kind not in {"or", "cc", "cg"}:
                    raise PipelineError(f"{video}: неизвестный тип DLC")
                provenance["presentation"] = {
                    "or": "original",
                    "cc": "color_copy",
                    "cg": "gray_copy",
                }[kind]
                doc = clip.split(".")[0]
            else:
                provenance["presentation"] = "physical"
                doc = clip
            if dataset == "midvholo":
                # psp05_01_02: тип, экземпляр, клип; остальные каталоги входят в video.
                doc = "_".join(clip.split("_")[:-1])
                template = clip.split("_")[0]
                gt_dir = markup_root / rel
                frames = []
                frame_tasks.append((directory, None, gt_dir, spec))
                family = "midvholo"
                provenance["metadata_sha256"] = metadata_hashes
            else:
                # Только clips.tar; photo/scan каталоги не используются.
                template = doc_type
                family = "midv2020"
                annotation = markup_root / rel.parent / (rel.name + ".json")
                frames = []
                frame_tasks.append((directory, annotation, None, spec))
            sequences.append(
                Sequence(
                    dataset,
                    video,
                    frames=frames,
                    provenance=provenance,
                    time_source=spec.get("time_source", "unknown"),
                    **identity(
                        dataset,
                        video,
                        f"{family}:{doc_type}:{doc}",
                        f"midv-template:{template}",
                        mappings,
                    ),
                )
            )
    from document_vos.parallel import ordered_map
    from document_vos.runtime import current_runtime

    runtime = current_runtime()
    admitted = []
    for sequence, frames, task in zip(
        sequences,
        ordered_map(_frame_list_task, frame_tasks, runtime.io_threads, threads=True),
        frame_tasks,
    ):
        sequence.frames = frames
        if spec.get("skip_empty_via") is True and task[1] and not any(f.has_gt for f in frames):
            if exclusions is not None:
                exclusions.append(
                    {
                        "video": sequence.key,
                        "reason": "empty_published_via_metadata",
                        "frames": len(frames),
                        "annotation": str(task[1]),
                        "annotation_sha256": sha256(task[1]),
                    }
                )
        else:
            admitted.append(sequence)
    return admitted


def prepare(cfg: dict, config_dir: Path, output: Path, selected: list[str] | None = None) -> dict:
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise PipelineError("Манифест/метаданные уже существуют; используйте новую версию")
    selected = selected or cfg.get("selected", [])
    if not selected or len(set(selected)) != len(selected) or set(selected) - DATASETS:
        raise PipelineError("Укажите непустой список уникальных известных датасетов в selected")
    mappings = {}
    mapping_path = cfg.get("relationships")
    if mapping_path:
        mappings = read_json(resolve_path(config_dir, mapping_path))
        if any(set(v) - {"document", "template", "group"} for v in mappings.values()):
            raise PipelineError("Таблица связей содержит неизвестные поля")
    sequences = []
    source_hashes = {}
    archive_receipts = {}
    exclusions = []
    for dataset in selected:
        spec = cfg.get("datasets", {}).get(dataset)
        if not isinstance(spec, dict) or "root" not in spec:
            raise PipelineError(f"Выбранный {dataset}: не задан локальный root")
        if "skip_empty_via" in spec and type(spec["skip_empty_via"]) is not bool:
            raise PipelineError(f"{dataset}.skip_empty_via: требуется true/false")
        root = resolve_path(config_dir, spec["root"])
        if not root.is_dir():
            raise PipelineError(f"Выбранный {dataset}: каталог не найден: {root}")
        if spec.get("require_complete_archives", False):
            from document_vos.data.download import dataset_archives

            for archive in dataset_archives(dataset):
                relative = Path(archive.target).relative_to(f"raw/{dataset}")
                if relative.parts == (root.name,):
                    relative = Path(".")
                receipt_path = root / relative / ".archive-receipt.json"
                if not receipt_path.is_file():
                    raise PipelineError(f"Нормализация полного набора не завершена: {receipt_path}")
                receipt = read_json(receipt_path)
                if receipt.get("status") != "VERIFIED" or receipt.get("archive") != archive.archive:
                    raise PipelineError(f"Неверная квитанция нормализации: {receipt_path}")
                archive_receipts[str(receipt_path)] = sha256(receipt_path)
        try:
            if "index" in spec:
                added = indexed(dataset, root, spec, mappings)
                source_hashes[str(root / spec["index"])] = sha256(root / spec["index"])
            elif dataset == "smartdoc2015":
                added = smartdoc(root, spec, mappings)
            elif dataset in {"sidtd", "custom"}:
                raise PipelineError(
                    f"{dataset}: необходим явный JSONL index с подтверждённым происхождением"
                )
            else:
                added = midv(dataset, root, spec, mappings, exclusions=exclusions)
        except (OSError, KeyError, TypeError, IndexError, AttributeError) as e:
            raise PipelineError(f"{dataset}: ошибка опубликованного формата/разметки: {e}") from e
        if not added:
            raise PipelineError(f"Выбранный {dataset}: нет подходящих размеченных физических видео")
        sequences.extend(added)
    stats = validate_data(sequences)
    stats.update(
        excluded_videos=len(exclusions), excluded_frames=sum(r["frames"] for r in exclusions)
    )
    save_manifest(output, sequences)
    meta = {
        "config": cfg,
        "selected": selected,
        "stats": stats,
        "manifest_sha256": sha256(output),
        "source_indices": source_hashes,
        "archive_receipts": archive_receipts,
        "exclusions": exclusions,
        "relationships_sha256": sha256(resolve_path(config_dir, mapping_path))
        if mapping_path
        else None,
    }
    write_json(output.with_suffix(".meta.json"), meta)
    return stats
