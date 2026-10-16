"""Единый JSONL: одна строка — одно видео, кадры уже упорядочены."""

import math
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import quad_mask, validate_quad
from .io import PipelineError, digest, read_jsonl, sha256, write_jsonl
from .parallel import DEFAULT_WORKERS, Progress, check_workers, ordered_map
from .validation_cache import ValidationCache, file_signature, frame_signature

DATASETS = {
    "smartdoc2015",
    "midv500",
    "midv2019",
    "midv2020",
    "midvlait",
    "midvholo",
    "dlc2021",
    "sidtd",
    "custom",
}


@dataclass
class Frame:
    id: str
    path: str
    time_s: float | None = None
    polygon: list[list[float]] | None = None
    mask: str | None = None

    @property
    def has_gt(self) -> bool:
        return self.polygon is not None or self.mask is not None


@dataclass
class Sequence:
    dataset: str
    video: str
    document: str
    template: str
    group: str
    frames: list[Frame]
    provenance: dict = field(default_factory=dict)
    time_source: str = "unknown"
    schema_version: int = 1

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.video}"

    @property
    def export_id(self) -> str:
        return f"{self.dataset}_{digest(self.key)[:16]}"

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: Path) -> list[Sequence]:
    sequences = []
    for row in read_jsonl(path):
        try:
            row = dict(row)
            row["frames"] = [Frame(**f) for f in row["frames"]]
            seq = Sequence(**row)
            for f in seq.frames:
                f.path = str(resolve_path(path.parent, f.path))
                if f.mask:
                    f.mask = str(resolve_path(path.parent, f.mask))
            sequences.append(seq)
        except (TypeError, KeyError) as e:
            raise PipelineError(f"Неверная схема манифеста {path}: {e}") from e
    if not sequences:
        raise PipelineError(f"Пустой манифест: {path}")
    return sequences


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return (root / p).resolve() if not p.is_absolute() else p.resolve()


def save_manifest(path: Path, sequences: list[Sequence]) -> None:
    write_jsonl(path, [s.to_dict() for s in sequences])


def load_binary(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not ((array == 0) | (array == 1)).all():
        raise PipelineError(f"Маска {path} должна иметь один канал и значения {{0, 1}}")
    if size and array.shape != (size[1], size[0]):
        raise PipelineError(f"Размер маски {path} {array.shape} не совпадает с кадром {size}")
    return array.astype(np.uint8)


def ground_truth(frame: Frame, size: tuple[int, int]) -> np.ndarray | None:
    if frame.mask:
        return load_binary(Path(frame.mask), size)
    if frame.polygon is not None:
        return quad_mask(frame.polygon, *size)
    return None


def first_mask(seq: Sequence) -> tuple[int, np.ndarray]:
    for i, f in enumerate(seq.frames):
        if f.has_gt:
            with Image.open(f.path) as image:
                gt = ground_truth(f, image.size)
            if gt is not None and gt.any():
                return i, gt
    raise PipelineError(f"{seq.key}: нет размеченной непустой маски для инициализации")


def _validate_frame(task: tuple) -> tuple:
    """Рабочий процесс получает пути; SQLite остаётся в родительском процессе."""
    video, frame, check_content, signature = task
    image = Path(frame.path)
    mask = Path(frame.mask) if frame.mask else None
    before = (file_signature(image), file_signature(mask) if mask else None)
    h = sha256(image) if check_content else None
    with Image.open(image) as im:
        im.load()
        gt = ground_truth(frame, im.size)
    empty = bool(not gt.any()) if gt is not None else None
    after = (file_signature(image), file_signature(mask) if mask else None)
    if before != after:
        raise PipelineError(f"{image}: файл изменился во время проверки")
    return video, frame, signature, h, empty


def validate_data(
    sequences: list[Sequence],
    check_content: bool = True,
    *,
    full: bool = False,
    cache_path: Path | None = None,
    refresh: bool = False,
    workers: int = DEFAULT_WORKERS,
    progress: bool = False,
) -> dict:
    """По умолчанию только структура. Полная проверка явно включается через full.

    Дорогие задачи параллельны; кеш, поиск дублей и счётчики ведёт один родитель.
    """
    check_workers(workers)
    stats = _validate_structure(sequences)
    if not full:
        return stats
    stats.update(
        empty_gt=0,
        repeated_frames_within_video=0 if check_content else None,
        validation="full",
        content_checked=check_content,
    )
    context = (
        ValidationCache(cache_path)
        if check_content and cache_path is not None
        else nullcontext(None)
    )
    hashes = {}
    nonempty = {seq.key: False for seq in sequences}
    bar = Progress("Полная проверка кадров", stats["frames"], workers, progress)

    def accept(video, frame, h, empty, cached):
        if check_content:
            if h in hashes and hashes[h] != video:
                raise PipelineError(f"Дублирующийся кадр между {hashes[h]} и {video}: {frame.path}")
            if h in hashes:
                stats["repeated_frames_within_video"] += 1
            hashes[h] = video
        if empty is not None:
            stats["empty_gt"] += int(empty)
            nonempty[video] |= not empty
        stats["cached_frames" if cached else "checked_frames"] += 1
        bar.advance()

    with context as cache:

        def jobs():
            for seq in sequences:
                for frame in seq.frames:
                    image = Path(frame.path)
                    mask = Path(frame.mask) if frame.mask else None
                    signature = frame_signature(image, mask, frame.polygon) if cache else None
                    result = cache.get(signature) if cache and not refresh else None
                    if result is not None:
                        accept(seq.key, frame, *result, cached=True)
                    else:
                        yield seq.key, frame, check_content, signature

        for video, frame, signature, h, empty in ordered_map(_validate_frame, jobs(), workers):
            if cache:
                image = Path(frame.path)
                mask = Path(frame.mask) if frame.mask else None
                if frame_signature(image, mask, frame.polygon) != signature:
                    raise PipelineError(f"{image}: файл изменился во время проверки")
                cache.put(signature, h, empty)
            accept(video, frame, h, empty, cached=False)
    for video, present in nonempty.items():
        if not present:
            raise PipelineError(f"{video}: нет размеченной непустой маски для инициализации")
    return stats


def _validate_structure(sequences: list[Sequence]) -> dict:
    keys, paths = set(), {}
    stats = {
        "videos": len(sequences),
        "frames": 0,
        "missing_gt": 0,
        "empty_gt": None,
        "repeated_frames_within_video": None,
        "validation": "structural",
        "content_checked": False,
        "cached_frames": 0,
        "checked_frames": 0,
    }
    for seq in sequences:
        if seq.schema_version != 1 or seq.dataset not in DATASETS:
            raise PipelineError(f"{seq.key}: неизвестная версия схемы или датасет")
        if seq.key in keys:
            raise PipelineError(f"Повтор видео: {seq.key}")
        keys.add(seq.key)
        if not all(
            isinstance(v, str) and v.strip()
            for v in [seq.video, seq.document, seq.template, seq.group]
        ):
            raise PipelineError(f"{seq.key}: обязательны document/template/group")
        if not seq.frames:
            raise PipelineError(f"{seq.key}: нет кадров")
        if seq.dataset == "sidtd":
            p = seq.provenance
            if (
                p.get("presentation") != "physical"
                or not p.get("annotation_source")
                or not p.get("origin")
                or not p.get("source_dataset")
            ):
                raise PipelineError(
                    f"{seq.key}: SIDTD требует physical, origin, source_dataset и annotation_source"
                )
            if str(p.get("source_dataset", "")).lower().replace("-", "") == "midv2020":
                raise PipelineError(f"{seq.key}: повтор MIDV-2020 в SIDTD запрещён")
        if seq.dataset == "dlc2021" and seq.provenance.get("presentation") not in {
            "original",
            "color_copy",
            "gray_copy",
        }:
            raise PipelineError(f"{seq.key}: DLC допускает только оригиналы и физические копии")
        ids = set()
        times = [f.time_s for f in seq.frames]
        if any(t is None for t in times) and not all(t is None for t in times):
            raise PipelineError(
                f"{seq.key}: время должно быть задано для всех кадров или ни одного"
            )
        if times[0] is not None:
            if seq.time_source == "unknown" or any(not math.isfinite(t) or t < 0 for t in times):
                raise PipelineError(f"{seq.key}: неверное время или неизвестный источник времени")
            if any(a >= b for a, b in zip(times, times[1:])):
                raise PipelineError(f"{seq.key}: время кадров должно строго возрастать")
        annotated = 0
        for f in seq.frames:
            if not isinstance(f.id, str) or not f.id or f.id in ids:
                raise PipelineError(f"{seq.key}: повтор или пустой id кадра {f.id}")
            ids.add(f.id)
            if f.mask and f.polygon is not None:
                raise PipelineError(f"{seq.key}/{f.id}: укажите маску или polygon, не оба")
            p = Path(f.path)
            if not p.is_file():
                raise PipelineError(f"{seq.key}/{f.id}: кадр не найден: {p}")
            resolved = str(p.resolve())
            if resolved in paths:
                raise PipelineError(f"Один файл кадра включён повторно: {p}")
            paths[resolved] = seq.key
            mask = Path(f.mask) if f.mask else None
            if mask is not None and not mask.is_file():
                raise PipelineError(f"{seq.key}/{f.id}: маска не найдена: {mask}")
            if f.polygon is not None:
                validate_quad(f.polygon)
            stats["frames"] += 1
            annotated += int(f.has_gt)
            stats["missing_gt"] += int(not f.has_gt)
        if annotated == 0:
            raise PipelineError(f"{seq.key}: нет необходимой разметки")
    stats["datasets"] = sorted({s.dataset for s in sequences})
    stats["documents"] = len({s.document for s in sequences})
    return stats


def natural_key(value: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", value)]
