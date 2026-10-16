"""Экспорт train-кадров с GT в формат официального VOSMergeTrainDataset."""

from pathlib import Path

import numpy as np
from PIL import Image

from document_vos.data import Frame, ground_truth, load_binary, load_manifest, validate_data
from document_vos.io import PipelineError, digest, new_directory, read_json, sha256, write_json
from document_vos.parallel import DEFAULT_WORKERS, Progress, check_workers, ordered_map
from document_vos.runtime import current_runtime, require_free_space
from document_vos.data.split import verify_training_split
from document_vos.data.validation_cache import ValidationCache, frame_signature


def _validate_export_frame(task: tuple[Path, Path, dict]) -> bool:
    image, mask, record = task
    if sha256(image) != record["image_sha256"] or sha256(mask) != record["mask_sha256"]:
        raise PipelineError(f"{image}: экспорт изменён")
    with Image.open(image) as im:
        if im.format != "JPEG" or im.mode != "RGB":
            raise PipelineError(f"{image}: ожидается RGB JPEG")
        gt = load_binary(mask, im.size)
    return bool(gt.any())


def _check_cached_export_task(task: tuple) -> tuple:
    video, signature, original = task
    present = _validate_export_frame(original)
    return video, signature, original, present


def _export_signature(task: tuple) -> str:
    image, mask, record = task
    return digest(
        [
            "training-export-validation/v1",
            frame_signature(image, mask, None),
            record["image_sha256"],
            record["mask_sha256"],
        ]
    )


def validate_export(
    root: Path,
    seq_length: int = 3,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: bool = False,
    refresh: bool = False,
) -> dict:
    check_workers(workers)
    metadata = read_json(root / "export.json")
    expected = {r["export_video"] for r in metadata["sequences"]}
    for kind in ["images", "masks"]:
        if {p.name for p in (root / kind).iterdir()} != expected:
            raise PipelineError(f"{kind}: каталог содержит лишние/недостающие видео")
    tasks, video_names = [], []
    for row in metadata["sequences"]:
        video = row["export_video"]
        records = row["frames"]
        if len(records) < seq_length:
            raise PipelineError(f"{video}: менее {seq_length} training-кадров с GT")
        names = [f"{i:06d}" for i in range(len(records))]
        if [r["name"] for r in records] != names:
            raise PipelineError(f"{video}: неверный порядок экспортных имён")
        if {p.name for p in (root / "images" / video).iterdir()} != {n + ".jpg" for n in names}:
            raise PipelineError(f"{video}: лишние/недостающие jpg")
        if {p.name for p in (root / "masks" / video).iterdir()} != {n + ".png" for n in names}:
            raise PipelineError(f"{video}: для каждого кадра нужна png-маска")
        for r in records:
            image = root / "images" / video / (r["name"] + ".jpg")
            mask = root / "masks" / video / (r["name"] + ".png")
            tasks.append((image, mask, r))
            video_names.append(video)
    if not tasks:
        raise PipelineError("Пустой training-экспорт")
    nonempty = {video: False for video in expected}
    bar = Progress("Проверка training-экспорта", len(tasks), workers, progress, endpoints_only=True)
    with ValidationCache(root / ".validation-cache.sqlite3") as cache:

        def jobs():
            for video, task in zip(video_names, tasks):
                signature = _export_signature(task)
                cached = None if refresh else cache.get(signature)
                if cached is None:
                    yield video, signature, task
                else:
                    if cached[0] != task[2]["image_sha256"] or cached[1] is None:
                        raise PipelineError("Неверная запись кеша training-экспорта")
                    nonempty[video] |= not cached[1]
                    bar.advance()

        for video, signature, task, present in ordered_map(
            _check_cached_export_task, jobs(), workers
        ):
            if _export_signature(task) != signature:
                raise PipelineError(f"Экспорт изменился во время проверки: {task[0]}")
            cache.put(signature, task[2]["image_sha256"], not present)
            nonempty[video] |= present
            bar.advance()
    for video, present in nonempty.items():
        if not present:
            raise PipelineError(f"{video}: нет объекта")
    return {"videos": len(expected), "frames": len(tasks)}


def _export_frame(task: tuple[Frame, Path, str, str, bool]) -> tuple[str, dict, bool]:
    frame, output, video, name, links = task
    image_path = output / "images" / video / (name + ".jpg")
    mask_path = output / "masks" / video / (name + ".png")
    with Image.open(frame.path) as im:
        gt = ground_truth(frame, im.size)
        nonempty = bool(gt.any())
        if links and im.format == "JPEG" and im.mode == "RGB":
            image_path.symlink_to(Path(frame.path).resolve())
        else:
            im.convert("RGB").save(image_path, quality=95, subsampling=0)
    if links and frame.mask and Path(frame.mask).suffix.lower() == ".png":
        mask_path.symlink_to(Path(frame.mask).resolve())
    else:
        Image.fromarray(np.asarray(gt, dtype=np.uint8)).save(mask_path)
    record = {
        "name": name,
        "source_id": frame.id,
        "source_path": frame.path,
        "time_s": frame.time_s,
        "image_sha256": sha256(image_path),
        "mask_sha256": sha256(mask_path),
    }
    return video, record, nonempty


def export_training(
    manifest: Path,
    split_dir: Path,
    output: Path,
    seq_length: int = 3,
    links: bool = True,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: bool = False,
) -> dict:
    check_workers(workers)
    if seq_length < 2:
        raise PipelineError("seq_length должен быть >= 2")
    verify_training_split(manifest, split_dir)
    sequences = load_manifest(manifest)
    validate_data(sequences)
    if max(sum(f.has_gt for f in s.frames) for s in sequences) < seq_length:
        raise PipelineError("Нет последовательностей с достаточным числом размеченных кадров")
    output = new_directory(output)
    metadata = {
        "manifest_sha256": sha256(manifest),
        "split_sha256": sha256(split_dir / "split.json"),
        "split_dir": str(split_dir.resolve()),
        "train_manifest": str(manifest.resolve()),
        "seq_length": seq_length,
        "sequences": [],
        "skipped_short": [],
    }
    tasks, exported, nonempty = [], {}, {}
    for seq in sequences:
        frames = [f for f in seq.frames if f.has_gt]
        if len(frames) < seq_length:
            metadata["skipped_short"].append(seq.key)
            continue
        video = seq.export_id
        for kind in ["images", "masks"]:
            (output / kind / video).mkdir(parents=True)
        row = {
            "source_video": seq.key,
            "export_video": video,
            "time_source": seq.time_source,
            "frames": [],
        }
        metadata["sequences"].append(row)
        exported[video] = row
        nonempty[video] = False
        for i, f in enumerate(frames):
            tasks.append((f, output, video, f"{i:06d}", links))
    bar = Progress("Экспорт кадров/масок", len(tasks), workers, progress)

    def guarded_tasks():
        runtime = current_runtime()
        width = min(workers * 2, runtime.queue_size)
        for start in range(0, len(tasks), width):
            wave = tasks[max(0, start - width) : start + width]
            if runtime.min_free_bytes:
                bound = 0
                for frame, _, _, _, links in wave:
                    with Image.open(frame.path) as image:
                        copied = not (links and image.format == "JPEG" and image.mode == "RGB")
                        bound += image.width * image.height * (4 if copied else 1) + 65536
                require_free_space(output, bound)
            yield from tasks[start : start + width]

    for video, record, present in ordered_map(_export_frame, guarded_tasks(), workers):
        exported[video]["frames"].append(record)
        nonempty[video] |= present
        bar.advance()
    for video, present in nonempty.items():
        if not present:
            raise PipelineError(f"{video}: нет объекта")
    write_json(output / "export.json", metadata)
    # Размеры/метки проверены при построении GT, имена заданы выше, хеши сохранены.
    # Не читаем только что созданный экспорт целиком второй раз.
    return {
        "videos": len(metadata["sequences"]),
        "frames": sum(len(row["frames"]) for row in metadata["sequences"]),
    }
