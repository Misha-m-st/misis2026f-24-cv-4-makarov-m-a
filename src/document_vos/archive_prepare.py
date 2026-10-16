"""Resumable, verified local normalization. No downloads; deletion is opt-in."""

import hashlib
import io
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .download import Dataset, DownloadError, dataset_archives, verify_archive
from .io import PipelineError, digest, read_json, read_jsonl, sha256, write_json
from .parallel import Progress, ordered_map
from .runtime import Runtime

RECEIPT = ".archive-receipt.json"
PROGRESS = ".archive-progress.jsonl"
PLAN = ".archive-plan.json"
RULES = "lossless-normalization/v1"


def check_space(root: Path, reserve: int, pending: int = 0) -> None:
    free = shutil.disk_usage(root).free
    if free < reserve + pending:
        raise PipelineError(
            f"Недостаточно места: доступно {free}, резерв {reserve}, ожидается {pending} байт"
        )


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise PipelineError(f"Небезопасный путь архива: {name}")
    if any(p.startswith(".archive-") for p in path.parts):
        raise PipelineError(f"Зарезервированный путь архива: {name}")
    return Path(*path.parts)


def members(archive: Path, dataset: Dataset) -> list[dict]:
    """Metadata only. TARs must be uncompressed for independent offset readers."""
    result = []
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise PipelineError(f"Ссылка в ZIP запрещена: {member.filename}")
                relative = _relative(member.filename)
                # MIDV-500 includes the document prefix; MIDV-2019 does not.
                document = Path(dataset.target).name
                if relative.parts[0] == document:
                    relative = Path(*relative.parts[1:])
                if not relative.parts:
                    continue
                converted = relative.suffix.lower() in {".tif", ".tiff"}
                result.append(
                    {
                        "source": member.filename,
                        "path": str(relative.with_suffix(".png") if converted else relative),
                        "size": member.file_size,
                        "directory": member.is_dir(),
                        "convert": converted,
                    }
                )
    else:
        with tarfile.open(archive, "r:") as stream:
            for member in stream:
                relative = _relative(member.name)
                if not member.isfile() and not member.isdir():
                    raise PipelineError(f"Специальный TAR member запрещён: {member.name}")
                if dataset.root_directory:
                    if relative.parts[0] != dataset.root_directory:
                        raise PipelineError(f"Неожиданный корень TAR: {member.name}")
                    relative = Path(*relative.parts[1:])
                elif dataset.name == "dlc2021" and relative.parts[0] == "clips":
                    relative = Path(*relative.parts[1:])
                if not relative.parts:
                    continue
                result.append(
                    {
                        "source": member.name,
                        "path": str(relative),
                        "size": member.size,
                        "directory": member.isdir(),
                        "convert": False,
                        "offset": member.offset_data,
                    }
                )
    paths = [row["path"] for row in result]
    if len(paths) != len(set(paths)):
        raise PipelineError(f"Коллизия путей после нормализации: {archive}")
    files = {r["path"] for r in result if not r["directory"]}
    for path in paths:
        if any(str(parent) in files for parent in Path(path).parents):
            raise PipelineError(f"Конфликт файла/каталога: {path}")
    return result


def _pixel_hash(image: Image.Image) -> str:
    array = np.asarray(image)
    return digest(
        {
            "mode": image.mode,
            "size": image.size,
            "dtype": str(array.dtype),
            "pixels": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _source(archive: Path, row: dict):
    """The caller closes both handles; each worker owns its archive handle."""
    if archive.suffix == ".zip":
        owner = zipfile.ZipFile(archive)
        return owner, owner.open(row["source"])
    owner = archive.open("rb")
    owner.seek(row["offset"])
    return owner, owner


def _materialize(task: tuple) -> dict:
    archive, root, row = task
    path = root / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".archive-part")
    owner, stream = _source(archive, row)
    source_hash = hashlib.sha256()
    try:
        if row["convert"]:
            data = stream.read(row["size"])
            if len(data) != row["size"]:
                raise PipelineError(f"Неполный member: {row['source']}")
            source_hash.update(data)
            with Image.open(io.BytesIO(data)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise PipelineError(f"Многостраничный TIFF: {row['source']}")
                image.load()
                pixels = _pixel_hash(image)
                image.save(temporary, format="PNG")
            with Image.open(temporary) as image:
                if _pixel_hash(image) != pixels:
                    raise PipelineError(f"TIFF/PNG отличаются: {row['source']}")
        else:
            pixels = None
            remaining = row["size"]
            with temporary.open("wb") as target:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise PipelineError(f"Неполный member: {row['source']}")
                    target.write(block)
                    source_hash.update(block)
                    remaining -= len(block)
        with temporary.open("rb") as target:
            os.fsync(target.fileno())
        output_hash = sha256(temporary)
        if not row["convert"] and output_hash != source_hash.hexdigest():
            raise PipelineError(f"Ошибка копирования: {row['source']}")
        temporary.replace(path)
        return {
            **row,
            "source_sha256": source_hash.hexdigest(),
            "sha256": output_hash,
            "pixel_sha256": pixels,
            "output_size": path.stat().st_size,
        }
    finally:
        stream.close()
        owner.close()


def _verify_output(task: tuple) -> str:
    root, row = task
    path = root / row["path"]
    if path.is_symlink() or sha256(path) != row["sha256"]:
        raise PipelineError(f"Сохранённый файл изменён: {path}")
    if row["convert"]:
        with Image.open(path) as image:
            if _pixel_hash(image) != row["pixel_sha256"]:
                raise PipelineError(f"Пиксели изменены: {path}")
    return row["path"]


def verify_receipt(root: Path, receipt: dict, runtime: Runtime) -> None:
    if receipt.get("rules") != RULES or receipt.get("status") != "VERIFIED":
        raise PipelineError(f"Неверная квитанция: {root}")
    rows = receipt["files"]
    expected = {r["path"] for r in rows}
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.name not in {RECEIPT, PROGRESS, PLAN}
    }
    if actual != expected:
        raise PipelineError(f"Состав сохранённых файлов не совпадает: {root}")
    list(ordered_map(_verify_output, ((root, r) for r in rows), runtime.workers))


def _normalize_archive(
    archive: Path,
    dataset: Dataset,
    target: Path,
    runtime: Runtime,
    *,
    delete_verified_zip: bool = False,
) -> dict:
    """An archive is committed atomically; only a verified ZIP may be deleted."""
    if target.exists():
        receipt = read_json(target / RECEIPT)
        if receipt["archive"] != dataset.archive or receipt["source_url"] != dataset.url:
            raise PipelineError(f"Квитанция относится к другому архиву: {target}")
        verify_receipt(target, receipt, runtime)
        if archive.exists():
            fingerprint = verify_archive(archive, dataset)
            if fingerprint != receipt["archive_sha256"]:
                raise PipelineError(f"Архив изменён: {archive}")
    else:
        fingerprint = verify_archive(archive, dataset)
        inventory = members(archive, dataset)
        staging = target.with_name(target.name + ".archive-staging")
        plan = {
            "rules": RULES,
            "archive_sha256": fingerprint,
            "inventory_sha256": digest(inventory),
        }
        if staging.exists():
            if read_json(staging / PLAN) != plan:
                raise PipelineError(f"Несовместимое возобновление: {staging}")
        else:
            staging.mkdir(parents=True)
            write_json(staging / PLAN, plan)
        progress_path = staging / PROGRESS
        if progress_path.exists():
            # A process crash may leave only the last JSON line incomplete.
            # Durable complete rows remain authoritative; unfinished jobs are redone.
            with progress_path.open("rb+") as journal:
                data = journal.read()
                if data and not data.endswith(b"\n"):
                    journal.truncate(data.rfind(b"\n") + 1)
                    journal.flush()
                    os.fsync(journal.fileno())
        completed = read_jsonl(progress_path) if progress_path.exists() else []
        done = {r["path"]: r for r in completed}
        if len(done) != len(completed):
            raise PipelineError(f"Повторная запись в журнале: {staging}")
        for row in inventory:
            path = staging / row["path"]
            if row["directory"]:
                path.mkdir(parents=True, exist_ok=True)
            elif row["path"] in done:
                saved = done[row["path"]]
                if any(saved[k] != v for k, v in row.items()):
                    raise PipelineError(f"Журнал не соответствует архиву: {row['path']}")
        list(ordered_map(_verify_output, ((staging, r) for r in completed), runtime.workers))
        pending = [r for r in inventory if not r["directory"] and r["path"] not in done]
        bar = Progress(dataset.archive, len(pending), runtime.workers, True, endpoints_only=True)
        with (staging / PROGRESS).open("a", encoding="utf-8") as journal:
            import json

            # Reserve space for a bounded wave before submitting it. This avoids
            # racing disk-space checks in individual workers.
            width = min(runtime.queue_size, runtime.workers * 2)

            def tasks():
                for start in range(0, len(pending), width):
                    wave = pending[start : start + width]
                    # Account for unfinished jobs from the preceding wave too.
                    preceding = pending[max(0, start - width) : start]
                    check_space(
                        staging,
                        runtime.min_free_bytes,
                        sum(r["size"] * 2 + 65536 for r in wave + preceding),
                    )
                    yield from ((archive, staging, r) for r in wave)

            for row in ordered_map(_materialize, tasks(), runtime.workers, queue_size=width):
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                completed.append(row)
                bar.advance()
        expected = {r["path"] for r in inventory if not r["directory"]}
        if {r["path"] for r in completed} != expected:
            raise PipelineError(f"Неполное сохранение архива: {archive}")
        receipt = {
            **plan,
            "status": "VERIFIED",
            "archive": dataset.archive,
            "source_url": dataset.url,
            "author_checksum": dataset.checksum,
            "author_algorithm": dataset.algorithm,
            "runtime": runtime.to_dict(),
            "files": sorted(completed, key=lambda r: r["path"]),
            "directories": [r["path"] for r in inventory if r["directory"]],
        }
        verify_receipt(staging, receipt, runtime)
        if sha256(archive) != fingerprint:
            raise PipelineError(f"Архив изменился во время обработки: {archive}")
        write_json(staging / RECEIPT, receipt)
        with (staging / RECEIPT).open("rb") as stream:
            os.fsync(stream.fileno())
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target)
        # Flush directory entries before deleting the only original archive.
        if hasattr(os, "O_DIRECTORY"):
            directories = [p for p in target.rglob("*") if p.is_dir()]
            for directory in [
                *sorted(directories, key=lambda p: len(p.parts), reverse=True),
                target,
                target.parent,
            ]:
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
    deleted = False
    if delete_verified_zip and archive.suffix == ".zip" and archive.exists():
        archive.unlink()
        deleted = True
    return {
        "archive": dataset.archive,
        "status": "VERIFIED",
        "deleted": deleted,
        "target": str(target),
        "files": len(receipt["files"]),
    }


def normalize_archive(
    archive: Path,
    dataset: Dataset,
    target: Path,
    runtime: Runtime,
    *,
    delete_verified_zip: bool = False,
) -> dict:
    from .locking import exclusive_lock

    with exclusive_lock(target.with_name(target.name + ".archive-lock")):
        staging = target.with_name(target.name + ".archive-staging")
        for root in [target, staging]:
            if root.is_symlink() or (
                root.exists() and any(p.is_symlink() for p in root.rglob("*"))
            ):
                raise PipelineError(f"Ссылка в каталоге нормализации запрещена: {root}")
        try:
            return _normalize_archive(
                archive, dataset, target, runtime, delete_verified_zip=delete_verified_zip
            )
        except (DownloadError, tarfile.TarError, zipfile.BadZipFile) as error:
            raise PipelineError(str(error)) from error


def prepare_archives(
    archives: Path,
    output: Path,
    selected: list[str],
    runtime: Runtime,
    *,
    execute: bool = False,
    delete_verified_zip: bool = False,
) -> dict:
    if type(delete_verified_zip) is not bool:
        raise PipelineError("delete_verified_zip: требуется true/false")
    if (
        not selected
        or len(selected) != len(set(selected))
        or set(selected) - {"midv500", "midv2019", "midvholo", "dlc2021"}
    ):
        raise PipelineError("Выберите уникальные midv500/midv2019/midvholo/dlc2021")
    results = []
    for name in selected:
        for dataset in dataset_archives(name):
            target = output / Path(dataset.target).relative_to("raw")
            if not execute:
                results.append(
                    {
                        "archive": str(archives / dataset.archive),
                        "target": str(target),
                        "exists": (archives / dataset.archive).is_file(),
                        "delete_after_verification": delete_verified_zip
                        and dataset.archive.endswith(".zip"),
                    }
                )
            else:
                results.append(
                    normalize_archive(
                        archives / dataset.archive,
                        dataset,
                        target,
                        runtime,
                        delete_verified_zip=delete_verified_zip,
                    )
                )
    return {
        "status": "COMPLETE" if execute else "PLANNED",
        "archives": results,
        "runtime": runtime.to_dict(),
    }
