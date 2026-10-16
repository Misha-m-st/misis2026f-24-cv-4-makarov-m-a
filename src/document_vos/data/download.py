"""Явная загрузка опубликованных архивов; стандартная библиотека Python и curl."""

import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


class DownloadError(Exception):
    """Ошибка загрузки, целостности или безопасной распаковки."""


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    archive: str
    size: int
    algorithm: str
    checksum: str | None
    target: str
    layout: tuple[str, ...] = ("images", "annotations")
    root_directory: str | None = None


DATASETS = {
    "smartdoc2015": Dataset(
        "smartdoc2015",
        "https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0/frames.tar.gz",
        "smartdoc-frames.tar.gz",
        1019404933,
        "sha256",
        "3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae",
        "raw/smartdoc2015",
    ),
    "midv2020": Dataset(
        "midv2020",
        "ftp://smartengines.com/midv-2020/dataset/clips.tar",
        "midv2020-clips.tar",
        10303651840,
        "md5",
        "752dc7770ef555d4e16273045c2f1726",
        "raw/midv2020/clips",
    ),
}
MARKER = ".document-vos-download.json"
SUPPORTED_DATASETS = ("smartdoc2015", "midv2020", "midv2019", "midvholo", "midv500", "dlc2021")


def dataset_archives(name: str) -> list[Dataset]:
    if name in DATASETS:
        return [DATASETS[name]]
    if name in {"midv2019", "midv500"}:
        with (
            files("document_vos.resources")
            .joinpath(f"{name}-archives.csv")
            .open(newline="") as stream
        ):
            return [
                Dataset(
                    name,
                    row["url"],
                    f"{name}-{row['archive']}",
                    int(row["bytes"]),
                    "md5",
                    row["md5"],
                    f"raw/{name}/{Path(row['archive']).stem}",
                    layout=("images", "ground_truth"),
                )
                for row in csv.DictReader(stream)
            ]
    if name == "midvholo":
        return [
            Dataset(
                name,
                f"ftp://smartengines.com/midv-holo/{part}.tar",
                f"holo-{part}.tar",
                size,
                "sha256",
                None,
                f"raw/midvholo/{part}",
                root_directory=part,
            )
            for part, size in [("images", 14166804480), ("markup", 31324160), ("metadata", 51200)]
        ]
    if name == "dlc2021":
        return [
            Dataset(
                name,
                "ftp://smartengines.com/dlc-2021/clips.tar",
                "dlc-clips.tar",
                17768312320,
                "md5",
                "0758a65d3ddccdd24eba25a98e4ba3c6",
                "raw/dlc2021/clips",
            )
        ]
    raise DownloadError(f"Неизвестный набор: {name}")


def log(message: str) -> None:
    print(message, flush=True)


def require_space(directory: Path, needed: int) -> None:
    free = shutil.disk_usage(directory).free
    if free < needed:
        raise DownloadError(
            f"Недостаточно места в {directory}: нужно ещё {needed / 1e9:.2f} ГБ, "
            f"свободно {free / 1e9:.2f} ГБ. Используйте --data-dir на другом диске."
        )


def verify_archive(path: Path, dataset: Dataset) -> str:
    size = path.stat().st_size
    if size != dataset.size:
        raise DownloadError(f"{path}: размер {size} байт вместо ожидаемых {dataset.size}")
    log(f"[{dataset.name}] Проверяю {dataset.algorithm.upper()} ({size / 1e9:.3f} ГБ)…")
    expected_hash = hashlib.new(dataset.algorithm)
    sha = hashlib.sha256() if dataset.algorithm != "sha256" else expected_hash
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            expected_hash.update(block)
            if sha is not expected_hash:
                sha.update(block)
    if (
        dataset.checksum is not None
        and expected_hash.hexdigest().lower() != dataset.checksum.lower()
    ):
        raise DownloadError(
            f"{path}: контрольная сумма НЕ совпадает. Файл сохранён; распаковка запрещена. "
            "Переместите повреждённый файл и повторите запуск для новой загрузки."
        )
    if dataset.checksum is not None:
        log(f"[{dataset.name}] Контрольная сумма: OK")
    else:
        log(f"[{dataset.name}] Размер: OK; SHA-256 сохранён. Сумма автора не опубликована.")
    return sha.hexdigest()


def download_archive(dataset: Dataset, archives: Path) -> tuple[Path, str]:
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / dataset.archive
    partial = archive.with_name(archive.name + ".part")
    if archive.exists():
        if not archive.is_file():
            raise DownloadError(f"Вместо архива найден каталог: {archive}")
        if archive.stat().st_size == dataset.size:
            fingerprint = verify_archive(archive, dataset)
            log(f"[{dataset.name}] Архив уже скачан; повторная загрузка не нужна.")
            return archive, fingerprint
        if archive.stat().st_size > dataset.size:
            raise DownloadError(
                f"{archive}: файл больше ожидаемого архива; он оставлен без изменений"
            )
        if partial.exists():
            raise DownloadError(
                f"Одновременно найдены неполные {archive} и {partial}; объединять их нельзя"
            )
        # Подхватываем также частичный файл от прежней ручной команды curl.
        archive.rename(partial)
    if partial.exists() and (not partial.is_file() or partial.stat().st_size > dataset.size):
        raise DownloadError(f"Неверный частичный архив: {partial}. Файл сохранён.")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset < dataset.size:
        require_space(archives, dataset.size - offset)
        log(f"[{dataset.name}] {dataset.url}")
        log(
            f"[{dataset.name}] Загрузка: {offset / 1e9:.3f} / {dataset.size / 1e9:.3f} ГБ → {partial}"
        )
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--continue-at",
            "-",
            "--progress-bar",
            "--output",
            str(partial),
            dataset.url,
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise DownloadError(
                f"curl завершился с кодом {result.returncode}. Частичный файл сохранён: {partial}. "
                "Повторите ту же команду скрипта, чтобы продолжить загрузку."
            )
    fingerprint = verify_archive(partial, dataset)
    partial.rename(archive)
    return archive, fingerprint


def ready_layout(root: Path, dataset: Dataset) -> bool:
    if dataset.name == "smartdoc2015":
        return (root / "metadata.csv.gz").is_file()
    if dataset.root_directory is not None:
        return root.name == dataset.root_directory and any(root.iterdir())
    return all((root / directory).is_dir() for directory in dataset.layout)


def find_dataset_root(staging: Path, dataset: Dataset) -> Path:
    # Некоторые tar-релизы добавляют clips/ или frames/; нормализуем до root из starter.yaml.
    pending = [(staging, 0)]
    matches = []
    while pending:
        path, depth = pending.pop()
        if ready_layout(path, dataset):
            matches.append(path)
            continue
        if depth < 3:
            pending.extend((p, depth + 1) for p in path.iterdir() if p.is_dir())
    if len(matches) != 1:
        raise DownloadError(
            f"{dataset.name}: не найден единственный ожидаемый корень данных в архиве"
        )
    return matches[0]


def extract_archive(dataset: Dataset, archive: Path, data_dir: Path, fingerprint: str) -> Path:
    target = data_dir / dataset.target
    marker = target / MARKER
    if marker.is_file():
        receipt = json.loads(marker.read_text(encoding="utf-8"))
        if (
            receipt.get("archive_sha256") == fingerprint
            and receipt.get("dataset") == dataset.name
            and ready_layout(target, dataset)
        ):
            log(f"[{dataset.name}] Уже распакован: {target}")
            return target
        raise DownloadError(
            f"{target}: отметка распаковки/структура отличается; существующие данные сохранены"
        )
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise DownloadError(
            f"{target}: уже содержит данные без отметки скрипта. Они не будут перезаписаны. "
            "Используйте --download-only или отдельный --data-dir."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{dataset.name}] Проверяю содержимое архива перед распаковкой…")
    staging = None
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zipped:
                members = zipped.infolist()
                for member in members:
                    path = Path(member.filename)
                    mode = member.external_attr >> 16
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or "\\" in member.filename
                        or (mode & 0o170000) not in {0, 0o040000, 0o100000}
                    ):
                        raise DownloadError(f"Небезопасный путь/ссылка/тип: {member.filename}")
                require_space(target.parent, sum(m.file_size for m in members))
                staging = Path(
                    tempfile.mkdtemp(prefix=f".{target.name}-unpack-", dir=target.parent)
                )
                log(f"[{dataset.name}] Распаковка → {target}…")
                zipped.extractall(staging)
                file_count = sum(not m.is_dir() for m in members)
        else:
            staging, file_count = extract_tar(archive, target, dataset)
        root = find_dataset_root(staging, dataset)
        receipt = {
            "dataset": dataset.name,
            "url": dataset.url,
            "archive_sha256": fingerprint,
            "archive_bytes": dataset.size,
            "source_checksum_verified": dataset.checksum is not None,
            "files": file_count,
        }
        (root / MARKER).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if target.exists():
            target.rmdir()  # Только ранее проверенный пустой каталог.
        root.rename(target)
        if staging.exists():
            # Сохраняем внешние README/LICENSE и папки обёртки, если root был вложенным.
            if any(staging.iterdir()):
                staging.rename(target / ".archive-extras")
            else:
                staging.rmdir()
        log(f"[{dataset.name}] Готово: {target}")
        return target
    finally:
        if staging is not None and staging.exists():
            # Удаляется только временная распаковка, созданная этим запуском.
            shutil.rmtree(staging)


def extract_tar(archive: Path, target: Path, dataset: Dataset) -> tuple[Path, int]:
    staging = None
    try:
        with tarfile.open(archive, "r:*") as tar:
            members = tar.getmembers()
            for member in members:
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise DownloadError(f"Небезопасный путь/ссылка/тип в архиве: {member.name}")
            require_space(target.parent, sum(m.size for m in members if m.isfile()))
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-unpack-", dir=target.parent))
            log(f"[{dataset.name}] Распаковка → {target} (может занять несколько минут)…")
            # Все пути проверены; ссылки/спецфайлы запрещены, staging новый и пустой.
            filters = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(staging, members=members, **filters)
        return staging, sum(m.isfile() for m in members)
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
