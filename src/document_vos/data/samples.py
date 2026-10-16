"""Fixed validation snippets, verified before extraction; never training downloads."""

import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from document_vos.assets import check_asset, fetch
from document_vos.config import DATASETS, resource
from document_vos.data import load_manifest, save_manifest, validate_data
from document_vos.io import PipelineError, read_json, sha256, write_json
from document_vos.locking import exclusive_lock


def catalog(path: Path | None = None) -> dict:
    import json

    result = read_json(path) if path else json.loads(resource("samples.json").read_text())
    if result.get("schema") != "document-vos/samples-v1" or not isinstance(
        result.get("samples"), list
    ):
        raise PipelineError("Неизвестный каталог примеров")
    seen = set()
    for entry in result["samples"]:
        check_asset(entry)
        if (type(entry.get('unpacked_bytes')) is not int or entry['unpacked_bytes'] < 1
                or not isinstance(entry.get('sequence'), str)
                or not entry['sequence'].startswith(entry.get('dataset', '') + '/')):
            raise PipelineError('Каталог: неверный распакованный размер/sequence')
        name = entry.get("id", "")
        if (
            not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or name in seen
            or entry.get("dataset") not in DATASETS
        ):
            raise PipelineError("Неверный/повторный ID или датасет в каталоге")
        seen.add(name)
    return result


def select(cat: dict, dataset: str, count: int) -> list[dict]:
    choices = [e for e in cat["samples"] if e["dataset"] == dataset]
    if type(count) is not int or not 1 <= count <= len(choices) or count > 2:
        raise PipelineError(
            f"{dataset}: доступны {len(choices)} фиксированных примера; count от 1 до 2"
        )
    return choices[:count]


def verify(directory: Path, entry: dict) -> None:
    receipt = read_json(directory / "receipt.json")
    if receipt.get("asset_sha256") != entry["sha256"]:
        raise PipelineError(f"Пример не соответствует каталогу: {directory}")
    for name, expected in receipt["files"].items():
        path = safe_path(name)
        full = directory / path
        if not full.is_file() or sha256(full) != expected:
            raise PipelineError(f"Повреждён пример: {full}")
    rows = load_manifest(directory / "manifest.jsonl")
    if len(rows) != 1 or rows[0].key != entry["sequence"]:
        raise PipelineError("Состав примера не совпадает с каталогом")
    for seq in rows:
        for frame in seq.frames:
            for name in [frame.path, frame.mask]:
                if name and not Path(name).is_relative_to(directory.resolve()):
                    raise PipelineError("Манифест примера выходит за его каталог")
    validate_data(rows)
    from document_vos.data import first_mask
    index, _ = first_mask(rows[0])
    if index != 0 or any(f.has_gt for f in rows[0].frames[1:]):
        raise PipelineError('Пример должен содержать только первую GT-маску')


def safe_path(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts or ":" in name:
        raise PipelineError(f"Небезопасный путь пакета: {name}")
    return Path(*path.parts)


def unpack(archive: Path, directory: Path, entry: dict) -> None:
    if directory.exists():
        verify(directory, entry)
        return
    with tempfile.TemporaryDirectory(prefix=".sample-", dir=directory.parent) as temp:
        stage = Path(temp)
        with zipfile.ZipFile(archive) as stream:
            names = set()
            total = 0
            for info in stream.infolist():
                relative = safe_path(info.filename)
                if relative.as_posix() in names or stat.S_ISLNK(info.external_attr >> 16):
                    raise PipelineError("Повторный путь или ссылка в пакете")
                names.add(relative.as_posix())
                total += info.file_size
                if total > entry["unpacked_bytes"]:
                    raise PipelineError("Пакет превышает объявленный распакованный размер")
                target = stage / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(info) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest)
        if total != entry["unpacked_bytes"]:
            raise PipelineError("Неверный распакованный размер")
        hashes = {str(p.relative_to(stage)): sha256(p) for p in stage.rglob("*") if p.is_file()}
        write_json(stage / "receipt.json", {"asset_sha256": entry["sha256"], "files": hashes})
        verify(stage, entry)
        stage.rename(directory)


def download(directory: Path, cat: dict, dataset: str, count: int, *, plan: bool = False) -> dict:
    entries = select(cat, dataset, count)
    result = {
        "status": "PLANNED" if plan else "COMPLETE",
        "samples": entries,
        "download_bytes": sum(e["bytes"] for e in entries),
        "output": str(directory),
        "scope": "validation subset; not an independent quality test",
    }
    if plan:
        return result
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory / ".samples.lock"):
        for entry in entries:
            target = directory / entry["id"]
            if target.exists():
                verify(target, entry)
                continue
            archive = fetch(entry, directory / "archives" / f"{entry['id']}.zip")
            unpack(archive, target, entry)
    return result


def manifest(directory: Path, target: Path, cat: dict, dataset: str, count: int) -> Path:
    sequences = []
    for entry in select(cat, dataset, count):
        source = directory / entry["id"]
        if not source.exists():
            raise PipelineError(
                f"Нет примера {entry['id']}: выполните document-vos download --sample "
                f"--dataset {dataset} --count {count}"
            )
        verify(source, entry)
        sequences.extend(load_manifest(source / "manifest.jsonl"))
    save_manifest(target, sequences)
    return target
