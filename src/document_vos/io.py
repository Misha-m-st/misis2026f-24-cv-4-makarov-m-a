"""Небольшие утилиты хранения. Результаты никогда не перезаписываются молча."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import yaml


class PipelineError(ValueError):
    """Понятная пользователю ошибка входных данных или протокола."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def mask_digest(mask) -> str:
    """Хеш размеров и бинарных байтов без большого промежуточного JSON."""
    h = hashlib.sha256(str(mask.shape).encode("ascii"))
    h.update(mask.astype("uint8").tobytes())
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected object")
                rows.append(row)
            except ValueError as e:
                raise PipelineError(f"{path}:{i}: неверный JSONL: {e}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def new_directory(path: Path) -> Path:
    if path.exists():
        raise PipelineError(f"Каталог уже существует: {path}. Выберите новую версию результата.")
    path.mkdir(parents=True)
    return path.resolve()


def versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in [
        "document-vos",
        "av",
        "numpy",
        "Pillow",
        "scipy",
        "matplotlib",
        "PyYAML",
        "hydra-core",
        "torch",
        "torchvision",
        "einops",
        "opencv-python-headless",
        "omegaconf",
        "thinplate",
        "tensorboard",
        "GitPython",
        "hickle",
        "tqdm",
        "pycocotools",
    ]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise PipelineError(f"{path}: конфигурация должна быть YAML-словарём")
    return cfg


def atomic_json(path: Path, value: Any) -> None:
    """Replace a status file atomically; immutable artifact writes use write_json."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)
