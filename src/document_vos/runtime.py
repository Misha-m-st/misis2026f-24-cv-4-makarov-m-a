"""Portable execution settings; experiment parameters live in their own config."""

import os
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from document_vos.io import PipelineError


@dataclass(frozen=True)
class Runtime:
    workers: int = 4
    io_threads: int = 2
    queue_size: int = 8
    threads_per_worker: int = 1
    dataloader_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = False
    device: str = "cuda"
    min_free_bytes: int = 0

    @classmethod
    def parse(cls, values: dict | None = None) -> "Runtime":
        values = {} if values is None else values
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise PipelineError("runtime: неизвестные поля или неверный YAML-словарь")
        result = cls(**values)
        for name in [
            "workers",
            "io_threads",
            "queue_size",
            "threads_per_worker",
            "prefetch_factor",
        ]:
            value = getattr(result, name)
            if type(value) is not int or value < 1:
                raise PipelineError(f"runtime.{name}: требуется положительное целое число")
        for name in ["dataloader_workers", "min_free_bytes"]:
            value = getattr(result, name)
            if type(value) is not int or value < 0:
                raise PipelineError(f"runtime.{name}: требуется неотрицательное целое число")
        for name in ["persistent_workers", "pin_memory"]:
            if type(getattr(result, name)) is not bool:
                raise PipelineError(f"runtime.{name}: требуется true/false")
        if result.device not in {"cpu", "cuda", "auto"}:
            raise PipelineError("runtime.device: используйте cpu, cuda или auto")
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    def loader_options(self, workers: int | None = None) -> dict:
        workers = self.dataloader_workers if workers is None else workers
        result = {"num_workers": workers, "pin_memory": self.pin_memory}
        if workers:
            result.update(
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                multiprocessing_context="spawn",
            )
        return result


_current = Runtime()


def current_runtime() -> Runtime:
    return _current


def require_free_space(path: Path, additional: int = 0) -> None:
    reserve = current_runtime().min_free_bytes
    if not reserve:
        return
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < reserve + additional:
        raise PipelineError(
            f"Недостаточно места в {path}: доступно {free}, "
            f"резерв {reserve}, ожидается {additional} байт"
        )


def configure(runtime: Runtime) -> None:
    """Call before importing numerical libraries or starting spawn workers."""
    global _current
    _current = runtime
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = str(runtime.threads_per_worker)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise PipelineError("CUDA запрошена явно, но недоступна")
    return requested


def install_loader_options(runtime: Runtime) -> None:
    """Configure the official loader without changing the pinned checkout."""
    from cutie.dataset import setup_training_data

    original = setup_training_data.DataLoader

    def loader(*args, **kwargs):
        kwargs.update(runtime.loader_options(kwargs.get("num_workers")))
        if kwargs["num_workers"] == 0:
            for name in ["prefetch_factor", "persistent_workers", "multiprocessing_context"]:
                kwargs.pop(name, None)
        return original(*args, **kwargs)

    setup_training_data.DataLoader = loader
