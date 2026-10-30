"""Portable execution settings; experiment parameters live in their own config."""

import os
import shutil
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path

from document_vos.io import PipelineError


@dataclass(frozen=True)
class Runtime:
    cpu_profile: str = "p6-smt12"
    workers: int = 6
    io_threads: int = 4
    queue_size: int = 8
    threads_per_worker: int = 1
    dataloader_workers: int = 6
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
        if result.cpu_profile != "p6-smt12":
            raise PipelineError("runtime.cpu_profile: требуется p6-smt12")
        if result.threads_per_worker != 1:
            raise PipelineError("p6-smt12: threads_per_worker должен быть 1")
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
                worker_init_fn=initialize_loader_worker,
            )
        return result


_current = Runtime()
_cores: list[dict] = []
_blas_controller = None


def select_cpu_cores(available: set[int], topology: Path = Path("/sys/devices/system/cpu")):
    """Select six complete two-way SMT cores, never partial sibling groups."""
    cores = {}
    try:
        for cpu in sorted(available):
            root = topology / f"cpu{cpu}" / "topology"
            siblings = set()
            for part in (root / "thread_siblings_list").read_text().strip().split(","):
                bounds = [int(n) for n in part.split("-")]
                siblings.update(range(bounds[0], bounds[-1] + 1))
            if len(siblings) != 2 or not siblings <= available:
                continue
            package = int((root / "physical_package_id").read_text())
            core = int((root / "core_id").read_text())
            cores[(package, core)] = {"package": package, "core": core, "cpus": sorted(siblings)}
    except (OSError, ValueError) as exc:
        raise PipelineError(f"p6-smt12: невозможно прочитать топологию CPU: {exc}") from exc
    selected = sorted(cores.values(), key=lambda item: item["cpus"])[:6]
    if len(selected) != 6 or len({cpu for item in selected for cpu in item["cpus"]}) != 12:
        raise PipelineError(
            f"p6-smt12: нужны 6 полных SMT-пар (12 CPU); доступны {len(cores)}; "
            f"affinity={sorted(available)}. Профиль автоматически не заменяется."
        )
    return selected


def configure_environment() -> None:
    """Inherited by spawn before numerical libraries are imported."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"


def configure_libraries(*, worker: bool = False) -> dict:
    """Set actual pools, including libraries imported before configure()."""
    import cv2
    import numpy  # noqa: F401 - load BLAS before applying the controller
    import torch
    from threadpoolctl import threadpool_limits

    global _blas_controller
    target = 1 if worker else 6
    torch.set_num_threads(target)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise PipelineError(
                "p6-smt12: нельзя установить PyTorch inter-op=1; "
                "запустите новый процесс до начала вычислений"
            ) from exc
    cv2.setNumThreads(1)
    _blas_controller = threadpool_limits(limits=1, user_api="blas")
    pools = library_snapshot()
    if (pools["torch_intra_op"], pools["torch_inter_op"], pools["opencv"]) != (target, 1, 1) or any(
        pool["num_threads"] != 1 for pool in pools["blas"]
    ):
        raise PipelineError(f"p6-smt12: библиотечные пулы не настроены: {pools}")
    return pools


def library_snapshot() -> dict:
    """Read actual settings without repairing them while collecting evidence."""
    import cv2
    import torch
    from threadpoolctl import threadpool_info

    pools = {
        "torch_intra_op": torch.get_num_threads(),
        "torch_inter_op": torch.get_num_interop_threads(),
        "opencv": cv2.getNumThreads(),
        "blas": [
            {key: pool[key] for key in ("internal_api", "prefix", "num_threads")}
            for pool in threadpool_info()
            if pool["user_api"] == "blas"
        ],
    }
    return pools


def initialize_loader_worker(worker_id: int, *, original=None) -> None:
    configure_environment()
    configure_libraries(worker=True)
    if original is not None:
        original(worker_id)


def runtime_snapshot() -> dict:
    """Measured execution evidence, not just requested YAML values."""
    return {
        "requested": current_runtime().to_dict(),
        "cpu_profile": current_runtime().cpu_profile,
        "affinity": sorted(os.sched_getaffinity(0)),
        "physical_cores": _cores,
        "pools": library_snapshot(),
        "workers": current_runtime().workers,
        "dataloader_workers": current_runtime().dataloader_workers,
        "io_threads": current_runtime().io_threads,
    }


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
    global _current, _cores
    Runtime.parse(runtime.to_dict())
    if not hasattr(os, "sched_getaffinity"):
        raise PipelineError("p6-smt12 требует Linux CPU affinity и SMT-топологию")
    cores = select_cpu_cores(set(os.sched_getaffinity(0)))
    cpus = {cpu for core in cores for cpu in core["cpus"]}
    try:
        # Also cover pools imported by an embedding process before this call.
        for thread in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(thread.name), cpus)
            except ProcessLookupError:
                pass  # A thread exited between enumeration and sched_setaffinity.
        if set(os.sched_getaffinity(0)) != cpus:
            raise OSError("affinity не совпадает с выбранными CPU")
    except OSError as exc:
        raise PipelineError(f"p6-smt12: невозможно установить CPU affinity: {exc}") from exc
    _current, _cores = runtime, cores
    configure_environment()
    configure_libraries()


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
    original = getattr(original, "_document_vos_original", original)

    def loader(*args, **kwargs):
        original_init = kwargs.get("worker_init_fn")
        kwargs.update(runtime.loader_options())
        if kwargs["num_workers"]:
            kwargs["worker_init_fn"] = partial(initialize_loader_worker, original=original_init)
        if kwargs["num_workers"] == 0:
            for name in ["prefetch_factor", "persistent_workers", "multiprocessing_context"]:
                kwargs.pop(name, None)
        return original(*args, **kwargs)

    loader._document_vos_original = original
    setup_training_data.DataLoader = loader
