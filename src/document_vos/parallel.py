"""Небольшой пул CPU-процессов с ограниченной очередью и стабильным порядком."""

import multiprocessing
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import monotonic
from typing import Callable, Iterable, Iterator, TypeVar

from document_vos.io import PipelineError

DEFAULT_WORKERS = 6
T = TypeVar("T")
R = TypeVar("R")


def check_workers(workers: int) -> None:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise PipelineError("workers должен быть целым числом >= 1; 1 — последовательный режим")


def _initialize_worker(threads: int = 1) -> None:
    from document_vos.runtime import configure_environment, configure_libraries

    configure_environment()
    configure_libraries(worker=True)


def ordered_map(
    function: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    *,
    queue_size: int | None = None,
    threads: bool = False,
) -> Iterator[R]:
    """Передаём пути/координаты, не изображения. В очереди не более 2×workers задач.

    Spawn не наследует SQLite-соединения и состояние CUDA. При workers=1
    функция выполняется в текущем процессе; результаты всегда в порядке входа.
    """
    check_workers(workers)
    from document_vos.runtime import current_runtime

    runtime = current_runtime()
    queue_size = min(workers * 2, runtime.queue_size) if queue_size is None else queue_size
    check_workers(queue_size)
    if workers == 1 and not threads:
        for item in items:
            yield function(item)
        return
    iterator = iter(items)
    try:
        first = next(iterator)
    except StopIteration:
        return
    if threads:
        pool = ThreadPoolExecutor(max_workers=workers)
    else:
        # Environment is inherited before spawn imports numpy/torch.
        from document_vos.runtime import configure_environment

        configure_environment()
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(runtime.threads_per_worker,),
        )
    pending = deque()
    try:
        pending.append(pool.submit(function, first))
        for _ in range(queue_size - 1):
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(pool.submit(function, next(iterator)))
            except StopIteration:
                pass
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


class Progress:
    """Прогресс/скорость/ETA в stderr: JSON-результат stdout остаётся чистым."""

    def __init__(
        self, label: str, total: int, workers: int, enabled: bool, *, endpoints_only: bool = False
    ):
        self.label, self.total, self.enabled = label, total, enabled
        self.done = 0
        self.endpoints_only = endpoints_only
        self.start = self.last = monotonic()
        if enabled:
            print(f"{label}: 0/{total}; workers={workers}", file=sys.stderr, flush=True)

    def advance(self) -> None:
        self.done += 1
        now = monotonic()
        if self.enabled and (
            self.done == self.total or (not self.endpoints_only and now - self.last >= 2)
        ):
            speed = self.done / max(now - self.start, 1e-9)
            remaining = (self.total - self.done) / speed
            print(
                f"{self.label}: {self.done}/{self.total} "
                f"({100 * self.done / max(1, self.total):.1f}%); "
                f"{speed:.1f}/с; осталось ≈{remaining / 60:.1f} мин",
                file=sys.stderr,
                flush=True,
            )
            self.last = now
