"""Параллельная обработка сохраняет результат, порядок и ограничение памяти."""

import os
from pathlib import Path
from time import monotonic, sleep

import pytest

from document_vos import parallel
from document_vos.data import validate_data
from document_vos.data.export import export_training, validate_export
from document_vos.data.split import split_manifest
from document_vos.io import PipelineError, read_json


def _probe(task):
    index, directory = task
    pid = os.getpid()
    (directory / str(pid)).touch()
    deadline = monotonic() + 15
    # Все четыре процесса действительно одновременно приступили к работе.
    while len(list(directory.iterdir())) < 4:
        if monotonic() > deadline:
            raise RuntimeError("Не запустились четыре рабочих процесса")
        sleep(0.01)
    sleep((3 - index) * 0.02)
    return index, pid, os.environ.get("OPENBLAS_NUM_THREADS")


def _identity(value):
    return value, os.getpid()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_reject_invalid_workers(workers):
    with pytest.raises(PipelineError, match="workers"):
        list(parallel.ordered_map(_identity, [1], workers))


def test_one_worker_runs_in_parent():
    assert list(parallel.ordered_map(_identity, [3, 1, 2], 1)) == [
        (3, os.getpid()),
        (1, os.getpid()),
        (2, os.getpid()),
    ]


def test_default_four_processes_and_order(tmp_path):
    results = list(
        parallel.ordered_map(_probe, [(i, tmp_path) for i in range(4)], parallel.DEFAULT_WORKERS)
    )
    assert [row[0] for row in results] == [0, 1, 2, 3]
    assert len({row[1] for row in results}) == 4
    assert all(row[1] != os.getpid() and row[2] == "1" for row in results)


def test_queue_does_not_submit_entire_dataset(monkeypatch):
    submitted = []

    class Future:
        def __init__(self, function, value):
            self.function, self.value = function, value

        def result(self):
            return self.function(self.value)

        def cancel(self):
            pass

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, value):
            submitted.append(value)
            return Future(function, value)

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", Pool)
    results = parallel.ordered_map(_identity, range(10000), 4)
    assert next(results)[0] == 0
    assert len(submitted) == 8
    results.close()


def test_serial_parallel_export_identical(tmp_path, manifest):
    split = tmp_path / "split"
    split_manifest(manifest, split)
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    stats = [
        export_training(split / "train.jsonl", split, out, workers=workers)
        for out, workers in zip(outputs, [1, 4])
    ]
    assert stats[0] == stats[1]
    assert read_json(outputs[0] / "export.json") == read_json(outputs[1] / "export.json")
    assert validate_export(outputs[0], workers=1) == validate_export(outputs[1], workers=4)
    for kind in ["images", "masks"]:
        for path in (outputs[0] / kind).rglob("*.*"):
            assert path.read_bytes() == (outputs[1] / path.relative_to(outputs[0])).read_bytes()


def test_parallel_failure_does_not_publish_complete_export(tmp_path, manifest, sequences):
    split = tmp_path / "split"
    split_manifest(manifest, split)
    from document_vos.data import load_manifest

    frame = load_manifest(split / "train.jsonl")[0].frames[0]
    Path(frame.path).write_bytes(b"invalid image")
    out = tmp_path / "failed"
    with pytest.raises(OSError):
        export_training(split / "train.jsonl", split, out, workers=4)
    assert not (out / "export.json").exists()


def test_full_validation_parallel_result_and_cache(tmp_path, sequences):
    serial = validate_data(sequences, full=True, workers=1)
    cache = tmp_path / "cache.sqlite3"
    result = validate_data(sequences, full=True, cache_path=cache, workers=4)
    assert result == serial
    subset = validate_data(sequences[:2], full=True, cache_path=cache, workers=4)
    assert subset["cached_frames"] == 8 and subset["checked_frames"] == 0
