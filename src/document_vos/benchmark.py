"""Explicit worker benchmark on fixed inputs. Never changes training settings."""

import hashlib
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image

from .data import load_manifest
from .io import PipelineError, digest, new_directory, sha256, write_json
from .parallel import check_workers, ordered_map
from .runtime import Runtime


def _decode(path: str) -> dict:
    with Image.open(path) as image:
        image = image.convert("RGB")
        checksum = digest(
            {"size": image.size, "pixels": hashlib.sha256(image.tobytes()).hexdigest()}
        )
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        rss = None
    return {"sha256": checksum, "worker_peak_rss": rss}


def benchmark(
    manifest: Path, output: Path, candidates: list[int], frames: int, runtime: Runtime
) -> dict:
    if frames < 1 or not candidates:
        raise PipelineError("benchmark: нужны кадры и кандидаты")
    for workers in candidates:
        check_workers(workers)
    sequences = load_manifest(manifest)
    # Round-robin clips so the same workload includes multiple datasets/resolutions.
    paths = []
    groups = defaultdict(list)
    for sequence in sequences:
        groups[sequence.dataset].append(sequence)
    candidates_paths = (
        sequence.frames[index].path
        for index in range(max(len(s.frames) for s in sequences))
        for position in range(max(map(len, groups.values())))
        for dataset in sorted(groups)
        if position < len(groups[dataset])
        for sequence in [groups[dataset][position]]
        if index < len(sequence.frames)
    )
    for path in candidates_paths:
        paths.append(path)
        if len(paths) == frames:
            break
    output = new_directory(output)
    baseline = None
    rows = []
    # Warm filesystem cache equally; this measures decode/checksum execution,
    # not a claim about cold-cache storage bandwidth.
    for path in paths:
        _decode(path)
    for workers in candidates:
        start = perf_counter()
        records = list(ordered_map(_decode, paths, workers))
        seconds = perf_counter() - start
        hashes = [r["sha256"] for r in records]
        baseline = hashes if baseline is None else baseline
        if baseline != hashes:
            raise PipelineError("Результат benchmark зависит от workers")
        rows.append(
            {
                "workers": workers,
                "seconds": seconds,
                "fps": len(paths) / seconds,
                "worker_peak_rss_native_units": max(
                    (r["worker_peak_rss"] for r in records if r["worker_peak_rss"] is not None),
                    default=None,
                ),
            }
        )
    result = {
        "status": "COMPLETE",
        "manifest_sha256": sha256(manifest),
        "frames": len(paths),
        "results": rows,
        "fastest_workers": min(rows, key=lambda r: r["seconds"])["workers"],
        "runtime": runtime.to_dict(),
        "input_paths": paths,
        "includes_process_startup": True,
        "applied_automatically": False,
    }
    write_json(output / "benchmark.json", result)
    return result
