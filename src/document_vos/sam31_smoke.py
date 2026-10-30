"""Bounded local GPU check: uv run python -m document_vos.sam31_smoke."""

import argparse
import json
import subprocess
import traceback
import uuid
from pathlib import Path

from document_vos.config import load
from document_vos.io import atomic_json, mask_digest, sha256


def ExternalPredictor(*args, **kwargs):
    from document_vos.benchmark import ExternalPredictor as predictor

    return predictor(*args, **kwargs)


def _input(*args, **kwargs):
    from document_vos.benchmark import _input as select_input

    return select_input(*args, **kwargs)


def load_manifest(*args, **kwargs):
    from document_vos.data import load_manifest as read

    return read(*args, **kwargs)


def first_mask(*args, **kwargs):
    from document_vos.data import first_mask as read

    return read(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split-dir", type=Path)
    args = parser.parse_args()
    cfg = load(args.config)
    from document_vos.runtime import Runtime, configure, runtime_snapshot

    configure(Runtime.parse(cfg["runtime"]))
    import numpy as np
    import torch
    from PIL import Image

    from document_vos.sam31 import cleanup_preserving_error

    output = args.output or Path(cfg["paths"]["results"]) / f"sam31-smoke-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "scope": "bounded_gpu_smoke",
        "frames": 5,
        "cpu_runtime": runtime_snapshot(),
    }
    try:
        # A busy GPU is a terminal skip; never wait for somebody else's job.
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report["gpu_processes_before"] = query.stdout.strip()
        if query.stdout.strip():
            report["status"] = "SKIPPED_GPU_BUSY"
            return 2
        manifest, scope, split_dir = _input(
            cfg,
            sample=False,
            input_dir=None,
            manifest=args.manifest,
            scope=None,
            split_dir=args.split_dir,
        )
        from document_vos.cutie.inference import validate_scope

        sequences = load_manifest(manifest)
        report["split_sha256"] = validate_scope(manifest, scope, split_dir, sequences)
        seq = next(seq for seq in sequences if len(seq.frames) - first_mask(seq)[0] >= 5)
        index, initial = first_mask(seq)
        frames = seq.frames[index : index + 5]
        weights = Path(cfg["benchmark"]["models"]["sam31"]["weights"])
        report.update(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "video": seq.key,
                "initialization_index": index,
                "initial_mask_sha256": mask_digest(initial),
                "weights_sha256": sha256(weights),
                "inputs": [
                    {"id": frame.id, "path": str(frame.path), "sha256": sha256(Path(frame.path))}
                    for frame in frames
                ],
                "autocast_before": torch.is_autocast_enabled("cuda"),
                "phase": "load_full_predictor",
            }
        )
        atomic_json(output / "status.json", report)
        predictor = ExternalPredictor("sam31", cfg, weights, "cuda")
        try:
            report.update(predictor.predictor.metadata)
            report["phase"] = "mask_initialization_and_propagation"
            atomic_json(output / "status.json", report)
            masks = predictor.predict_sequence(frames, initial, output / ".model-cache")
            predictor.sync()
            for i, mask in enumerate(masks):
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"{i:06d}.png")
            Image.fromarray(initial.astype(np.uint8) * 255).save(output / "initial-mask.png")
            report["vram"] = predictor.memory_stats()
            report["active_sessions_after_propagation"] = len(
                predictor.predictor.predictor._all_inference_states
            )
            if report["active_sessions_after_propagation"]:
                raise RuntimeError("SAM 3.1 leaked an inference session")
            report["saved_masks"] = len(masks)
        finally:
            cleanup_preserving_error(predictor.close)
        report["autocast_after"] = torch.is_autocast_enabled("cuda")
        if report["autocast_after"] != report["autocast_before"]:
            raise RuntimeError("SAM 3.1 leaked an autocast context")
        report.update(status="COMPLETE", phase="closed")
        return 0
    except BaseException as exc:
        report.update(
            status="FAILED_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        if torch.cuda.is_initialized():
            report["cuda_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "allocated_bytes_peak": torch.cuda.max_memory_allocated(),
            }
        return 1
    finally:
        atomic_json(output / "status.json", report)
        print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
