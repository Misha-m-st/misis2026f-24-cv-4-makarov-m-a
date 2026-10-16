"""Короткий GPU-замер официального Trainer; снижение нагрузки только после CUDA OOM."""

import argparse
import logging
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from .cutie import (
    CUTIE_REVISION,
    activate_checkout,
    check_checkout,
    compose_official,
    disable_downloads_and_require_full_weights,
    local_weights,
)
from .io import PipelineError, new_directory, read_json, sha256, versions, write_json
from .parallel import check_workers
from .split import verify_training_split
from .training import training_config, validate_training_config


def candidates() -> list[dict]:
    """Официальные spatial/temporal параметры сохраняются до batch=1."""
    return [
        {"batch_size": b, "seq_length": 8, "num_ref_frames": 3, "crop_size": [480, 480]}
        for b in [16, 8, 4, 2, 1]
    ] + [
        {"batch_size": 1, "seq_length": t, "num_ref_frames": r, "crop_size": [s, s]}
        for t, r, s in [(5, 3, 480), (3, 2, 480), (3, 2, 384)]
    ]


def should_continue(status: str) -> bool:
    if status not in {"OOM", "PASS"}:
        raise PipelineError(f"Замер остановлен: {status}; это не CUDA OOM")
    return status == "OOM"


def worker(case: Path, warmup: int, steps: int) -> None:
    import numpy as np
    import torch

    cfg = OmegaConf.load(case / "resolved.yaml")
    root = check_checkout(Path(os.environ["DOCUMENT_VOS_CUTIE"]))
    validate_training_config(
        OmegaConf.to_container(cfg, resolve=True),
        Path(cfg.data.vos_datasets.base),
        Path(cfg.weights),
    )
    activate_checkout(root)
    disable_downloads_and_require_full_weights()
    from cutie.dataset.setup_training_data import setup_main_training_datasets
    from cutie.model.trainer import Trainer
    from cutie.utils.logger import TensorboardLogger

    result = {"status": "FAILED", "steps": [], "optimizer_updates": 0}
    started = time.perf_counter()
    try:
        torch.cuda.set_device(0)
        torch.distributed.init_process_group("nccl")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.cuda.reset_peak_memory_stats()
        logger = TensorboardLogger(str(case), logging.getLogger(), enabled_tb=False)
        trainer = Trainer(cfg, cfg.main_training, logger, str(case))
        trainer.load_weights(cfg.weights)
        trainer.train()

        def updated(*_):
            result["optimizer_updates"] += 1

        trainer.optimizer.register_step_post_hook(updated)
        _, sampler, loader = setup_main_training_datasets(
            cfg, cfg.main_training.max_skip_schedule[0]
        )
        # Ограничиваем CPU prefetch; batch и официальный dataset/Trainer неизменны.
        loader.prefetch_factor = 1
        sampler.set_epoch(0)
        iterator = iter(loader)
        for i in range(warmup + steps):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            data = next(iterator)
            loaded = time.perf_counter()
            trainer.train_integrator.reset_except_hooks()
            before = result["optimizer_updates"]
            trainer.do_pass(data, i)
            torch.cuda.synchronize()
            end = time.perf_counter()
            values = trainer.train_integrator.values
            loss = values["total_loss"] / trainer.train_integrator.counts["total_loss"]
            if not math.isfinite(loss):
                raise PipelineError("Нечисловой loss; нагрузку не понижаем")
            row = {
                "iteration": i,
                "warmup": i < warmup,
                "loader_seconds": loaded - begin,
                "training_seconds": end - loaded,
                "full_seconds": end - begin,
                "loss": loss,
                "optimizer_updated": result["optimizer_updates"] > before,
                "amp_scale": trainer.scaler.get_scale(),
                "videos": list(data["info"]["name"]),
                "frames_by_temporal_index": data["info"]["frames"],
            }
            result["steps"].append(row)
            print(f"step {i + 1}/{warmup + steps}: {row}", flush=True)
            del data
        measured = result["steps"][warmup:]
        if not all(r["optimizer_updated"] for r in measured):
            raise PipelineError(
                "AMP пропустил измеряемый optimizer step; нужен разбор, не снижение"
            )
        result.update(
            status="PASS",
            mean_training_seconds=sum(r["training_seconds"] for r in measured) / steps,
            mean_full_seconds=sum(r["full_seconds"] for r in measured) / steps,
        )
    except BaseException as e:
        result.update(
            status="OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "FAILED",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    finally:
        result.update(
            elapsed_seconds=time.perf_counter() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(),
            total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
        write_json(case / "runtime.json", result)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cutie", type=Path, default=Path("external/Cutie"))
    p.add_argument("--weights", type=Path, default=Path("weights/cutie-base-mega.pth"))
    p.add_argument("--export", type=Path, default=Path("data/training-v1"))
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--case", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.case:
        worker(args.case, args.warmup, args.steps)
        return
    check_workers(args.workers)
    if args.workers < 1 or args.warmup < 1 or args.steps < 1 or args.warmup + args.steps >= 100:
        p.error("workers/warmup/steps >= 1; сумма шагов < 100 (без periodic saves/logging)")
    if args.output is None:
        p.error("Укажите новый --output")
    root, weights, export = (
        check_checkout(args.cutie),
        local_weights(args.weights),
        args.export.resolve(),
    )
    metadata = read_json(export / "export.json")
    verify_training_split(Path(metadata["train_manifest"]), Path(metadata["split_dir"]))
    if sha256(Path(metadata["train_manifest"])) != metadata["manifest_sha256"]:
        raise PipelineError("Манифест изменён после экспорта")
    if sha256(Path(metadata["split_dir"]) / "split.json") != metadata["split_sha256"]:
        raise PipelineError("Split изменён после экспорта")
    import torch

    if not torch.cuda.is_available():
        raise PipelineError("CUDA недоступна")
    output = new_directory(args.output)
    official = compose_official(root, "train_config")
    official_resolved = OmegaConf.to_container(official, resolve=True)
    write_json(output / "official-config.json", official_resolved)
    summary = {
        "protocol": "descending-on-CUDA-OOM-stop-at-first-PASS",
        "scope": "capacity smoke, not quality or full training qualification",
        "export_validation": "manifest/split provenance + official loader on sampled batches; not full rehash",
        "cutie_revision": CUTIE_REVISION,
        "weights_sha256": sha256(weights),
        "export_sha256": sha256(export / "export.json"),
        "manifest_sha256": metadata["manifest_sha256"],
        "versions": versions(),
        "seed": 42,
        "warmup": args.warmup,
        "measured_steps": args.steps,
        "loader_workers": args.workers,
        "prefetch_factor": 1,
        "cases": [],
    }
    for index, candidate in enumerate(candidates()):
        case = (
            output
            / f"{index:02d}-b{candidate['batch_size']}-t{candidate['seq_length']}-s{candidate['crop_size'][0]}"
        )
        case.mkdir()
        cfg = training_config(root, weights, export, case, candidate, workers=args.workers)
        # Сохраняем исходные loss/optimizer/scheduler и остальные параметры main_training.
        cfg["main_training"] = {
            **official_resolved["main_training"],
            **candidate,
            "num_objects": 1,
            "merge_probability": 0.0,
            "enabled": True,
        }
        cfg["data"]["main_training"]["num_iterations"] = official_resolved["main_training"][
            "num_iterations"
        ]
        cfg["data"]["main_training"]["lr_schedule_steps"] = official_resolved["main_training"][
            "lr_schedule_steps"
        ]
        cfg["cudnn_benchmark"] = official_resolved["cudnn_benchmark"]
        validate_training_config(cfg, export, weights)
        OmegaConf.save(OmegaConf.create(cfg), case / "resolved.yaml")
        env = os.environ.copy()
        env.update(
            DOCUMENT_VOS_CUTIE=str(root),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "-m",
            "document_vos.capacity",
            "--case",
            str(case),
            "--warmup",
            str(args.warmup),
            "--steps",
            str(args.steps),
        ]
        print(
            f"Проверяю {case.name} (официальный Trainer, AMP, workers={args.workers})", flush=True
        )
        with (case / "run.log").open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        runtime = (
            read_json(case / "runtime.json")
            if (case / "runtime.json").is_file()
            else {"status": "FAILED", "error": "Нет runtime.json; см. run.log"}
        )
        runtime.update(
            config=candidate, directory=str(case), subprocess_returncode=process.returncode
        )
        if process.returncode and runtime["status"] == "PASS":
            runtime["status"] = "FAILED"
        summary["cases"].append(runtime)
        write_json(case / "summary-so-far.json", summary)
        print(f"{case.name}: {runtime['status']}", flush=True)
        if runtime["status"] not in {"PASS", "OOM"}:
            write_json(output / "summary.json", summary)
        if not should_continue(runtime["status"]):
            break
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
