"""Launcher оставляет официальный main_training неизменным; выполняется torchrun."""

import os
import runpy
from pathlib import Path

from document_vos.io import PipelineError, read_json, sha256, write_json


def main() -> None:
    plan_path = Path(os.environ["DOCUMENT_VOS_TRAIN_PLAN"])
    plan = read_json(plan_path)
    cfg = plan["config"]
    from document_vos.runtime import Runtime, configure, runtime_snapshot

    runtime = Runtime.parse(cfg.get("document_vos_runtime", {}))
    configure(runtime)
    from document_vos.cutie import (
        activate_checkout,
        check_checkout,
        disable_downloads_and_require_full_weights,
    )
    from document_vos.cutie.training import validate_training_config

    root = check_checkout(Path(os.environ["DOCUMENT_VOS_CUTIE"]))
    if sha256(plan_path.parent / "config/train.yaml") != plan["config_sha256"]:
        raise PipelineError("Конфигурация обучения изменена после preflight")
    weights = Path(cfg["weights"])
    if sha256(weights) != plan["weights_sha256"]:
        raise PipelineError("Начальные weights изменены после preflight")
    validate_training_config(cfg, Path(cfg["data"]["vos_datasets"]["base"]), weights)
    activate_checkout(root)
    disable_downloads_and_require_full_weights()
    from document_vos.runtime import install_loader_options

    install_loader_options(runtime)
    import torch

    preflight = "--cfg" in __import__("sys").argv
    if not preflight:
        from document_vos.cutie.training_progress import install_metrics_output

        install_metrics_output()
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    status, error = "COMPLETE", None
    try:
        runpy.run_path(str(root / "cutie/train.py"), run_name="__main__")
    except BaseException as e:
        status = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "FAILED"
        error = str(e)
        raise
    finally:
        if not preflight:
            write_json(
                plan_path.parent / "runtime.json",
                {
                    "cpu_runtime": runtime_snapshot(),
                    "status": status,
                    "error": error,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            )


if __name__ == "__main__":
    main()
