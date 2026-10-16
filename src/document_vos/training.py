"""Подготовка/проверка resolved Hydra-конфига и явный запуск официального обучения."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from .cutie import CUTIE_REVISION, check_checkout, compose_official, local_weights
from .export import validate_export
from .io import PipelineError, digest, new_directory, read_json, sha256, versions, write_json
from .parallel import DEFAULT_WORKERS
from .runtime import Runtime, current_runtime, require_free_space, resolve_device
from .split import verify_training_split
from .training_progress import run_with_progress

TRAIN_DEFAULTS = {
    "batch_size": 2,
    "seq_length": 8,
    "crop_size": [480, 480],
    "amp": True,
    "learning_rate": 1e-5,
    "num_iterations": 10000,
    "num_ref_frames": 3,
    "lr_schedule": "constant",
}


def training_config(
    root: Path,
    weights: Path,
    export: Path,
    output: Path,
    options: dict,
    seed: int = 42,
    *,
    workers: int = DEFAULT_WORKERS,
    runtime: Runtime | None = None,
) -> dict:
    if type(workers) is not int or workers < 0:
        raise PipelineError("DataLoader workers должен быть >= 0")
    runtime = replace(runtime or current_runtime(), dataloader_workers=workers)
    if set(options) - TRAIN_DEFAULTS.keys():
        raise PipelineError(
            "Разрешены только параметры из TRAIN_DEFAULTS; обязательные guards не переопределяются"
        )
    settings = {**TRAIN_DEFAULTS, **options}
    if settings["batch_size"] < 1 or settings["seq_length"] < 2 or settings["num_iterations"] < 1:
        raise PipelineError("Неверный batch/seq_length/число итераций")
    if not 0 < settings["learning_rate"] < 1:
        raise PipelineError("Неверный learning_rate")
    if (
        len(settings["crop_size"]) != 2
        or settings["crop_size"][0] != settings["crop_size"][1]
        or settings["crop_size"][0] < 16
        or settings["crop_size"][0] % 16
    ):
        raise PipelineError("Официальный loader требует квадратный crop, кратный 16")
    if not 1 <= settings["num_ref_frames"] < settings["seq_length"]:
        raise PipelineError("num_ref_frames должен быть между 1 и seq_length-1")
    cfg = compose_official(root, "train_config")
    OmegaConf.set_struct(cfg, False)
    cfg.pre_training.enabled = False
    for k, v in {**settings, "enabled": True, "num_objects": 1, "merge_probability": 0.0}.items():
        cfg.main_training[k] = v
    cfg.data = {
        "image_datasets": {"base": str(export)},
        "pre_training": {"datasets": []},
        "vos_datasets": {
            "base": str(export),
            "DOCUMENTS": {
                "image_directory": "images",
                "mask_directory": "masks",
                "frame_interval": 1,
                "subset": None,
                "empty_masks": None,
                "multiplier": 1,
            },
        },
        "main_training": {
            "datasets": ["DOCUMENTS"],
            "num_iterations": settings["num_iterations"],
            "lr_schedule_steps": [],
        },
    }
    cfg.weights = str(weights)
    cfg.checkpoint = None
    cfg.seed = seed
    cfg.num_workers = workers
    cfg.document_vos_runtime = runtime.to_dict()
    cfg.cudnn_benchmark = False
    cfg.log_text_interval = 25
    cfg.exp_id = "documents"
    cfg.save_weights_interval = min(1000, settings["num_iterations"])
    cfg.save_checkpoint_interval = cfg.save_weights_interval
    result = OmegaConf.to_container(cfg, resolve=True)
    result["hydra"] = {"run": {"dir": str(output / "official")}, "job": {"chdir": False}}
    validate_training_config(result, export, weights)
    return result


def validate_training_config(cfg: dict, export: Path, weights: Path) -> None:
    stage = cfg["main_training"]
    if cfg["pre_training"]["enabled"] is not False or stage["enabled"] is not True:
        raise PipelineError("Разрешён только main_training без pre_training")
    if stage["num_objects"] != 1 or stage["merge_probability"] != 0.0:
        raise PipelineError("Нужны num_objects=1 и merge_probability=0.0")
    if (
        Path(cfg["weights"]).resolve() != local_weights(weights)
        or cfg.get("checkpoint") is not None
    ):
        raise PipelineError("Нужны явные начальные weights без resume checkpoint")
    data = cfg["data"]
    if data["main_training"]["datasets"] != ["DOCUMENTS"] or set(data["vos_datasets"]) != {
        "base",
        "DOCUMENTS",
    }:
        raise PipelineError("Обучение разрешено только на экспортированных документах")
    d = data["vos_datasets"]["DOCUMENTS"]
    if (
        Path(data["vos_datasets"]["base"]).resolve() != export.resolve()
        or d["image_directory"] != "images"
        or d["mask_directory"] != "masks"
        or d["subset"] is not None
        or d["multiplier"] != 1
    ):
        raise PipelineError("Resolved data-конфигурация не соответствует экспорту")


def train(
    root: Path,
    weights: Path,
    export: Path,
    output: Path,
    options: dict,
    seed: int = 42,
    execute: bool = False,
    *,
    workers: int = DEFAULT_WORKERS,
    progress: bool = False,
    runtime: Runtime | None = None,
    refresh_export: bool = False,
) -> dict:
    runtime = replace(runtime or current_runtime(), dataloader_workers=workers)
    if type(workers) is not int or workers < 0:
        raise PipelineError("DataLoader workers должен быть >= 0")
    root, weights, export = check_checkout(root), local_weights(weights), export.resolve()
    metadata = read_json(export / "export.json")
    verify_training_split(Path(metadata["train_manifest"]), Path(metadata["split_dir"]))
    if sha256(Path(metadata["train_manifest"])) != metadata["manifest_sha256"]:
        raise PipelineError("Train-манифест изменён после экспорта")
    if sha256(Path(metadata["split_dir"]) / "split.json") != metadata["split_sha256"]:
        raise PipelineError("Split изменён после экспорта")
    validate_export(
        export,
        options.get("seq_length", TRAIN_DEFAULTS["seq_length"]),
        workers=runtime.workers,
        progress=progress,
        refresh=refresh_export,
    )
    if execute:
        try:
            device = resolve_device(runtime.device)
        except ImportError as e:
            raise PipelineError("Для --execute установите extra cutie и CUDA PyTorch") from e
        if device != "cuda":
            raise PipelineError(
                "Официальный training launcher требует CUDA; CPU поддержан для данных/инференса"
            )
    require_free_space(output, weights.stat().st_size * 4)
    output = new_directory(output)
    cfg = training_config(
        root, weights, export, output, options, seed, workers=workers, runtime=runtime
    )
    config_path = output / "config"
    config_path.mkdir()
    OmegaConf.save(OmegaConf.create(cfg), config_path / "train.yaml")
    plan = {
        "status": "PLANNED",
        "cutie_revision": CUTIE_REVISION,
        "seed": seed,
        "weights_sha256": sha256(weights),
        "export_sha256": sha256(export / "export.json"),
        "manifest_sha256": metadata["manifest_sha256"],
        "split_sha256": metadata["split_sha256"],
        "config_sha256": sha256(config_path / "train.yaml"),
        "versions": versions(),
        "peak_vram": "NOT_MEASURED",
        "config": cfg,
        "runtime": runtime.to_dict(),
        "export_validation": {
            "cache": str(export / ".validation-cache.sqlite3"),
            "refresh": refresh_export,
            "policy": "expected hashes + inode/size/mtime/ctime + validation version",
        },
    }
    write_json(output / "plan.json", plan)
    env = os.environ.copy()
    env.update(
        DOCUMENT_VOS_CUTIE=str(root),
        DOCUMENT_VOS_TRAIN_PLAN=str(output / "plan.json"),
        LOCAL_RANK="0",
        WORLD_SIZE="1",
    )
    # Hydra itself resolves the exact config used by the official entrypoint, without training.
    args = ["--config-path", str(config_path), "--config-name", "train"]
    proof = subprocess.run(
        [sys.executable, "-m", "document_vos.training_worker", *args, "--cfg", "job", "--resolve"],
        env=env,
        text=True,
        capture_output=True,
    )
    (output / "hydra-proof.log").write_text(proof.stdout + proof.stderr, encoding="utf-8")
    if proof.returncode:
        raise PipelineError(f"Hydra preflight FAILED; см. {output / 'hydra-proof.log'}")
    resolved = OmegaConf.to_container(OmegaConf.create(proof.stdout), resolve=True)
    validate_training_config(resolved, export, weights)
    requested = {k: v for k, v in cfg.items() if k != "hydra"}
    if digest(resolved) != digest(requested):
        raise PipelineError("Hydra resolved config отличается от проверенного плана")
    OmegaConf.save(OmegaConf.create(resolved), output / "resolved.yaml")
    if not execute:
        return {"status": "PLANNED", "hydra": "VERIFIED", "output": str(output)}
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "document_vos.training_worker",
        *args,
    ]
    with (output / "training.log").open("x", encoding="utf-8") as log:
        returncode = run_with_progress(command, env, log, cfg["main_training"]["num_iterations"])
    saved = {
        str(p.resolve()): sha256(p)
        for p in (output / "official").glob("*.pth")
        if "_ckpt_" not in p.name
    }
    runtime = read_json(output / "runtime.json") if (output / "runtime.json").exists() else {}
    successful = returncode == 0 and saved and runtime.get("status") == "COMPLETE"
    result = {
        "status": "COMPLETE" if successful else "FAILED",
        "returncode": returncode,
        "command": command,
        "plan_sha256": sha256(output / "plan.json"),
        "saved_weights": saved,
        "runtime": runtime,
    }
    write_json(output / "result.json", result)
    if not successful:
        raise PipelineError(f"Обучение FAILED (в том числе при OOM); см. {output / 'training.log'}")
    return result
