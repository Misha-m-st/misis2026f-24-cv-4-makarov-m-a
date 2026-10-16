"""Explicit, resumable orchestration of existing CLI stages, with durable status."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import yaml

from .data import load_manifest, resolve_path
from .io import PipelineError, digest, load_config, read_json, sha256, write_json
from .locking import exclusive_lock
from .runtime import Runtime


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Stages:
    def __init__(self, output: Path, signature: dict, *, resume_failed: bool = False):
        self.output = output
        self.resume_failed = resume_failed
        self.lock = threading.Lock()
        self.path = output / "status.json"
        self.state = (
            read_json(self.path)
            if self.path.exists()
            else {
                "signature": signature,
                "stages": {},
                "status": "RUNNING",
            }
        )
        old_signature = self.state["signature"]
        if old_signature != signature:
            old_config = {k: v for k, v in old_signature.items() if k != "source_sha256"}
            new_config = {k: v for k, v in signature.items() if k != "source_sha256"}
            old_sources = old_signature.get("source_sha256", {})
            new_sources = signature.get("source_sha256", {})
            changed_sources = {
                name
                for name in old_sources.keys() | new_sources.keys()
                if old_sources.get(name) != new_sources.get(name)
            }
            if (
                not resume_failed
                or old_config != new_config
                or changed_sources != {"cli.py", "retraining.py"}
                or not any(
                    stage.get("status") == "FAILED" for stage in self.state["stages"].values()
                )
            ):
                raise PipelineError(
                    "Конфигурация или исходники изменились; нужен новый retraining output"
                )
            self.state.setdefault("source_revisions", []).append(
                {
                    "previous_retraining_py": old_sources["retraining.py"],
                    "current_retraining_py": new_sources["retraining.py"],
                }
            )
            self.state["signature"] = signature
        if resume_failed:
            if not any(stage.get("status") == "FAILED" for stage in self.state["stages"].values()):
                raise PipelineError("--resume-failed задан, но в этом запуске нет FAILED-этапа")
            for stage_name, stage in self.state["stages"].items():
                if stage["status"] == "COMPLETE":
                    if any(
                        not Path(path).is_file() or sha256(Path(path)) != expected
                        for path, expected in stage.get("outputs", {}).items()
                    ):
                        raise PipelineError(
                            f"Нельзя продолжить: артефакты завершённого этапа изменились: {stage_name}"
                        )
                    if not Path(stage["log"]).is_file():
                        raise PipelineError(
                            f"Нельзя продолжить: отсутствует лог этапа {stage_name}"
                        )
        self.state["status"] = "RUNNING"
        self.save()

    def save(self):
        _atomic_json(self.path, self.state)

    def run(self, name: str, arguments: list[str], outputs: list[Path]) -> dict:
        command = [sys.executable, "-m", "document_vos", *map(str, arguments)]
        with self.lock:
            previous = self.state["stages"].get(name)
            if previous:
                if previous["command"] != command:
                    raise PipelineError(f"Команда этапа изменилась: {name}")
                if previous["status"] != "COMPLETE":
                    if previous["status"] != "FAILED" or not self.resume_failed:
                        raise PipelineError(
                            f"Этап {name} прерван/FAILED; изучите лог и выберите новый output"
                        )
                    attempt = previous.get("retry_count", 0) + 1
                    history = previous.get("attempt_history", []) + [previous.copy()]
                    entry = {
                        "status": "RUNNING",
                        "command": command,
                        "log": str(self.output / f"{name}.retry-{attempt}.log"),
                        "retry_count": attempt,
                        "attempt_history": history,
                    }
                    self.state["stages"][name] = entry
                    self.save()
                else:
                    if any(
                        not Path(p).is_file() or sha256(Path(p)) != h
                        for p, h in previous["outputs"].items()
                    ):
                        raise PipelineError(f"Артефакты этапа изменились: {name}")
                    return previous
            else:
                entry = {
                    "status": "RUNNING",
                    "command": command,
                    "log": str(self.output / f"{name}.log"),
                }
                self.state["stages"][name] = entry
                self.save()
        print(f"[{name}] start; log={entry['log']}", flush=True)
        start = perf_counter()
        with Path(entry["log"]).open("x", encoding="utf-8") as stream:
            code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT)
        with self.lock:
            entry.update(
                returncode=code,
                seconds=perf_counter() - start,
                status="COMPLETE" if code == 0 else "FAILED",
            )
            if code == 0:
                for path in outputs:
                    if not path.is_file():
                        entry["status"] = "FAILED"
                        entry["error"] = f"Отсутствует обязательный результат: {path}"
                if entry["status"] == "COMPLETE":
                    entry["outputs"] = {str(p): sha256(p) for p in outputs}
            self.save()
        print(f"[{name}] {entry['status']}; {entry['seconds']:.1f}s", flush=True)
        if entry["status"] != "COMPLETE":
            raise PipelineError(f"Этап {name} FAILED; см. {entry['log']}")
        return entry


def probe_export(root: Path, export: Path, output: Path, crop_size: int = 480) -> dict:
    """Read one real augmented official training sample from every source dataset."""
    from .cutie import activate_checkout, check_checkout

    # The pinned dataset reads LOCAL_RANK even outside torch.distributed.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    activate_checkout(check_checkout(root))
    from cutie.dataset.vos_dataset import VOSMergeTrainDataset

    metadata = read_json(export / "export.json")
    dataset_config = {
        "DOCUMENTS": {
            "im_root": str(export / "images"),
            "gt_root": str(export / "masks"),
            "max_skip": 5,
            "subset": None,
            "empty_masks": None,
            "multiplier": 1,
        }
    }
    loader = VOSMergeTrainDataset(
        dataset_config,
        seq_length=metadata["seq_length"],
        max_num_obj=1,
        size=crop_size,
        merge_probability=0,
    )
    video_dataset = {
        r["export_video"]: r["source_video"].split("/", 1)[0] for r in metadata["sequences"]
    }
    examples = {}
    for index, (_, video, _) in enumerate(loader.video_frames):
        dataset = video_dataset[video]
        if dataset not in examples:
            sample = loader[index]
            if sample["info"]["num_objects"] != 1:
                raise PipelineError(f"Официальный loader не нашёл один объект: {dataset}")
            examples[dataset] = {
                "video": video,
                "shape": list(sample["rgb"].shape),
                "num_objects": sample["info"]["num_objects"],
            }
        if len(examples) == len(set(video_dataset.values())):
            break
    result = {
        "status": "COMPLETE",
        "datasets": examples,
        "export_sha256": sha256(export / "export.json"),
    }
    write_json(output, result)
    return result


def run_retraining(
    cfg: dict,
    base: Path,
    runtime: Runtime,
    *,
    execute: bool = False,
    resume_failed: bool = False,
) -> dict:
    def path(key):
        return resolve_path(base, cfg[key])

    data_config, training_config = path("data_config"), path("training_config")
    data = load_config(data_config)
    training = load_config(training_config)
    manifest = resolve_path(data_config.parent, data["output"])
    export = resolve_path(training_config.parent, training["export"])
    trained = resolve_path(training_config.parent, training["output"])
    weights = resolve_path(training_config.parent, training["weights"])
    root = resolve_path(training_config.parent, training["cutie_root"])
    output, split, previous = path("output"), path("split_dir"), path("previous_split")
    if not execute:
        return {
            "status": "PLANNED",
            "datasets": data["selected"],
            "manifest": str(manifest),
            "split": str(split),
            "export": str(export),
            "training": str(trained),
            "iterations": training["training"]["num_iterations"],
            "runtime": runtime.to_dict(),
            "normalization": "Run prepare-archives explicitly before retrain; no deletion here",
        }
    signature = {
        "config": cfg,
        "data_sha256": sha256(data_config),
        "training_sha256": sha256(training_config),
        "runtime": runtime.to_dict(),
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "configuration_files": {
            str(path(key)): sha256(path(key))
            for key in [
                "data_config",
                "training_config",
                "inference_config",
                "evaluation_runtime_config",
            ]
            if cfg.get(key)
        },
        "initial_weights_sha256": sha256(weights),
        "previous_split_sha256": sha256(previous / "split.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".run.lock"):
        stages = Stages(output, signature, resume_failed=resume_failed)
        runtime_path = output / "runtime.yaml"
        if not runtime_path.exists():
            runtime_path.write_text(
                yaml.safe_dump({"runtime": runtime.to_dict()}), encoding="utf-8"
            )
        common = ["--runtime-config", str(runtime_path)]
        try:
            stages.run(
                "prepare",
                ["prepare", "--config", str(data_config), *common],
                [manifest, manifest.with_suffix(".meta.json")],
            )
            stages.run(
                "validate", ["validate-data", "--manifest", str(manifest), "--full", *common], []
            )
            stages.run(
                "split",
                [
                    "split",
                    "--manifest",
                    str(manifest),
                    "--previous",
                    str(previous),
                    "--output",
                    str(split),
                    "--seed",
                    str(training.get("seed", 42)),
                    "--val-fraction",
                    str(cfg.get("val_fraction", 0.2)),
                    *common,
                ],
                [split / "split.json", split / "train.jsonl", split / "validation.jsonl"],
            )
            stages.run(
                "export",
                [
                    "export-training",
                    "--manifest",
                    str(split / "train.jsonl"),
                    "--split-dir",
                    str(split),
                    "--output",
                    str(export),
                    "--seq-length",
                    str(training["training"]["seq_length"]),
                    *common,
                ],
                [export / "export.json"],
            )
            counts = {}
            for label in ["train", "validation"]:
                counts[label] = {}
                for sequence in load_manifest(split / f"{label}.jsonl"):
                    row = counts[label].setdefault(sequence.dataset, {"videos": 0, "frames": 0})
                    row["videos"] += 1
                    row["frames"] += len(sequence.frames)
                if set(counts[label]) != set(data["selected"]):
                    raise PipelineError(f"Не все выбранные датасеты присутствуют в {label}")
            _atomic_json(output / "dataset-counts.json", counts)
            stages.run(
                "official-loader",
                [
                    "probe-training",
                    "--cutie-root",
                    str(root),
                    "--export",
                    str(export),
                    "--crop-size",
                    str(training["training"]["crop_size"][0]),
                    "--output",
                    str(output / "loader.json"),
                    *common,
                ],
                [output / "loader.json"],
            )
            stages.run(
                "preflight",
                [
                    "train",
                    "--config",
                    str(training_config),
                    "--output",
                    str(output / "preflight"),
                    *common,
                ],
                [output / "preflight/resolved.yaml"],
            )
            timings = []
            for workers in cfg.get("smoke_workers", [runtime.dataloader_workers]):
                name = f"smoke-workers-{workers}"
                folder = output / name
                entry = stages.run(
                    name,
                    [
                        "train",
                        "--config",
                        str(training_config),
                        "--output",
                        str(folder),
                        "--iterations",
                        str(cfg.get("smoke_iterations", 100)),
                        "--workers",
                        str(workers),
                        "--execute",
                        *common,
                    ],
                    [folder / "result.json"],
                )
                result = read_json(folder / "result.json")
                if result["status"] != "COMPLETE":
                    raise PipelineError(f"CUDA smoke не завершён: {name}")
                # Check actual saved weights by an inference initialization too.
                checkpoint = next(iter(result["saved_weights"]))
                smoke_manifest = output / "smoke-validation.jsonl"
                if not smoke_manifest.exists():
                    from .data import save_manifest

                    sequence = load_manifest(split / "validation.jsonl")[0]
                    sequence.frames = sequence.frames[:8]
                    save_manifest(smoke_manifest, [sequence])
                # Independent scope is reserved for custom videos. Use a local
                # split receipt to keep the normal validation contract intact.
                smoke_split = output / "smoke-split"
                if not smoke_split.exists():
                    smoke_split.mkdir()
                    write_json(
                        smoke_split / "split.json", {"validation_sha256": sha256(smoke_manifest)}
                    )
                stages.run(
                    name + "-reload",
                    [
                        "infer",
                        "--config",
                        str(path("inference_config")),
                        "--manifest",
                        str(smoke_manifest),
                        "--split-dir",
                        str(smoke_split),
                        "--weights",
                        checkpoint,
                        "--output",
                        str(folder / "reload"),
                        *common,
                    ],
                    [folder / "reload/run.json"],
                )
                timings.append((entry["seconds"], workers))
            chosen = min(timings)[1]
            chosen_runtime = {**runtime.to_dict(), "dataloader_workers": chosen}
            chosen_path = output / "chosen-runtime.yaml"
            if not chosen_path.exists():
                chosen_path.write_text(
                    yaml.safe_dump({"runtime": chosen_runtime}), encoding="utf-8"
                )
            stages.run(
                "training",
                [
                    "train",
                    "--config",
                    str(training_config),
                    "--execute",
                    "--workers",
                    str(chosen),
                    "--runtime-config",
                    str(chosen_path),
                ],
                [trained / "result.json"],
            )
            result = read_json(trained / "result.json")
            checkpoint = next(iter(result["saved_weights"]))
            references = [
                ("baseline", weights, None),
                ("v1", path("previous_weights"), path("previous_training")),
                ("v2", Path(checkpoint), trained),
            ]
            # At most one CPU evaluation pool overlaps one GPU inference.
            # Its separate runtime profile shares the resource budget explicitly.
            evaluation_runtime = (
                path("evaluation_runtime_config")
                if cfg.get("evaluation_runtime_config")
                else runtime_path
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = None
                for label, split_root in [("new-validation", split), ("old-validation", previous)]:
                    for model, checkpoint, lineage in references:
                        run = output / label / model
                        args = [
                            "infer",
                            "--config",
                            str(path("inference_config")),
                            "--manifest",
                            str(split_root / "validation.jsonl"),
                            "--split-dir",
                            str(split_root),
                            "--weights",
                            str(checkpoint),
                            "--output",
                            str(run),
                            *common,
                        ]
                        if lineage:
                            args += ["--training-run", str(lineage)]
                        stages.run(f"{label}-{model}", args, [run / "run.json"])
                        if pending:
                            pending.result()
                        pending = pool.submit(
                            stages.run,
                            f"{label}-{model}-eval",
                            [
                                "evaluate",
                                "--run",
                                str(run),
                                "--output",
                                str(run) + "-eval",
                                "--runtime-config",
                                str(evaluation_runtime),
                            ],
                            [Path(str(run) + "-eval") / "evaluation.json"],
                        )
                if pending:
                    pending.result()
            for label in ["new-validation", "old-validation"]:
                for left, right in [("baseline", "v1"), ("baseline", "v2"), ("v1", "v2")]:
                    folder = output / label
                    destination = folder / f"{left}-vs-{right}"
                    stages.run(
                        f"{label}-{left}-vs-{right}",
                        [
                            "compare",
                            "--baseline",
                            str(folder / f"{left}-eval/evaluation.json"),
                            "--finetuned",
                            str(folder / f"{right}-eval/evaluation.json"),
                            "--output",
                            str(destination),
                            *common,
                        ],
                        [destination / "comparison.json"],
                    )
            stages.state["status"] = "COMPLETE"
            stages.save()
            return {"status": "COMPLETE", "output": str(output), "config_sha256": digest(signature)}
        except BaseException as error:
            stages.state.update(status="FAILED", error=str(error))
            stages.save()
            raise
