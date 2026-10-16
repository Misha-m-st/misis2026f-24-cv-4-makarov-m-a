"""Small public CLI. Diagnostics live under `tools`."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from document_vos.config import DATASETS, load, validate
from document_vos.io import PipelineError


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cutie для видео документов")
    p.add_argument("--config", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["setup", "download", "prepare", "train", "infer", "reproduce"]:
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        if name in {"train", "infer", "prepare", "reproduce"}:
            s.add_argument("--output", type=Path)
            s.add_argument("--workers", type=int)
        if name in {"setup", "train", "infer", "reproduce"}:
            s.add_argument("--cutie-root", type=Path)
            s.add_argument("--device", choices=["cuda", "cpu", "auto"])
        if name in {"setup", "prepare"}:
            s.add_argument("--sam", action="store_true")
        if name in {"setup", "download"}:
            s.add_argument("--catalog", type=Path)
        if name in {"download", "infer"}:
            s.add_argument("--sample", action="store_true")
            s.add_argument("--dataset", nargs="+", choices=DATASETS)
            s.add_argument("--count", type=int)
        if name == "download":
            s.add_argument("--plan", action="store_true")
            s.add_argument("--archives", type=Path)
        if name == "prepare":
            s.add_argument("target", choices=["datasets", "videos"])
            s.add_argument("--input-dir", type=Path)
            s.add_argument("--dataset", nargs="+", choices=DATASETS)
            s.add_argument(
                "--step", type=int, help="SAM: разреженная разметка вместо только первого кадра"
            )
        if name in {"train", "infer"}:
            s.add_argument("--weights", type=Path)
            s.add_argument("--seed", type=int)
        if name == "train":
            s.add_argument("--export", type=Path)
            s.add_argument("--plan", action="store_true")
            s.add_argument("--batch-size", type=int)
            s.add_argument("--seq-length", type=int)
            s.add_argument("--iterations", type=int)
            s.add_argument("--learning-rate", type=float)
        if name == "infer":
            s.add_argument("--model", choices=["baseline", "finetuned"])
            s.add_argument("--input-dir", type=Path)
            s.add_argument("--manifest", type=Path)
            s.add_argument("--split-dir", type=Path)
            s.add_argument("--scope", choices=["validation", "independent"])
            s.add_argument("--mem-every", type=int)
            s.add_argument("--max-internal-size", type=int)
        if name == "reproduce":
            s.add_argument(
                "--resume", type=Path, help="Проверить входы и продолжить завершённые этапы"
            )
    t = sub.add_parser("tools", add_help=False, help="Диагностики, проверка данных, сборка Release")
    t.add_argument("arguments", nargs=argparse.REMAINDER)
    return p


def dispatch(args: argparse.Namespace) -> dict:
    cfg = load(args.config)

    def option(name, default=None):
        return getattr(args, name, default)

    def cli_path(name):
        value = option(name)
        return value.expanduser().resolve() if value is not None else None

    for flag, key in [("cutie_root", "cutie"), ("archives", "archives")]:
        if cli_path(flag):
            cfg["paths"][key] = str(cli_path(flag))
    if option("device"):
        cfg["runtime"]["device"] = args.device
    if option("workers") is not None:
        if args.command != "train" or args.workers != 0:
            cfg["runtime"]["workers"] = args.workers
        cfg["runtime"]["dataloader_workers"] = args.workers
    if option("model"):
        cfg["model"] = args.model
    if args.command == "infer" and option("dataset") and not args.sample:
        raise PipelineError("infer --dataset требует --sample")
    if option("dataset"):
        if option("sample"):
            if len(args.dataset) != 1:
                raise PipelineError("--sample принимает ровно один --dataset")
            cfg["samples"]["dataset"] = args.dataset[0]
        else:
            cfg["datasets"]["selected"] = args.dataset
    if option("count") is not None:
        if not option("sample"):
            raise PipelineError("--count используется только с --sample")
        cfg["samples"]["count"] = args.count
    if option("catalog") is not None and args.command == "download":
        if not args.sample:
            raise PipelineError("--catalog используется с download --sample")
        cfg["samples"]["catalog"] = str(cli_path("catalog"))
    for arg, key in [
        ("batch_size", "batch_size"),
        ("seq_length", "seq_length"),
        ("iterations", "num_iterations"),
        ("learning_rate", "learning_rate"),
    ]:
        if option(arg) is not None:
            cfg["training"][key] = option(arg)
    for key in ["mem_every", "max_internal_size"]:
        if option(key) is not None:
            cfg["inference"][key] = option(key)
    if option("step") is not None:
        if not option("sam"):
            raise PipelineError("--step требует --sam")
        cfg["sam"]["step"] = args.step
    validate(cfg)
    if option("seed") is not None and args.seed < 0:
        raise PipelineError("seed должен быть >=0")
    from document_vos.runtime import Runtime, configure

    configure(Runtime.parse(cfg["runtime"]))
    name = args.command
    if name == "setup":
        from document_vos.setup import setup

        return setup(cfg, sam=args.sam, catalog=cli_path("catalog"))
    if name == "download":
        if args.sample:
            from document_vos.data.samples import catalog, download

            return download(
                Path(cfg["paths"]["samples"]),
                catalog(Path(cfg["samples"]["catalog"]) if cfg["samples"]["catalog"] else None),
                cfg["samples"]["dataset"],
                cfg["samples"]["count"],
                plan=args.plan,
            )
        from document_vos.data.download import dataset_archives, download_archive
        from document_vos.locking import exclusive_lock

        entries = [e for d in cfg["datasets"]["selected"] for e in dataset_archives(d)]
        result = {
            "status": "PLANNED" if args.plan else "COMPLETE",
            "download_bytes": sum(e.size for e in entries),
            "archives": [
                {
                    "dataset": e.name,
                    "url": e.url,
                    "bytes": e.size,
                    "checksum": e.checksum,
                    "algorithm": e.algorithm,
                    "path": str(Path(cfg["paths"]["archives"]) / e.archive),
                }
                for e in entries
            ],
        }
        if not args.plan:
            directory = Path(cfg["paths"]["archives"])
            with exclusive_lock(directory / ".download.lock"):
                for entry in entries:
                    download_archive(entry, directory)
        return result
    if name == "prepare":
        if args.target == "videos":
            from document_vos.videos import prepare

            return prepare(
                cli_path("input_dir") or Path(cfg["paths"]["videos"]),
                Path(cfg["paths"]["prepared"]),
                cfg,
                sam=args.sam,
                output=cli_path("output"),
            )
        if args.sam or args.input_dir or args.step:
            raise PipelineError("prepare datasets не принимает настройки видео/SAM")
        from document_vos.data.preparation import prepare

        return prepare(
            cfg, canonical=cfg["datasets"]["selected"] == DATASETS, output=cli_path("output")
        )
    if name == "infer":
        if args.sample and (args.scope or args.split_dir):
            raise PipelineError(
                "--sample использует фиксированную validation, без --scope/--split-dir"
            )
        if args.input_dir and args.manifest:
            raise PipelineError("--input-dir и --manifest несовместимы")
        from document_vos.workflows import inference

        return inference(
            cfg,
            sample=args.sample,
            input_dir=cli_path("input_dir"),
            manifest=cli_path("manifest"),
            split_dir=cli_path("split_dir"),
            scope=args.scope,
            weights=cli_path("weights"),
            output=cli_path("output"),
            seed=args.seed if args.seed is not None else cfg["preparation"]["seed"],
        )
    if name == "train":
        from document_vos.workflows import training

        return training(
            cfg,
            export=cli_path("export"),
            weights=cli_path("weights"),
            output=cli_path("output"),
            seed=args.seed if args.seed is not None else cfg["preparation"]["seed"],
            plan=args.plan,
        )
    if name == "reproduce":
        from document_vos.workflows import reproduce

        return reproduce(cfg, output=cli_path("output"), resume=cli_path("resume"))
    raise PipelineError(f"Неизвестная команда: {name}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "tools":
        if len(argv) > 1 and argv[1] == "capacity":
            from document_vos.capacity import main as capacity

            sys.argv = [sys.argv[0], *argv[2:]]
            capacity()
            return
        if len(argv) > 1 and argv[1] == "smoke":
            from document_vos.smoke import main as smoke

            smoke(argv[2:])
            return
        if len(argv) > 1 and argv[1] == "sam-preview":
            from document_vos.sam.preview_sam import main as preview

            sys.argv = [sys.argv[0], *argv[2:]]
            preview()
            return
        if len(argv) > 1 and argv[1] == "build-release":
            from document_vos.release import main as build

            build(argv[2:])
            return
        from document_vos.diagnostics import main as diagnostics

        sys.argv = [sys.argv[0], *argv[1:]]
        diagnostics()
        return
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        ImportError,
        yaml.YAMLError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
