"""Единая CLI. YAML задаёт значения по умолчанию, явные аргументы имеют приоритет."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from document_vos.io import PipelineError, load_config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cutie для сегментации документов (без автоматического скачивания)",
        epilog="Также: tools smoke --help; tools build-release --help; tools sam-preview --help; tools capacity --help",
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "prepare",
        "split",
        "validate-data",
        "export-training",
        "train",
        "infer",
        "evaluate",
        "compare",
        "report",
        "prepare-archives",
        "benchmark",
        "probe-training",
        "render",
    ]:
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, help="YAML-конфигурация; CLI переопределяет её")
        s.add_argument("--runtime-config", type=Path, help="Переносимый YAML-профиль исполнения")
        if name != "validate-data":
            s.add_argument("--output", type=Path, help="Новый файл/каталог результата")
        if name in {"split", "validate-data", "export-training", "infer"}:
            s.add_argument("--manifest", type=Path)
        if name in {"validate-data", "export-training", "evaluate", "train"}:
            s.add_argument(
                "--workers",
                type=int,
                help="Число CPU-процессов, по умолчанию 4; 1 — последовательная обработка",
            )
        if name == "validate-data":
            s.add_argument(
                "--full",
                action="store_true",
                default=None,
                help="Явно проверить хеши, пиксели и GT; по умолчанию только структура",
            )
            s.add_argument("--cache", type=Path, help="SQLite-кеш полной проверки")
            s.add_argument(
                "--refresh",
                action="store_true",
                default=None,
                help="С --full: повторить проверку всех файлов, обновив кеш",
            )
        if name == "prepare":
            s.add_argument("--datasets", nargs="+", help="Явный выбранный список датасетов")
        if name == "prepare-archives":
            s.add_argument("--archives", type=Path)
            s.add_argument("--datasets", nargs="+")
            s.add_argument("--execute", action="store_true", help="Подготовить локальные архивы")
            s.add_argument("--delete-verified-zip", action="store_true", default=None)
        if name == "benchmark":
            s.add_argument("--manifest", type=Path)
            s.add_argument("--candidates", nargs="+", type=int)
            s.add_argument("--frames", type=int)
        if name == "probe-training":
            s.add_argument("--cutie-root", type=Path)
            s.add_argument("--export", type=Path)
            s.add_argument("--crop-size", type=int)
        if name in {"split", "train", "infer"}:
            s.add_argument("--seed", type=int)
        if name == "split":
            s.add_argument("--val-fraction", type=float)
            s.add_argument("--previous", type=Path)
        if name in {"export-training", "infer"}:
            s.add_argument("--split-dir", type=Path)
        if name == "export-training":
            s.add_argument("--seq-length", type=int)
            s.add_argument(
                "--copy", action="store_true", help="Копировать совместимые файлы вместо ссылок"
            )
        if name in {"train", "infer"}:
            s.add_argument("--cutie-root", type=Path)
            s.add_argument("--weights", type=Path, help="Обязательный локальный checkpoint")
        if name == "train":
            s.add_argument("--export", type=Path)
            s.add_argument(
                "--refresh-export",
                action="store_true",
                default=None,
                help="Повторить полную проверку экспорта, игнорируя кеш",
            )
            s.add_argument(
                "--execute", action="store_true", help="Явно разрешить длительное GPU-обучение"
            )
            s.add_argument("--batch-size", type=int)
            s.add_argument("--seq-length", type=int)
            s.add_argument("--iterations", type=int)
            s.add_argument("--learning-rate", type=float)
        if name == "infer":
            s.add_argument("--input-dir", type=Path, help="Подготовленный телефонный test set")
            s.add_argument("--scope", choices=["validation", "independent"])
            s.add_argument("--device", choices=["cuda", "cpu"])
            s.add_argument("--training-run", type=Path)
            s.add_argument("--mem-every", type=int)
            s.add_argument("--max-internal-size", type=int)
        if name == "render":
            s.add_argument("--run", type=Path, required=True)
            s.add_argument("--other", type=Path)
            s.add_argument("--fps", type=float, default=30)
            s.add_argument("--alpha", type=float, default=0.35)
        if name == "evaluate":
            s.add_argument("--run", type=Path)
        if name == "compare":
            s.add_argument("--baseline", type=Path)
            s.add_argument("--finetuned", type=Path)
        if name == "report":
            s.add_argument("--source", type=Path)
    return p


def dispatch(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config) if args.config else {}
    from document_vos.runtime import Runtime, configure, resolve_device

    runtime_values = dict(cfg.get("runtime", {}))
    if args.runtime_config:
        runtime_cfg = load_config(args.runtime_config)
        runtime_values.update(runtime_cfg.get("runtime", runtime_cfg))
    runtime = Runtime.parse(runtime_values)
    configure(runtime)
    base = args.config.resolve().parent if args.config else Path.cwd()

    def value(name, default=None, required=False):
        found = getattr(args, name, None)
        found = cfg.get(name, default) if found is None else found
        if required and found is None:
            raise PipelineError(f"Необходимо --{name.replace('_', '-')} или поле {name} в YAML")
        return found

    def path(name, required=True):
        cli_value = getattr(args, name, None)
        found = value(name, required=required)
        if found is None:
            return None
        from document_vos.data import resolve_path

        return resolve_path(Path.cwd() if cli_value is not None else base, str(found))

    def workers(training=False):
        explicit = getattr(args, "workers", None)
        key = "dataloader_workers" if training else "workers"
        if explicit is not None:
            return explicit
        if key in runtime_values:
            return getattr(runtime, key)
        return cfg.get("workers", getattr(runtime, key))

    name = args.command
    if name == "probe-training":
        from document_vos.cutie.probe import probe_export

        return probe_export(
            path("cutie_root"), path("export"), path("output"), value("crop_size", 480)
        )
    if name == "prepare-archives":
        from document_vos.data.archive_prepare import prepare_archives

        return prepare_archives(
            path("archives"),
            path("output"),
            value("datasets", []),
            runtime,
            execute=args.execute,
            delete_verified_zip=value("delete_verified_zip", False),
        )
    if name == "benchmark":
        from document_vos.evaluation.benchmark import benchmark

        return benchmark(
            path("manifest"),
            path("output"),
            value("candidates", [1, 2, 4]),
            value("frames", 64),
            runtime,
        )
    if name == "prepare":
        from document_vos.data.adapters import prepare

        return prepare(cfg, base, path("output"), args.datasets)
    if name == "validate-data":
        from document_vos.data import load_manifest, validate_data

        manifest = path("manifest")
        full, refresh = value("full", False), value("refresh", False)
        if refresh and not full:
            raise PipelineError("--refresh используется только вместе с --full")
        cache = path("cache", False) or manifest.parent / ".validation-cache.sqlite3"
        return validate_data(
            load_manifest(manifest),
            full=full,
            cache_path=cache,
            refresh=refresh,
            workers=workers(),
            progress=True,
        )
    if name == "split":
        from document_vos.data.split import split_manifest

        return split_manifest(
            path("manifest"),
            path("output"),
            value("seed", 42),
            value("val_fraction", 0.2),
            path("previous", False),
        )
    if name == "export-training":
        from document_vos.data.export import export_training

        return export_training(
            path("manifest"),
            path("split_dir"),
            path("output"),
            value("seq_length", 3),
            not args.copy,
            workers=workers(),
            progress=True,
        )
    if name == "train":
        from document_vos.cutie.training import train

        options = dict(cfg.get("training", {}))
        for arg, key in [
            ("batch_size", "batch_size"),
            ("seq_length", "seq_length"),
            ("iterations", "num_iterations"),
            ("learning_rate", "learning_rate"),
        ]:
            if getattr(args, arg) is not None:
                options[key] = getattr(args, arg)
        return train(
            path("cutie_root"),
            path("weights"),
            path("export"),
            path("output"),
            options,
            value("seed", 42),
            args.execute,
            workers=workers(training=True),
            progress=True,
            runtime=runtime,
            refresh_export=value("refresh_export", False),
        )
    if name == "infer":
        from document_vos.cutie.inference import infer

        if args.input_dir is not None and args.manifest is not None:
            raise PipelineError("Одновременные --input-dir и --manifest запрещены")
        input_dir = None if args.manifest is not None else path("input_dir", False)
        # Выбор папки через CLI заменяет также scope старого YAML для validation.
        if input_dir is None:
            scope = value("scope", "validation")
        elif args.scope is not None:
            scope = args.scope
        elif args.input_dir is not None:
            scope = "independent"
        else:
            scope = value("scope", "independent")
        if input_dir and scope != "independent":
            raise PipelineError("--input-dir требует independent; validation запрещён")
        options = dict(cfg.get("inference", {}))
        for key in ["mem_every", "max_internal_size"]:
            if getattr(args, key) is not None:
                options[key] = getattr(args, key)
        return infer(
            None if input_dir else path("manifest"),
            path("output"),
            path("cutie_root"),
            path("weights"),
            options,
            scope,
            path("split_dir", False),
            value("seed", 42),
            resolve_device(
                args.device
                or (
                    runtime.device
                    if "device" in runtime_values
                    else value("device", runtime.device)
                )
            ),
            path("training_run", False),
            input_dir=input_dir,
        )
    if name == "render":
        from document_vos.rendering import bundle

        return bundle(
            path("run"), path("output"), other=path("other", False), fps=args.fps, alpha=args.alpha
        )
    if name == "evaluate":
        from document_vos.evaluation import evaluate

        return evaluate(path("run"), path("output"), workers=workers(), progress=True)
    if name == "compare":
        from document_vos.evaluation.comparison import compare

        return compare(path("baseline"), path("finetuned"), path("output"))
    from document_vos.evaluation.reports import report

    return report(path("source"), path("output"))


def main() -> None:
    args = parser().parse_args()
    try:
        result = dispatch(args)
    except (ValueError, OSError, KeyError, TypeError, ImportError, yaml.YAMLError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
