"""Local dataset normalization and reusable, fingerprinted manifest/split/export."""

from dataclasses import replace
from pathlib import Path

from document_vos.config import DATASETS
from document_vos.io import PipelineError, digest, new_directory, read_json, write_json
from document_vos.locking import exclusive_lock
from document_vos.runs import inventory, source_signature, verify_inventory


def roots(cfg: dict) -> dict[str, Path]:
    raw = Path(cfg["paths"]["raw"])
    defaults = {
        name: raw / name / ("clips" if name in {"midv2020", "dlc2021"} else "")
        for name in cfg["datasets"]["selected"]
    }
    return {
        name: Path(cfg["datasets"]["roots"].get(name, default))
        for name, default in defaults.items()
    }


def ensure_local(cfg: dict, *, normalize: bool) -> dict[str, Path]:
    from document_vos.data.archive_prepare import prepare_archives
    from document_vos.data.download import dataset_archives, extract_archive, verify_archive
    from document_vos.runtime import Runtime

    result = roots(cfg)
    selected = list(result)
    archives = Path(cfg["paths"]["archives"])
    missing = []
    for name, root in result.items():
        if root.is_dir() and any(root.iterdir()):
            continue
        missing.extend(
            str(archives / a.archive)
            for a in dataset_archives(name)
            if not (archives / a.archive).is_file()
        )
    if missing:
        raise PipelineError(
            f"Нет локальных данных/архивов ({len(missing)}). "
            "Выполните: uv run document-vos download --dataset " + " ".join(selected)
        )
    if not normalize:
        return result
    for name, root in result.items():
        if root.is_dir() and any(root.iterdir()):
            continue
        if name in {"smartdoc2015", "midv2020"}:
            if name in cfg["datasets"]["roots"]:
                raise PipelineError(
                    f"Нет заданного root {root}; скачайте/распакуйте данные или исправьте root"
                )
            for entry in dataset_archives(name):
                path = archives / entry.archive
                extract_archive(
                    replace(entry, target=root.name), path, root.parent, verify_archive(path, entry)
                )
        else:
            if name in cfg["datasets"]["roots"]:
                raise PipelineError(f"Нет заданного root {root}; укажите нормализованный root")
            prepare_archives(
                archives,
                Path(cfg["paths"]["raw"]),
                [name],
                Runtime.parse(cfg["runtime"]),
                execute=True,
            )
    return result


def prepare(cfg: dict, *, canonical: bool = False, output: Path | None = None) -> dict:
    from document_vos.data.adapters import prepare as make_manifest
    from document_vos.data.export import export_training
    from document_vos.data.recipe import published
    from document_vos.data.recipe import split as recipe_split
    from document_vos.data.split import split_manifest

    if canonical and (
        cfg["datasets"]["selected"] != DATASETS
        or cfg["preparation"] != {"seed": 42, "val_fraction": 0.2, "seq_length": 8}
    ):
        raise PipelineError(
            "Подготовка опубликованного рецепта требует неизменённых данных/параметров"
        )
    local = ensure_local(cfg, normalize=True)
    # Hash actual input bytes, not just mtimes; no silent reuse after changed frames/annotations.
    inputs = {name: inventory(root) for name, root in local.items()}
    signature = {
        "inputs": inputs,
        "preparation": cfg["preparation"],
        "canonical": canonical,
        "source_sha256": source_signature(),
    }
    target = output or Path(cfg["paths"]["prepared"]) / "datasets" / digest(signature)[:20]
    if any(target.resolve().is_relative_to(root.resolve()) for root in local.values()):
        raise PipelineError("Подготовленный output должен находиться вне исходных датасетов")
    with exclusive_lock(target.parent / f".{target.name}.lock"):
        if target.exists():
            receipt = read_json(target / "receipt.json")
            if receipt["signature"] != signature:
                raise PipelineError("Входы подготовки изменились; нужен новый output")
            current = inventory(target)
            current.pop('receipt.json', None)
            if current != receipt['files']:
                raise PipelineError(f'Подготовленные артефакты изменились: {target}')
            verify_inventory(target, receipt["files"], exact=False)
            return {"status": "EXISTING", "output": str(target)}
        new_directory(target)
        spec = {
            "selected": list(local),
            "datasets": {
                name: {"root": str(root), **({"skip_empty_via": True} if name == "dlc2021" else {})}
                for name, root in local.items()
            },
        }
        make_manifest(spec, Path.cwd(), target / "manifest.jsonl")
        if canonical:
            meta = read_json(target / "manifest.meta.json")
            exclusions = [
                {k: v for k, v in row.items() if k != "annotation"} for row in meta["exclusions"]
            ]
            if exclusions != published()["exclusions"]:
                raise PipelineError("Исключения DLC не совпадают с опубликованным рецептом")
            recipe_split(target / "manifest.jsonl", target / "split")
        else:
            p = cfg["preparation"]
            split_manifest(
                target / "manifest.jsonl", target / "split", p["seed"], p["val_fraction"]
            )
        export_training(
            target / "split" / "train.jsonl",
            target / "split",
            target / "training-export",
            cfg["preparation"]["seq_length"],
            workers=cfg["runtime"]["workers"],
            progress=True,
        )
        after = {name: inventory(root) for name, root in local.items()}
        if after != inputs:
            raise PipelineError("Исходные данные изменились во время подготовки")
        write_json(target / "receipt.json", {"signature": signature, "files": inventory(target)})
        return {"status": "COMPLETE", "output": str(target)}
