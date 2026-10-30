"""Public workflows, composed from the integrity-checked scientific core."""

from pathlib import Path

from document_vos.config import DEFAULTS
from document_vos.io import PipelineError, digest, read_json, sha256
from document_vos.locking import exclusive_lock
from document_vos.runs import Run, inventory


def path(cfg: dict, name: str) -> Path:
    return Path(cfg["paths"][name])


def weight(cfg: dict, model: str, explicit: Path | None = None) -> Path:
    configured = cfg.get("benchmark", {}).get("models", {}).get(model, {}).get("weights")
    if explicit is None and configured is not None:
        explicit = Path(configured)
    if explicit:
        if not explicit.is_file():
            raise PipelineError(
                f"Нет весов {model}: {explicit}. Проверьте benchmark.models.{model}.weights "
                "в config.yaml или явный --weights для infer."
            )
        return explicit.resolve()
    from document_vos.setup import models

    entry = models()["models"][model]
    target = path(cfg, "weights") / entry["filename"]
    if not target.is_file():
        raise PipelineError(
            f"Нет весов {model}: {target}. Укажите существующий checkpoint через "
            f"benchmark.models.{model}.weights в config.yaml или выполните "
            f"uv run document-vos setup --model {model}."
        )
    if sha256(target) != entry["sha256"]:
        raise PipelineError(
            f"{model}: веса отличаются от каталога; используйте явный --weights для своих весов"
        )
    return target


def inference(
    cfg: dict,
    *,
    sample: bool = False,
    input_dir: Path | None = None,
    manifest: Path | None = None,
    split_dir: Path | None = None,
    scope: str | None = None,
    weights: Path | None = None,
    output: Path | None = None,
    seed: int = 42,
) -> dict:
    from document_vos.cutie import check_checkout
    from document_vos.cutie.inference import infer, snapshot_inputs
    from document_vos.runtime import resolve_device

    if scope == "validation" and split_dir is None:
        raise PipelineError("Validation требует --split-dir")
    if input_dir is not None and scope == "validation":
        raise PipelineError("--input-dir требует independent")
    model_weights = weight(cfg, cfg["model"], weights)
    root = check_checkout(path(cfg, "cutie"))
    source = {}
    if sample:
        if input_dir or manifest:
            raise PipelineError("--sample несовместим с --input-dir/--manifest")
        from document_vos.data.samples import catalog, select, verify

        selected = cfg["samples"]
        cat = catalog(Path(selected["catalog"]) if selected["catalog"] else None)
        entries = select(cat, selected["dataset"], selected["count"])
        for entry in entries:
            directory = path(cfg, "samples") / entry["id"]
            if not directory.is_dir():
                raise PipelineError(
                    "Нет примера: выполните uv run document-vos download --sample "
                    f"--dataset {selected['dataset']} --count {selected['count']}"
                )
            verify(directory, entry)
        source = {"samples": entries, "scope": "validation_sample"}
    elif manifest:
        from document_vos.data import load_manifest

        source = {
            "manifest_sha256": sha256(manifest),
            "inputs": snapshot_inputs(load_manifest(manifest)),
        }
    else:
        from document_vos.videos import prepare

        directory = input_dir or path(cfg, "videos")
        if (directory / "manifest.jsonl").is_file():
            prepared = directory
        else:
            prepared = Path(prepare(directory, path(cfg, "prepared"), cfg)["output"])
        from document_vos.data.input_folder import select_input_folder

        sequences, selection = select_input_folder(prepared)
        source = {
            "prepared": str(prepared),
            "inputs": snapshot_inputs(sequences),
            "selection": selection,
        }
    source.update(weights_sha256=sha256(model_weights), cutie_root=str(root))
    run = Run(path(cfg, "results"), "infer", cfg, source, output=output)
    with exclusive_lock(run.output / ".workflow.lock"):
        if sample:

            def assemble(target):
                target.mkdir()
                from document_vos.data.samples import manifest as sample_manifest

                sample_manifest(
                    path(cfg, "samples"),
                    target / "manifest.jsonl",
                    cat,
                    selected["dataset"],
                    selected["count"],
                )

            assembly = run.stage("inputs", assemble)
            manifest = assembly / "manifest.jsonl"
            scope = "validation_sample"
        kwargs = {"input_dir": prepared} if not sample and not manifest else {}
        result = run.stage(
            "inference",
            lambda target: infer(
                manifest,
                target,
                root,
                model_weights,
                cfg["inference"],
                scope or ("validation" if split_dir else "independent"),
                split_dir,
                seed,
                resolve_device(cfg["runtime"]["device"]),
                **kwargs,
            ),
        )
        if (result / "predictions.jsonl").is_file():

            def rendering(target):
                from document_vos.rendering import bundle

                bundle(result, target, **cfg["visualization"])

            run.stage("render", rendering)
        core_status = read_json(result / "run.json")["status"]
        return run.finish(
            status="NO_ELIGIBLE_VIDEOS" if core_status == "NO_ELIGIBLE_VIDEOS" else "COMPLETE"
        )


def training(
    cfg: dict,
    *,
    export: Path | None = None,
    weights: Path | None = None,
    output: Path | None = None,
    seed: int = 42,
    plan: bool = False,
) -> dict:
    from document_vos.cutie import check_checkout
    from document_vos.cutie.training import train
    from document_vos.data.preparation import prepare

    root, initial = check_checkout(path(cfg, "cutie")), weight(cfg, "baseline", weights)
    if export is None:
        prepared = Path(
            prepare(cfg, canonical=cfg["datasets"]["selected"] == DEFAULTS["datasets"]["selected"])[
                "output"
            ]
        )
        export = prepared / "training-export"
    signature = {"export": inventory(export), "weights_sha256": sha256(initial), "seed": seed}
    run = Run(path(cfg, "results"), "train", cfg, signature, output=output)
    with exclusive_lock(run.output / ".workflow.lock"):
        run.stage(
            "checkpoints",
            lambda target: train(
                root,
                initial,
                export,
                target,
                cfg["training"],
                seed,
                not plan,
                workers=cfg["runtime"]["dataloader_workers"],
                runtime=__import__("document_vos.runtime", fromlist=["Runtime"]).Runtime.parse(
                    cfg["runtime"]
                ),
                progress=True,
            ),
            live=True,
        )
        if plan:
            run.state["status"] = "PLANNED"
            run.save()
            return {"status": "PLANNED", "output": str(run.output)}
        return run.finish()


def reproduce(cfg: dict, *, output: Path | None = None, resume: Path | None = None) -> dict:
    from document_vos.config import DATASETS
    from document_vos.cutie import check_checkout
    from document_vos.data.preparation import ensure_local, prepare
    from document_vos.data.recipe import published
    from document_vos.runtime import resolve_device

    for key in ["training", "preparation", "inference"]:
        if cfg[key] != DEFAULTS[key]:
            raise PipelineError(
                f"{key}: параметры отличаются от опубликованного рецепта; используйте train/infer"
            )
    if cfg["datasets"]["selected"] != DATASETS:
        raise PipelineError("reproduce требует все шесть опубликованных датасетов")
    root = check_checkout(path(cfg, "cutie"))
    initial = weight(cfg, "baseline")
    resolve_device(cfg["runtime"]["device"])
    local = ensure_local(cfg, normalize=True)
    signature = {
        "datasets": {name: inventory(p) for name, p in local.items()},
        "baseline_sha256": sha256(initial),
        "recipe": digest(published()),
    }
    destination = resume or output
    lock = (
        destination.parent / f".{destination.name}.lock"
        if destination
        else path(cfg, "results") / ".reproduce.lock"
    )
    with exclusive_lock(lock):
        run = Run(path(cfg, "results"), "reproduce", cfg, signature, output=output, resume=resume)
        prepared = run.stage("prepare", lambda target: prepare(cfg, canonical=True, output=target))
        return paired(run, cfg, prepared, root, initial)


def paired(run: Run, cfg: dict, prepared: Path, root: Path, initial: Path) -> dict:
    """Same stage implementation for the published recipe and explicitly named smoke runs."""
    from document_vos.cutie.inference import infer
    from document_vos.cutie.training import train
    from document_vos.evaluation import evaluate
    from document_vos.evaluation.comparison import compare
    from document_vos.evaluation.reports import report
    from document_vos.runtime import Runtime, resolve_device

    trained = run.stage(
        "checkpoints",
        lambda target: train(
            root,
            initial,
            prepared / "training-export",
            target,
            cfg["training"],
            42,
            True,
            workers=cfg["runtime"]["dataloader_workers"],
            runtime=Runtime.parse(cfg["runtime"]),
            progress=True,
        ),
        live=True,
    )
    weights = trained / "official" / "documents_main_training_last.pth"
    evaluations = {}
    for model, model_weights in [("baseline", initial), ("finetuned", weights)]:
        predicted = run.stage(
            model,
            lambda target, w=model_weights: infer(
                prepared / "split" / "validation.jsonl",
                target,
                root,
                w,
                cfg["inference"],
                "validation",
                prepared / "split",
                42,
                resolve_device(cfg["runtime"]["device"]),
                trained if w == weights else None,
            ),
        )
        evaluations[model] = run.stage(
            f"{model}-metrics",
            lambda target, p=predicted: evaluate(
                p, target, workers=cfg["runtime"]["workers"], progress=True
            ),
        )
    paired_dir = run.stage(
        "comparison",
        lambda target: compare(
            evaluations["baseline"] / "evaluation.json",
            evaluations["finetuned"] / "evaluation.json",
            target,
        ),
    )
    run.stage("report", lambda target: report(paired_dir / "comparison.json", target))
    return run.finish()
