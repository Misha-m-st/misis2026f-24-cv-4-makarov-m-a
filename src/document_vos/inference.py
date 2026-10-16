"""GT передаётся один раз; все последующие кадры, включая кадры без GT, обрабатываются."""

import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image

from .cutie import CUTIE_REVISION, CutiePredictor, check_checkout, inference_config, local_weights
from .data import Sequence, first_mask, load_binary, load_manifest, save_manifest, validate_data
from .io import (
    PipelineError,
    digest,
    mask_digest,
    new_directory,
    read_json,
    sha256,
    versions,
    write_json,
    write_jsonl,
)


class Predictor(Protocol):
    def reset(self) -> None: ...
    def sync(self) -> None: ...
    def step(
        self, image: np.ndarray, mask: np.ndarray | None = None, end: bool = False
    ) -> np.ndarray: ...
    def memory_stats(self) -> dict: ...


def _snapshot_frame(task: tuple) -> dict:
    video, frame = task
    return {
        "video": video,
        "frame": frame.id,
        "image_sha256": sha256(Path(frame.path)),
        "mask_sha256": sha256(Path(frame.mask)) if frame.mask else None,
    }


def snapshot_inputs(sequences) -> list[dict]:
    from .parallel import ordered_map
    from .runtime import current_runtime

    return list(
        ordered_map(
            _snapshot_frame,
            ((s.key, f) for s in sequences for f in s.frames),
            current_runtime().io_threads,
            threads=True,
        )
    )


def _read_image(frame) -> np.ndarray:
    with Image.open(frame.path) as image:
        return np.asarray(image.convert("RGB"))


def _save_prediction(task: tuple) -> dict:
    prediction, path, record = task
    Image.fromarray(prediction.astype(np.uint8)).save(path)
    return {**record, "sha256": sha256(path)}


def validate_scope(
    manifest: Path, scope: str, split_dir: Path | None, sequences: list[Sequence] | None = None
) -> str | None:
    sequences = load_manifest(manifest) if sequences is None else sequences
    if scope == "independent":
        if any(s.dataset != "custom" for s in sequences):
            raise PipelineError("Independent-run принимает только отдельный custom-манифест")
        return None
    if scope != "validation" or split_dir is None:
        raise PipelineError("Validation-run требует --split-dir для подтверждения выборки")
    meta = read_json(split_dir / "split.json")
    if sha256(manifest) != meta["validation_sha256"]:
        raise PipelineError("Инференс validation требует неизменённый validation-манифест")
    if any(s.dataset == "custom" for s in sequences):
        raise PipelineError("Custom-видео оцениваются отдельно от validation")
    return sha256(split_dir / "split.json")


def propagate(
    manifest: Path,
    output: Path,
    predictor: Predictor,
    metadata: dict,
    *,
    sequences: list[Sequence] | None = None,
    selection: dict | None = None,
) -> dict:
    """Движок с тестируемым контрактом; CLI всегда использует настоящий CutiePredictor."""
    sequences = load_manifest(manifest) if sequences is None else sequences
    validate_data(sequences)
    output = new_directory(output)
    manifest = output / "selected-manifest.jsonl"
    save_manifest(manifest, sequences)
    if selection is not None:
        write_json(output / "selection.json", selection)
        metadata = {**metadata, "selection": selection}
    assets = snapshot_inputs(sequences)
    initializations = []
    predictions = []
    frame_count, model_seconds = 0, 0.0
    full_start = perf_counter()
    try:
        for seq in sequences:
            if selection is None:
                index, initial = first_mask(seq)
            else:
                index = 0
                with Image.open(seq.frames[0].path) as image:
                    initial = load_binary(Path(seq.frames[0].mask), image.size)
                if not initial.any():
                    raise PipelineError(f"{seq.key}: начальная маска стала пустой после отбора")
            initializations.append(
                {
                    "video": seq.key,
                    "frame": seq.frames[index].id,
                    "index": index,
                    "mask_sha256": mask_digest(initial),
                    "skipped_before_init": index,
                }
            )
            predictor.reset()
            directory = output / "predictions" / seq.export_id
            directory.mkdir(parents=True)
            from .parallel import ordered_map
            from .runtime import current_runtime

            runtime = current_runtime()

            def predict_frames():
                nonlocal frame_count, model_seconds
                # A reader and the writer share the configured I/O budget.
                readers = max(1, runtime.io_threads // 2)
                for i, image in enumerate(
                    ordered_map(
                        _read_image, seq.frames[index:], readers, threads=runtime.io_threads > 1
                    ),
                    index,
                ):
                    frame = seq.frames[i]
                    predictor.sync()
                    start = perf_counter()
                    pred = predictor.step(
                        image, initial if i == index else None, end=(i == len(seq.frames) - 1)
                    )
                    predictor.sync()
                    elapsed = perf_counter() - start
                    model_seconds += getattr(predictor, "last_model_seconds", elapsed)
                    if pred.shape != image.shape[:2] or not ((pred == 0) | (pred == 1)).all():
                        raise PipelineError(
                            f"{seq.key}/{frame.id}: неверный размер/значения предсказания"
                        )
                    path = directory / f"{i:06d}.png"
                    from .runtime import require_free_space

                    require_free_space(output, (pred.size + 65536) * runtime.queue_size)
                    frame_count += 1
                    yield (
                        pred.copy(),
                        path,
                        {
                            "video": seq.key,
                            "frame": frame.id,
                            "index": i,
                            "path": str(path.relative_to(output)),
                            "initialization": i == index,
                        },
                    )

            writers = max(1, runtime.io_threads - max(1, runtime.io_threads // 2))
            predictions.extend(
                ordered_map(
                    _save_prediction, predict_frames(), writers, threads=runtime.io_threads > 1
                )
            )

        predictor.sync()
        full_seconds = perf_counter() - full_start
        if assets != snapshot_inputs(sequences):
            raise PipelineError("Исходные данные изменились во время инференса")
        write_jsonl(output / "predictions.jsonl", predictions)
        signature = {
            "manifest_sha256": sha256(manifest),
            "inputs_sha256": digest(assets),
            "initializations": initializations,
            "inference": metadata["inference"],
            "seed": metadata["seed"],
            "scope": metadata["scope"],
            "split_sha256": metadata.get("split_sha256"),
            "cutie_revision": metadata["cutie_revision"],
            "device": metadata["device"],
            "versions": metadata["versions"],
        }
        result = {
            **metadata,
            "status": "COMPLETE",
            "manifest": str(manifest.resolve()),
            "signature": signature,
            "inputs": assets,
            "predictions_sha256": sha256(output / "predictions.jsonl"),
            "timing": {
                "model_calls": frame_count,
                "propagated_frames": frame_count - len(sequences),
                "model_seconds": model_seconds,
                "full_seconds": full_seconds,
                "model_fps": frame_count / model_seconds,
                "full_fps": frame_count / full_seconds,
                "definition": "model: InferenceCore.step with CUDA sync; full: load/convert/transfer/step/save",
                "includes_initialization": True,
                "warmup_excluded": False,
            },
            "vram": predictor.memory_stats(),
        }
        write_json(output / "run.json", result)
        return {
            "status": "COMPLETE",
            "videos": len(sequences),
            "frames": frame_count,
            "output": str(output),
        }
    except BaseException as e:
        if not (output / "run.json").exists():
            write_json(output / "run.json", {**metadata, "status": "FAILED", "error": str(e)})
        raise


def infer(
    manifest: Path | None,
    output: Path,
    root: Path,
    weights: Path,
    overrides: dict,
    scope: str,
    split_dir: Path | None,
    seed: int = 42,
    device: str = "cuda",
    training_run: Path | None = None,
    *,
    input_dir: Path | None = None,
) -> dict:
    if output.exists():
        raise PipelineError(f"Каталог уже существует: {output}. Выберите новую версию результата.")
    selection, sequences = None, None
    if input_dir is not None:
        if manifest is not None or scope != "independent":
            raise PipelineError("input_dir требует independent и не допускает manifest")
        from .input_folder import select_input_folder

        sequences, selection = select_input_folder(input_dir)
        print(
            f"Видео включено: {selection['included']}; исключено: {selection['excluded']}",
            file=sys.stderr,
        )
        manifest = input_dir / "manifest.jsonl"
        if not sequences:
            output = new_directory(output)
            write_jsonl(output / "selected-manifest.jsonl", [])
            write_json(output / "selection.json", selection)
            result = {
                "status": "NO_ELIGIBLE_VIDEOS",
                "videos": 0,
                "selection": selection,
                "output": str(output),
                "model_loaded": False,
            }
            write_json(output / "run.json", result)
            return result
    if manifest is None:
        raise PipelineError("Необходим manifest или input_dir")
    sequences = load_manifest(manifest) if sequences is None else sequences
    split_hash = validate_scope(manifest, scope, split_dir, sequences)
    validate_data(sequences)
    root, weights = check_checkout(root), local_weights(weights)
    cfg = inference_config(root, overrides)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    metadata = {
        "cutie_revision": CUTIE_REVISION,
        "weights_sha256": sha256(weights),
        "weights": str(weights),
        "inference": cfg,
        "seed": seed,
        "device": device,
        "versions": versions(),
        "scope": scope,
        "split_sha256": split_hash,
        "backend": "official-cutie",
        "runtime": __import__("document_vos.runtime", fromlist=["current_runtime"])
        .current_runtime()
        .to_dict(),
    }
    if training_run:
        result = read_json(training_run / "result.json")
        plan = read_json(training_run / "plan.json")
        if (
            result.get("status") != "COMPLETE"
            or result["plan_sha256"] != sha256(training_run / "plan.json")
            or metadata["weights_sha256"] not in result["saved_weights"].values()
        ):
            raise PipelineError("Finetuned weights не подтверждены завершённым training-run")
        if scope == "validation" and plan["split_sha256"] != split_hash:
            # Comparing historical weights on an extended split is safe only
            # after validating their original training provenance and disjointness.
            from .split import assert_disjoint, verify_training_split

            export = Path(plan["config"]["data"]["vos_datasets"]["base"])
            exported = read_json(export / "export.json")
            trained = Path(exported["train_manifest"])
            original_split = Path(exported["split_dir"])
            verify_training_split(trained, original_split)
            if (
                sha256(trained) != plan["manifest_sha256"]
                or sha256(original_split / "split.json") != plan["split_sha256"]
            ):
                raise PipelineError("Исходный train/split исторической модели изменён")
            assert_disjoint(load_manifest(trained), sequences)
        metadata["training"] = {
            "initial_weights_sha256": plan["weights_sha256"],
            "plan_sha256": result["plan_sha256"],
            "result_sha256": sha256(training_run / "result.json"),
            "split_sha256": plan["split_sha256"],
        }
    predictor = CutiePredictor(root, weights, cfg, device)
    return propagate(
        manifest, output, predictor, metadata, sequences=sequences, selection=selection
    )
