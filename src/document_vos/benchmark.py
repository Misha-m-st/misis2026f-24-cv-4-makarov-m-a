"""Cross-architecture video inference and comparison, isolated from the Cutie path."""

from __future__ import annotations

import csv
import gc
import html
import io
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from document_vos.config import DEFAULTS
from document_vos.cutie import CUTIE_REVISION, CutiePredictor, check_checkout
from document_vos.cutie.inference import propagate, snapshot_inputs, validate_scope
from document_vos.data import load_manifest
from document_vos.io import PipelineError, atomic_json, read_json, sha256, versions, write_json
from document_vos.runs import Run
from document_vos.timing import ComputeTimer, speed_compatibility

MODELS = ("baseline", "finetuned", "efficienttam", "livos", "sam31")


def _frames_directory(frames, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    for index, frame in enumerate(frames):
        with Image.open(frame.path) as image:
            image.convert("RGB").save(directory / f"{index:06d}.jpg", format="JPEG", quality=95)
    return directory


class ExternalPredictor:
    """Sequence API normalized to ordered binary masks at input resolution."""

    match_cutie_size = None

    def __init__(self, model: str, cfg: dict, weights: Path, device: str, *, match_cutie_size=None):
        self.name, self.options, self.weights, self.device = (
            model,
            cfg["benchmark"]["models"][model],
            weights,
            device,
        )
        self.repo_root = Path(cfg["paths"]["external_models"]) / model
        self.predictor = None
        self.match_cutie_size = match_cutie_size
        self.geometry_records = []
        self.last_model_seconds = 0.0
        import torch

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._load()
        if match_cutie_size is not None:
            from document_vos.benchmark_geometry import adapt_efficienttam, adapt_sam31

            try:
                if self.name == "efficienttam":
                    adapt_efficienttam(self.predictor)
                elif self.name == "sam31":
                    adapt_sam31(self.predictor.predictor)
            except BaseException:
                from document_vos.sam31 import cleanup_preserving_error

                cleanup_preserving_error(self.close)
                raise

    def _load(self):
        try:
            import torch
        except ImportError as exc:
            raise PipelineError(
                f"{self.name}: PyTorch недоступен; выполните setup --model {self.name}"
            ) from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise PipelineError(
                f"{self.name}: CUDA недоступна; выберите --device cpu либо подготовьте CUDA"
            )
        if self.name == "efficienttam":
            try:
                from efficient_track_anything.build_efficienttam import (
                    build_efficienttam_video_predictor,
                )
            except ImportError as exc:
                raise PipelineError(
                    "EfficientTAM API недоступен; выполните uv run document-vos setup --model efficienttam"
                ) from exc
            self.predictor = build_efficienttam_video_predictor(
                "configs/efficienttam/efficienttam_s.yaml",
                str(self.weights),
                device=self.device,
                vos_optimized=self.options["compile"],
                hydra_overrides_extra=[
                    "++model.compile_image_encoder="
                    + str(bool(self.options.get("compile", False))).lower()
                ],
            )
            if type(self.predictor).__name__ == "EfficientTAMVideoPredictorVOS":
                # Keep memory-attention outputs alive across the independently
                # compiled SAM heads on current PyTorch CUDA Graphs runtimes.
                self.predictor.memory_attention.register_forward_hook(
                    lambda _module, _args, output: output.clone()
                )
        elif self.name == "sam31":
            if self.device != "cuda":
                raise PipelineError(
                    "Официальный SAM 3.1 multiplex adapter в этом релизе требует CUDA"
                )
            try:
                from sam3.model_builder import build_sam3_multiplex_video_predictor
            except ImportError as exc:
                raise PipelineError(
                    "SAM 3.1 API недоступен; выполните uv run document-vos setup --model sam31"
                ) from exc
            if not self.options["amp"] or self.options["dtype"] != "bfloat16":
                raise PipelineError("SAM 3.1 требует amp=true и dtype=bfloat16 (штатный BF16)")
            if self.options.get("offload_state_to_cpu", False):
                raise PipelineError(
                    "SAM 3.1 multiplex: offload_state_to_cpu=true не поддерживается"
                )
            from document_vos.sam31 import MaskSessionPredictor

            self.predictor = MaskSessionPredictor(
                build_sam3_multiplex_video_predictor(
                    checkpoint_path=str(self.weights),
                    use_fa3=self.options["use_fa3"],
                    compile=self.options["compile"],
                    async_loading_frames=False,
                )
            )
        elif self.name == "livos":
            try:
                from livos.model.livos_wrapper import LIVOS
                from livos.model.utils import resnet
            except ImportError as exc:
                raise PipelineError(
                    "LiVOS API недоступен; выполните uv run document-vos setup --model livos"
                ) from exc
            # Build the architecture without pretrained ResNet weights; the complete
            # LiVOS checkpoint below must provide every parameter.
            constructors = {name: getattr(resnet, name) for name in ("resnet18", "resnet50")}

            def without_pretrained(constructor):
                def build(*args, **kwargs):
                    kwargs["pretrained"] = False
                    return constructor(*args, **kwargs)

                return build

            for name, constructor in constructors.items():
                setattr(resnet, name, without_pretrained(constructor))
            try:
                self.predictor = LIVOS(model_type="base").to(self.device).eval()
            finally:
                for name, constructor in constructors.items():
                    setattr(resnet, name, constructor)
            try:
                state = torch.load(self.weights, map_location="cpu", weights_only=True)
                self.predictor.load_weights(state)
                target_state = self.predictor.state_dict()
                if set(state) != set(target_state) or any(
                    state[key].shape != target_state[key].shape for key in target_state
                ):
                    missing = sorted(set(target_state) - set(state))[:8]
                    unexpected = sorted(set(state) - set(target_state))[:8]
                    raise ValueError(f"missing={missing}; unexpected={unexpected}")
                self.predictor.load_state_dict(state, strict=True)
            except Exception as exc:
                raise PipelineError(f"LiVOS checkpoint неполон или несовместим: {exc}") from exc
        else:
            raise PipelineError(f"Неизвестная модель: {self.name}")

    def reset(self):
        return None

    def sync(self):
        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()

    def memory_stats(self):
        import torch

        if self.device != "cuda":
            return {
                "device": self.device,
                "allocated_bytes_peak": None,
                "reserved_bytes_peak": None,
            }
        return {
            "device": self.device,
            "allocated_bytes_peak": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes_peak": int(torch.cuda.max_memory_reserved()),
        }

    def predict_sequence(self, frames, initial_mask, cache_dir):
        """Compatibility API for standalone diagnostics; propagation uses the iterator."""
        return list(self.iter_sequence(frames, initial_mask, cache_dir))

    def iter_sequence(self, frames, initial_mask, cache_dir):
        """Yield ordered masks as soon as each frame is ready, with bounded inputs."""
        if cache_dir.exists():
            raise PipelineError(f"Временный model input уже существует: {cache_dir}")
        import torch
        import torch.nn.functional as F

        from document_vos.benchmark_inputs import FrameStore, Geometry

        timer = ComputeTimer(self.sync)
        self.last_model_seconds = 0.0
        store = geometry = None
        try:
            model_mask = initial_mask
            if self.match_cutie_size is not None:
                with Image.open(frames[0].path) as image:
                    geometry = Geometry.cutie(
                        (image.height, image.width),
                        self.match_cutie_size,
                        stride=14 if self.name == "sam31" else 16,
                    )
                normalization = {}
                if self.name == "efficienttam":
                    normalization = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
                elif self.name == "sam31":
                    normalization = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
                store = FrameStore(frames, geometry, **normalization)
                self.geometry_records.append(geometry.to_dict())
                model_mask = geometry.mask(initial_mask)
                from document_vos.benchmark_geometry import set_geometry, set_sam31_geometry

                if self.name == "efficienttam":
                    set_geometry(self.predictor, geometry, 16)
                elif self.name == "sam31":
                    set_sam31_geometry(self.predictor.predictor, geometry)
            # Native mode retains the legacy JPEG directory API. The matched
            # benchmark decodes original RGB files, as CUTIE does.
            frame_input = store
            if store is None and self.name != "livos":
                frame_input = _frames_directory(frames, cache_dir)
            enabled = self.device == "cuda" and self.options.get("amp", False)
            dtype = torch.bfloat16 if self.options.get("dtype") == "bfloat16" else torch.float16
            precision = (
                torch.autocast("cuda", dtype=dtype, enabled=enabled)
                if self.device == "cuda"
                else nullcontext()
            )
            with precision:
                targets = []
                if self.name == "efficienttam":
                    iterator = self._efficienttam(frame_input, model_mask, timer)
                elif self.name == "sam31":
                    shapes = []
                    for frame in frames:
                        if geometry is not None:
                            shapes.append(geometry.padded)
                        else:
                            with Image.open(frame.path) as image:
                                shapes.append((image.height, image.width))
                    model = self.predictor.predictor.model
                    targets = [
                        (model, "run_backbone_and_detection"),
                        *(
                            (model.tracker, method)
                            for method in ("forward_image", "track_step", "_encode_new_memory")
                        ),
                    ]

                    def sam_masks():
                        api = (
                            getattr(self.predictor, "iter_predict", None) or self.predictor.predict
                        )
                        yield from api(frame_input, model_mask, shapes, self.options)

                    iterator = sam_masks()
                else:
                    iterator = self._livos(frames, initial_mask, torch, F, timer, store=store)
                count = 0
                try:
                    with timer.wrap(targets):
                        for count, mask in enumerate(iterator, 1):
                            if count > len(frames):
                                raise PipelineError(f"{self.name}: лишние предсказания")
                            if geometry is not None and self.name != "livos":
                                mask = (
                                    (geometry.restore_logits(mask) > 0).astype(np.uint8)
                                    if self.name == "efficienttam"
                                    else geometry.restore_binary(mask)
                                )
                            with Image.open(frames[count - 1].path) as image:
                                shape = (image.height, image.width)
                            if mask.shape != shape:
                                mask = np.asarray(
                                    Image.fromarray(mask).resize(
                                        shape[::-1], Image.Resampling.NEAREST
                                    )
                                )
                            self.last_model_seconds = timer.seconds
                            yield (mask >= 0.5).astype(np.uint8)
                    if count != len(frames):
                        raise PipelineError(f"{self.name}: неполные предсказания по кадрам")
                finally:
                    iterator.close()
        finally:
            self.last_model_seconds = timer.seconds
            if store is not None:
                store.close()
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _efficienttam(self, frame_input, initial_mask, timer):
        import torch

        state = None
        try:
            with timer.wrap(
                [
                    (self.predictor, method)
                    for method in ("forward_image", "track_step", "_encode_new_memory")
                ]
            ):
                if self.device == "cuda" and self.options.get("compile", False):
                    torch.compiler.cudagraph_mark_step_begin()
                state = self.predictor.init_state(
                    video_path=frame_input
                    if self.match_cutie_size is not None
                    else str(frame_input),
                    offload_video_to_cpu=self.options.get("offload_video_to_cpu", True),
                    offload_state_to_cpu=self.options.get("offload_state_to_cpu", False),
                )
                self.predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=1, mask=initial_mask.astype(bool)
                )
                expected = 0
                stream = self.predictor.propagate_in_video(state)
                while True:
                    if self.device == "cuda" and self.options.get("compile", False):
                        torch.compiler.cudagraph_mark_step_begin()
                    try:
                        frame_idx, obj_ids, logits = next(stream)
                    except StopIteration:
                        break
                    if frame_idx != expected or list(obj_ids) != [1]:
                        raise PipelineError("EfficientTAM: неверный порядок кадров или ID объекта")
                    expected += 1
                    scores = logits[0, 0].detach().float().cpu().numpy()
                    yield (
                        scores
                        if self.match_cutie_size is not None
                        else (scores > 0).astype(np.uint8)
                    )
        finally:
            if state is not None:
                self.predictor.reset_state(state)

    def close(self):
        if self.name == "sam31" and self.predictor is not None:
            from document_vos.sam31 import cleanup_preserving_error

            predictor, self.predictor = self.predictor, None
            cleanup_preserving_error(predictor.close)

    def _livos(self, frames, initial_mask, torch, F, timer, *, store=None):
        with torch.inference_mode():
            from livos.eval import InferenceCore
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode

            # Match LiVOS VideoReader/ResultSaver transforms and final argmax.
            core = InferenceCore(self.predictor)
            size = int(self.options["max_internal_size"])
            image_to_tensor = transforms.ToTensor()
            image_resize = transforms.Resize(
                size, interpolation=InterpolationMode.BILINEAR, antialias=True
            )
            mask_resize = transforms.Resize(
                size, interpolation=InterpolationMode.NEAREST, antialias=True
            )
            from document_vos.cutie.inference import _read_image
            from document_vos.parallel import ordered_map
            from document_vos.runtime import current_runtime

            runtime = current_runtime()
            images = (
                (store[index] for index in range(len(frames)))
                if store is not None
                else ordered_map(
                    _read_image,
                    frames,
                    max(1, runtime.io_threads // 2),
                    threads=runtime.io_threads > 1,
                )
            )
            for index, image in enumerate(images):
                original = store.geometry.original if store is not None else image.shape[:2]
                tensor = image if store is not None else image_resize(image_to_tensor(image))
                mask = None
                objects = None
                if index == 0:
                    mask = (
                        torch.from_numpy(store.geometry.mask(initial_mask)).long()
                        if store is not None
                        else mask_resize(torch.from_numpy(initial_mask.copy()).long().unsqueeze(0))[
                            0
                        ]
                    )
                    mask[mask > 0] = 1
                    objects = [1]
                tensor = tensor.to(self.device)
                mask = mask.to(self.device) if mask is not None else None
                with timer.measure():
                    prob = core.step(tensor, mask, objects, is_last_frame=index == len(frames) - 1)
                if store is not None:
                    prob = store.geometry.crop(prob)
                probabilities = F.interpolate(
                    prob.unsqueeze(1), original, mode="bilinear", align_corners=False
                )[:, 0]
                yield (probabilities.argmax(dim=0) == 1).to(torch.uint8).cpu().numpy()


def _input(cfg, *, sample, input_dir, manifest, scope, split_dir):
    if sample:
        if input_dir or manifest:
            raise PipelineError("--sample несовместим с --input-dir/--manifest")
        from document_vos.data.samples import catalog, select, verify
        from document_vos.data.samples import manifest as make_manifest

        selected = cfg["samples"]
        cat = catalog(Path(selected["catalog"]) if selected["catalog"] else None)
        entries = select(cat, selected["dataset"], selected["count"])
        for entry in entries:
            directory = Path(cfg["paths"]["samples"]) / entry["id"]
            if not directory.is_dir():
                raise PipelineError("Нет примера; выполните document-vos download --sample")
            verify(directory, entry)
        manifest_dir = Path(cfg["paths"]["prepared"]) / "benchmark-sample-manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{__import__('uuid').uuid4().hex}.jsonl"
        make_manifest(
            Path(cfg["paths"]["samples"]),
            manifest_path,
            cat,
            selected["dataset"],
            selected["count"],
        )
        return manifest_path, "validation_sample", None
    if manifest:
        actual_scope = scope or ("validation" if split_dir else "independent")
        return manifest, actual_scope, split_dir
    if input_dir:
        from document_vos.data import save_manifest
        from document_vos.data.input_folder import select_input_folder
        from document_vos.videos import prepare

        folder = (
            input_dir
            if (input_dir / "manifest.jsonl").exists()
            else Path(prepare(input_dir, Path(cfg["paths"]["prepared"]), cfg)["output"])
        )
        sequences, _ = select_input_folder(folder)
        if not sequences:
            raise PipelineError("Нет пригодных видео с первой маской")
        manifest_dir = Path(cfg["paths"]["prepared"]) / "benchmark-input-manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        selected_manifest = manifest_dir / f"{__import__('uuid').uuid4().hex}.jsonl"
        save_manifest(selected_manifest, sequences)
        return selected_manifest, "independent", None
    if scope == "independent":
        raise PipelineError("independent требует --input-dir либо --manifest")
    from document_vos.data.preparation import prepare

    prepared = Path(
        prepare(cfg, canonical=cfg["datasets"]["selected"] == DEFAULTS["datasets"]["selected"])[
            "output"
        ]
    )
    selected_split = split_dir or prepared / "split"
    return selected_split / "validation.jsonl", "validation", selected_split


def _run_model(
    cfg,
    model,
    manifest,
    scope,
    split_dir,
    output,
    seed,
    device,
    weights=None,
    *,
    match_cutie_size=None,
    benchmark_protocol_plan=None,
):
    from document_vos.runtime import Runtime, configure

    configure(Runtime.parse(cfg["runtime"]))
    import torch

    from document_vos.workflows import weight

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.manual_seed_all(seed)

    options = cfg["benchmark"]["models"][model]
    if model in {"baseline", "finetuned"}:
        weights = weight(cfg, model, weights)
        root = check_checkout(Path(cfg["paths"]["cutie"]))
        from document_vos.cutie import inference_config

        predictor = CutiePredictor(
            root,
            weights,
            inference_config(root, {**cfg["inference"], **options["inference"]}),
            device,
        )
        revision = CUTIE_REVISION
        backend = "official-cutie"
        inf_cfg = inference_config(root, {**cfg["inference"], **options["inference"]})
    else:
        weights = weights or Path(options["weights"])
        if not weights.is_file():
            raise PipelineError(
                f"Нет весов {model}: выполните uv run document-vos setup --model {model}"
            )
        predictor = ExternalPredictor(
            model, cfg, weights, device, match_cutie_size=match_cutie_size
        )
    try:
        if model not in {"baseline", "finetuned"}:
            from document_vos.config import resource

            catalog = json.loads(resource("external_models.json").read_text(encoding="utf-8"))
            revision = catalog[model]["revision"]
            backend = model
            resolution = (
                int(options["max_internal_size"])
                if model == "livos"
                else getattr(predictor.predictor, "image_size", "architecture default")
            )
            inf_cfg = {**options, "internal_resolution": resolution}
            if model == "efficienttam":
                inf_cfg["predictor_class"] = type(predictor.predictor).__name__
                inf_cfg["vos_optimized"] = (
                    inf_cfg["predictor_class"] == "EfficientTAMVideoPredictorVOS"
                )
            if model == "sam31":
                inf_cfg.update(predictor.predictor.metadata)
            if match_cutie_size is not None:
                from document_vos.benchmark_geometry import ADAPTER_VERSION

                inf_cfg.update(
                    {
                        "resolution_policy": "cutie",
                        "reference_max_internal_size": match_cutie_size,
                        "internal_resolution": "per-sequence; see geometry",
                        "geometry_adapter": ADAPTER_VERSION,
                        "geometry": predictor.geometry_records,
                        "input_decode": "original RGB; no JPEG transcoding",
                        "input_storage": "bounded CPU prefetch",
                    }
                )
                if model == "sam31":
                    inf_cfg["initialization_type"] = (
                        "original_binary_mask with CUTIE resize/padding"
                    )
                    inf_cfg["tracker_mask_input_resolution"] = "per-sequence rectangular grid"
        sequences = load_manifest(manifest)
        split_hash = validate_scope(manifest, scope, split_dir, sequences)
        metadata = {
            "model": model,
            "backend": backend,
            "backend_revision": revision,
            "cutie_revision": revision if model in {"baseline", "finetuned"} else None,
            "weights": str(weights.resolve()),
            "weights_sha256": sha256(weights),
            "inference": inf_cfg,
            "seed": seed,
            "device": device,
            "versions": versions(),
            "scope": scope,
            "split_sha256": split_hash,
            "runtime": __import__("document_vos.runtime", fromlist=["current_runtime"])
            .current_runtime()
            .to_dict(),
        }
        if benchmark_protocol_plan is not None:
            from document_vos.timing import execution_environment

            metadata["benchmark_protocol_plan"] = benchmark_protocol_plan
            metadata["execution_environment"] = execution_environment(device)
        return propagate(manifest, output, predictor, metadata, sequences=sequences)
    finally:
        if isinstance(predictor, ExternalPredictor):
            predictor.close()


def inference(
    cfg: dict,
    model: str,
    *,
    sample=False,
    input_dir=None,
    manifest=None,
    split_dir=None,
    scope=None,
    output=None,
    seed=42,
    weights=None,
):
    from document_vos.runtime import Runtime, configure

    configure(Runtime.parse(cfg["runtime"]))
    if model not in MODELS:
        raise PipelineError(f"Неизвестная модель: {model}")
    manifest_path, scope, split_dir = _input(
        cfg, sample=sample, input_dir=input_dir, manifest=manifest, scope=scope, split_dir=split_dir
    )
    device = __import__("document_vos.runtime", fromlist=["resolve_device"]).resolve_device(
        cfg["runtime"]["device"]
    )
    result_root = Path(cfg["paths"]["results"])
    if output:
        target = output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = result_root / f"infer-{model}-{__import__('uuid').uuid4().hex[:8]}"
    result = _run_model(
        cfg, model, manifest_path, scope, split_dir, target, seed, device, weights=weights
    )
    from document_vos.rendering import bundle

    bundle(target, target / "visualization", **cfg["visualization"])
    return result


def _compare_evaluations(evaluations: dict[str, dict]) -> dict:
    names = list(evaluations)
    first = evaluations[names[0]]
    reference_run = first["run"]
    reference_frames = {(r["video"], r["frame"]) for r in first["frames"]}
    reference_initializations = reference_run["signature"]["initializations"]
    for name, data in evaluations.items():
        run = data["run"]
        if run["signature"]["inputs_sha256"] != reference_run["signature"]["inputs_sha256"]:
            raise PipelineError(f"{name}: входные кадры/маски отличаются от первого участника")
        if run["signature"]["manifest_sha256"] != reference_run["signature"]["manifest_sha256"]:
            raise PipelineError(f"{name}: манифест отличается от первого участника")
        if run["signature"]["initializations"] != reference_initializations:
            raise PipelineError(f"{name}: первая маска/точка инициализации отличаются")
        if data["metric_protocol"] != first["metric_protocol"]:
            raise PipelineError(f"{name}: протокол метрик отличается")
        if {(r["video"], r["frame"]) for r in data["frames"]} != reference_frames:
            raise PipelineError(f"{name}: набор оцениваемых кадров отличается")
    return {
        "status": "COMPLETE",
        "participants": names,
        "metric_protocol": first["metric_protocol"],
        "selection": first["selection"],
        "evaluations": evaluations,
        "speed_compatibility": speed_compatibility(
            {name: data["run"] for name, data in evaluations.items()}
        ),
    }


def benchmark(cfg: dict, *, models=None, resume: Path | None = None, manifest=None, split_dir=None):
    from document_vos.runtime import Runtime, configure

    configure(Runtime.parse(cfg["runtime"]))
    selected = list(models or cfg["benchmark"]["selected"])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(MODELS):
        raise PipelineError("--models: укажите уникальные поддерживаемые модели")
    from document_vos.timing import benchmark_plan

    plan = benchmark_plan(cfg, selected)
    match_cutie_size = None
    if cfg["benchmark"]["resolution"] == "cutie":
        match_cutie_size = cfg["inference"]["max_internal_size"]
        for model in selected:
            if model in {"baseline", "finetuned"}:
                actual = cfg["benchmark"]["models"][model]["inference"].get(
                    "max_internal_size", match_cutie_size
                )
                if actual != match_cutie_size:
                    raise PipelineError(
                        "benchmark.resolution=cutie требует общего max_internal_size у CUTIE"
                    )
    manifest_path, scope, split_dir = _input(
        cfg,
        sample=False,
        input_dir=None,
        manifest=manifest
        or (Path(cfg["benchmark"]["manifest"]) if cfg["benchmark"]["manifest"] else None),
        scope=None,
        split_dir=split_dir
        or (Path(cfg["benchmark"]["split_dir"]) if cfg["benchmark"]["split_dir"] else None),
    )
    if scope == "validation" and split_dir is None:
        raise PipelineError("Validation benchmark требует подтверждающий split_dir")
    sequences = load_manifest(manifest_path)
    inputs = {
        "manifest_sha256": sha256(manifest_path),
        "inputs": snapshot_inputs(sequences),
        "models": selected,
        "scope": scope,
    }
    resolved = json.loads(json.dumps(cfg))
    resolved["benchmark"]["selected"] = selected
    run = Run(Path(cfg["paths"]["results"]), "benchmark", resolved, inputs, resume=resume)
    evaluations = {}
    failures = {}
    from document_vos.evaluation import evaluate
    from document_vos.rendering import bundle
    from document_vos.runtime import resolve_device

    device = resolve_device(cfg["runtime"]["device"])
    for model in selected:

        def action(target, model=model):
            try:
                inference_dir = target / "inference"
                _run_model(
                    cfg,
                    model,
                    manifest_path,
                    scope,
                    split_dir,
                    inference_dir,
                    cfg["benchmark"]["seed"],
                    device,
                    match_cutie_size=match_cutie_size,
                    benchmark_protocol_plan=plan,
                )
                result = evaluate(
                    inference_dir, target / "evaluation", workers=cfg["runtime"]["workers"]
                )
                data = read_json(Path(result["output"]) / "evaluation.json")
                evaluations[model] = data
                bundle(inference_dir, target / "visualization", **cfg["visualization"])
                write_json(
                    target / "participant.json",
                    {
                        "status": "COMPLETE",
                        "model": model,
                        "inference": str(inference_dir),
                        "evaluation": str(target / "evaluation"),
                    },
                )
            except Exception as exc:
                failures[model] = f"{type(exc).__name__}: {exc}"
                write_json(
                    target / "participant.json",
                    {"status": "FAILED", "model": model, "error": failures[model]},
                )
                raise

        try:
            stage = run.stage(model, action)
        except Exception as exc:
            failures[model] = f"{type(exc).__name__}: {exc}"
            continue
        participant = read_json(stage / "participant.json")
        if participant["status"] == "COMPLETE":
            evaluations[model] = read_json(stage / "evaluation" / "evaluation.json")
        else:
            failures[model] = participant["error"]
    status = "COMPLETE" if not failures and len(evaluations) == len(selected) else "PARTIAL"
    result = {
        "status": status,
        "speed_compatibility": speed_compatibility(
            {name: item["run"] for name, item in evaluations.items()}
        ),
        "participants": selected,
        "failures": failures,
        "evaluations": {
            name: {"path": f"{name}/evaluation/evaluation.json", "summary": values["summary"]}
            for name, values in evaluations.items()
        },
    }
    if evaluations:
        try:
            result.update(_compare_evaluations(evaluations))
            result["status"] = status
            result["deltas_vs_cutie"] = {}
            for model, data in evaluations.items():
                result["deltas_vs_cutie"][model] = {}
                candidate = data["summary"]["primary_equal_datasets"]
                for reference in ("baseline", "finetuned"):
                    if model == reference or reference not in evaluations:
                        continue
                    ref = evaluations[reference]["summary"]["primary_equal_datasets"]
                    result["deltas_vs_cutie"][model][reference] = {
                        metric: candidate[metric] - ref[metric]
                        if candidate[metric] is not None and ref[metric] is not None
                        else None
                        for metric in ("J", "F", "JF", "BIoU")
                    }
        except PipelineError as exc:
            result["status"] = "INCOMPATIBLE"
            result["comparison_error"] = str(exc)
    atomic_json(run.output / "benchmark.json", result)
    _write_csv(run.output / "comparison.csv", selected, result)
    _write_html(run.output / "comparison.html", result)
    if failures:
        # Failed stages remain retryable by --resume; successful stages are
        # inventory-verified and reused on the next attempt.
        run.state["status"] = result["status"]
        run.save()
        final = {
            "status": result["status"],
            "output": str(run.output.resolve()),
            "summary": str(run.output / "summary.md"),
            "index": str(run.output / "index.html"),
        }
    else:
        final = run.finish(status=result["status"])
    _write_html(run.output / "index.html", result)
    return final


def _write_html(path: Path, result: dict) -> None:
    rows = []
    speed = result.get("speed_compatibility") or speed_compatibility(
        {name: item.get("run", {}) for name, item in result["evaluations"].items()}
    )
    for model, item in result["evaluations"].items():
        summary = item.get("summary")
        if summary is None:
            continue
        values = summary["primary_equal_datasets"]
        run = item.get("run", {})
        inference = run.get("inference", {})
        runtime_settings = {
            key: inference[key]
            for key in (
                "internal_resolution",
                "max_internal_size",
                "amp",
                "dtype",
                "compile",
                "offload_video_to_cpu",
                "offload_state_to_cpu",
                "use_fa3",
            )
            if key in inference
        }
        timing = run.get("timing", {})
        speed_info = speed["participants"].get(model, {"status": "INCOMPATIBLE", "reasons": []})
        inference_dir = Path(item["run_dir"]) if item.get("run_dir") else None
        visualization_link = ""
        if inference_dir is not None:
            visualization = inference_dir.parent / "visualization"
            if visualization.is_dir():
                relative = visualization.relative_to(path.parent)
                visualization_link = (
                    f'<a href="{html.escape(relative.as_posix())}/">visual examples</a><br>'
                )
        rows.append(
            "<tr><th>"
            + visualization_link
            + html.escape(model)
            + "</th>"
            + f"<td>{html.escape(json.dumps(runtime_settings, ensure_ascii=False))}</td>"
            + f"<td>{html.escape(speed_info['status'] + ': ' + '; '.join(speed_info['reasons']))}</td>"
            + f"<td>{timing.get('model_fps', 'NOT_MEASURED')}</td>"
            + f"<td>{timing.get('full_fps', 'NOT_MEASURED')}</td>"
            + "".join(
                f"<td>{values[metric]:.4f}</td>" if values[metric] is not None else "<td>—</td>"
                for metric in ("J", "F", "JF", "BIoU")
            )
            + "</tr>"
        )
    table = "".join(rows) or "<tr><td colspan=9>Нет завершённых оценок</td></tr>"
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Сравнение document-vos</title>'
        "<h1>Сравнение VOS моделей</h1><p>Статус: " + result["status"] + "</p>"
        "<p>Совместимость скорости: " + speed["status"] + ". Несовместимые FPS — исторические "
        "измерения; они не используются для сравнительных выводов. Метрики качества оцениваются отдельно.</p>"
        "<table border=1><thead><tr><th>Модель</th><th>Разрешённые настройки</th>"
        "<th>Совместимость FPS</th><th>FPS модели (измерено)</th><th>FPS полного прохода (измерено)</th><th>J</th><th>F</th><th>J&amp;F</th>"
        "<th>Boundary IoU</th></tr></thead><tbody>" + table + "</tbody></table>"
        "<h2>Дельты относительно Cutie</h2><pre>"
        + html.escape(json.dumps(result.get("deltas_vs_cutie", {}), ensure_ascii=False, indent=2))
        + "</pre>"
        "<h2>Ошибки</h2><pre>"
        + html.escape(json.dumps(result.get("failures", {}), ensure_ascii=False, indent=2))
        + "</pre>",
        encoding="utf-8",
    )


def _write_csv(path: Path, selected: list[str], result: dict) -> None:
    fields = [
        "model",
        "status",
        "J",
        "F",
        "JF",
        "BIoU",
        "model_fps",
        "full_fps",
        "speed_compatibility",
        "speed_incompatibility_reasons",
        "timing_protocol",
        "inference_settings",
        "delta_vs_baseline",
        "delta_vs_finetuned",
        "error",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    speed = result.get("speed_compatibility") or speed_compatibility(
        {name: item.get("run", {}) for name, item in result["evaluations"].items()}
    )
    for model in selected:
        item = result["evaluations"].get(model, {})
        summary = item.get("summary", {})
        values = summary.get("primary_equal_datasets", {})
        run = item.get("run", {})
        deltas = result.get("deltas_vs_cutie", {}).get(model, {})
        writer.writerow(
            {
                "model": model,
                "status": "COMPLETE" if summary else "FAILED",
                **{metric: values.get(metric) for metric in ("J", "F", "JF", "BIoU")},
                "model_fps": run.get("timing", {}).get("model_fps"),
                "full_fps": run.get("timing", {}).get("full_fps"),
                "speed_compatibility": speed["participants"]
                .get(model, {})
                .get("status", "INCOMPATIBLE"),
                "speed_incompatibility_reasons": "; ".join(
                    speed["participants"].get(model, {}).get("reasons", [])
                ),
                "timing_protocol": run.get("timing", {}).get("protocol_version"),
                "inference_settings": json.dumps(run.get("inference", {}), ensure_ascii=False),
                "delta_vs_baseline": json.dumps(deltas.get("baseline"), ensure_ascii=False),
                "delta_vs_finetuned": json.dumps(deltas.get("finetuned"), ensure_ascii=False),
                "error": result.get("failures", {}).get(model),
            }
        )
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")
