"""Versioned compute-only timers for pinned model APIs; no vendor changes."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter

from document_vos.io import PipelineError, digest

TIMING_PROTOCOL = "document-vos/compute-v3"
BENCHMARK_PROTOCOL = "document-vos/benchmark-fps-v1"
MODEL_BOUNDARIES = {
    "official-cutie": "InferenceCore.step",
    "livos": "InferenceCore.step",
    "efficienttam": "forward_image + track_step + _encode_new_memory (nested calls counted once)",
    "sam31": "run_backbone_and_detection + tracker.forward_image + tracker.track_step + tracker._encode_new_memory (nested calls counted once)",
}
FULL_BOUNDARY = (
    "sequence processing through completed PNG writes and final CUDA synchronization; "
    "includes input decode/conversion, JPEG preparation when required, mask export and cache cleanup; "
    "excludes checkpoint load, input preflight/hashes, evaluator and visualization; no extra warmup"
)


def protocol(backend: str) -> dict:
    return {
        "protocol_version": TIMING_PROTOCOL,
        "model_boundary": MODEL_BOUNDARIES[backend],
        "full_boundary": FULL_BOUNDARY,
        "synchronization": "CUDA sync before and after each outermost model call and full pass",
    }


def benchmark_plan(cfg: dict, models: list[str]) -> dict:
    """Common, reproducible comparison contract, including all participants' settings."""
    from copy import deepcopy

    from document_vos.benchmark_geometry import ADAPTER_VERSION
    from document_vos.runs import source_signature

    return {
        "version": BENCHMARK_PROTOCOL,
        "timing_protocol": TIMING_PROTOCOL,
        "source_sha256": source_signature(),
        "resolution_policy": cfg["benchmark"]["resolution"],
        "seed": cfg["benchmark"]["seed"],
        "geometry_adapter": ADAPTER_VERSION if cfg["benchmark"]["resolution"] == "cutie" else None,
        "input_pipeline": "bounded-original-rgb-v1"
        if cfg["benchmark"]["resolution"] == "cutie"
        else "native-model-input-v1",
        "reference_inference": deepcopy(cfg["inference"]),
        "models": {name: deepcopy(cfg["benchmark"]["models"][name]) for name in sorted(models)},
        "runtime": deepcopy(cfg["runtime"]),
    }


def execution_environment(device: str) -> dict:
    import torch

    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_autocast_active": bool(torch.is_autocast_enabled("cuda")),
    }


def seal_benchmark_protocol(plan: dict, metadata: dict) -> dict:
    """Seal after processing so the effective per-sequence geometry is included."""
    execution = {
        key: metadata.get(key)
        for key in (
            "model",
            "backend",
            "backend_revision",
            "weights_sha256",
            "seed",
            "device",
            "inference",
            "execution_environment",
        )
    }
    return {
        "plan": plan,
        "plan_sha256": digest(plan),
        "execution": execution,
        "execution_sha256": digest(execution),
    }


def _benchmark_evidence(run: dict) -> tuple[dict | None, list[str]]:
    proof = run.get("benchmark_protocol")
    if not isinstance(proof, dict):
        return None, ["missing benchmark protocol signature"]
    plan, execution = proof.get("plan"), proof.get("execution")
    if not isinstance(plan, dict) or not isinstance(execution, dict):
        return None, ["missing benchmark protocol plan or execution evidence"]
    if (
        plan.get("version") != BENCHMARK_PROTOCOL
        or plan.get("timing_protocol") != TIMING_PROTOCOL
        or not plan.get("source_sha256")
        or plan.get("resolution_policy") not in {"cutie", "native"}
        or proof.get("plan_sha256") != digest(plan)
        or proof.get("execution_sha256") != digest(execution)
    ):
        return None, ["invalid or different benchmark protocol signature"]
    keys = (
        "model",
        "backend",
        "backend_revision",
        "weights_sha256",
        "seed",
        "device",
        "inference",
        "execution_environment",
    )
    if any(execution.get(key) != run.get(key) for key in keys):
        return None, ["execution evidence differs from run metadata"]
    model = execution.get("model")
    models = plan.get("models")
    if not isinstance(models, dict) or model not in models:
        return None, ["model absent from benchmark protocol plan"]
    expected_backend = "official-cutie" if model in {"baseline", "finetuned"} else model
    if (
        execution.get("backend") != expected_backend
        or execution.get("seed") != plan.get("seed")
        or not execution.get("weights_sha256")
    ):
        return None, ["missing or different backend, seed or checkpoint evidence"]
    options, inference = models[model], execution.get("inference")
    if not isinstance(options, dict) or not isinstance(inference, dict):
        return None, ["missing model execution settings"]
    if model in {"baseline", "finetuned"}:
        expected = {**plan.get("reference_inference", {}), **options.get("inference", {})}
        if "long_term" in options.get("inference", {}):
            expected["long_term"] = {
                **plan.get("reference_inference", {}).get("long_term", {}),
                **options["inference"]["long_term"],
            }
    else:
        expected = {key: value for key, value in options.items() if key != "weights"}
    if any(inference.get(key) != value for key, value in expected.items()):
        return None, ["effective model settings differ from benchmark plan"]
    if plan.get("resolution_policy") == "cutie" and model not in {"baseline", "finetuned"}:
        if (
            inference.get("resolution_policy") != "cutie"
            or inference.get("geometry_adapter") != plan.get("geometry_adapter")
            or inference.get("reference_max_internal_size")
            != plan.get("reference_inference", {}).get("max_internal_size")
            or not inference.get("geometry")
        ):
            return None, ["missing or different effective resolution/geometry evidence"]
        from document_vos.benchmark_inputs import Geometry

        geometry = inference["geometry"]
        reference_size = inference["reference_max_internal_size"]
        if (
            not isinstance(geometry, list)
            or type(reference_size) is not int
            or (reference_size != -1 and reference_size < 16)
        ):
            return None, ["invalid per-sequence geometry evidence"]
        content_geometry = []
        for item in geometry:
            if not isinstance(item, dict):
                return None, ["invalid per-sequence geometry evidence"]
            original = item.get("original_hw")
            if (
                not isinstance(original, (list, tuple))
                or len(original) != 2
                or any(type(x) is not int or x < 1 for x in original)
            ):
                return None, ["invalid original frame geometry"]
            expected_geometry = Geometry.cutie(
                tuple(original),
                reference_size,
                stride=14 if model == "sam31" else 16,
            ).to_dict()
            if digest(item) != digest(expected_geometry):
                return None, ["effective geometry does not match CUTIE resolution policy"]
            content_geometry.append({"original_hw": original, "resized_hw": item["resized_hw"]})
    else:
        content_geometry = None
    if model == "efficienttam" and inference.get("vos_optimized") != bool(options.get("compile")):
        return None, ["EfficientTAM optimized predictor differs from requested compile setting"]
    env = execution.get("execution_environment")
    if (
        not isinstance(env, dict)
        or not env.get("device_name")
        or not env.get("torch_version")
        or "cuda_version" not in env
    ):
        return None, ["missing actual execution environment"]
    inputs = run.get("signature")
    if (
        not isinstance(inputs, dict)
        or not inputs.get("inputs_sha256")
        or not inputs.get("manifest_sha256")
        or not isinstance(inputs.get("initializations"), list)
        or not inputs["initializations"]
    ):
        return None, ["missing input and initialization evidence"]
    if content_geometry is not None and len(content_geometry) != len(inputs["initializations"]):
        return None, ["geometry count differs from initialized sequences"]
    workload = {key: inputs[key] for key in ("inputs_sha256", "manifest_sha256", "initializations")}
    return {
        "plan_sha256": proof["plan_sha256"],
        "execution": execution,
        "workload": digest(workload),
        "content_geometry": digest(content_geometry) if content_geometry is not None else None,
    }, []


class ComputeTimer:
    """One shared depth across encoder/tracker wrappers prevents nested double counting."""

    def __init__(self, sync, clock=None):
        self.sync = sync
        self.clock = clock or perf_counter
        self.seconds = 0.0
        self.calls = 0
        self.depth = 0

    @contextmanager
    def measure(self):
        outer = self.depth == 0
        if outer:
            self.sync()
            start = self.clock()
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if outer:
                self.sync()
                self.seconds += self.clock() - start
                self.calls += 1

    @contextmanager
    def wrap(self, targets):
        """Temporarily wrap bound instance methods; restore even after errors."""
        originals = []
        try:
            for owner, name in targets:
                method = getattr(owner, name, None)
                if not callable(method):
                    raise PipelineError(f"{TIMING_PROTOCOL}: pinned API отсутствует: {name}")
                local = name in vars(owner)
                originals.append((owner, name, local, method))

                @wraps(method)
                def timed(*args, _method=method, **kwargs):
                    with self.measure():
                        return _method(*args, **kwargs)

                setattr(owner, name, timed)
            yield self
        finally:
            for owner, name, local, method in reversed(originals):
                if local:
                    setattr(owner, name, method)
                else:
                    delattr(owner, name)


def speed_compatibility(runs: dict[str, dict]) -> dict:
    """Quality compatibility is independent. Unknown speed evidence fails closed."""
    participants = {}
    keys = {}
    benchmark_evidence = {}
    for name, run in runs.items():
        reasons = []
        timing = run.get("timing", {})
        backend = run.get("backend", "official-cutie")
        if backend not in MODEL_BOUNDARIES or any(
            timing.get(key) != value for key, value in protocol(backend).items()
        ):
            reasons.append("missing or different timing protocol/boundaries")
        evidence, evidence_reasons = _benchmark_evidence(run)
        reasons.extend(evidence_reasons)
        if evidence is not None:
            benchmark_evidence[name] = evidence
        measured = run.get("cpu_runtime", {})
        pools = measured.get("pools", {})
        cores = measured.get("physical_cores", [])
        affinity = measured.get("affinity", [])
        core_cpus = [cpu for core in cores for cpu in core.get("cpus", [])]
        if (
            measured.get("cpu_profile") != "p6-smt12"
            or len(cores) != 6
            or any(len(core.get("cpus", [])) != 2 for core in cores)
            or len(set(core_cpus)) != 12
            or sorted(core_cpus) != affinity
            or pools.get("torch_intra_op") != 6
            or pools.get("torch_inter_op") != 1
            or pools.get("opencv") != 1
            or not pools.get("blas")
            or any(pool.get("num_threads") != 1 for pool in pools.get("blas", []))
            or any(
                measured.get(key) != value
                for key, value in (("workers", 6), ("dataloader_workers", 6), ("io_threads", 4))
            )
        ):
            reasons.append("missing or different actual p6-smt12 CPU profile/pools/workers")
        participants[name] = {
            "status": "INCOMPATIBLE" if reasons else "COMPATIBLE",
            "reasons": reasons,
        }
        if not reasons:
            keys[name] = {
                "affinity": affinity,
                "physical_cores": cores,
                "pools": {key: value for key, value in pools.items() if key != "blas"},
                "device": run.get("device"),
            }
    if keys and any(key != next(iter(keys.values())) for key in keys.values()):
        for name in keys:
            participants[name] = {
                "status": "INCOMPATIBLE",
                "reasons": ["actual CPU pools/affinity/device differ between runs"],
            }
    if benchmark_evidence:
        plans = {evidence["plan_sha256"] for evidence in benchmark_evidence.values()}
        workloads = {evidence["workload"] for evidence in benchmark_evidence.values()}
        geometries = {
            evidence["content_geometry"]
            for evidence in benchmark_evidence.values()
            if evidence["content_geometry"] is not None
        }
        devices = {
            (
                evidence["execution"]["execution_environment"]["device_name"],
                evidence["execution"]["execution_environment"]["torch_version"],
                evidence["execution"]["execution_environment"]["cuda_version"],
            )
            for evidence in benchmark_evidence.values()
        }
        same_model = {}
        for name, evidence in benchmark_evidence.items():
            model = evidence["execution"]["model"]
            same_model.setdefault(model, []).append((name, digest(evidence["execution"])))
        different_models = {
            model for model, entries in same_model.items() if len({item[1] for item in entries}) > 1
        }
        for name, evidence in benchmark_evidence.items():
            reasons = participants[name]["reasons"]
            if len(plans) > 1:
                reasons.append("benchmark protocol plans differ")
            if len(workloads) > 1:
                reasons.append("input frames, manifest or initial masks differ")
            if len(geometries) > 1:
                reasons.append("resized content geometry differs between models")
            if len(devices) > 1:
                reasons.append("GPU or PyTorch/CUDA versions differ")
            if evidence["execution"]["model"] in different_models:
                reasons.append("effective settings or geometry differ for the same model")
            if reasons:
                participants[name]["status"] = "INCOMPATIBLE"
    return {
        "status": "COMPATIBLE"
        if participants and all(item["status"] == "COMPATIBLE" for item in participants.values())
        else "INCOMPATIBLE",
        "protocol_version": TIMING_PROTOCOL,
        "participants": participants,
    }
