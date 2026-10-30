"""CPU-only evidence for topology, inherited pools, timing boundaries and old artifacts."""

import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from document_vos.io import PipelineError
from document_vos.runtime import select_cpu_cores
from document_vos.timing import ComputeTimer, protocol, speed_compatibility


def topology(root, pairs):
    for core, pair in enumerate(pairs):
        for cpu in pair:
            path = root / f"cpu{cpu}" / "topology"
            path.mkdir(parents=True)
            (path / "thread_siblings_list").write_text(",".join(map(str, pair)))
            (path / "physical_package_id").write_text("0")
            (path / "core_id").write_text(str(core))


def test_selects_complete_pairs_with_nonconsecutive_siblings(tmp_path):
    pairs = [(0, 5), (1, 2), (3, 4), (6, 7), (8, 9), (10, 11), (12,), (13,)]
    topology(tmp_path, pairs)
    assert [core["cpus"] for core in select_cpu_cores(set(range(14)), tmp_path)] == [
        list(p) for p in pairs[:6]
    ]
    for available in (set(range(11)), set(range(14)) - {5}, set(range(6))):
        with pytest.raises(PipelineError, match="6 полных SMT-пар"):
            select_cpu_cores(available, tmp_path)
    with pytest.raises(PipelineError, match="топологию"):
        select_cpu_cores({99}, tmp_path)


def run_python(script, tmp_path):
    path = tmp_path / "probe.py"
    path.write_text(script)
    result = subprocess.run([sys.executable, str(path)], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("before", [True, False])
def test_real_profile_before_after_import_and_idempotence(tmp_path, before):
    result = run_python(
        ("import torch, numpy, cv2\n" if before else "")
        + """
import json, os, subprocess, sys
from document_vos.runtime import Runtime, configure, runtime_snapshot
configure(Runtime())
a = runtime_snapshot()
configure(Runtime())
assert runtime_snapshot() == a
assert all(os.sched_getaffinity(int(t)) == set(a['affinity']) for t in os.listdir('/proc/self/task'))
child = subprocess.check_output([sys.executable, '-c', 'import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True)
assert json.loads(child) == a['affinity']
assert os.environ['OMP_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '1'
print(json.dumps(a))
""",
        tmp_path,
    )
    assert result["cpu_profile"] == "p6-smt12"
    assert len(result["affinity"]) == 12 and len(result["physical_cores"]) == 6
    assert result["pools"]["torch_intra_op"] == 6
    assert result["pools"]["torch_inter_op"] == result["pools"]["opencv"] == 1
    assert result["pools"]["blas"] and all(
        pool["num_threads"] == 1 for pool in result["pools"]["blas"]
    )
    assert (result["workers"], result["dataloader_workers"], result["io_threads"]) == (6, 6, 4)


def test_interop_failure_is_not_silently_ignored(tmp_path):
    result = run_python(
        """
import json, torch
from document_vos.runtime import Runtime, configure
from document_vos.io import PipelineError
torch.set_num_interop_threads(2)
try:
    configure(Runtime())
except PipelineError as exc:
    print(json.dumps(str(exc)))
else:
    raise AssertionError('must reject an immutable inter-op pool')
""",
        tmp_path,
    )
    assert "inter-op=1" in result


def test_restricted_affinity_fails_without_fallback(tmp_path):
    result = run_python(
        """
import json, os
from document_vos.runtime import Runtime, configure
from document_vos.io import PipelineError
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
try:
    configure(Runtime())
except PipelineError as exc:
    print(json.dumps(str(exc)))
else:
    raise AssertionError('must fail')
""",
        tmp_path,
    )
    assert "6 полных SMT-пар" in result


def test_spawn_cpu_and_loader_workers_inherit_profile_and_single_pools(tmp_path):
    result = run_python(
        """
import json, os, sys, types
from document_vos.runtime import Runtime, configure, configure_libraries, install_loader_options
from document_vos.parallel import ordered_map

def probe(i):
    return {'affinity': sorted(os.sched_getaffinity(0)), 'pools': configure_libraries(worker=True)}

def seed(i):
    assert configure_libraries(worker=True)['torch_intra_op'] == 1
    print('' if False else '', end='')

def loader(*args, **kwargs):
    from torch.utils.data import DataLoader
    return DataLoader(*args, **kwargs)

if __name__ == '__main__':
    configure(Runtime())
    affinity = sorted(os.sched_getaffinity(0))
    workers = list(ordered_map(probe, range(2), 2))
    fake = types.SimpleNamespace(DataLoader=loader)
    sys.modules['cutie.dataset'] = types.SimpleNamespace(setup_training_data=fake)
    install_loader_options(Runtime(dataloader_workers=2))
    install_loader_options(Runtime(dataloader_workers=2))
    from torch.utils.data import Dataset
    # A picklable existing dataset, while preserving its upstream worker seeding callback.
    import torch
    batches = list(fake.DataLoader(torch.utils.data.TensorDataset(torch.arange(3)), worker_init_fn=seed))
    assert len(batches) == 3
    print(json.dumps({'affinity': affinity, 'workers': workers}))
""",
        tmp_path,
    )
    for worker in result["workers"]:
        assert worker["affinity"] == result["affinity"]
        assert (
            worker["pools"]["torch_intra_op"]
            == worker["pools"]["torch_inter_op"]
            == worker["pools"]["opencv"]
            == 1
        )
        assert all(pool["num_threads"] == 1 for pool in worker["pools"]["blas"])


class Clock:
    def __init__(self):
        self.now = 0
        self.events = []

    def __call__(self):
        return self.now

    def work(self, amount):
        self.now += amount

    def sync(self):
        self.events.append(self.now)


def test_compute_timer_nested_calls_sync_and_restore_on_failure():
    clock = Clock()
    timer = ComputeTimer(clock.sync, clock)

    class Model:
        def encoder(self):
            clock.work(2)

        def memory(self):
            clock.work(3)

        def step(self):
            self.encoder()
            self.memory()
            clock.work(5)
            return "same-output"

    model = Model()
    with timer.wrap([(model, "encoder"), (model, "memory"), (model, "step")]):
        model.encoder()  # first-frame initialization
        assert model.step() == "same-output"
        assert model.step() == "same-output"
    assert timer.seconds == 22 and timer.calls == 3
    assert clock.events == [0, 2, 2, 12, 12, 22]
    assert not vars(model)
    with pytest.raises(RuntimeError):
        with timer.wrap([(model, "step")]):
            model.step()
            raise RuntimeError("original")
    assert not vars(model) and timer.depth == 0


def test_external_timers_exclude_jpeg_and_export_include_frame_zero(tmp_path, monkeypatch):
    from document_vos import benchmark, timing

    clock = Clock()
    monkeypatch.setattr(timing, "perf_counter", clock)
    original_prepare = benchmark._frames_directory

    def prepare(*args):
        clock.work(100)
        return original_prepare(*args)

    monkeypatch.setattr(benchmark, "_frames_directory", prepare)

    class Model:
        def forward_image(self):
            clock.work(2)

        def _encode_new_memory(self):
            clock.work(1)

        def track_step(self):
            clock.work(3)
            self._encode_new_memory()

        def init_state(self, **kwargs):
            clock.work(100)  # decode/convert inputs
            self.forward_image()  # frame zero, not a warmup
            return {}

        def add_new_mask(self, **kwargs):
            self.track_step()
            self._encode_new_memory()  # first-frame preflight

        def propagate_in_video(self, state):
            for i in range(2):
                if i:
                    self.forward_image()
                    self.track_step()
                clock.work(100)  # output conversion/export outside compute region
                yield i, [1], torch.ones((1, 1, 6, 9))

        def reset_state(self, state):
            clock.work(100)

    frame = tmp_path / "input.png"
    Image.new("RGB", (9, 6)).save(frame)
    adapter = benchmark.ExternalPredictor.__new__(benchmark.ExternalPredictor)
    adapter.name, adapter.device, adapter.options = "efficienttam", "cpu", {}
    adapter.predictor = Model()
    masks = adapter.predict_sequence(
        [SimpleNamespace(path=frame)] * 2, np.ones((6, 9)), tmp_path / "cache"
    )
    assert adapter.last_model_seconds == 13
    assert clock.now == 513
    assert len(masks) == 2 and all(np.array_equal(mask, np.ones((6, 9))) for mask in masks)
    assert not (tmp_path / "cache").exists()


def measured_run():
    from copy import deepcopy

    from document_vos.config import DEFAULTS
    from document_vos.timing import benchmark_plan, seal_benchmark_protocol

    cfg = deepcopy(DEFAULTS)
    run = {
        "model": "baseline",
        "backend_revision": "pinned",
        "weights_sha256": "checkpoint-hash",
        "seed": cfg["benchmark"]["seed"],
        "inference": deepcopy(cfg["inference"]),
        "signature": {
            "inputs_sha256": "inputs",
            "manifest_sha256": "manifest",
            "initializations": [{"video": "v", "mask_sha256": "mask"}],
        },
        "execution_environment": {
            "device_name": "test GPU",
            "torch_version": "test",
            "cuda_version": "test",
            "matmul_tf32": False,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
        },
        "device": "cuda",
        "backend": "official-cutie",
        "timing": protocol("official-cutie"),
        "cpu_runtime": {
            "cpu_profile": "p6-smt12",
            "affinity": list(range(12)),
            "physical_cores": [
                {"core": i, "package": 0, "cpus": [2 * i, 2 * i + 1]} for i in range(6)
            ],
            "pools": {
                "torch_intra_op": 6,
                "torch_inter_op": 1,
                "opencv": 1,
                "blas": [{"prefix": "libopenblas", "internal_api": "openblas", "num_threads": 1}],
            },
            "workers": 6,
            "dataloader_workers": 6,
            "io_threads": 4,
        },
    }
    run["benchmark_protocol"] = seal_benchmark_protocol(benchmark_plan(cfg, ["baseline"]), run)
    return run


def test_speed_compatibility_requires_matching_signed_geometry_and_execution():
    from document_vos.config import DEFAULTS
    from document_vos.timing import benchmark_plan, seal_benchmark_protocol

    cfg = deepcopy(DEFAULTS)
    cfg["benchmark"]["models"]["efficienttam"]["compile"] = True
    plan = benchmark_plan(cfg, ["efficienttam"])
    run = measured_run()
    run.update(
        model="efficienttam",
        backend="efficienttam",
        timing=protocol("efficienttam"),
        inference={
            **cfg["benchmark"]["models"]["efficienttam"],
            "vos_optimized": True,
            "resolution_policy": "cutie",
            "geometry_adapter": plan["geometry_adapter"],
            "reference_max_internal_size": cfg["inference"]["max_internal_size"],
            "geometry": [
                {
                    "original_hw": [1080, 1920],
                    "resized_hw": [480, 853],
                    "padded_hw": [480, 864],
                    "padding_lrtb": [5, 6, 0, 0],
                }
            ],
        },
    )
    run["benchmark_protocol"] = seal_benchmark_protocol(plan, run)
    assert speed_compatibility({"new": run, "same": deepcopy(run)})["status"] == "COMPATIBLE"

    shared_plan = benchmark_plan(cfg, ["baseline", "efficienttam"])
    baseline = measured_run()
    baseline["benchmark_protocol"] = seal_benchmark_protocol(shared_plan, baseline)
    combined = deepcopy(run)
    combined["benchmark_protocol"] = seal_benchmark_protocol(shared_plan, combined)
    assert (
        speed_compatibility({"baseline": baseline, "efficienttam": combined})["status"]
        == "COMPATIBLE"
    )

    from document_vos.benchmark_inputs import Geometry

    shared_plan = benchmark_plan(cfg, ["efficienttam", "sam31"])
    combined["benchmark_protocol"] = seal_benchmark_protocol(shared_plan, combined)
    sam = deepcopy(run)
    sam.update(
        model="sam31",
        backend="sam31",
        timing=protocol("sam31"),
        inference={
            **cfg["benchmark"]["models"]["sam31"],
            "resolution_policy": "cutie",
            "geometry_adapter": shared_plan["geometry_adapter"],
            "reference_max_internal_size": 480,
            "geometry": [Geometry.cutie((1080, 1920), 480, stride=14).to_dict()],
        },
    )
    sam["benchmark_protocol"] = seal_benchmark_protocol(shared_plan, sam)
    assert speed_compatibility({"efficienttam": combined, "sam31": sam})["status"] == "COMPATIBLE"
    sam["inference"]["geometry"] = [Geometry.cutie((1080, 1918), 480, stride=14).to_dict()]
    sam["benchmark_protocol"] = seal_benchmark_protocol(shared_plan, sam)
    assert speed_compatibility({"efficienttam": combined, "sam31": sam})["status"] == "INCOMPATIBLE"

    old = deepcopy(run)
    old.pop("benchmark_protocol")
    old["inference"].update(compile=False, internal_resolution=1024, vos_optimized=False)
    assert speed_compatibility({"new": run, "old": old})["status"] == "INCOMPATIBLE"

    for change in ("resolution", "compile", "adapter"):
        other = deepcopy(run)
        changed_plan = deepcopy(plan)
        if change == "resolution":
            changed_plan["resolution_policy"] = "native"
        elif change == "compile":
            changed_plan["models"]["efficienttam"]["compile"] = False
            other["inference"].update(compile=False, vos_optimized=False)
        else:
            changed_plan["geometry_adapter"] = "different-adapter"
            other["inference"]["geometry_adapter"] = "different-adapter"
        other["benchmark_protocol"] = seal_benchmark_protocol(changed_plan, other)
        assert speed_compatibility({"new": run, "other": other})["status"] == "INCOMPATIBLE"

    changed = deepcopy(run)
    changed["inference"]["geometry"][0]["resized_hw"] = [480, 854]
    changed["benchmark_protocol"] = seal_benchmark_protocol(plan, changed)
    assert speed_compatibility({"new": run, "changed": changed})["status"] == "INCOMPATIBLE"
    changed = deepcopy(run)
    changed["signature"]["inputs_sha256"] = "different inputs"
    assert speed_compatibility({"new": run, "changed": changed})["status"] == "INCOMPATIBLE"
    changed = deepcopy(run)
    changed["weights_sha256"] = "different checkpoint"
    changed["benchmark_protocol"] = seal_benchmark_protocol(plan, changed)
    assert speed_compatibility({"new": run, "changed": changed})["status"] == "INCOMPATIBLE"
    changed = deepcopy(run)
    changed["seed"] += 1
    changed["benchmark_protocol"] = seal_benchmark_protocol(plan, changed)
    assert speed_compatibility({"changed": changed})["status"] == "INCOMPATIBLE"
    changed = deepcopy(run)
    changed["backend"] = "livos"
    changed["timing"] = protocol("livos")
    changed["benchmark_protocol"] = seal_benchmark_protocol(plan, changed)
    assert speed_compatibility({"changed": changed})["status"] == "INCOMPATIBLE"
    changed = deepcopy(run)
    changed["benchmark_protocol"]["plan"]["resolution_policy"] = "native"
    assert speed_compatibility({"tampered": changed})["status"] == "INCOMPATIBLE"


def test_old_or_different_speed_evidence_does_not_discard_quality(tmp_path):
    from test_model_comparison import evaluation

    from document_vos.benchmark import _compare_evaluations, _write_csv, _write_html

    good = measured_run()
    assert (
        speed_compatibility({"baseline": good, "finetuned": deepcopy(good)})["status"]
        == "COMPATIBLE"
    )
    for field in ("cpu_runtime", "timing"):
        old = deepcopy(good)
        old.pop(field)
        assert speed_compatibility({"baseline": good, "old": old})["status"] == "INCOMPATIBLE"
    other = deepcopy(good)
    other["timing"]["model_boundary"] = "whole sequence including JPEG"
    assert speed_compatibility({"baseline": good, "old": other})["status"] == "INCOMPATIBLE"
    other = deepcopy(good)
    other["cpu_runtime"]["affinity"] = list(range(12, 24))
    assert speed_compatibility({"baseline": good, "old": other})["status"] == "INCOMPATIBLE"
    data = evaluation()
    data["summary"] = {"primary_equal_datasets": {"J": 1, "F": 1, "JF": 1, "BIoU": 1}}
    result = _compare_evaluations({"baseline": data, "finetuned": deepcopy(data)})
    assert result["status"] == "COMPLETE"
    assert result["speed_compatibility"]["status"] == "INCOMPATIBLE"
    _write_csv(tmp_path / "comparison.csv", ["baseline", "finetuned"], result)
    _write_html(tmp_path / "comparison.html", result)
    assert "INCOMPATIBLE" in (tmp_path / "comparison.csv").read_text()
    assert "INCOMPATIBLE" in (tmp_path / "comparison.html").read_text()


def test_resume_rejects_old_protocol_and_changed_actual_cpu(tmp_path, monkeypatch):
    from document_vos import runtime
    from document_vos.io import read_json
    from document_vos.runs import Run

    evidence = measured_run()["cpu_runtime"]
    monkeypatch.setattr(runtime, "runtime_snapshot", lambda: deepcopy(evidence))
    run = Run(tmp_path, "benchmark", {}, {})
    Run(tmp_path, "benchmark", {}, {}, resume=run.output)
    state_path = run.output / "status.json"
    original = state_path.read_bytes()
    state = read_json(state_path)
    del state["signature"]["execution"]
    state_path.write_text(json.dumps(state))
    with pytest.raises(PipelineError, match="CPU-профиль или протокол"):
        Run(tmp_path, "benchmark", {}, {}, resume=run.output)
    state_path.write_bytes(original)
    evidence["affinity"] = [99]
    with pytest.raises(PipelineError, match="CPU-профиль или протокол"):
        Run(tmp_path, "benchmark", {}, {}, resume=run.output)
    assert state_path.read_bytes() == original


def test_cutie_step_keeps_compute_only_boundary(monkeypatch):
    import time

    from document_vos.cutie import CutiePredictor

    clock = Clock()
    monkeypatch.setattr(time, "perf_counter", clock)

    class Core:
        def step(self, *args, **kwargs):
            clock.work(2)
            return torch.ones((6, 9), dtype=torch.long)

        def output_prob_to_mask(self, value):
            clock.work(20)
            return value

    predictor = CutiePredictor.__new__(CutiePredictor)
    predictor.torch, predictor.device = torch, torch.device("cpu")
    predictor.cfg = SimpleNamespace(amp=False)
    predictor.processor = Core()

    def sync():
        clock.sync()
        clock.work(1)

    predictor.sync = sync
    mask = predictor.step(np.zeros((6, 9, 3), dtype=np.uint8), np.ones((6, 9)))
    assert predictor.last_model_seconds == 3  # step + final CUDA sync
    assert clock.now == 24
    assert np.array_equal(mask, np.ones((6, 9)))
    assert clock.events == [0, 3]


def test_full_timer_waits_for_png_and_final_sync_but_excludes_preflight(
    tmp_path, sequences, monkeypatch
):
    from document_vos import runtime
    from document_vos.cutie import inference
    from document_vos.data import save_manifest
    from document_vos.io import read_json

    clock = Clock()
    monkeypatch.setattr(inference, "perf_counter", clock)
    monkeypatch.setattr(runtime, "_current", runtime.Runtime(io_threads=1))
    seq = sequences[0]
    seq.frames = seq.frames[:2]
    manifest = tmp_path / "manifest.jsonl"
    save_manifest(manifest, [seq])
    monkeypatch.setattr(inference, "validate_data", lambda *a, **kw: clock.work(100))

    def snapshot(*args):
        clock.work(100)
        return []

    monkeypatch.setattr(inference, "snapshot_inputs", snapshot)
    original_save = inference._save_prediction

    def save(task):
        clock.work(10)
        return original_save(task)

    monkeypatch.setattr(inference, "_save_prediction", save)

    class Predictor:
        last_model_seconds = 4

        def reset(self):
            clock.work(5)

        def sync(self):
            clock.work(1)

        def predict_sequence(self, frames, mask, cache):
            clock.work(100)  # JPEG/input processing
            clock.work(4)  # compute, including first frame
            clock.work(100)  # cache cleanup and mask conversion
            return [mask.copy() for _ in frames]

        def memory_stats(self):
            return {}

    meta = {"inference": {}, "seed": 42, "scope": "validation", "device": "cpu", "versions": {}}
    output = tmp_path / "run"
    inference.propagate(manifest, output, Predictor(), meta)
    run = read_json(output / "run.json")
    assert run["timing"]["model_seconds"] == 4
    assert (
        run["timing"]["full_seconds"] == 231
    )  # reset 5 + 204 + seq sync 1 + writes 20 + final sync 1
    assert clock.now == 532  # preflight 200 + initial sync 1 + full 231 + postflight 100
    assert len(list((output / "predictions").rglob("*.png"))) == 2
    assert run["timing"]["includes_initialization"] and not run["timing"]["warmup_excluded"]


def test_livos_reads_original_inputs_without_jpeg_preparation(tmp_path, monkeypatch):
    from document_vos import benchmark, timing

    clock = Clock()
    monkeypatch.setattr(timing, "perf_counter", clock)
    module = SimpleNamespace()

    class Core:
        def __init__(self, model):
            pass

        def step(self, image, mask, objects, is_last_frame):
            clock.work(2)
            return torch.stack([torch.zeros_like(image[0]), torch.ones_like(image[0])])

    module.InferenceCore = Core
    monkeypatch.setitem(sys.modules, "livos.eval", module)

    def forbidden(*args):
        raise AssertionError("LiVOS must not prepare JPEG")

    monkeypatch.setattr(benchmark, "_frames_directory", forbidden)
    adapter = benchmark.ExternalPredictor.__new__(benchmark.ExternalPredictor)
    adapter.name, adapter.device, adapter.options = "livos", "cpu", {"max_internal_size": 6}
    adapter.predictor = object()
    frame = tmp_path / "input.png"
    Image.new("RGB", (9, 6)).save(frame)
    masks = adapter.predict_sequence(
        [SimpleNamespace(path=frame)] * 2, np.ones((6, 9)), tmp_path / "cache"
    )
    assert adapter.last_model_seconds == 4
    assert len(masks) == 2 and all(np.array_equal(mask, np.ones((6, 9))) for mask in masks)


def test_sam_timer_covers_first_frame_and_restores_boundaries(tmp_path, monkeypatch):
    from document_vos import benchmark, timing

    clock = Clock()
    monkeypatch.setattr(timing, "perf_counter", clock)

    class Tracker:
        def forward_image(self):
            clock.work(2)

        def track_step(self):
            clock.work(3)
            self._encode_new_memory()

        def _encode_new_memory(self):
            clock.work(1)

    class Model:
        tracker = Tracker()

        def run_backbone_and_detection(self):
            self.tracker.forward_image()
            clock.work(2)

    model = Model()

    def predict(directory, mask, shapes, options):
        clock.work(100)  # input/session preparation
        model.run_backbone_and_detection()  # frame zero, includes nested encoder
        model.tracker.track_step()
        model.tracker._encode_new_memory()  # first-frame memory preflight
        model.run_backbone_and_detection()
        model.tracker.track_step()
        clock.work(100)  # output conversion and session teardown
        return [mask.copy() for _ in shapes]

    adapter = benchmark.ExternalPredictor.__new__(benchmark.ExternalPredictor)
    adapter.name, adapter.device, adapter.options = "sam31", "cpu", {}
    adapter.predictor = SimpleNamespace(predictor=SimpleNamespace(model=model), predict=predict)
    frame = tmp_path / "input.png"
    Image.new("RGB", (9, 6)).save(frame)
    masks = adapter.predict_sequence(
        [SimpleNamespace(path=frame)] * 2, np.ones((6, 9)), tmp_path / "cache"
    )
    assert adapter.last_model_seconds == 17
    assert clock.now == 217
    assert len(masks) == 2 and all(np.array_equal(mask, np.ones((6, 9))) for mask in masks)
    assert not vars(model) and not vars(model.tracker)


def test_all_entrypoints_apply_same_actual_profile(tmp_path, manifest, monkeypatch):
    from document_vos import benchmark, cutie, runtime, sam31_smoke
    from document_vos.cli import dispatch, parser
    from document_vos.config import load
    from document_vos.cutie import training, training_worker
    from document_vos.io import sha256

    captured = []

    def capture(*args, **kwargs):
        measured = runtime.runtime_snapshot()
        captured.append(
            {
                key: measured[key]
                for key in (
                    "cpu_profile",
                    "affinity",
                    "physical_cores",
                    "pools",
                    "workers",
                    "dataloader_workers",
                    "io_threads",
                )
            }
        )
        return {"status": "COMPLETE"}

    cfg = load()
    monkeypatch.setattr(benchmark, "benchmark", capture)
    dispatch(parser().parse_args(["benchmark"]))
    monkeypatch.setattr(benchmark, "_input", lambda *a, **kw: (manifest, "independent", None))
    monkeypatch.setattr(benchmark, "_run_model", capture)
    monkeypatch.setattr("document_vos.rendering.bundle", lambda *a, **kw: None)
    cfg["runtime"]["device"] = "cpu"
    benchmark.inference(cfg, "livos", manifest=manifest, output=tmp_path / "infer")

    # An occupied GPU exercises the standalone SAM setup without loading a checkpoint.
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(tmp_path / "sam-smoke")])
    monkeypatch.setattr(
        sam31_smoke.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="busy")
    )
    assert sam31_smoke.main() == 2
    report = json.loads((tmp_path / "sam-smoke/status.json").read_text())
    captured.append({key: report["cpu_runtime"][key] for key in captured[0]})

    # Execute the training worker's real preflight path up to the pinned entrypoint.
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"local")
    config = tmp_path / "config/train.yaml"
    config.parent.mkdir()
    config.write_text("config")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "config": {
                    "document_vos_runtime": runtime.Runtime().to_dict(),
                    "weights": str(weights),
                    "data": {"vos_datasets": {"base": str(tmp_path)}},
                },
                "config_sha256": sha256(config),
                "weights_sha256": sha256(weights),
            }
        )
    )
    monkeypatch.setenv("DOCUMENT_VOS_TRAIN_PLAN", str(plan))
    monkeypatch.setenv("DOCUMENT_VOS_CUTIE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["worker", "--cfg"])
    monkeypatch.setattr(cutie, "check_checkout", lambda root: root)
    monkeypatch.setattr(cutie, "activate_checkout", lambda *a: None)
    monkeypatch.setattr(cutie, "disable_downloads_and_require_full_weights", lambda: None)
    monkeypatch.setattr(training, "validate_training_config", lambda *a: None)
    monkeypatch.setattr(runtime, "install_loader_options", lambda *a: None)
    monkeypatch.setattr(training_worker.runpy, "run_path", capture)
    training_worker.main()
    assert len(captured) == 4 and all(item == captured[0] for item in captured)
