from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from document_vos.benchmark import ExternalPredictor
from document_vos.benchmark_inputs import FrameStore, Geometry
from document_vos.config import DEFAULTS, validate
from document_vos.io import PipelineError


@pytest.mark.parametrize(
    "shape,limit,resized,padded",
    [
        ((1080, 1920), 480, (480, 853), (480, 864)),
        ((1920, 1080), 480, (853, 480), (864, 480)),
        ((241, 399), 480, (241, 399), (256, 400)),
        ((241, 399), -1, (241, 399), (256, 400)),
    ],
)
def test_geometry_matches_cutie_resize_and_padding(shape, limit, resized, padded):
    import torch
    import torch.nn.functional as F

    geometry = Geometry.cutie(shape, limit)
    assert geometry.resized == resized and geometry.padded == padded
    rgb = np.random.default_rng(1).integers(0, 255, (*shape, 3), dtype=np.uint8)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    if shape != resized:
        tensor = F.interpolate(tensor[None], resized, mode="bilinear", align_corners=False)[0]
    torch.testing.assert_close(geometry.image(rgb), F.pad(tensor, geometry.padding), rtol=0, atol=0)
    mask = np.ones(shape, dtype=np.uint8)
    assert np.array_equal(geometry.restore_binary(geometry.mask(mask)), mask)


def test_sam_padding_preserves_same_image_content_and_reports_stride_difference():
    geometry = Geometry.cutie((1080, 1920), 480, stride=14)
    assert geometry.resized == (480, 853)
    assert geometry.padded == (490, 854)
    assert geometry.crop(geometry.mask(np.ones(geometry.original, np.uint8))).all()


def test_frame_store_prefetch_is_bounded_and_closes(tmp_path, monkeypatch):
    from document_vos import runtime

    monkeypatch.setattr(
        runtime, "current_runtime", lambda: SimpleNamespace(queue_size=3, io_threads=4)
    )
    path = tmp_path / "rgb.png"
    Image.new("RGB", (48, 32), (20, 100, 200)).save(path)
    frames = [SimpleNamespace(path=path)] * 20
    store = FrameStore(frames, Geometry.cutie((32, 48), 480))
    try:
        for index in [0, 1, 2, 10, 3, 19]:
            assert store[index].shape == (3, 32, 48)
            assert len(store.pending) <= 3
        assert store.peak_pending <= 3
    finally:
        store.close()
    assert store.closed and not store.pending
    with pytest.raises(IndexError):
        store[0]


def test_efficienttam_streams_without_jpeg_and_cleans_up_on_early_close(tmp_path, monkeypatch):
    import torch

    from document_vos import benchmark

    def forbidden(*_args):
        raise AssertionError("matched benchmark must decode original images")

    monkeypatch.setattr(benchmark, "_frames_directory", forbidden)
    monkeypatch.setattr("document_vos.benchmark_geometry.set_geometry", lambda *_args: None)
    events = []

    class Predictor:
        def init_state(self, *, video_path, **kwargs):
            self.store = video_path
            assert isinstance(video_path, FrameStore)
            assert video_path[0].shape == (3, 32, 48)
            return {}

        def add_new_mask(self, **kwargs):
            assert kwargs["mask"].shape == (32, 48)

        def propagate_in_video(self, state):
            for i in range(10):
                events.append(i)
                yield i, [1], torch.ones((1, 1, 32, 48))

        def reset_state(self, state):
            events.append("reset")

        def forward_image(self):
            pass

        def track_step(self):
            pass

        def _encode_new_memory(self):
            pass

    adapter = ExternalPredictor.__new__(ExternalPredictor)
    adapter.name, adapter.device, adapter.options = "efficienttam", "cpu", {}
    adapter.match_cutie_size = 480
    adapter.geometry_records = []
    adapter.predictor = Predictor()
    path = tmp_path / "original.png"
    Image.new("RGB", (48, 32)).save(path)
    stream = adapter.iter_sequence(
        [SimpleNamespace(path=path)] * 10, np.ones((32, 48), np.uint8), tmp_path / "cache"
    )
    assert next(stream).shape == (32, 48)
    assert events == [0]
    stream.close()
    assert events == [0, "reset"]
    assert adapter.predictor.store.closed
    assert not (tmp_path / "cache").exists()


def test_resolution_option_is_strict_and_benchmark_only(tmp_path, monkeypatch):
    from document_vos import benchmark
    from document_vos.cli import dispatch, parser

    cfg = deepcopy(DEFAULTS)
    assert cfg["benchmark"]["resolution"] == "cutie"
    cfg["benchmark"]["resolution"] = "stretch"
    with pytest.raises(PipelineError, match="benchmark.resolution"):
        validate(cfg)
    monkeypatch.chdir(tmp_path)
    captured = []
    monkeypatch.setattr(benchmark, "benchmark", lambda cfg, **kw: captured.append(cfg))
    dispatch(parser().parse_args(["benchmark", "--resolution", "native"]))
    assert captured[0]["benchmark"]["resolution"] == "native"
    with pytest.raises(SystemExit):
        parser().parse_args(["infer", "--resolution", "cutie"])
    assert ExternalPredictor.match_cutie_size is None


def test_pinned_geometry_adapter_rejects_changed_upstream_source(monkeypatch, tmp_path):
    from document_vos import benchmark_geometry

    def method(self):
        return (self.image_size, self.image_size)

    pin = tmp_path / "pins.json"
    import json

    key = f"{method.__module__}.{method.__qualname__}"
    pin.write_text(json.dumps({key: "wrong-hash"}))
    monkeypatch.setattr(benchmark_geometry, "resource", lambda _: pin)
    with pytest.raises(PipelineError, match="pinned API изменился"):
        benchmark_geometry._bind(SimpleNamespace(), method)


def test_propagation_sends_masks_to_writer_before_sequence_finishes(
    tmp_path, manifest, sequences, monkeypatch
):
    import threading

    from document_vos.cutie import inference

    written = threading.Event()
    save = inference._save_prediction

    def save_and_signal(task):
        result = save(task)
        written.set()
        return result

    monkeypatch.setattr(inference, "_save_prediction", save_and_signal)
    closed = []

    class Predictor:
        last_model_seconds = 4.0

        def reset(self):
            pass

        def sync(self):
            pass

        def memory_stats(self):
            return {}

        def predict_sequence(self, *args):
            raise AssertionError("propagation must use streaming API")

        def iter_sequence(self, frames, mask, cache):
            try:
                yield mask.copy()
                assert written.wait(2), "writer did not receive the first mask"
                for _ in frames[1:]:
                    yield mask.copy()
            finally:
                closed.append(True)

    result = inference.propagate(
        manifest,
        tmp_path / "streamed",
        Predictor(),
        {"inference": {}, "seed": 42, "scope": "validation", "device": "cpu", "versions": {}},
        sequences=sequences[:1],
    )
    assert result["frames"] == 4 and closed == [True]


def test_propagation_seals_effective_benchmark_evidence(tmp_path, manifest, sequences, monkeypatch):
    from test_cpu_profile_timing import measured_run

    from document_vos.cutie.inference import propagate
    from document_vos.io import read_json
    from document_vos.timing import benchmark_plan

    cfg = deepcopy(DEFAULTS)
    saved = measured_run()
    monkeypatch.setattr("document_vos.runtime.runtime_snapshot", lambda: saved["cpu_runtime"])

    class Predictor:
        last_model_seconds = 0.001

        def reset(self):
            pass

        def sync(self):
            pass

        def memory_stats(self):
            return {}

        def step(self, image, mask=None, end=False):
            return np.ones(image.shape[:2], dtype=np.uint8)

    meta = {
        "model": "baseline",
        "backend": "official-cutie",
        "backend_revision": "pinned",
        "weights_sha256": "checkpoint-hash",
        "device": "cuda",
        "inference": deepcopy(cfg["inference"]),
        "execution_environment": saved["execution_environment"],
        "benchmark_protocol_plan": benchmark_plan(cfg, ["baseline"]),
        "seed": 42,
        "scope": "independent",
        "versions": {},
    }
    propagate(manifest, tmp_path / "run", Predictor(), meta, sequences=sequences[:1])
    run = read_json(tmp_path / "run" / "run.json")
    assert run["benchmark_protocol"]["execution"]["inference"] == run["inference"]
    assert run["speed_compatibility"]["status"] == "COMPATIBLE"
