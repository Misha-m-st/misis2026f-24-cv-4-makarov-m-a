from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from document_vos.benchmark import ExternalPredictor
from document_vos.config import load
from document_vos.io import PipelineError
from document_vos.sam31 import MaskSessionPredictor, cleanup_preserving_error


def new_state():
    return dict(
        orig_height=6,
        orig_width=9,
        device="cpu",
        num_frames=3,
        feature_cache={},
        tracker_metadata={},
        action_history=[],
        sam2_inference_states=[],
        cached_frame_outputs={},
    )


def fake_predictor():
    tracker = SimpleNamespace(per_obj_inference=False, input_mask_size=4)
    model = SimpleNamespace(tracker=tracker, world_size=1, rank=0, image_size=1008)
    model.init_state = Mock(side_effect=lambda **kw: new_state())
    model._initialize_metadata = Mock(
        side_effect=lambda: {
            "obj_ids_per_gpu": [np.array([], dtype=np.int64)],
            "num_obj_per_gpu": np.zeros(1, dtype=np.int64),
            "obj_id_to_score": {},
            "obj_id_to_sam2_score_frame_wise": defaultdict(dict),
        }
    )
    model._prepare_backbone_feats = Mock()
    model._init_new_sam2_state = Mock(side_effect=lambda state: {})
    model.add_action_history = lambda state, kind, **kw: state["action_history"].append(
        {"type": kind, **kw}
    )
    model._cache_frame_outputs = lambda state, index, masks: state[
        "cached_frame_outputs"
    ].__setitem__(index, masks)
    tracker.add_new_masks = Mock(
        side_effect=lambda **kw: (0, [1], None, kw["masks"][:, None].float() * 20 - 10)
    )
    tracker.propagate_in_video_preflight = Mock()
    predictor = SimpleNamespace(model=model, _all_inference_states={})
    predictor._extend_expiration_time = Mock()
    predictor._get_session = lambda session: predictor._all_inference_states[session]

    def handle(request):
        if request["type"] == "close_session":
            predictor._all_inference_states.pop(request["session_id"], None)
        return {"is_success": True}

    predictor.handle_request = Mock(side_effect=handle)
    predictor.handle_stream_request = Mock(return_value=iter([]))
    return predictor


def seed(bridge, mask=None):
    bridge.handle_request({"type": "start_session", "session_id": "s", "resource_path": "frames"})
    mask = np.ones((6, 9), dtype=np.uint8) if mask is None else mask
    return bridge.handle_request(
        {"type": "add_mask", "session_id": "s", "frame_index": 0, "obj_id": 1, "mask": mask}
    )


def test_session_registration_original_mask_metadata_and_action():
    predictor = fake_predictor()
    bridge = MaskSessionPredictor(predictor)
    mask = np.zeros((6, 9), dtype=np.uint8)
    mask[1:5, 2:7] = 1
    seed(bridge, mask)
    predictor.model.init_state.assert_called_once_with(
        resource_path="frames", offload_video_to_cpu=False, async_loading_frames=False
    )
    session = predictor._all_inference_states["s"]
    assert session["start_time"] <= session["last_use_time"]
    state = session["state"]
    passed = predictor.model.tracker.add_new_masks.call_args.kwargs["masks"]
    assert tuple(passed.shape) == (1, 6, 9)
    np.testing.assert_array_equal(passed[0], mask)
    assert passed.data_ptr() == torch.as_tensor(mask).data_ptr()
    assert state["action_history"] == [{"type": "add", "frame_idx": 0, "obj_ids": [1]}]
    assert state["tracker_metadata"]["obj_ids_all_gpu"].tolist() == [1]
    assert state["tracker_metadata"]["num_obj_per_gpu"].tolist() == [1]
    assert state["tracker_metadata"]["max_obj_id"] == 1
    assert state["tracker_metadata"]["obj_id_to_score"] == {1: 1.0}
    assert state["tracker_metadata"]["obj_id_to_sam2_score_frame_wise"][0][1].item() == 1
    np.testing.assert_array_equal(state["cached_frame_outputs"][0][1][0], mask)
    predictor.model.tracker.propagate_in_video_preflight.assert_called_once_with(
        state["sam2_inference_states"][0], run_mem_encoder=True
    )
    assert bridge.metadata["internal_resolution"] == 1008
    with pytest.raises(PipelineError, match="новую"):
        bridge.handle_request(
            {"type": "add_mask", "session_id": "s", "frame_index": 0, "obj_id": 1, "mask": mask}
        )


@pytest.mark.parametrize("mask", [np.zeros((6, 9)), np.ones((5, 9)), np.full((6, 9), 0.5)])
def test_bridge_rejects_invalid_masks_before_features(mask):
    predictor = fake_predictor()
    with pytest.raises(PipelineError, match="маска"):
        seed(MaskSessionPredictor(predictor), mask)
    predictor.model._prepare_backbone_feats.assert_not_called()


def test_session_offload_rejected_and_other_requests_delegated():
    predictor = fake_predictor()
    bridge = MaskSessionPredictor(predictor)
    with pytest.raises(PipelineError, match="offload_state"):
        bridge.handle_request({"type": "start_session", "offload_state_to_cpu": True})
    predictor.model.init_state.assert_not_called()
    bridge.handle_request({"type": "reset_session", "session_id": "s"})
    predictor.handle_request.assert_called_once()


def output(index, ids=(1,), masks=None):
    return {
        "frame_index": index,
        "outputs": {
            "out_obj_ids": np.array(ids, dtype=np.int64),
            "out_binary_masks": np.ones((len(ids), 6, 9), dtype=bool) if masks is None else masks,
        },
    }


def predict(predictor):
    return MaskSessionPredictor(predictor).predict(
        "frames",
        np.ones((6, 9), dtype=np.uint8),
        [(6, 9)] * 3,
        {"offload_video_to_cpu": True, "offload_state_to_cpu": False},
    )


def test_propagation_selects_id_and_empty_object_is_empty_mask():
    predictor = fake_predictor()
    binary = np.zeros((2, 6, 9), dtype=bool)
    binary[1, 2:4, 3:7] = True
    predictor.handle_stream_request.return_value = iter(
        [output(2, (8, 1), binary), output(0), output(1, ())]
    )
    masks = predict(predictor)
    np.testing.assert_array_equal(masks[2], binary[1])
    assert not masks[1].any()
    assert masks[0].all()
    assert not predictor._all_inference_states
    request = predictor.handle_stream_request.call_args.args[0]
    assert request["propagation_direction"] == "forward"
    assert request["start_frame_index"] == 0


@pytest.mark.parametrize(
    "items",
    [
        [output(0)],
        [output(0), output(0)],
        [output(-1)],
        [output(3)],
        [output(0.5)],
        [output(True)],
        [output(0, (1, 1))],
        [output(0, (1,), np.ones((2, 6, 9)))],
        [output(0, (1,), np.ones((1, 3, 9)))],
        [output(0, (1,), np.full((1, 6, 9), 0.5))],
    ],
)
def test_bad_outputs_fail_and_close_session(items):
    predictor = fake_predictor()
    predictor.handle_stream_request.return_value = iter(items)
    with pytest.raises(PipelineError):
        predict(predictor)
    assert not predictor._all_inference_states
    assert predictor.handle_request.call_args.args[0]["type"] == "close_session"


@pytest.mark.parametrize("phase", ["start", "mask", "stream"])
def test_oom_is_preserved_even_when_session_cleanup_fails(phase):
    predictor = fake_predictor()
    error = torch.cuda.OutOfMemoryError("original OOM")
    if phase == "start":
        predictor.model.init_state.side_effect = error
    elif phase == "mask":
        predictor.model.tracker.add_new_masks.side_effect = error
    else:
        predictor.handle_stream_request.side_effect = error
    predictor.handle_request.side_effect = RuntimeError("cleanup error")
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        predict(predictor)
    assert caught.value is error
    assert any("cleanup error" in note for note in error.__notes__)
    assert predictor.handle_request.call_args.args[0]["type"] == "close_session"


def test_autocast_cleanup_reverse_order_idempotent_and_preserves_error():
    predictor = fake_predictor()
    exited = []
    predictor.bf16_context = SimpleNamespace(__exit__=lambda *args: exited.append("predictor"))
    predictor.model.tracker.bf16_context = SimpleNamespace(
        __exit__=lambda *args: exited.append("tracker")
    )
    bridge = MaskSessionPredictor(predictor)
    bridge.close()
    bridge.close()
    assert exited == ["predictor", "tracker"]
    error = ValueError("inference")
    with pytest.raises(ValueError) as caught:
        try:
            raise error
        finally:
            cleanup_preserving_error(lambda: (_ for _ in ()).throw(RuntimeError("close")))
    assert caught.value is error


def test_builder_arguments_and_external_cleanup(monkeypatch, tmp_path):
    import sys

    predictor = fake_predictor()
    builder = Mock(return_value=predictor)
    monkeypatch.setitem(
        sys.modules,
        "sam3.model_builder",
        SimpleNamespace(build_sam3_multiplex_video_predictor=builder),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    cfg = load()
    adapter = ExternalPredictor("sam31", cfg, tmp_path / "local.pt", "cuda")
    builder.assert_called_once_with(
        checkpoint_path=str(tmp_path / "local.pt"),
        use_fa3=False,
        compile=False,
        async_loading_frames=False,
    )
    assert adapter.predictor.metadata["effective_precision"] == "cuda_autocast_bfloat16"
    adapter.close()
    assert adapter.predictor is None
    for setting in ({"amp": False}, {"dtype": "float16"}, {"offload_state_to_cpu": True}):
        cfg["benchmark"]["models"]["sam31"].update(load()["benchmark"]["models"]["sam31"])
        cfg["benchmark"]["models"]["sam31"].update(setting)
        with pytest.raises(PipelineError):
            ExternalPredictor("sam31", cfg, tmp_path / "local.pt", "cuda")
    assert builder.call_count == 1


def test_external_cache_cleanup_after_bridge_error(tmp_path):
    adapter = ExternalPredictor.__new__(ExternalPredictor)
    adapter.name, adapter.device, adapter.options = "sam31", "cpu", {}
    adapter.predictor = SimpleNamespace(predict=Mock(side_effect=RuntimeError("model")))

    def noop(*args, **kwargs):
        return None

    adapter.predictor.predictor = SimpleNamespace(
        model=SimpleNamespace(
            run_backbone_and_detection=noop,
            tracker=SimpleNamespace(forward_image=noop, track_step=noop, _encode_new_memory=noop),
        )
    )
    frame = tmp_path / "frame.png"
    Image.new("RGB", (9, 6)).save(frame)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="model"):
        adapter.predict_sequence([SimpleNamespace(path=frame)], np.ones((6, 9)), cache)
    assert not cache.exists()


def test_pinned_real_classes_keep_mask_conditioning_and_choose_instance_propagation(monkeypatch):
    pytest.importorskip("sam3")
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
    from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo

    # Real add_new_masks, state registration and memory preflight; stub neural computation only.
    tracker = Sam3VideoTrackingMultiplexDemo.__new__(Sam3VideoTrackingMultiplexDemo)
    torch.nn.Module.__init__(tracker)
    for key, value in dict(
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        per_obj_inference=False,
        input_mask_size=4,
        is_dynamic_model=True,
        add_all_frames_to_correct_as_cond=False,
        non_overlap_masks_for_output=False,
        fill_hole_area=0,
        clear_non_cond_mem_around_input=False,
    ).items():
        setattr(tracker, key, value)
    tracker.multiplex_controller = SimpleNamespace(
        get_state=lambda **kw: SimpleNamespace(total_valid_entries=1)
    )
    observed = []

    def neural(**kw):
        observed.append(kw["mask_inputs"].clone())
        return {"pred_masks": torch.zeros((1, 1, 4, 4))}, None

    monkeypatch.setattr(tracker, "_run_single_frame_inference", neural)
    mem_calls = []

    def consolidate(state, frame_idx, is_cond, run_mem_encoder, **kw):
        mem_calls.append(run_mem_encoder)
        return dict(state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][frame_idx])

    monkeypatch.setattr(tracker, "_consolidate_temp_output_across_obj", consolidate)
    monkeypatch.setattr(tracker, "_add_output_per_object", lambda *args: None)
    model = Sam3MultiplexTrackingWithInteractivity.__new__(Sam3MultiplexTrackingWithInteractivity)
    torch.nn.Module.__init__(model)
    model.tracker, model.world_size, model.rank = tracker, 1, 0
    model.is_multiplex, model.masklet_confirmation_enable = True, True
    model.image_size = 1008
    monkeypatch.setattr(model, "_prepare_backbone_feats", lambda *args, **kw: None)
    real_init = tracker.init_state

    def init_state(**kw):
        state = real_init(**kw)
        state["device"] = state["storage_device"] = torch.device("cpu")
        return state

    monkeypatch.setattr(tracker, "init_state", init_state)
    predictor = fake_predictor()
    predictor.model = model
    monkeypatch.setattr(model, "init_state", lambda **kw: new_state(), raising=False)
    mask = np.zeros((6, 9), dtype=np.uint8)
    mask[1:5, 2:7] = 1
    seed(MaskSessionPredictor(predictor), mask)
    state = predictor._all_inference_states["s"]["state"]
    assert model.parse_action_history_for_propagation(state) == ("propagation_partial", [1])
    tracker_state = state["sam2_inference_states"][0]
    assert tracker_state["first_ann_frame_idx"] == 0
    assert tracker_state["consolidated_frame_inds"]["cond_frame_outputs"] == {0}
    assert tracker_state["tracking_has_started"]
    np.testing.assert_array_equal(tracker_state["mask_inputs_per_obj"][0][0][0, 0], mask)
    np.testing.assert_array_equal(state["cached_frame_outputs"][0][1][0], mask)
    assert observed[0].shape == (1, 1, 4, 4)
    assert True in mem_calls
    # A second native preflight must not lose the conditioning input or assert.
    tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
    assert 0 in tracker_state["mask_inputs_per_obj"][0]
    # Exercise native partial propagation and its cache merge on later frames.
    # An absent cache entry silently discards the tracked object in this revision.
    state["previous_stages_out"] = [None] * state["num_frames"]
    monkeypatch.setattr(
        model,
        "_propogate_tracker_one_frame_local_gpu",
        lambda *args, **kw: ([1], torch.ones((1, 2, 2)), torch.ones(1)),
        raising=False,
    )

    def postprocess(state, out, **kw):
        return {
            "out_obj_ids": list(out["obj_id_to_mask"]),
            "out_binary_masks": list(out["obj_id_to_mask"].values()),
        }

    monkeypatch.setattr(model, "_postprocess_output", postprocess)
    propagated = list(
        model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=3, reverse=False)
    )
    assert [index for index, _ in propagated] == [0, 1, 2]
    assert all(out["out_obj_ids"] == [1] for _, out in propagated)
    assert all(out["out_binary_masks"][0].all() for _, out in propagated)
    assert 0 in tracker_state["mask_inputs_per_obj"][0]


def test_smoke_busy_gpu_skips_without_loading_or_waiting(monkeypatch, tmp_path):
    import json
    import sys

    from document_vos import sam31_smoke

    output = tmp_path / "smoke"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    monkeypatch.setattr(
        sam31_smoke.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout="123, other-training-job, 4000 MiB\n")),
    )
    loader = Mock(side_effect=AssertionError("must not load model"))
    inputs = Mock(side_effect=AssertionError("must not prepare data"))
    monkeypatch.setattr(sam31_smoke, "ExternalPredictor", loader)
    monkeypatch.setattr(sam31_smoke, "_input", inputs)
    assert sam31_smoke.main() == 2
    assert json.loads((output / "status.json").read_text())["status"] == "SKIPPED_GPU_BUSY"
    loader.assert_not_called()
    inputs.assert_not_called()


def test_close_attempts_both_contexts_after_cleanup_failure():
    predictor = fake_predictor()
    bridge = MaskSessionPredictor(predictor)
    seed(bridge)
    predictor.handle_request.side_effect = RuntimeError("session cleanup")
    predictor.bf16_context = SimpleNamespace(__exit__=Mock())
    predictor.model.tracker.bf16_context = SimpleNamespace(__exit__=Mock())
    with pytest.raises(RuntimeError, match="session cleanup"):
        bridge.close()
    predictor.bf16_context.__exit__.assert_called_once_with(None, None, None)
    predictor.model.tracker.bf16_context.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("failure_phase", [None, "load", "propagate"])
def test_smoke_rewrites_status_and_keeps_original_failure(monkeypatch, tmp_path, failure_phase):
    import json
    import sys

    from document_vos import sam31_smoke
    from document_vos.cutie import inference

    output = tmp_path / "smoke"
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    monkeypatch.setattr(
        sam31_smoke.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=""))
    )
    monkeypatch.setattr(
        sam31_smoke,
        "_input",
        lambda *args, **kw: (tmp_path / "manifest.jsonl", "validation", tmp_path),
    )
    frames = [SimpleNamespace(id=str(i), path=tmp_path / f"{i}.jpg") for i in range(5)]
    monkeypatch.setattr(
        sam31_smoke,
        "load_manifest",
        lambda path: [SimpleNamespace(key="test/video", frames=frames)],
    )
    monkeypatch.setattr(sam31_smoke, "first_mask", lambda seq: (0, np.ones((6, 9), np.uint8)))
    monkeypatch.setattr(inference, "validate_scope", lambda *args: "split-hash")
    monkeypatch.setattr(sam31_smoke, "sha256", lambda path: "file-hash")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    error = torch.cuda.OutOfMemoryError("original GPU failure")
    predictor = SimpleNamespace(
        predictor=SimpleNamespace(
            metadata={"internal_resolution": 1008},
            predictor=SimpleNamespace(_all_inference_states={}),
        ),
        predict_sequence=Mock(return_value=[np.ones((6, 9), np.uint8) for _ in range(5)]),
        sync=Mock(),
        memory_stats=Mock(return_value={}),
        close=Mock(),
    )
    builder = Mock(return_value=predictor)
    if failure_phase == "load":
        builder.side_effect = error
    elif failure_phase == "propagate":
        predictor.predict_sequence.side_effect = error
    monkeypatch.setattr(sam31_smoke, "ExternalPredictor", builder)
    writes = []
    atomic = sam31_smoke.atomic_json

    def record_status(path, report):
        atomic(path, report)
        writes.append(json.loads(path.read_text()))

    monkeypatch.setattr(sam31_smoke, "atomic_json", record_status)
    assert sam31_smoke.main() == (1 if failure_phase else 0)
    report = json.loads((output / "status.json").read_text())
    assert writes[0]["phase"] == "load_full_predictor"
    if failure_phase:
        assert report["status"] == "FAILED_OOM"
        assert report["error"] == "OutOfMemoryError: original GPU failure"
        assert "original GPU failure" in (output / "error.log").read_text()
        assert "FileExistsError" not in (output / "error.log").read_text()
    else:
        assert report["status"] == "COMPLETE"
        assert report["saved_masks"] == 5
        assert report["active_sessions_after_propagation"] == 0
        assert len(list(output.glob("*.png"))) == 6
    if failure_phase != "load":
        assert writes[1]["phase"] == "mask_initialization_and_propagation"
        predictor.close.assert_called_once()
