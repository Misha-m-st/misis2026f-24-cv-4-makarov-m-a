"""Mask/session compatibility for SAM 3.1 revision f1647e8 (no vendor edits).

Only a fresh, single-GPU session with object 1 on frame 0 is supported.
The full detector + tracker remains owned by the official predictor.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from document_vos.io import PipelineError

MASK_BRIDGE_VERSION = "sam31-mask-bridge/v1"


def cleanup_preserving_error(action):
    """Run cleanup without replacing an active inference/OOM exception."""
    original = sys.exception()
    try:
        action()
    except BaseException as exc:
        if original is None:
            raise
        original.add_note(f"SAM 3.1 cleanup failed: {type(exc).__name__}: {exc}")


class MaskSessionPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self._closed = False

    @property
    def metadata(self):
        model = self.predictor.model
        return {
            "mask_bridge_version": MASK_BRIDGE_VERSION,
            "initialization_type": "original_binary_mask",
            "effective_precision": "cuda_autocast_bfloat16",
            "internal_resolution": int(model.image_size),
            "tracker_mask_input_resolution": int(model.tracker.input_mask_size),
        }

    @torch.inference_mode()
    def handle_request(self, request):
        if request["type"] == "start_session":
            if request.get("offload_state_to_cpu", False):
                raise PipelineError(
                    "SAM 3.1 multiplex: offload_state_to_cpu=true не поддерживается"
                )
            session_id = request.get("session_id") or str(uuid.uuid4())
            registry = self.predictor._all_inference_states
            if session_id in registry:
                raise PipelineError("SAM 3.1: session_id уже зарегистрирован")
            state = self.predictor.model.init_state(
                resource_path=request["resource_path"],
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                async_loading_frames=False,
            )
            now = time.time()
            registry[session_id] = {
                "state": state,
                "session_id": session_id,
                "start_time": now,
                "last_use_time": now,
            }
            self.predictor._extend_expiration_time(registry[session_id])
            return {"session_id": session_id}
        if request["type"] == "add_mask":
            session = self.predictor._get_session(request["session_id"])
            self.predictor._extend_expiration_time(session)
            return self._add_mask(session["state"], request)
        return self.predictor.handle_request(request)

    def _add_mask(self, state, request):
        model = self.predictor.model
        if (
            request["frame_index"] != 0
            or request["obj_id"] != 1
            or state["action_history"]
            or state["tracker_metadata"]
            or state["sam2_inference_states"]
            or model.world_size != 1
            or model.rank != 0
            or model.tracker.per_obj_inference
        ):
            raise PipelineError(
                "SAM 3.1 mask bridge требует новую single-GPU сессию, кадр 0, объект 1"
            )
        mask = torch.as_tensor(request["mask"])
        shape = (state["orig_height"], state["orig_width"])
        if (
            tuple(mask.shape) != shape
            or not torch.all((mask == 0) | (mask == 1)).item()
            or not mask.any().item()
        ):
            raise PipelineError("SAM 3.1: нужна непустая бинарная маска исходного размера")

        metadata = state["tracker_metadata"]
        metadata.update(model._initialize_metadata())
        model._prepare_backbone_feats(state, 0, reverse=False)
        tracker_state = model._init_new_sam2_state(state)
        state["sam2_inference_states"].append(tracker_state)
        metadata["obj_ids_per_gpu"][0] = np.array([1], dtype=np.int64)
        metadata["num_obj_per_gpu"][0] = 1
        metadata["obj_ids_all_gpu"] = np.array([1], dtype=np.int64)
        metadata["max_obj_id"] = 1
        model.add_action_history(state, "add", frame_idx=0, obj_ids=[1])
        metadata["obj_id_to_score"][1] = 1.0
        metadata["obj_id_to_sam2_score_frame_wise"][0][1] = torch.tensor(
            1.0, dtype=torch.float32, device=state["device"]
        )
        # Pass the caller's original pixels; only the tracker resizes internally.
        frame_idx, ids, _, video_masks = model.tracker.add_new_masks(
            inference_state=tracker_state, frame_idx=0, obj_ids=[1], masks=mask[None]
        )
        if frame_idx != 0 or list(ids) != [1] or tuple(video_masks.shape) != (1, 1, *shape):
            raise PipelineError("SAM 3.1: неверный ответ tracker.add_new_masks")
        model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
        # Do not call the point handler's mask-input cleanup: it erases conditioning.
        # Native _build_sam2_output drops refinements for uncached frames. Empty
        # merge slots let instance propagation populate them without VG detections.
        for index in range(state["num_frames"]):
            model._cache_frame_outputs(state, index, {})
        obj_masks = {1: video_masks[0] > 0}
        model._cache_frame_outputs(state, 0, obj_masks)
        return {"frame_index": 0}

    def handle_stream_request(self, request):
        return self.predictor.handle_stream_request(request)

    def predict(self, frame_dir, initial_mask, frame_shapes, options):
        masks = [None] * len(frame_shapes)
        for index, mask in self._iter_indexed(frame_dir, initial_mask, frame_shapes, options):
            masks[index] = mask
        return masks

    def iter_predict(self, frame_dir, initial_mask, frame_shapes, options):
        from contextlib import closing

        with closing(self._iter_indexed(frame_dir, initial_mask, frame_shapes, options)) as stream:
            for expected, (index, mask) in enumerate(stream):
                if index != expected:
                    raise PipelineError("SAM 3.1: streaming требует прямого порядка кадров")
                yield mask

    def _iter_indexed(self, frame_dir, initial_mask, frame_shapes, options):
        # Choose the ID before initialization so even a partial start can be closed.
        session_id = str(uuid.uuid4())
        try:
            self.handle_request(
                {
                    "type": "start_session",
                    "session_id": session_id,
                    "resource_path": str(frame_dir)
                    if isinstance(frame_dir, (Path, str))
                    else frame_dir,
                    "offload_video_to_cpu": options["offload_video_to_cpu"],
                    "offload_state_to_cpu": options["offload_state_to_cpu"],
                }
            )
            self.handle_request(
                {
                    "type": "add_mask",
                    "session_id": session_id,
                    "frame_index": 0,
                    "obj_id": 1,
                    "mask": initial_mask,
                }
            )
            seen = set()
            for item in self.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": len(frame_shapes),
                }
            ):
                index = item["frame_index"]
                if (
                    isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    or not 0 <= index < len(frame_shapes)
                    or index in seen
                ):
                    raise PipelineError(f"SAM 3.1: неверный/повторный индекс кадра {index}")
                out = item["outputs"]
                ids = _numpy(out["out_obj_ids"])
                binary = _numpy(out["out_binary_masks"])
                if (
                    ids.ndim != 1
                    or not np.issubdtype(ids.dtype, np.integer)
                    or len(np.unique(ids)) != len(ids)
                    or binary.shape != (len(ids), *frame_shapes[index])
                    or not np.all((binary == 0) | (binary == 1))
                ):
                    raise PipelineError("SAM 3.1: несогласованные ID/бинарные маски/размеры")
                selected = np.flatnonzero(ids == 1)
                seen.add(index)
                yield (
                    index,
                    (
                        binary[selected[0]].astype(np.uint8)
                        if len(selected)
                        else np.zeros(frame_shapes[index], dtype=np.uint8)
                    ),
                )
            if len(seen) != len(frame_shapes):
                raise PipelineError("SAM 3.1: неполные предсказания по кадрам")
        finally:
            cleanup_preserving_error(
                lambda: self.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for session_id in list(self.predictor._all_inference_states):
            try:
                self.handle_request({"type": "close_session", "session_id": session_id})
            except BaseException as exc:
                errors.append(exc)
        # Builder enters tracker context first, predictor context second.
        for owner in (self.predictor, self.predictor.model.tracker):
            context = getattr(owner, "bf16_context", None)
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
