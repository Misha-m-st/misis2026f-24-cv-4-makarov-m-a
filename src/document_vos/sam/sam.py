"""Official SAM 3 image model in the preparation process, with local weights only."""

import importlib.metadata
import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import GeometryOptions, binary, recover_quad

REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
INSTANCE_SELECTION = "largest-area/nonoverlapping-v2"


def check_installation() -> None:
    distribution = importlib.metadata.distribution("sam3")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError(f"SAM 3 must be installed from pinned revision {REVISION}")


def limit_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
    min_score: float = 0.5,
    max_masks: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove overlapping smaller instances, then cap the largest survivors."""
    if max_masks < 1 or not 0 <= min_score <= 1:
        raise ValueError("max_masks must be positive and min_score must be in [0, 1]")
    masks, scores = np.asarray(masks), np.asarray(scores)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != shape or scores.shape != (len(masks),):
        raise ValueError("invalid SAM instance output shapes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("SAM returned invalid instance confidence")
    indices = np.flatnonzero(scores >= min_score)
    if not len(indices):
        return np.empty((0, *shape), np.uint8), scores[indices], indices
    accepted = np.stack([binary(masks[index], shape) for index in indices])
    areas = accepted.sum(axis=(1, 2), dtype=np.int64)
    order = np.lexsort((indices, -scores[indices], -areas))
    occupied = np.zeros(shape, dtype=bool)
    survivors = []
    for index in order:
        if not np.any(occupied & accepted[index]):
            survivors.append(index)
        # An overlap with any larger confident mask removes this instance,
        # even if that larger mask was itself removed by an earlier overlap.
        occupied |= accepted[index].astype(bool)
    selected = np.asarray(survivors[:max_masks], dtype=np.int64)
    return accepted[selected], scores[indices[selected]], indices[selected]


def choose_page_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    shape: tuple[int, int],
    min_score: float,
    geometry: GeometryOptions,
    max_masks: int = 1,
) -> tuple[np.ndarray, dict]:
    model_instances = len(scores)
    masks, scores, indices = limit_masks(masks, scores, shape, min_score, max_masks)
    if not len(masks):
        raise ValueError("SAM found no document instances above the confidence threshold")
    # The pipeline has one document. Do not silently replace the largest mask
    # with a smaller instance if its geometry cannot be recovered.
    recover_quad(masks[0], geometry)
    return masks[0], {
        "selection": INSTANCE_SELECTION,
        "selected_instance": int(indices[0]),
        "instance_score": float(scores[0]),
        "selected_area": int(masks[0].sum()),
        "instances": len(masks),
        "model_instances": model_instances,
        "max_masks": max_masks,
        "min_score": min_score,
        "retained_instances": indices.tolist(),
    }


class SAM:
    def __init__(
        self,
        weights: Path,
        device: str = "cuda",
        prompt: str = "sheet of paper",
        min_score: float = 0.5,
        geometry: GeometryOptions | None = None,
        max_masks: int = 1,
    ):
        if not weights.is_file():
            raise ValueError(
                "provide an existing local SAM checkpoint; weights are never downloaded"
            )
        if not prompt.strip() or not 0 <= min_score <= 1:
            raise ValueError("invalid SAM prompt/confidence thresholds")
        if max_masks < 1:
            raise ValueError("max_masks must be positive")
        check_installation()
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        # Build without a checkpoint to enforce strict loading ourselves; upstream loads non-strictly.
        self.model = build_sam3_image_model(
            checkpoint_path=None,
            load_from_HF=False,
            device="cpu",
            compile=False,
            enable_inst_interactivity=False,
        )
        checkpoint = torch.load(weights.resolve(), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise ValueError("SAM checkpoint must be a state dictionary")
        state = {
            k.removeprefix("detector."): v
            for k, v in checkpoint.items()
            if k.startswith("detector.")
        }
        if not state:
            raise ValueError("checkpoint has no SAM image detector weights")
        # Official checkpoints also include the SAM 1 interactive feature neck.
        # It is absent when enable_inst_interactivity=False; only that unused
        # subtree may be discarded. All active detector keys remain strict.
        state = {
            k: v
            for k, v in state.items()
            if not k.startswith("backbone.vision_backbone.sam2_convs.")
        }
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=min_score)
        self.device, self.prompt = device, prompt
        self.min_score = min_score
        self.geometry = geometry or GeometryOptions()
        self.max_masks = max_masks
        self.last_instance_selection = {}

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        masks, scores = self.predict_instances(rgb)
        mask, info = choose_page_mask(
            masks,
            scores,
            rgb.shape[:2],
            self.min_score,
            self.geometry,
            self.max_masks,
        )
        info["selected_instance"] = self.last_instance_selection["original_indices"][0]
        info["model_instances"] = self.last_instance_selection["model_instances"]
        info["retained_instances"] = self.last_instance_selection["original_indices"]
        return mask, info

    def predict_instances(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Visible masks ranked by area and capped, before geometric recovery."""
        import torch

        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        )
        with torch.inference_mode(), amp:
            state = self.processor.set_image(Image.fromarray(rgb))
            state = self.processor.set_text_prompt(prompt=self.prompt, state=state)
        raw_masks = state["masks"].detach().cpu().numpy()
        raw_scores = state["scores"].detach().float().cpu().numpy()
        masks, scores, indices = limit_masks(
            raw_masks, raw_scores, rgb.shape[:2], self.min_score, self.max_masks
        )
        self.last_instance_selection = {
            "model_instances": len(raw_scores),
            "original_indices": indices.tolist(),
            "max_masks": self.max_masks,
            "selection": INSTANCE_SELECTION,
        }
        return masks, scores


def compatibility() -> dict:
    import cv2
    import torch
    import torchvision

    result = {
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "sam_revision": REVISION,
        "model": "SAM 3 image",
        "real_inference": {
            "status": "NOT_RUN",
            "reason": "this compatibility probe checks imports only; it does not run inference",
        },
    }
    try:
        check_installation()
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
        from sam3.model_builder import build_sam3_image_model  # noqa: F401

        result["image_import"] = {"status": "PASS"}
    except Exception:
        result["image_import"] = {"status": "BLOCKED", "error": traceback.format_exc()}
    return result
