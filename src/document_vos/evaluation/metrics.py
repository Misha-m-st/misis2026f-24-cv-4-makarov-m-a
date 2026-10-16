"""J, DAVIS F и Boundary IoU на исходном разрешении, значения в диапазоне [0,1]."""

import math

import numpy as np
from scipy import ndimage

from document_vos.io import PipelineError

METRICS = ("J", "F", "JF", "BIoU")
METRIC_PROTOCOL = {
    "version": 1,
    "F_tolerance_diagonal": 0.008,
    "F_boundary": "DAVIS seg2bmap",
    "BIoU_band_diagonal": 0.02,
    "BIoU_erosion": "3x3 zero-padded square",
    "empty_policy": "both=1, one=0",
    "initial_frame": "excluded",
    "decay": "DAVIS four overlapping intervals, int64 indices",
    "aggregation": "frames -> equal videos in dataset -> equal datasets",
}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """Однопиксельная граница DAVIS, с его соглашением на последней строке/столбце."""
    east, south, southeast = (np.zeros_like(mask) for _ in range(3))
    east[:, :-1] = mask[:, 1:]
    south[:-1] = mask[1:]
    southeast[:-1, :-1] = mask[1:, 1:]
    b = (mask ^ east) | (mask ^ south) | (mask ^ southeast)
    b[-1] = mask[-1] ^ east[-1]
    b[:, -1] = mask[:, -1] ^ south[:, -1]
    b[-1, -1] = False
    return b


def boundary_f(a: np.ndarray, b: np.ndarray, *, diagonal: float | None = None) -> float:
    ab, bb = boundary_map(a), boundary_map(b)
    na, nb = int(ab.sum()), int(bb.sum())
    if na == 0 or nb == 0:
        return float(na == nb)
    radius = max(1, math.ceil(0.008 * (math.hypot(*a.shape) if diagonal is None else diagonal)))
    # Euclidean distance <= radius эквивалентно dilation диском DAVIS.
    precision = np.count_nonzero(ab & (ndimage.distance_transform_edt(~bb) <= radius)) / na
    recall = np.count_nonzero(bb & (ndimage.distance_transform_edt(~ab) <= radius)) / nb
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def boundary_band(mask: np.ndarray, *, diagonal: float | None = None) -> np.ndarray:
    radius = max(1, round(0.02 * (math.hypot(*mask.shape) if diagonal is None else diagonal)))
    eroded = ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), bool), iterations=radius, border_value=0
    )
    return mask & ~eroded


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape or gt.ndim != 2 or min(gt.shape) < 1:
        raise PipelineError("Метрики требуют две маски одинакового исходного размера H×W")
    for m in [pred, gt]:
        if not ((m == 0) | (m == 1)).all():
            raise PipelineError("Метрики принимают только бинарные маски {0,1}")
    a, b = pred.astype(bool), gt.astype(bool)
    if not a.any() or not b.any():
        value = float(bool(a.any()) == bool(b.any()))
        return dict.fromkeys(METRICS, value)
    diagonal = math.hypot(*a.shape)
    # Remove only shared zero background. The halo preserves DAVIS boundary
    # edge rules; both radii still use the original image diagonal.
    foreground = a | b
    rows = np.flatnonzero(foreground.any(axis=1))
    columns = np.flatnonzero(foreground.any(axis=0))
    y0, y1 = max(0, int(rows[0]) - 1), min(a.shape[0], int(rows[-1]) + 2)
    x0, x1 = max(0, int(columns[0]) - 1), min(a.shape[1], int(columns[-1]) + 2)
    a, b = a[y0:y1, x0:x1].copy(), b[y0:y1, x0:x1].copy()
    j, f = iou(a, b), boundary_f(a, b, diagonal=diagonal)
    return {
        "J": j,
        "F": f,
        "JF": (j + f) / 2,
        "BIoU": iou(boundary_band(a, diagonal=diagonal), boundary_band(b, diagonal=diagonal)),
    }


def decay(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, float)
    if not np.isfinite(arr).all():
        raise PipelineError("Decay не принимает пропуски/NaN")
    ids = (np.round(np.linspace(1, len(arr), 5) + 1e-10) - 1).astype(np.int64)
    return float(arr[ids[0] : ids[1] + 1].mean() - arr[ids[3] : ids[4] + 1].mean())
