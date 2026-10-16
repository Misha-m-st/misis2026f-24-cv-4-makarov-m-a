import numpy as np
import pytest

from document_vos.evaluation.metrics import boundary_band, boundary_f, iou, mask_metrics


@pytest.mark.parametrize("shape", [(1, 1), (1, 31), (31, 1), (17, 23), (127, 211), (480, 640)])
def test_cropped_metrics_equal_full_resolution(shape):
    rng = np.random.default_rng(42)
    pairs = [
        (np.zeros(shape, np.uint8), np.ones(shape, np.uint8)),
        (np.ones(shape, np.uint8), np.ones(shape, np.uint8)),
    ]
    for index in range(12):
        a, b = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        for mask in [a, b]:
            y, x = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            mask[
                y : min(shape[0], y + max(1, shape[0] // 3)),
                x : min(shape[1], x + max(1, shape[1] // 3)),
            ] = 1
        if index % 3 == 0:
            a[0, 0], b[-1, -1] = 1, 1
        pairs.append((a, b))
    pairs.append(
        (rng.integers(0, 2, shape, dtype=np.uint8), rng.integers(0, 2, shape, dtype=np.uint8))
    )
    for a, b in pairs:
        if not a.any() or not b.any():
            expected = dict.fromkeys(
                ["J", "F", "JF", "BIoU"], float(bool(a.any()) == bool(b.any()))
            )
        else:
            aa, bb = a.astype(bool), b.astype(bool)
            j, f = iou(aa, bb), boundary_f(aa, bb)
            expected = {
                "J": j,
                "F": f,
                "JF": (j + f) / 2,
                "BIoU": iou(boundary_band(aa), boundary_band(bb)),
            }
        assert mask_metrics(a, b) == expected
