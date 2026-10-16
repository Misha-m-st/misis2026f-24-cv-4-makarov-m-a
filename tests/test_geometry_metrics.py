import numpy as np
import pytest

from document_vos.evaluation import aggregate, temporal_curve
from document_vos.evaluation.metrics import METRICS, boundary_band, decay, mask_metrics
from document_vos.geometry import clip_polygon, quad_mask
from document_vos.io import PipelineError


def test_polygon_clipped_as_intersection():
    # Простой clamp дал бы треугольник с неправильным пересечением нижней границы.
    points = [[-10, 5], [5, -10], [20, 5], [5, 20]]
    p = clip_polygon(points, 11, 11)
    assert len(p) == 4
    assert set(p) == {(0, 0), (10, 0), (10, 10), (0, 10)}
    assert quad_mask(points, 11, 11).sum() == 121
    assert quad_mask([[20, 20], [30, 20], [30, 30], [20, 30]], 11, 11).sum() == 0
    assert quad_mask([[-3, 1], [0, 1], [0, 5], [-3, 5]], 11, 11).sum() == 0


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [5, 5], [0, 5], [5, 0]],
        [[0, 0]] * 4,
        [[0, 0], [5, 0], [1, 1], [0, 5]],
        [[0, 0], [5, 0], [5, float("nan")], [0, 5]],
    ],
)
def test_bad_geometry(points):
    with pytest.raises(PipelineError):
        quad_mask(points, 10, 10)


def test_metrics_empty_shifted_and_edges():
    a = np.zeros((20, 20), np.uint8)
    a[5:10, 5:10] = 1
    b = np.zeros_like(a)
    b[5:10, 6:11] = 1
    empty = np.zeros_like(a)
    assert mask_metrics(a, a) == dict.fromkeys(METRICS, 1.0)
    assert mask_metrics(empty, empty) == dict.fromkeys(METRICS, 1.0)
    assert mask_metrics(a, empty) == dict.fromkeys(METRICS, 0.0)
    assert mask_metrics(empty, a) == dict.fromkeys(METRICS, 0.0)
    assert mask_metrics(a, b)["J"] == pytest.approx(20 / 30)
    assert mask_metrics(a, b)["F"] == 1
    assert 0 < mask_metrics(a, b)["BIoU"] < 1
    full = np.ones((20, 20), bool)
    boundary = boundary_band(full)
    assert boundary[0].all() and boundary[-1].all()
    assert boundary[:, 0].all() and not boundary[10, 10]
    assert mask_metrics(full, full)["BIoU"] == 1
    for shape in [(1, 1), (1, 10), (10, 1)]:
        mask = np.ones(shape, np.uint8)
        assert mask_metrics(mask, mask) == dict.fromkeys(METRICS, 1.0)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 20, 300, 1000])
def test_decay_short_long_and_constant(n):
    assert decay([0.42] * n) == pytest.approx(0)
    values = np.linspace(1, 0, n)
    ids = np.rint(np.linspace(1, n, 5) + 1e-10).astype(int) - 1
    expected = np.mean(values[: ids[1] + 1]) - np.mean(values[ids[3] :])
    assert decay(values.tolist()) == pytest.approx(expected)
    assert decay(values.tolist()) >= 0
    assert decay([]) is None


def test_equal_datasets_unequal_videos_and_lengths():
    rows, videos = [], []
    for d, v, score, n in [("a", "a1", 0, 100), ("a", "a2", 1, 1), ("b", "b1", 1, 2)]:
        common = dict(
            dataset=d,
            video=v,
            nonempty_gt=True,
            normalized_time=0.25,
            **dict.fromkeys(METRICS, score),
        )
        rows.extend([common] * n)
        videos.append(
            {
                **common,
                "evaluated_frames": n,
                **dict.fromkeys(["J_decay", "F_decay", "BIoU_decay"], 0),
            }
        )
    result = aggregate(rows, videos)
    assert result["primary_equal_datasets"]["J"] == 0.75
    assert result["equal_all_videos"]["J"] == pytest.approx(2 / 3)
    assert result["all_frames"]["J"] == pytest.approx(3 / 103)
    assert result["nonempty_gt_primary"]["J"] == 0.75
    curve = temporal_curve(rows)
    assert curve[2]["BIoU"] == 0.75 and curve[2]["videos"] == 3
    assert curve[0]["BIoU"] is None
