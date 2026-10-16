"""Conservative planar-page hypothesis from supported straight boundary segments."""

from dataclasses import asdict, dataclass
from itertools import combinations

import cv2
import numpy as np

GEOMETRY_METHOD = "supported-boundary-lines/v4-clipped"


def clip_polygon(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Intersect a polygon with the image; do not clamp its original corners."""
    h, w = shape
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    for axis, bound, sign in [
        (0, 0.0, 1),
        (0, float(w - 1), -1),
        (1, 0.0, 1),
        (1, float(h - 1), -1),
    ]:
        if not len(points):
            break
        result = []
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            inside_a, inside_b = sign * (a[axis] - bound) >= 0, sign * (b[axis] - bound) >= 0
            if inside_a:
                result.append(a)
            if inside_a != inside_b:
                result.append(a + (b - a) * ((bound - a[axis]) / (b[axis] - a[axis])))
        points = np.asarray(result, dtype=np.float64).reshape(-1, 2)
    return points


def rasterize_quad(quad: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    polygon = clip_polygon(quad, shape)
    mask = np.zeros(shape, np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask, polygon


def clip_segment(
    a: np.ndarray, b: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    direction = b - a
    lo, hi = 0.0, 1.0
    for axis, maximum in [(0, w - 1), (1, h - 1)]:
        if abs(direction[axis]) < 1e-12:
            if not 0 <= a[axis] <= maximum:
                return None
        else:
            t0, t1 = sorted([-a[axis] / direction[axis], (maximum - a[axis]) / direction[axis]])
            lo, hi = max(lo, t0), min(hi, t1)
            if lo >= hi:
                return None
    return a + lo * direction, a + hi * direction


def visible_side_support(
    boundary: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance: float, shape: tuple[int, int]
) -> float:
    segment = clip_segment(a, b, shape)
    if segment is None:
        return 0.0
    return line_support(boundary, *segment, tolerance)


@dataclass(frozen=True)
class GeometryOptions:
    line_tolerance_px: float = 2.5
    min_segment_fraction: float = 0.10
    min_side_support: float = 0.55
    min_visible_containment: float = 0.995
    max_area_ratio: float = 1.65
    min_area_fraction: float = 0.005
    min_angle_degrees: float = 20.0
    ambiguity_margin: float = 0.02


def binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != shape or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError(
            f"invalid mask: expected one-channel 0/1 with shape {shape}; got {mask.shape}"
        )
    if not mask.any():
        raise ValueError("visible mask is empty")
    return mask.astype(np.uint8)


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # A line is [normal_x, normal_y, offset], with n.x = offset.
    matrix = np.stack([a[:2], b[:2]])
    if abs(np.linalg.det(matrix)) < 1e-6:
        raise ValueError("parallel page sides")
    return np.linalg.solve(matrix, np.array([a[2], b[2]]))


def line_support(boundary: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance: float) -> float:
    direction = b - a
    length = np.linalg.norm(direction)
    unit = direction / length
    relative = boundary - a
    along = relative @ unit
    distance = abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
    near = along[(distance <= tolerance) & (along >= 0) & (along <= length)]
    if not len(near):
        return 0.0
    # Measure coverage along the side, not merely the count of nearby pixels.
    bins = max(10, int(length / max(1.0, tolerance)))
    occupied = np.unique(np.minimum(bins - 1, (near / length * bins).astype(int)))
    return len(occupied) / bins


def equivalent_quads(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    return (
        min(
            float(np.linalg.norm(a - np.roll(b, shift, axis=0), axis=1).max()) for shift in range(4)
        )
        <= tolerance
    )


def equivalent_polygons(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    """Compare observable boundaries, including polygons with clipped corners."""
    if len(a) < 3 or len(b) < 3:
        return False

    def distance(points: np.ndarray, polygon: np.ndarray) -> float:
        starts = polygon
        vectors = np.roll(polygon, -1, axis=0) - starts
        lengths = np.sum(vectors * vectors, axis=1)
        relative = points[:, None, :] - starts[None, :, :]
        t = np.sum(relative * vectors[None, :, :], axis=2) / np.maximum(lengths, 1e-12)
        projected = starts[None, :, :] + np.clip(t, 0, 1)[:, :, None] * vectors[None, :, :]
        return float(np.linalg.norm(points[:, None, :] - projected, axis=2).min(axis=1).max())

    return max(distance(a, b), distance(b, a)) <= tolerance


def recover_quad(
    visible: np.ndarray, options: GeometryOptions | None = None
) -> tuple[np.ndarray, dict]:
    options = options or GeometryOptions()
    if options.line_tolerance_px <= 0 or not 0 < options.min_side_support <= 1:
        raise ValueError("invalid geometry thresholds")
    if not 0 < options.min_visible_containment <= 1 or options.max_area_ratio < 1:
        raise ValueError("invalid geometry thresholds")
    if not 0 < options.min_segment_fraction < 1 or not 0 < options.min_area_fraction < 1:
        raise ValueError("invalid geometry thresholds")
    visible = np.asarray(visible)
    if visible.ndim != 2:
        raise ValueError("visible mask must be two-dimensional")
    visible = binary(visible, visible.shape)
    h, w = visible.shape
    area = int(visible.sum())
    if area < options.min_area_fraction * h * w:
        raise ValueError("visible page area is too small")
    contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.concatenate(contours).reshape(-1, 2).astype(np.float64)
    # Image clipping creates artificial boundary pixels. They are not evidence
    # for a physical page side and must not enter line fitting/support checks.
    page_boundary = boundary[
        (boundary[:, 0] > 0)
        & (boundary[:, 0] < w - 1)
        & (boundary[:, 1] > 0)
        & (boundary[:, 1] < h - 1)
    ]
    hull = cv2.convexHull(boundary.astype(np.float32)).reshape(-1, 2)
    # Ignore boundary detail already within the line-fitting error tolerance.
    vertices = (
        cv2.approxPolyDP(hull, options.line_tolerance_px, True).reshape(-1, 2).astype(np.float64)
    )
    if len(vertices) < 4:
        raise ValueError("boundary has fewer than four vertices")
    diagonal = float(np.hypot(h, w))
    lines = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if any(
            abs(a[axis] - edge) < 0.5 and abs(b[axis] - edge) < 0.5
            for axis, edge in [(0, 0), (0, w - 1), (1, 0), (1, h - 1)]
        ):
            continue
        vector = b - a
        length = np.linalg.norm(vector)
        if length < options.min_segment_fraction * diagonal:
            continue
        normal = np.array([-vector[1], vector[0]]) / length
        offset = normal @ a
        near = page_boundary[abs(page_boundary @ normal - offset) <= options.line_tolerance_px]
        if len(near) < 4:
            continue
        vx, vy, x, y = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        normal = np.array([-vy, vx], dtype=float)
        lines.append(np.array([*normal, normal @ np.array([x, y])]))
    if len(lines) < 4:
        raise ValueError("fewer than four supported page sides")
    if len(lines) > 12:
        raise ValueError("boundary has too many supported sides for a bounded hypothesis")
    candidates = []
    for indexes in combinations(range(len(lines)), 4):
        chosen = [lines[i] for i in indexes]
        try:
            quad = np.stack([intersection(chosen[i - 1], chosen[i]) for i in range(4)])
        except ValueError:
            continue
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
            continue
        edges = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(edges, axis=1)
        if (lengths < options.min_segment_fraction * diagonal).any():
            continue
        incoming = -np.roll(edges, 1, axis=0)
        cosines = (incoming * edges).sum(axis=1) / (np.roll(lengths, 1) * lengths)
        angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
        if (angles < options.min_angle_degrees).any() or (
            angles > 180 - options.min_angle_degrees
        ).any():
            continue
        supports = [
            visible_side_support(
                page_boundary, quad[i], quad[(i + 1) % 4], options.line_tolerance_px, visible.shape
            )
            for i in range(4)
        ]
        if min(supports) < options.min_side_support:
            continue
        mask, _ = rasterize_quad(quad, visible.shape)
        containment = float(np.count_nonzero(mask & visible) / area)
        ratio = float(mask.sum() / area)
        if (
            containment < options.min_visible_containment
            or not 1 <= ratio <= options.max_area_ratio
        ):
            continue
        score = float(np.mean(supports) - 0.1 * (ratio - 1))
        # Canonical clockwise order with topmost corner first, for provenance.
        start = np.lexsort((quad[:, 0], quad[:, 1]))[0]
        quad = np.roll(quad, -int(start), axis=0)
        candidates.append((score, mask, quad, supports, containment, ratio))
    if not candidates:
        raise ValueError("no quadrilateral passes side-support, containment and expansion gates")
    candidates.sort(key=lambda c: c[0], reverse=True)
    distinct = []
    for candidate in candidates:
        if not any(
            equivalent_polygons(
                clip_polygon(candidate[2], visible.shape),
                clip_polygon(other[2], visible.shape),
                options.line_tolerance_px,
            )
            for other in distinct
        ):
            distinct.append(candidate)
    best = distinct[0]
    if len(distinct) > 1 and best[0] - distinct[1][0] < options.ambiguity_margin:
        raise ValueError("ambiguous outer contour: multiple similarly supported quadrilaterals")
    score, mask, quad, supports, containment, ratio = best
    _, clipped = rasterize_quad(quad, visible.shape)
    return mask, {
        "method": GEOMETRY_METHOD,
        "target": "planar full-page hypothesis",
        "quad": quad.tolist(),
        "clipped_polygon": clipped.tolist(),
        "out_of_frame": bool(
            (quad < 0).any() or (quad[:, 0] > w - 1).any() or (quad[:, 1] > h - 1).any()
        ),
        "side_support": supports,
        "visible_containment": containment,
        "area_ratio": ratio,
        "geometry_score": score,
        "visible_pixels": area,
        "extrapolated_pixels": int(np.count_nonzero(mask & (1 - visible))),
        "thresholds": asdict(options),
        "candidate_count": len(candidates),
        "distinct_candidate_count": len(distinct),
        "interpretation": "geometric pseudo-label, not observed ground truth behind occluders",
    }
