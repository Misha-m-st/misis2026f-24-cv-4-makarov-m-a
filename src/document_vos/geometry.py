"""Четырёхугольник пересекается с кадром до растеризации; координаты не зажимаются."""

import numpy as np
from PIL import Image, ImageDraw

from .io import PipelineError


def validate_quad(points: list[list[float]]) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise PipelineError("Нужны четыре конечные координаты (x, y) по обходу контура")
    edges = np.roll(p, -1, axis=0) - p
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(crosses > 1e-8) or np.all(crosses < -1e-8)):
        raise PipelineError("Вырожденный, невыпуклый или самопересекающийся четырёхугольник")
    return p


def clip_polygon(points: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    """Sutherland–Hodgman; координаты пиксельных центров: [0,w-1] × [0,h-1]."""
    if width < 1 or height < 1:
        raise PipelineError("Размер кадра должен быть положительным")
    polygon = [tuple(v) for v in validate_quad(points)]
    for axis, limit, sign in [(0, 0, 1), (0, width - 1, -1), (1, 0, 1), (1, height - 1, -1)]:
        output = []
        if not polygon:
            break
        previous = polygon[-1]
        for current in polygon:
            inside = sign * (current[axis] - limit) >= 0
            prev_inside = sign * (previous[axis] - limit) >= 0
            if inside != prev_inside:
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                q = [previous[j] + t * (current[j] - previous[j]) for j in (0, 1)]
                q[axis] = float(limit)
                output.append(tuple(q))
            if inside:
                output.append(current)
            previous = current
        polygon = output
    return polygon


def quad_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    polygon = clip_polygon(points, width, height)
    mask = Image.new("L", (width, height), 0)
    # Не превращаем касание края в фиктивный объект.
    if len(polygon) >= 3:
        p = np.asarray(polygon)
        area = abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2
        if area > 1e-8:
            ImageDraw.Draw(mask).polygon(polygon, fill=1)
    return np.asarray(mask, dtype=np.uint8)
