"""PyAV video I/O: one decoded frame per PTS, explicit display rotation, VFR MP4."""

import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from document_vos.io import PipelineError


def extract(video: Path, directory: Path) -> tuple[list[dict], dict]:
    if directory.exists():
        raise PipelineError(f"Каталог кадров уже существует: {directory}")
    directory.mkdir(parents=True)
    rows, pts, bases = [], [], []
    rotation, size, oriented = None, None, None
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise PipelineError(f"Нет видеопотока: {video}")
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise PipelineError("Отсутствует исходный PTS/time_base")
            time = float(frame.pts * frame.time_base)
            if not math.isfinite(time) or time < 0 or (rows and time <= rows[-1]["time_s"]):
                raise PipelineError("Исходные timestamps должны строго возрастать и быть >=0")
            angle = float(frame.rotation)
            if not math.isfinite(angle) or abs(angle / 90 - round(angle / 90)) > 1e-6:
                raise PipelineError(f"Неподдерживаемый поворот: {angle}")
            rgb = frame.to_ndarray(format="rgb24")
            # DISPLAYMATRIX rotation is counterclockwise (same convention as FFmpeg).
            rgb = np.rot90(rgb, round(angle / 90))
            if rotation is None:
                rotation, size, oriented = (
                    angle,
                    [frame.width, frame.height],
                    [rgb.shape[1], rgb.shape[0]],
                )
            if angle != rotation or [rgb.shape[1], rgb.shape[0]] != oriented:
                raise PipelineError("Размер или поворот изменяется внутри видео")
            path = directory / f"{i:06d}.png"
            Image.fromarray(rgb).save(path)
            rows.append({"id": f"{i:06d}", "path": str(path.resolve()), "time_s": time})
            pts.append(frame.pts)
            bases.append(str(frame.time_base))
    if not rows:
        raise PipelineError("Нет декодированных кадров")
    return rows, {
        "time_base": bases[0],
        "source_pts": pts,
        "source_time_bases": bases,
        "rotation_degrees": rotation,
        "source_size": size,
        "oriented_size": oriented,
    }


def encode(path: Path, images, times: list[float], *, fps: float = 30) -> dict:
    """Encode with a microsecond clock, no frame-rate resampling; verify by decoding."""
    if path.exists():
        raise PipelineError(f"Видео уже существует: {path}")
    if not times or any(not math.isfinite(t) for t in times):
        raise PipelineError("Нет корректных времён для рендеринга")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise PipelineError("Время визуализации должно строго возрастать")
    clock = Fraction(1, 1000000)
    expected = [round((t - times[0]) / clock) for t in times]
    if any(a >= b for a, b in zip(expected, expected[1:])):
        raise PipelineError("Интервал кадров меньше точности MP4 clock")
    path.parent.mkdir(parents=True, exist_ok=True)
    count, stream, original_size = 0, None, None
    with av.open(
        str(path), "w", options={"movflags": "+faststart", "video_track_timescale": "1000000"}
    ) as out:
        for image in images:
            if count >= len(times):
                raise PipelineError("Слишком много кадров для timestamps")
            rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
            if stream is None:
                original_size = [rgb.shape[1], rgb.shape[0]]
                stream = out.add_stream(
                    "libx264", rate=Fraction(str(fps)).limit_denominator(100000)
                )
                stream.width = rgb.shape[1] + rgb.shape[1] % 2
                stream.height = rgb.shape[0] + rgb.shape[0] % 2
                stream.pix_fmt = "yuv420p"
                stream.time_base = clock
                stream.codec_context.time_base = clock
                stream.options = {"crf": "18", "preset": "fast", "bf": "0", "threads": "1"}
            if [rgb.shape[1], rgb.shape[0]] != original_size:
                raise PipelineError("Размер кадров меняется")
            # Pad only the rendered MP4; model frames and PNG masks keep original size.
            rgb = np.pad(rgb, ((0, rgb.shape[0] % 2), (0, rgb.shape[1] % 2), (0, 0)), mode="edge")
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
            frame.pts, frame.time_base = expected[count], clock
            for packet in stream.encode(frame):
                out.mux(packet)
            count += 1
        if stream is None or count != len(times):
            raise PipelineError("Число кадров не совпадает с timestamps")
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(path)) as check:
        actual = [float(f.pts * f.time_base) for f in check.decode(video=0)]
    wanted = [n * float(clock) for n in expected]
    if len(actual) != len(wanted) or any(abs(a - b) > 1.1e-6 for a, b in zip(actual, wanted)):
        raise PipelineError("MP4 не сохранил число кадров или timestamps")
    return {
        "frames": count,
        "source_times_s": times,
        "output_times_s": actual,
        "time_origin_s": times[0],
        "timestamp_tolerance_s": 1.1e-6,
        "original_size": original_size,
        "encoded_size": [stream.width, stream.height],
        "backend": f"PyAV {av.__version__}",
        "verified": True,
    }
