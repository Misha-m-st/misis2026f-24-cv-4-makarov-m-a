"""FFmpeg decoding: one PNG per source frame, source presentation timestamps."""

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"{argv[0]} exited {result.returncode}: {result.stderr[-4000:]}")
    return result.stdout


def extract(video: Path, directory: Path, ffmpeg: str, ffprobe: str) -> tuple[list[dict], dict]:
    probe = json.loads(
        command(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "stream=width,height,time_base:stream_tags=rotate:stream_side_data=rotation:"
                "frame=best_effort_timestamp,pts",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = probe["streams"][0]
    time_base = Fraction(stream["time_base"])
    try:
        pts = [int(f.get("best_effort_timestamp", f.get("pts"))) for f in probe["frames"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("missing/invalid source presentation timestamps") from exc
    times = [float(p * time_base) for p in pts]
    if not times or any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("missing/invalid source presentation timestamps")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("source presentation timestamps must strictly increase")
    rotation = float(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])
    if not math.isfinite(rotation) or rotation % 90:
        raise ValueError(f"unsupported phone rotation: {rotation}")
    size = (stream["width"], stream["height"])
    if rotation % 180:
        size = size[::-1]
    directory.mkdir(parents=True)
    # Autorotation is FFmpeg's default; no scale, fps or resampling filter.
    command(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-start_number",
            "0",
            str(directory / "%06d.png"),
        ]
    )
    paths = sorted(directory.glob("*.png"))
    if len(paths) != len(times):
        raise ValueError(f"decoded frame count {len(paths)} != timestamp count {len(times)}")
    frames = []
    for i, (path, t) in enumerate(zip(paths, times)):
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"frame {i}: size {image.size} != oriented source {size}")
        frames.append({"id": f"{i:06d}", "path": str(path.resolve()), "time_s": t})
    return frames, {
        "time_base": str(time_base),
        "source_pts": pts,
        "rotation_degrees": rotation,
        "source_size": [stream["width"], stream["height"]],
        "oriented_size": list(size),
    }
