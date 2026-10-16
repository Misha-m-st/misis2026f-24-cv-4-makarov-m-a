"""Render saved Cutie predictions with upstream overlay_davis and FFmpeg."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(root, video_keys=None):
    result = {}
    for row in read_rows(root / "predictions.jsonl"):
        if video_keys is not None and row["video"] not in video_keys:
            continue
        key = (row["video"], row["frame"])
        if key in result:
            raise ValueError(f"Duplicate prediction: {key}")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid prediction: {path}")
        result[key] = (path, row)
    return result


def frame_times(sequence, fallback_fps=None):
    """Keep source timestamps, or use explicitly requested visualization timing."""
    source = [frame["time_s"] for frame in sequence["frames"]]
    if all(time is not None for time in source):
        times = [float(time) for time in source]
        basis = "source manifest timestamps"
    elif fallback_fps is not None:
        times = [index / fallback_fps for index in range(len(source))]
        basis = f"visualization only at {fallback_fps:g} fps; source timestamps unavailable"
    else:
        raise ValueError("Missing timestamps; specify --fallback-fps for visualization playback")
    if len(times) < 2 or not all(math.isfinite(t) for t in times):
        raise ValueError("Expected at least two frames with finite timestamps")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Expected strictly increasing timestamps")
    return times, basis


def mask_image(path, row, source_size, display_size):
    if digest(path) != row["sha256"]:
        raise ValueError(f"Prediction hash mismatch: {path}")
    with Image.open(path) as image:
        if image.size != source_size:
            raise ValueError(f"Prediction size mismatch: {path}")
        pixels = np.asarray(image)
        if pixels.ndim != 2 or not np.isin(pixels, [0, 1]).all():
            raise ValueError(f"Expected a single-channel 0/1 mask: {path}")
        return np.asarray(image.resize(display_size, Image.Resampling.NEAREST))


def probe(path):
    data = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,duration:frame=best_effort_timestamp_time",
            "-show_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(data)


def render(args):
    for command in ("ffmpeg", "ffprobe"):
        if not shutil.which(command):
            raise ValueError(f"Required executable is missing: {command}")
    baseline, finetuned = args.baseline.resolve(), args.finetuned.resolve()
    sequences = read_rows(baseline / "selected-manifest.jsonl")
    other = read_rows(finetuned / "selected-manifest.jsonl")
    if args.video:
        selected = set(args.video)
        sequences = [s for s in sequences if f"{s['dataset']}/{s['video']}" in selected]
        other = [s for s in other if f"{s['dataset']}/{s['video']}" in selected]
        if {f"{s['dataset']}/{s['video']}" for s in sequences} != selected:
            raise ValueError("Some --video keys are absent from the baseline manifest")

    # Match the actual input frames and their timestamps, not hashed folder names.
    def signature(rows):
        return [
            (s["dataset"], s["video"], [(f["id"], f["path"], f["time_s"]) for f in s["frames"]])
            for s in rows
        ]

    if signature(sequences) != signature(other):
        raise ValueError("The two runs do not contain identical input sequences")
    video_keys = {f"{s['dataset']}/{s['video']}" for s in sequences}
    maps = [predictions(root, video_keys) for root in (baseline, finetuned)]
    expected = set()
    for seq in sequences:
        key = f"{seq['dataset']}/{seq['video']}"
        frame_times(seq, args.fallback_fps)
        for index, frame in enumerate(seq["frames"]):
            pair = (key, frame["id"])
            if pair in expected or not Path(frame["path"]).is_file():
                raise ValueError(f"Duplicate or missing source frame: {pair}")
            expected.add(pair)
            for mapping in maps:
                if pair not in mapping or mapping[pair][1]["index"] != index:
                    raise ValueError(f"Missing or misaligned prediction: {pair}")
    if any(set(mapping) != expected for mapping in maps):
        raise ValueError("Prediction set differs from the selected manifest")

    # The upstream CPU visualization module also creates Torch palette constants.
    # Hide CUDA before importing it; no model, checkpoint, or inference is loaded.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(args.cutie.resolve()))
    from gui.interactive_utils import overlay_davis

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = (
        ImageFont.truetype(font_path, 23) if Path(font_path).is_file() else ImageFont.load_default()
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "IN_PROGRESS",
        "renderer": "Cutie gui.interactive_utils.overlay_davis",
        "baseline": str(baseline),
        "finetuned": str(finetuned),
        "panel_height_limit": args.panel_height,
        "panel_width_limit": args.panel_width,
        "finetuned_label": args.finetuned_label,
        "overlay_alpha": 0.5,
        "mask_resize": "nearest",
        "frame_resize": "lanczos",
        "timing": "per-video timing basis; MP4 time base 1/90000",
        "fallback_fps": args.fallback_fps,
        "audio": False,
        "inference": False,
        "smoke": args.smoke,
        "input_hashes": {
            f"{root.name}/{name}": digest(root / name)
            for root in (baseline, finetuned)
            for name in ("selected-manifest.jsonl", "predictions.jsonl")
        },
        "cutie_revision": subprocess.check_output(
            ["git", "-C", str(args.cutie), "rev-parse", "HEAD"], text=True
        ).strip(),
        "videos": [],
    }
    report_path = output / "render.json"

    def save_report():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save_report()
    try:
        for seq in sequences[:1] if args.smoke else sequences:
            key = f"{seq['dataset']}/{seq['video']}"
            frames = seq["frames"][:12] if args.smoke else seq["frames"]
            source_video = seq.get("provenance", {}).get("source_video", {})
            source_path = source_video.get("path") if isinstance(source_video, dict) else None
            source_name = Path(source_path).stem if source_path else seq["dataset"]
            name = re.sub(r"[^\w.\-]+", "_", f"{source_name}_{seq['video']}_comparison")
            with Image.open(frames[0]["path"]) as image:
                source_size = image.size
            scale = min(1.0, args.panel_height / source_size[1])
            if args.panel_width is not None:
                scale = min(scale, args.panel_width / source_size[0])
            height = max(2, int(source_size[1] * scale / 2) * 2)
            width = max(2, int(source_size[0] * scale / 2) * 2)
            size = (width, height)
            all_times, timing_basis = frame_times(seq, args.fallback_fps)
            times = all_times[: len(frames)]
            source_timing = timing_basis == "source manifest timestamps"
            # Concat durations have microsecond precision. Quantize absolute PTS
            # first, so rounding each interval cannot accumulate across a video.
            times_us = [round((time - times[0]) * 1_000_000) for time in times]
            durations = [(b - a) / 1_000_000 for a, b in zip(times_us, times_us[1:])]
            durations.append(float(np.median(durations)))
            print(
                f"Rendering {source_name}: {len(frames)} frames, panels {width}x{height}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix=".frames-", dir=output) as temp:
                temp = Path(temp)
                concat = []
                for index, frame in enumerate(frames):
                    with Image.open(frame["path"]) as image:
                        if image.size != source_size:
                            raise ValueError(f"Source size changed: {frame['path']}")
                        original = np.asarray(
                            image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
                        )
                    panels = [original]
                    rows = []
                    for mapping in maps:
                        path, row = mapping[(key, frame["id"])]
                        mask = mask_image(path, row, source_size, size)
                        panels.append(overlay_davis(original, mask, alpha=0.5))
                        rows.append(row)
                    canvas = Image.new("RGB", (width * 3, height + 72), (20, 24, 30))
                    draw = ImageDraw.Draw(canvas)
                    for column, (label, panel) in enumerate(
                        zip(("Исходный кадр", "Cutie — оригинальная", args.finetuned_label), panels)
                    ):
                        draw.text((column * width + 12, 7), label, font=font, fill="white")
                        canvas.paste(Image.fromarray(panel), (column * width, 40))
                    note = (
                        " · начальная заданная маска"
                        if all(r.get("initialization") for r in rows)
                        else ""
                    )
                    time_note = (
                        f"{times[index]:.3f} с"
                        if source_timing
                        else f"просмотр {args.fallback_fps:g} кадр/с; исходное время неизвестно"
                    )
                    draw.text(
                        (12, height + 43),
                        f"Кадр {frame['id']} · {time_note}{note}",
                        font=font,
                        fill="white",
                    )
                    filename = f"{index:06d}.jpg"
                    canvas.save(temp / filename, quality=95, subsampling=0)
                    if index == min(30, len(frames) - 1):
                        canvas.save(output / f"{name}.jpg", quality=95, subsampling=0)
                    concat.extend(
                        [
                            f"file '{filename}'",
                            "option framerate 90000",
                            f"duration {durations[index]:.9f}",
                        ]
                    )
                    if (index + 1) % 100 == 0:
                        print(f"  {index + 1}/{len(frames)} frames", flush=True)
                (temp / "frames.txt").write_text("\n".join(concat) + "\n")
                video = output / f"{name}.mp4"
                log = output / f"{name}.ffmpeg.log"
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostdin",
                    "-n",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(temp / "frames.txt"),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-x264-params",
                    "fps=30/1",
                    "-preset",
                    "fast",
                    "-crf",
                    str(args.crf),
                    "-threads",
                    str(args.threads),
                    "-pix_fmt",
                    "yuv420p",
                    "-fps_mode",
                    "vfr",
                    "-bf",
                    "0",
                    "-enc_time_base",
                    "1:90000",
                    "-video_track_timescale",
                    "90000",
                    "-bsf:v",
                    f"setts=duration={round(durations[-1] * 90000)}",
                    "-movflags",
                    "+faststart",
                    str(video),
                ]
                with log.open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                measured = probe(video)
                info = measured["streams"][0]
                actual_times = [float(f["best_effort_timestamp_time"]) for f in measured["frames"]]
                if int(info["nb_read_frames"]) != len(frames) or len(actual_times) != len(frames):
                    raise ValueError(f"Encoded frame count mismatch: {video}")
                error = max(
                    abs(t - (expected_time - times[0]))
                    for t, expected_time in zip(actual_times, times)
                )
                if error > 0.0001 or (info["width"], info["height"]) != canvas.size:
                    raise ValueError(f"Encoded timestamps or dimensions mismatch: {video}")
                report["videos"].append(
                    {
                        "video": key,
                        "source_name": source_name,
                        "path": video.name,
                        "preview": f"{name}.jpg",
                        "frames": len(frames),
                        "source_size": source_size,
                        "encoded_size": canvas.size,
                        "duration_s": float(info["duration"]),
                        "timing_basis": timing_basis,
                        "max_timestamp_error_s": error,
                        "sha256": digest(video),
                        "verified": True,
                    }
                )
                save_report()
                print(
                    f"Verified {video.name}: {len(frames)} frames, PTS error {error:.6f}s",
                    flush=True,
                )
        playlist = ["#EXTM3U"]
        for video in report["videos"]:
            playlist.extend(
                [f"#EXTINF:{video['duration_s']:.6f},{video['source_name']}", video["path"]]
            )
        (output / "comparison.m3u").write_text("\n".join(playlist) + "\n")
        report["status"] = "COMPLETE"
        save_report()
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        save_report()
        raise
    print(f"Saved {len(report['videos'])} videos to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PROJECT / "results/phone-baseline-v5")
    parser.add_argument("--finetuned", type=Path, default=PROJECT / "results/phone-finetuned-v5")
    parser.add_argument("--cutie", type=Path, default=PROJECT / "external/Cutie")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; never overwrites results"
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=1080,
        help="Preview height; 1920 for full-height phone frames",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--video", action="append", help="Select dataset/video key; repeat for multiple sequences"
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        help="Explicit playback rate when timestamps are missing; not original capture timing",
    )
    parser.add_argument("--panel-width", type=int, help="Optional maximum width of each panel")
    parser.add_argument("--finetuned-label", default="Cutie — дообученная")
    parser.add_argument("--threads", type=int, default=4, help="FFmpeg encoding threads")
    parser.add_argument(
        "--smoke", action="store_true", help="Render only the first 12 frames of the first video"
    )
    args = parser.parse_args()
    if args.panel_height < 100 or not 0 <= args.crf <= 51:
        parser.error("panel-height must be >= 100 and crf must be in 0..51")
    if args.panel_width is not None and args.panel_width < 100:
        parser.error("panel-width must be >= 100")
    if args.threads < 1:
        parser.error("threads must be >= 1")
    if args.fallback_fps is not None and (
        not math.isfinite(args.fallback_fps) or not 0 < args.fallback_fps <= 120
    ):
        parser.error("fallback-fps must be finite and in (0, 120]")
    render(args)


if __name__ == "__main__":
    main()
