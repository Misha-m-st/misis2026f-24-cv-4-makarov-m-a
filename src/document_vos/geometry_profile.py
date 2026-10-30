"""Bounded real-video geometry-transition diagnostic; not a full FPS benchmark."""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from document_vos.benchmark import ExternalPredictor
from document_vos.config import load
from document_vos.cutie.inference import propagate
from document_vos.data import ground_truth, load_manifest, save_manifest
from document_vos.evaluation import evaluate
from document_vos.io import PipelineError, atomic_json, read_json, read_jsonl, sha256, versions
from document_vos.runs import source_signature
from document_vos.runtime import Runtime, configure, runtime_snapshot


def select_clips(manifest, frame_limit):
    """First fully annotated landscape and portrait clips, in that order."""
    selected = {}
    for seq in load_manifest(manifest):
        index = next((i for i, frame in enumerate(seq.frames) if frame.has_gt), None)
        if index is None:
            continue
        frames = seq.frames[index : index + frame_limit]
        if len(frames) != frame_limit or not all(frame.has_gt for frame in frames):
            continue
        with Image.open(frames[0].path) as im:
            width, height = im.size
        orientation = "landscape" if width > height else "portrait" if height > width else None
        if orientation and orientation not in selected:
            item = deepcopy(seq)
            item.frames = frames
            selected[orientation] = item
        if len(selected) == 2:
            return [selected["landscape"], selected["portrait"]]
    raise PipelineError("Нужны полностью размеченные landscape и portrait клипы")


class TracedPredictor:
    def __init__(self, predictor, model):
        self.predictor, self.model = predictor, model
        self.last_model_seconds = 0.0
        self.events = []
        self.reset_calls = 0
        if model == "efficienttam":
            original = predictor.predictor.reset_state

            def reset(state):
                try:
                    return original(state)
                finally:
                    self.reset_calls += 1

            predictor.predictor.reset_state = reset

    def reset(self):
        self.predictor.reset()

    def sync(self):
        self.predictor.sync()

    def memory_stats(self):
        return self.predictor.memory_stats()

    def iter_sequence(self, frames, mask, cache_dir):
        event = {"frames": len(frames), "cumulative_model_seconds_at_emission": []}
        self.events.append(event)
        stream = self.predictor.iter_sequence(frames, mask, cache_dir)
        start = perf_counter()
        try:
            for result in stream:
                event["cumulative_model_seconds_at_emission"].append(
                    self.predictor.last_model_seconds
                )
                self.last_model_seconds = self.predictor.last_model_seconds
                yield result
        finally:
            stream.close()
            self.last_model_seconds = self.predictor.last_model_seconds
            self.predictor.sync()
            event["sequence_wall_seconds_including_writer_overlap"] = perf_counter() - start
            event["model_seconds"] = self.last_model_seconds
            event["geometry"] = self.predictor.geometry_records[-1]
            if self.model == "efficienttam":
                event["reset_calls_cumulative"] = self.reset_calls
            else:
                event["active_sessions_after_sequence"] = len(
                    self.predictor.predictor.predictor._all_inference_states
                )


def contact_sheet(seq, run_dir, output):
    predictions = {(r["video"], r["index"]): r for r in read_jsonl(run_dir / "predictions.jsonl")}
    panel_w, panel_h = 480, 330
    canvas = Image.new("RGB", (panel_w * 3, panel_h + 34), "white")
    draw = ImageDraw.Draw(canvas)
    for column, index in enumerate((0, len(seq.frames) // 2, len(seq.frames) - 1)):
        frame = seq.frames[index]
        with Image.open(frame.path) as source:
            original = source.convert("RGB")
        gt = ground_truth(frame, original.size)
        prediction = predictions[(seq.key, index)]
        with Image.open(run_dir / prediction["path"]) as mask_image:
            mask = np.array(mask_image, dtype=np.uint8)
        original.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        shape = original.size
        pred_img = Image.fromarray(mask * 255).resize(shape, Image.Resampling.NEAREST)
        gt_img = Image.fromarray(gt * 255).resize(shape, Image.Resampling.NEAREST)
        red = Image.new("RGB", shape, (255, 32, 32))
        original.paste(red, mask=pred_img.point(lambda x: int(x * 0.30)))
        edge = gt_img.filter(ImageFilter.FIND_EDGES).point(lambda x: 255 if x else 0)
        original.paste(Image.new("RGB", shape, (0, 220, 255)), mask=edge)
        x = column * panel_w + (panel_w - shape[0]) // 2
        y = (panel_h - shape[1]) // 2
        canvas.paste(original, (x, y))
        draw.text(
            (column * panel_w + 8, panel_h + 8),
            f"frame {index}: prediction red / GT cyan",
            fill="black",
        )
    canvas.save(output)


def run(args):
    cfg = load(args.config)
    configure(Runtime.parse(cfg["runtime"]))
    import torch

    if not torch.cuda.is_available():
        raise PipelineError("CUDA недоступна")
    random.seed(cfg["benchmark"]["seed"])
    np.random.seed(cfg["benchmark"]["seed"])
    torch.manual_seed(cfg["benchmark"]["seed"])
    torch.cuda.manual_seed_all(cfg["benchmark"]["seed"])
    torch.backends.cudnn.benchmark = False
    if not 8 <= args.frames <= 32:
        raise PipelineError("frames должен быть от 8 до 32")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    report = {
        "status": "RUNNING",
        "model": args.model,
        "pid": os.getpid(),
        "source_sha256": source_signature(),
        "manifest_sha256": sha256(args.manifest),
        "split_sha256": sha256(args.split_dir / "split.json"),
        "frames_per_clip": args.frames,
        "scope": "bounded diagnostic, not a full benchmark or statistically stable FPS",
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "passes": {},
    }
    atomic_json(status_path, report)
    predictor = None
    try:
        split = read_json(args.split_dir / "split.json")
        if report["manifest_sha256"] != split.get("validation_sha256"):
            raise PipelineError("Манифест не соответствует validation split")
        sequences = select_clips(args.manifest, args.frames)
        selected_manifest = output / "selected-manifest.jsonl"
        save_manifest(selected_manifest, sequences)
        weights = Path(cfg["benchmark"]["models"][args.model]["weights"])
        report["weights_sha256"] = sha256(weights)
        report["selection"] = [
            {
                "video": seq.key,
                "frames": [f.id for f in seq.frames],
                "first_frame": f.path if (f := seq.frames[0]) else None,
            }
            for seq in sequences
        ]
        atomic_json(status_path, report)
        predictor = ExternalPredictor(
            args.model, cfg, weights, "cuda", match_cutie_size=cfg["inference"]["max_internal_size"]
        )
        traced = TracedPredictor(predictor, args.model)
        report["runtime"] = runtime_snapshot()
        report["predictor_class"] = (
            type(predictor.predictor).__name__
            if args.model == "efficienttam"
            else type(predictor.predictor.predictor).__name__
        )
        for pass_name in ("cold_process", "repeat_same_process"):
            print(f"{args.model}: {pass_name}", flush=True)
            # Keep each run's geometry evidence local to that pass while
            # retaining the model and compiled components in this process.
            predictor.geometry_records.clear()
            inf_dir = output / pass_name / "inference"
            inf_dir.parent.mkdir()
            meta = {
                "model": args.model,
                "backend": args.model,
                "backend_revision": "see pinned external_models.json",
                "weights": str(weights),
                "weights_sha256": report["weights_sha256"],
                "inference": {
                    **cfg["benchmark"]["models"][args.model],
                    "resolution_policy": "cutie",
                    "reference_max_internal_size": cfg["inference"]["max_internal_size"],
                    "geometry": predictor.geometry_records,
                    "diagnostic": True,
                },
                "seed": cfg["benchmark"]["seed"],
                "device": "cuda",
                "versions": versions(),
                "scope": "validation_subset_diagnostic",
                "split_sha256": report["split_sha256"],
                "runtime": Runtime.parse(cfg["runtime"]).to_dict(),
            }
            start_event = len(traced.events)
            propagate(selected_manifest, inf_dir, traced, meta, sequences=sequences)
            result = read_json(inf_dir / "run.json")
            events = traced.events[start_event:]
            if len(events) != 2 or any(
                len(x["cumulative_model_seconds_at_emission"]) != args.frames for x in events
            ):
                raise PipelineError("Неполные результаты двух клипов")
            if args.model == "sam31" and any(e["active_sessions_after_sequence"] for e in events):
                raise PipelineError("SAM 3.1 оставил активную сессию")
            if args.model == "efficienttam":
                expected_start = 0 if pass_name == "cold_process" else 2
                if [e["reset_calls_cumulative"] for e in events] != [
                    expected_start + 1,
                    expected_start + 2,
                ]:
                    raise PipelineError("EfficientTAM не сбросил состояние после каждого клипа")
            for sequence, event in zip(sequences, events):
                event["video"] = sequence.key
                times = event["cumulative_model_seconds_at_emission"]
                event["first_output_model_seconds"] = times[0]
                event["remaining_after_first_output_model_seconds"] = (
                    event["model_seconds"] - times[0]
                )
                event["fps_all_model_calls"] = args.frames / event["model_seconds"]
                # SAM can buffer outputs: emission index need not equal a compute frame.
                event["after_first_output_fps"] = (
                    (args.frames - 1) / event["remaining_after_first_output_model_seconds"]
                    if args.model == "efficienttam"
                    and event["remaining_after_first_output_model_seconds"] > 0
                    else None
                )
            report["passes"][pass_name] = {
                "timing": result["timing"],
                "vram": result["vram"],
                "sequences": events,
                "evaluation": "PENDING",
            }
            atomic_json(status_path, report)
            evaluate(inf_dir, output / pass_name / "evaluation", workers=cfg["runtime"]["workers"])
            evaluation = read_json(output / pass_name / "evaluation" / "evaluation.json")
            report["passes"][pass_name]["evaluation"] = {
                "summary": evaluation["summary"],
                "videos": evaluation["videos"],
                "evaluated_frames": evaluation["selection"]["evaluated_frames"],
            }
            for orientation, sequence in zip(("landscape", "portrait"), sequences):
                contact_sheet(sequence, inf_dir, output / pass_name / f"{orientation}.png")
            atomic_json(status_path, report)
        predictor.close()
        predictor = None
        report["status"] = "COMPLETE"
        report["interpretation"] = (
            "Two real clips, two ordered passes in one process; repeat is a diagnostic warm pass, "
            "not steady-state proof. First process uses a fresh Inductor disk cache. "
            "CUDA-synchronized compute timer includes the first frame and compilation. "
            "SAM can buffer output so per-emission deltas are not per-frame timings."
        )
        atomic_json(status_path, report)
        return report
    except BaseException as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(status_path, report)
        raise
    finally:
        if predictor is not None:
            predictor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/split-v2/validation.jsonl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/split-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["efficienttam", "sam31"], required=True)
    parser.add_argument("--frames", type=int, default=12)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
