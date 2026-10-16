"""Optional SAM3 preparation with geometric pseudo-labels and admission previews."""

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from ..io import sha256, write_json, write_jsonl
from ..video import extract
from .geometry import GEOMETRY_METHOD, GeometryOptions, binary, recover_quad

ANNOTATION_FAILURE_POLICY = "initial-required/skip-later-v1"


class Annotator(Protocol):
    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]: ...


def prepare(
    videos: list[Path],
    output: Path,
    predictor: Annotator,
    *,
    step: int | None = None,
    identity: dict | None = None,
    geometry: GeometryOptions | None = None,
    recreate: bool = False,
) -> dict:
    if step is not None and step < 1:
        raise ValueError("annotation step must be positive")
    geometry = geometry or GeometryOptions()
    videos, output = [v.resolve() for v in videos], output.resolve()
    if not videos or len(set(videos)) != len(videos):
        raise ValueError("supply distinct local videos")
    if any(v.is_relative_to(output) for v in videos):
        raise ValueError("source videos must be outside the output directory")
    inputs = {
        "schema": "phone-testset/sam-geometry-v1",
        "step": step,
        "model": identity or {},
        "geometry": asdict(geometry),
        "geometry_method": GEOMETRY_METHOD,
        "annotation_failure_policy": ANNOTATION_FAILURE_POLICY,
        "videos": [{"path": str(v), "sha256": sha256(v)} for v in videos],
    }
    if output.exists():
        old = json.loads((output / "preparation.json").read_text())
        if old.get("schema") != inputs["schema"]:
            raise ValueError("refusing to overwrite an unrelated dataset directory")
        if not recreate:
            if old != inputs:
                raise ValueError(
                    "preparation inputs changed; use new output or explicit --recreate"
                )
            if not all(
                (output / name).is_file()
                for name in ["manifest.jsonl", "annotations.jsonl", "preparation_result.json"]
            ):
                raise ValueError("preparation incomplete; use new output or explicit --recreate")
            return {"status": "EXISTING", "output": str(output), "review_pngs_restored": 0}
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "review_initial").mkdir()
    write_json(output / "preparation.json", inputs)
    rows, annotations = [], []
    for source in inputs["videos"]:
        video = Path(source["path"])
        name = hashlib.sha256(str(video).encode()).hexdigest()[:16]
        info = {
            "status": "eligible",
            "reason": None,
            "annotation_step": step,
            "annotation_failure_policy": ANNOTATION_FAILURE_POLICY,
            "annotation_failures": [],
        }
        row = {
            "schema_version": 1,
            "dataset": "custom",
            "video": name,
            "document": name,
            "template": name,
            "group": name,
            "frames": [],
            "time_source": "PyAV source PTS * stream time_base",
            "provenance": {
                "source_video": source,
                "model": identity or {},
                "phone_testset": info,
                "annotation_source": f"SAM 3 + {GEOMETRY_METHOD}",
                "annotation_kind": "geometric_pseudo_labels",
                "manual_mask_verification": False,
                "target": "planar full-page hypothesis including inferred occluded pixels",
            },
        }
        initial, initial_quad, initial_visible = None, None, None
        try:
            frames, timing = extract(video, output / "frames" / name)
            row["frames"] = frames
            info.update(timing)
            coverage = {
                "sampled_after_initial": len(range(0, len(frames), step or len(frames))) - 1,
                "accepted_after_initial": 0,
                "failed_after_initial": 0,
                "fraction": None,
            }
            info["annotation_coverage"] = coverage
            raw_dir, masks_dir = output / "visible_masks" / name, output / "masks" / name
            raw_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)
            for i in range(0, len(frames), step or len(frames)):
                evidence = {"video": f"custom/{name}", "index": i, "frame": frames[i]["id"]}
                try:
                    with Image.open(frames[i]["path"]) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    visible, model_info = predictor.predict(rgb)
                    visible = binary(visible, rgb.shape[:2])
                    raw_path = raw_dir / f"{i:06d}.png"
                    Image.fromarray(visible).save(raw_path)
                    evidence.update(
                        model=model_info,
                        visible_mask=str(raw_path),
                        visible_sha256=sha256(raw_path),
                    )
                    mask, geometry_info = recover_quad(visible, geometry)
                    path = masks_dir / f"{i:06d}.png"
                    Image.fromarray(mask).save(path)
                    frames[i]["mask"] = str(path)
                    evidence.update(
                        status="accepted",
                        geometry=geometry_info,
                        mask=str(path),
                        mask_sha256=sha256(path),
                    )
                    if i == 0:
                        initial, initial_quad = mask, geometry_info["quad"]
                        initial_visible = visible
                    else:
                        coverage["accepted_after_initial"] += 1
                except Exception as exc:
                    frames[i].pop("mask", None)
                    evidence.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    info["annotation_failures"].append(
                        {"index": i, "frame": frames[i]["id"], "reason": evidence["reason"]}
                    )
                    if i == 0:
                        raise ValueError(f"annotation frame {i}: {evidence['reason']}") from exc
                    coverage["failed_after_initial"] += 1
                finally:
                    annotations.append(evidence)
                    if coverage["sampled_after_initial"]:
                        coverage["fraction"] = (
                            coverage["accepted_after_initial"] / coverage["sampled_after_initial"]
                        )
            with Image.open(frames[0]["path"]) as image:
                preview = np.asarray(image.convert("RGB")).copy()
            selected = initial.astype(bool)
            preview[selected] = (0.65 * preview[selected] + 0.35 * np.array([0, 255, 0])).astype(
                np.uint8
            )
            cv2.polylines(preview, [np.rint(initial_quad).astype(np.int32)], True, (255, 160, 0), 2)
            contours, _ = cv2.findContours(
                initial_visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(preview, contours, -1, (0, 100, 255), 1)
            Image.fromarray(preview).save(output / "review_initial" / f"{name}.png")
        except Exception as exc:
            info.update(status="ineligible", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    write_jsonl(output / "manifest.jsonl", rows)
    write_jsonl(output / "annotations.jsonl", annotations)
    eligible = sum(row["provenance"]["phone_testset"]["status"] == "eligible" for row in rows)
    result = {
        "status": "COMPLETE" if eligible else "NO_ELIGIBLE_VIDEOS",
        "videos": len(rows),
        "eligible": eligible,
        "ineligible": len(rows) - eligible,
        "failed_annotations": sum(
            len(row["provenance"]["phone_testset"]["annotation_failures"]) for row in rows
        ),
        "annotation_failure_policy": ANNOTATION_FAILURE_POLICY,
        "output": str(output),
        "annotation": "automatic geometric pseudo-labels; no manual verification",
    }
    write_json(output / "preparation_result.json", result)
    return result
