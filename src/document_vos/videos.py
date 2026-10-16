"""Prepare user videos with explicit initial PNG/corners, or optional SAM3."""

from pathlib import Path

import numpy as np
from PIL import Image

from document_vos.data import Frame, Sequence, save_manifest
from document_vos.geometry import quad_mask, validate_quad
from document_vos.io import PipelineError, digest, new_directory, read_json, sha256, write_json
from document_vos.locking import exclusive_lock
from document_vos.runs import inventory, verify_inventory
from document_vos.video import extract

EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def inputs(directory: Path, *, sam: bool = False) -> tuple[list[Path], dict]:
    videos = sorted(
        p.resolve() for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file()
    )
    if not videos:
        raise PipelineError(
            f"Нет видео в {directory}. Добавьте видео и .mask.png либо .corners.json"
        )
    sources = []
    for video in videos:
        mask, corners = video.with_suffix(".mask.png"), video.with_suffix(".corners.json")
        if mask.exists() and corners.exists():
            raise PipelineError(f"Неоднозначная первая маска: {mask} и {corners}")
        if sam and (mask.exists() or corners.exists()):
            raise PipelineError(f"--sam и ручной источник маски одновременно: {video}")
        if not sam and not mask.exists() and not corners.exists():
            raise PipelineError(f"Необходимо {mask.name} либо {corners.name}")
        source = {"video": str(video), "sha256": sha256(video)}
        for name, path in [("mask", mask), ("corners", corners)]:
            if path.exists():
                source[name] = {"path": str(path), "sha256": sha256(path)}
        sources.append(source)
    return videos, {"sources": sources, "sam": sam}


def initial_mask(source: dict, size: tuple[int, int]) -> np.ndarray:
    if "mask" in source:
        with Image.open(source["mask"]["path"]) as image:
            mask = np.asarray(image)
        if (
            mask.ndim != 2
            or mask.shape != (size[1], size[0])
            or not (np.isin(mask, [0, 1]).all() or np.isin(mask, [0, 255]).all())
        ):
            raise PipelineError("Первая маска: один канал, размер кадра, значения 0/1 либо 0/255")
        mask = (mask > 0).astype("uint8")
    else:
        points = read_json(Path(source["corners"]["path"]))
        if isinstance(points, dict):
            if set(points) != {"corners"}:
                raise PipelineError('corners.json: ожидается список либо {"corners": [...]}')
            points = points["corners"]
        validate_quad(points)
        mask = quad_mask(points, *size)
    if not mask.any():
        raise PipelineError("Первая маска пустая")
    return mask


def prepare(
    directory: Path, prepared: Path, cfg: dict, *, sam: bool = False, output: Path | None = None
) -> dict:
    videos, signature = inputs(directory, sam=sam)
    from document_vos.runs import source_signature

    signature.update(settings=cfg["sam"] if sam else {}, source_sha256=source_signature())
    if sam:
        path = Path(cfg["sam"]["weights"])
        if not path.is_file():
            raise PipelineError("Нет весов SAM3: выполните document-vos setup --sam")
        signature["weights_sha256"] = sha256(path)
    target = output or prepared / "videos" / digest(signature)[:20]
    if any(p.is_relative_to(target.resolve()) for p in videos):
        raise PipelineError("Исходные видео должны находиться вне output")
    with exclusive_lock(target.parent / f".{target.name}.lock"):
        if target.exists():
            receipt = read_json(target / "receipt.json")
            if receipt["inputs"] != signature:
                raise PipelineError("Входы подготовки изменились; выберите новый output")
            # A deleted review PNG is an admission decision, never restored or treated as corruption.
            current = {
                k: v
                for k, v in inventory(target).items()
                if k != "receipt.json" and not k.startswith("review_initial/")
            }
            if current != receipt["files"]:
                raise PipelineError(f"Подготовленный артефакт изменился: {target}")
            verify_inventory(target, receipt["files"], exact=False)
            result = read_json(target / "preparation_result.json")
            return {
                "status": "EXISTING" if result["status"] == "COMPLETE" else result["status"],
                "output": str(target),
                "review_pngs_restored": 0,
            }
        if sam:
            from .sam.prepare import prepare as sam_prepare
            from .sam.sam import SAM

            settings = cfg["sam"]
            from document_vos.runtime import resolve_device

            model = SAM(
                Path(settings["weights"]),
                device=resolve_device(cfg["runtime"]["device"]),
                prompt=settings["prompt"],
                min_score=settings["min_score"],
                max_masks=settings["max_masks"],
            )
            sam_prepare(videos, target, model, step=settings["step"], identity=signature)
        else:
            new_directory(target)
            sequences = []
            (target / "review_initial").mkdir()
            for video, source in zip(videos, signature["sources"]):
                name = digest(str(video))[:16]
                frames, timing = extract(video, target / "frames" / name)
                with Image.open(frames[0]["path"]) as image:
                    rgb = np.asarray(image.convert("RGB"))
                    mask = initial_mask(source, image.size)
                path = target / "masks" / f"{name}.png"
                path.parent.mkdir(exist_ok=True)
                Image.fromarray(mask).save(path)
                frames[0]["mask"] = str(path.resolve())
                from document_vos.rendering import overlay

                overlay(rgb, mask, 0.35).save(target / "review_initial" / f"{name}.png")
                sequences.append(
                    Sequence(
                        "custom",
                        name,
                        name,
                        name,
                        name,
                        [Frame(**f) for f in frames],
                        provenance={
                            "source_video": source,
                            "annotation_kind": "user_initial_mask",
                            "phone_testset": {"status": "eligible", **timing},
                        },
                        time_source="PyAV source PTS * stream time_base",
                    )
                )
            save_manifest(target / "manifest.jsonl", sequences)
            write_json(
                target / "preparation_result.json", {"status": "COMPLETE", "videos": len(videos)}
            )
        _, after = inputs(directory, sam=sam)
        if after["sources"] != signature["sources"]:
            raise PipelineError("Исходные видео/маски изменились во время подготовки")
        files = {k: v for k, v in inventory(target).items() if not k.startswith("review_initial/")}
        write_json(target / "receipt.json", {"inputs": signature, "files": files})
        result = read_json(target / "preparation_result.json")
        return {"status": result["status"], "output": str(target), "review_pngs_restored": 0}
