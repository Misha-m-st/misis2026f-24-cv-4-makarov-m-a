"""Отбор готового набора по PNG допуска и маске первого кадра."""

from pathlib import Path

from PIL import Image

from .data import Sequence, load_binary, load_manifest
from .io import PipelineError, sha256


def select_input_folder(directory: Path) -> tuple[list[Sequence], dict]:
    directory = directory.resolve()
    manifest = directory / "manifest.jsonl"
    sequences = load_manifest(manifest)
    included, decisions = [], []
    for seq in sequences:
        reason = None
        info = seq.provenance.get("phone_testset", {})
        review = directory / "review_initial" / f"{seq.video}.png"
        if seq.dataset != "custom" or not info:
            reason = "not_phone_testset_sequence"
        elif Path(seq.video).name != seq.video or seq.video in {".", ".."}:
            reason = "invalid_review_path"
        elif info.get("status") != "eligible":
            reason = info.get("reason") or "sequence_ineligible"
        elif not review.is_file():
            reason = "review_png_missing"
        elif not seq.frames or not seq.frames[0].mask or seq.frames[0].polygon is not None:
            reason = "initial_mask_missing"
        else:
            try:
                with Image.open(seq.frames[0].path) as image:
                    initial = load_binary(Path(seq.frames[0].mask), image.size)
                if not initial.any():
                    reason = "initial_mask_empty"
            except (OSError, ValueError) as exc:
                reason = f"initial_mask_invalid: {exc}"
        if reason is None:
            included.append(seq)
        decisions.append({"video": seq.key, "included": reason is None, "reason": reason})
    if len({s.key for s in sequences}) != len(sequences):
        raise PipelineError("Повтор видео в индексе телефонного набора")
    return included, {
        "input_dir": str(directory),
        "source_manifest_sha256": sha256(manifest),
        "included": len(included),
        "excluded": len(sequences) - len(included),
        "decisions": decisions,
        "annotation_coverage": [
            {
                "video": s.key,
                **s.provenance["phone_testset"].get("annotation_coverage", {}),
                "failures": s.provenance["phone_testset"].get("annotation_failures", []),
            }
            for s in included
            if "annotation_coverage" in s.provenance["phone_testset"]
        ],
        "annotation_sources": [
            {
                "video": s.key,
                "source": s.provenance.get("annotation_source", "unspecified"),
                "kind": s.provenance.get("annotation_kind", "unspecified"),
                "manual_mask_verification": s.provenance.get("manual_mask_verification"),
                "target": s.provenance.get("target", "unspecified"),
            }
            for s in included
        ],
    }
