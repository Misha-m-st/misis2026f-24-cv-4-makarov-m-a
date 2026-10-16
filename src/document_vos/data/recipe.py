"""Portable assignment of the published recipe, independent of previous experiments."""

import json
from pathlib import Path

from document_vos.config import resource
from document_vos.data import load_manifest, save_manifest
from document_vos.io import PipelineError, digest, new_directory, sha256, write_json
from document_vos.data.split import assert_disjoint


def identity(seq) -> str:
    # Source paths are intentionally excluded: coordinates, order and identities remain pinned.
    return digest(
        {
            "key": seq.key,
            "document": seq.document,
            "template": seq.template,
            "group": seq.group,
            "time_source": seq.time_source,
            "frames": [
                {
                    "id": f.id,
                    "time_s": f.time_s,
                    "polygon": f.polygon,
                    "mask": Path(f.mask).name if f.mask else None,
                }
                for f in seq.frames
            ],
        }
    )


def source_identity(seq) -> str:
    return digest(
        [
            {
                "id": f.id,
                "sha256": sha256(Path(f.path)),
                "mask_sha256": sha256(Path(f.mask)) if f.mask else None,
            }
            for f in seq.frames
        ]
    )


def published() -> dict:
    return json.loads(resource("recipe.json").read_text())


def split(manifest: Path, output: Path) -> dict:
    recipe = published()
    sequences = load_manifest(manifest)
    by_key = {s.key: s for s in sequences}
    expected = {r["key"]: r for r in recipe["sequences"]}
    if len(by_key) != len(sequences) or by_key.keys() != expected.keys():
        raise PipelineError("Состав данных отличается от опубликованного рецепта")
    for key, seq in by_key.items():
        if (
            identity(seq) != expected[key]["sha256"]
            or source_identity(seq) != expected[key]["source_sha256"]
        ):
            raise PipelineError(f"Изменились аннотации/порядок/связи: {key}")
    assigned = {
        label: sorted(
            [s for s in sequences if expected[s.key]["split"] == label], key=lambda s: s.key
        )
        for label in ["train", "validation"]
    }
    assert_disjoint(assigned["train"], assigned["validation"])
    output = new_directory(output)
    for label, rows in assigned.items():
        save_manifest(output / f"{label}.jsonl", rows)
    result = {
        "seed": 42,
        "manifest_sha256": sha256(manifest),
        "train_sha256": sha256(output / "train.jsonl"),
        "validation_sha256": sha256(output / "validation.jsonl"),
        "counts": {k: len(v) for k, v in assigned.items()},
        "recipe_sha256": digest(recipe),
        "policy": "published portable assignment",
    }
    write_json(output / "split.json", result)
    return result
