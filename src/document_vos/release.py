"""Build a local, unpublished Release bundle from explicitly supplied existing evidence."""

import argparse
import copy
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from document_vos.config import DATASETS, DEFAULTS, resource
from document_vos.data import first_mask, load_manifest
from document_vos.data.recipe import identity, source_identity
from document_vos.data.split import assert_disjoint
from document_vos.io import (
    PipelineError,
    digest,
    new_directory,
    read_json,
    sha256,
    write_json,
    write_jsonl,
)

BASE_URL = "https://github.com/Misha-m-st/misis2026f-24-cv-4-makarov-m-a/releases/download/document-vos-0.1"
SOURCES = {
    "smartdoc2015": {
        "url": "https://github.com/jchazalon/smartdoc15-ch1-dataset",
        "attribution": "Burie, Chazalon, Coustaty, Eskenazi, Luqman, Mehri, Nayef, Ogier, Prum, Rusinol. ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc), 2015.",
        "terms": "CC BY 4.0; original frames, first GT quad rasterized to binary mask",
    },
    "midv2020": {
        "url": "ftp://smartengines.com/midv-2020/",
        "attribution": "Bulatov et al. MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis, 2021.",
        "terms": "CC BY-SA 2.5 Generic, https://creativecommons.org/licenses/by-sa/2.5/. Face images: Generated Photos, https://generated.photos/.",
        "terms_source": "https://docs.google.com/forms/d/e/1FAIpQLSdxB1gvdVlRcARUlMolTJzyqY93XBZHhwiBwkDx8BDyMIPWIg/viewform",
    },
    "midv500": {
        "url": "ftp://smartengines.com/midv-500/",
        "attribution": "Bulatov et al. MIDV-500: A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream, 2018.",
        "terms": "Original document image notices apply; see origins in dataset. No project relicensing.",
    },
    "midv2019": {
        "url": "ftp://smartengines.com/midv-500/extra/midv-2019/",
        "attribution": "Bulatov et al. MIDV-2019: Challenges of the modern mobile-based document OCR, 2019.",
        "terms": "Original MIDV-500 document image notices apply; no project relicensing.",
    },
    "midvholo": {
        "url": "https://github.com/SmartEngines/midv-holo",
        "attribution": "Smart Engines, MIDV-Holo. Face images: Generated Photos, https://generated.photos/.",
        "terms": "CC BY-SA 2.5 Generic, https://creativecommons.org/licenses/by-sa/2.5/.",
    },
    "dlc2021": {
        "url": "ftp://smartengines.com/dlc-2021/",
        "attribution": "Polevoy, Sigareva, Ershova, Arlazarov, Nikolaev, Ming, Luqman, Burie. Document Liveness Challenge Dataset (DLC-2021), J. Imaging 8(7):181, 2022. Face images: Generated Photos, https://generated.photos/.",
        "terms": "CC BY-SA 2.5 Generic, https://creativecommons.org/licenses/by-sa/2.5/.",
        "terms_source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9323793/",
    },
}


def sample_asset(seq, output: Path, name: str, recipe: dict) -> dict:
    """Package original frames and exactly one GT mask, retaining dataset admission metadata."""
    dataset = seq.dataset
    stage = output / ".staging" / name
    (stage / "frames").mkdir(parents=True)
    row = seq.to_dict()
    row["provenance"] = {
        "validation_sample": {"id": name, "recipe_sha256": digest(recipe)},
        "annotation_kind": "original dataset GT; first mask only",
        "presentation": seq.provenance.get("presentation"),
        "source": SOURCES[dataset],
    }
    row["frames"] = []
    checksums = {}
    for n, f in enumerate(seq.frames):
        source = Path(f.path)
        relative = f"frames/{n:06d}{source.suffix.lower()}"
        shutil.copyfile(source, stage / relative)
        checksums[relative] = sha256(stage / relative)
        row["frames"].append(
            {
                "id": f.id,
                "path": relative,
                "time_s": f.time_s,
                "polygon": None,
                "mask": "initial.png" if n == 0 else None,
            }
        )
    init, mask = first_mask(seq)
    if init != 0:
        raise PipelineError(f"Пример не имеет непустой первой GT-маски: {seq.key}")
    Image.fromarray(mask.astype(np.uint8)).save(stage / "initial.png")
    write_jsonl(stage / "manifest.jsonl", [row])
    write_json(
        stage / "source.json",
        {
            "sequence": seq.key,
            "split": "validation",
            "frame_sha256": checksums,
            "first_mask_sha256": sha256(stage / "initial.png"),
            "time_source": seq.time_source,
            "source": SOURCES[dataset],
            "original_pixels_preserved": True,
            "recipe_sha256": digest(recipe),
        },
    )
    (stage / "NOTICE.txt").write_text(
        json.dumps(SOURCES[dataset], ensure_ascii=False, indent=2)
        + "\nValidation subset, not independent evaluation.\n",
        encoding="utf-8",
    )
    if dataset == "smartdoc2015":
        root = Path(seq.frames[0].path)
        while root.parent != root and not (root / "LICENCE").is_file():
            root = root.parent
        if (root / "LICENCE").is_file():
            shutil.copyfile(root / "LICENCE", stage / "ORIGINAL-LICENCE.txt")
    elif dataset == "midvholo":
        # Retain per-document origin notices, without extra images/annotations.
        source_root = Path(seq.frames[0].path)
        for parent in source_root.parents:
            if (parent / "origins").is_dir():
                for origin in (parent / "origins").rglob("*"):
                    if origin.is_file() and origin.suffix.lower() in {
                        ".lst",
                        ".txt",
                        ".json",
                    }:
                        dest = stage / "origins" / origin.relative_to(parent / "origins")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(origin, dest)
                break
    unpacked = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    archive = output / f"{name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipout:
        for file in sorted(stage.rglob("*")):
            if file.is_file():
                info = zipfile.ZipInfo(
                    file.relative_to(stage).as_posix(), date_time=(2026, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                zipout.writestr(info, file.read_bytes())
    entry = {
        "id": name,
        "dataset": dataset,
        "sequence": seq.key,
        "url": f"{BASE_URL}/{archive.name}",
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
        "unpacked_bytes": unpacked,
        "frames": len(seq.frames),
        "source_frame_bytes": sum(Path(f.path).stat().st_size for f in seq.frames),
        "source": SOURCES[dataset],
    }
    shutil.rmtree(stage)
    return entry


def build(
    split: Path,
    metadata: Path,
    training: Path,
    baseline: Path,
    output: Path,
    *,
    write_resources: bool = False,
) -> dict:
    labels = {label: load_manifest(split / f"{label}.jsonl") for label in ["train", "validation"]}
    split_meta = read_json(split / "split.json")
    for label in labels:
        if sha256(split / f"{label}.jsonl") != split_meta[f"{label}_sha256"]:
            raise PipelineError("Исходное разбиение изменилось")
    assert_disjoint(labels["train"], labels["validation"])
    meta = read_json(metadata)
    counts = {k: len(v) for k, v in labels.items()}
    total_frames = sum(len(s.frames) for rows in labels.values() for s in rows)
    if (
        counts != {"train": 2836, "validation": 717}
        or total_frames != 200106
        or len(meta["exclusions"]) != 21
    ):
        raise PipelineError("Состав не соответствует опубликованному рецепту")
    plan = read_json(training / "plan.json")
    result = read_json(training / "result.json")
    checkpoint = training / "official" / "documents_main_training_last.pth"
    if (
        result["status"] != "COMPLETE"
        or result["plan_sha256"] != sha256(training / "plan.json")
        or sha256(checkpoint) not in result["saved_weights"].values()
        or plan["split_sha256"] != sha256(split / "split.json")
        or plan["weights_sha256"] != sha256(baseline)
    ):
        raise PipelineError("Не подтверждён завершённый training checkpoint")
    config = plan["config"]
    stage = config["main_training"]
    if (
        config["seed"] != 42
        or config["pre_training"]["enabled"]
        or stage["num_objects"] != 1
        or stage["merge_probability"] != 0
        or any(stage[k] != v for k, v in DEFAULTS["training"].items())
    ):
        raise PipelineError("Training-run отличается от публикуемого рецепта")
    output = new_directory(output)
    assignments = [
        {"key": s.key, "split": label, "sha256": identity(s), "source_sha256": source_identity(s)}
        for label, rows in labels.items()
        for s in rows
    ]
    exclusions = [{k: v for k, v in r.items() if k != "annotation"} for r in meta["exclusions"]]
    archives = []
    for receipt_name in meta.get("archive_receipts", {}):
        receipt = read_json(Path(receipt_name))
        archives.append(
            {
                k: receipt[k]
                for k in ["archive", "archive_sha256", "author_checksum"]
                if k in receipt
            }
        )
    from document_vos.data.download import dataset_archives

    author_sources = [
        {
            "dataset": name,
            "url": a.url,
            "bytes": a.size,
            "algorithm": a.algorithm,
            "checksum": a.checksum,
        }
        for name in DATASETS
        for a in dataset_archives(name)
    ]
    recipe = {
        "schema": "document-vos/recipe-v1",
        "counts": counts,
        "frames": total_frames,
        "seed": 42,
        "exclusions": exclusions,
        "sequences": sorted(assignments, key=lambda r: r["key"]),
        "sources": author_sources,
        "local_source_archives": archives,
    }
    write_json(output / "recipe.json", recipe)
    resolved = copy.deepcopy(plan["config"])
    resolved["weights"] = "${paths.weights}/cutie-base-mega.pth"
    for key in ["image_datasets", "vos_datasets"]:
        resolved["data"][key]["base"] = "${paths.prepared}/training-export"
    if "hydra" in resolved:
        resolved["hydra"]["run"]["dir"] = "${paths.results}/checkpoints/official"
    (output / "resolved-training.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    samples = []
    for dataset in DATASETS:
        choices = [s for s in labels["validation"] if s.dataset == dataset]
        if dataset == "smartdoc2015":
            fixed = ["background01/patent001", "background02/patent002"]
            choices = [next(s for s in choices if s.video == key) for key in fixed]
        else:
            choices.sort(key=lambda s: (sum(Path(f.path).stat().st_size for f in s.frames), s.key))
        for index, seq in enumerate(choices[:2], 1):
            name = f"{dataset}-{index:02d}"
            samples.append(sample_asset(seq, output, name, recipe))
    shutil.rmtree(output / ".staging")
    cat = {"schema": "document-vos/samples-v1", "release_status": "UNPUBLISHED", "samples": samples}
    write_json(output / "samples.json", cat)
    model_cat = {"schema": "document-vos/models-v1", "release_status": "UNPUBLISHED", "models": {}}
    for name, src, filename, url in [
        (
            "baseline",
            baseline,
            "cutie-base-mega.pth",
            "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
        ),
        ("finetuned", checkpoint, "cutie-document.pth", f"{BASE_URL}/cutie-document.pth"),
    ]:
        shutil.copyfile(src, output / filename)
        model_cat["models"][name] = {
            "filename": filename,
            "url": url,
            "bytes": src.stat().st_size,
            "sha256": sha256(src),
        }
    write_json(output / "models.json", model_cat)
    for name in ["MODEL_CARD.md", "THIRD_PARTY.md"]:
        (output / name).write_text(resource(name).read_text(encoding="utf-8"), encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file()),
        encoding="utf-8",
    )
    if write_resources:
        for name in ["recipe.json", "samples.json", "models.json", "resolved-training.yaml"]:
            shutil.copyfile(output / name, Path(__file__).parent / "resources" / name)
    return {
        "status": "PREPARED_NOT_PUBLISHED",
        "output": str(output),
        "samples": len(samples),
        "sample_bytes": sum(e["bytes"] for e in samples),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["split", "metadata", "training", "baseline", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--write-resources", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
