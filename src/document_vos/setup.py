"""Explicit dependency acquisition; normal workflows never download weights."""

import json
import subprocess
import tempfile
from pathlib import Path

from document_vos.assets import fetch
from document_vos.config import resource
from document_vos.cutie import CUTIE_REVISION, check_checkout
from document_vos.io import PipelineError, sha256, versions, write_json
from document_vos.locking import exclusive_lock


def models(catalog: Path | None = None) -> dict:
    return json.loads(catalog.read_text() if catalog else resource("models.json").read_text())


def setup(cfg: dict, *, sam: bool = False, catalog: Path | None = None) -> dict:
    entries = models(catalog)["models"]
    from document_vos.assets import check_asset

    if set(entries) != {"baseline", "finetuned"}:
        raise PipelineError("Каталог моделей должен содержать baseline и finetuned")
    for entry in entries.values():
        check_asset(entry)
        if Path(entry["filename"]).name != entry["filename"]:
            raise PipelineError("Небезопасное имя весов")
    root = Path(cfg["paths"]["cutie"])
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root.parent / ".cutie-setup.lock"):
        if not root.exists():
            with tempfile.TemporaryDirectory(prefix=".cutie-", dir=root.parent) as temp:
                stage = Path(temp) / "Cutie"
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--no-checkout",
                        "https://github.com/hkchengrex/Cutie.git",
                        str(stage),
                    ],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(stage), "checkout", "--detach", CUTIE_REVISION], check=True
                )
                check_checkout(stage)
                stage.rename(root)
        check_checkout(root)
    if sam:
        return setup_sam(cfg)
    weights = Path(cfg["paths"]["weights"])
    downloaded = []
    for model, entry in entries.items():
        path = fetch(entry, weights / entry["filename"])
        downloaded.append({"model": model, "path": str(path), "sha256": sha256(path)})
    import av
    import torch

    return {
        "status": "COMPLETE",
        "cutie_revision": CUTIE_REVISION,
        "models": downloaded,
        "cuda_available": torch.cuda.is_available(),
        "versions": versions(),
        "pyav": av.__version__,
    }


def setup_sam(cfg: dict) -> dict:
    # Public Cutie remains usable without installing the optional SAM dependency.
    entry = json.loads(resource("sam.json").read_text())
    target = Path(cfg["sam"]["weights"])
    if not target.exists():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            subprocess.run(["uv", "sync", "--locked", "--extra", "sam"], check=True)
            from huggingface_hub import hf_hub_download
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(
            hf_hub_download(entry["repository"], entry["filename"], revision=entry["revision"])
        )
        import shutil

        if sha256(source) != entry["sha256"]:
            raise PipelineError("Веса SAM из Hugging Face отличаются от закреплённых")
        partial = target.with_name(target.name + ".part")
        with partial.open("xb") as dest, source.open("rb") as src:
            shutil.copyfileobj(src, dest)
        partial.rename(target)
    receipt = target.with_suffix(".receipt.json")
    actual = sha256(target)
    if actual != entry["sha256"] or target.stat().st_size != entry["bytes"]:
        raise PipelineError("Веса SAM отличаются от закреплённого checkpoint")
    if receipt.exists():
        from document_vos.io import read_json

        if read_json(receipt)["sha256"] != actual:
            raise PipelineError("Веса SAM изменились")
    else:
        write_json(
            receipt,
            {
                "sha256": actual,
                "source": "facebook/sam3",
                "revision": entry["revision"],
                "annotation_kind": "unverified geometric pseudo-labels",
            },
        )
    return {
        "status": "COMPLETE",
        "weights": str(target),
        "sha256": actual,
        "next": "uv run --extra sam document-vos prepare videos --sam",
    }
