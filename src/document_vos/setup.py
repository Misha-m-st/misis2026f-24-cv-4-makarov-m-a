"""Explicit dependency acquisition; normal workflows never download weights."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from document_vos.assets import fetch
from document_vos.config import resource
from document_vos.cutie import CUTIE_REVISION, check_checkout
from document_vos.io import PipelineError, sha256, versions, write_json
from document_vos.locking import exclusive_lock


def models(catalog: Path | None = None) -> dict:
    return json.loads(catalog.read_text() if catalog else resource("models.json").read_text())


def setup(
    cfg: dict, *, sam: bool = False, catalog: Path | None = None, model: str | None = None
) -> dict:
    if model in {"efficienttam", "livos", "sam31"}:
        return setup_external(cfg, model)
    entries = models(catalog)["models"]
    from document_vos.assets import check_asset

    if set(entries) != {"baseline", "finetuned"}:
        raise PipelineError("Каталог моделей должен содержать baseline и finetuned")
    if model is not None:
        if model not in entries:
            raise PipelineError(f"Неизвестная модель: {model}")
        entries = {model: entries[model]}
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


def _fetch_unpinned(
    url: str, target: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None
) -> dict:
    """Download under a lock; retain an interrupted .part and record first-seen hash."""
    import urllib.error
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(target.with_name(target.name + ".lock")):
        if not target.exists():
            partial = target.with_name(target.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                response = urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=60
                )
            except (urllib.error.URLError, OSError) as exc:
                raise PipelineError(f"Не удалось получить {url}: {exc}; .part сохранён") from exc
            with response:
                if offset and response.status != 206:
                    raise PipelineError("Источник не поддержал продолжение .part")
                with partial.open("ab" if offset else "xb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise PipelineError(f"Размер checkpoint неверен: {partial}; .part сохранён")
            actual = sha256(partial)
            if expected_sha256 and actual != expected_sha256:
                raise PipelineError(f"SHA-256 checkpoint неверен: {partial}; .part сохранён")
            partial.rename(target)
        actual = sha256(target)
        if expected_sha256 and actual != expected_sha256:
            raise PipelineError(f"SHA-256 checkpoint неверен: {target}")
        receipt = target.with_suffix(target.suffix + ".receipt.json")
        record = {
            "path": str(target),
            "sha256": actual,
            "bytes": target.stat().st_size,
            "source": url,
            "verification": "publisher-pinned"
            if expected_sha256
            else "first-download-hash; publisher checksum unavailable",
        }
        if expected_bytes is not None and record["bytes"] != expected_bytes:
            raise PipelineError(f"Размер checkpoint неверен: {target}")
        if receipt.exists():
            from document_vos.io import read_json

            if read_json(receipt) != record:
                raise PipelineError(f"Receipt изменился: {receipt}")
        else:
            write_json(receipt, record)
        return record


def setup_external(cfg: dict, model: str) -> dict:
    """Install only the selected external code and checkpoint; inference stays offline."""
    from document_vos.config import resource

    catalog = json.loads(resource("external_models.json").read_text(encoding="utf-8"))
    item = catalog[model]
    root = Path(cfg["paths"]["external_models"]) / model
    root.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root.parent / f".{model}-setup.lock"):
        if not root.exists():
            with tempfile.TemporaryDirectory(prefix=f".{model}-", dir=root.parent) as temp:
                stage = Path(temp) / model
                subprocess.run(
                    ["git", "clone", "--no-checkout", item["repository"], str(stage)], check=True
                )
                subprocess.run(
                    ["git", "-C", str(stage), "checkout", "--detach", item["revision"]], check=True
                )
                stage.rename(root)
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != item["revision"]:
            raise PipelineError(f"{model}: исходники отличаются от закреплённой версии")
    # Install selected package into the project's active interpreter without replacing torch.
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "-e", str(root)],
        check=True,
    )
    target = Path(cfg["benchmark"]["models"][model]["weights"])
    if model == "efficienttam":
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "iopath==0.1.10"],
            check=True,
        )
    if model == "sam31":
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "huggingface-hub==0.36.0",
                "iopath==0.1.10",
                "decord==0.6.0",
                "timm==1.0.22",
                "psutil==7.1.2",
                "setuptools==80.9.0",
            ],
            check=True,
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise PipelineError(
                "SAM 3.1: установите optional dependency huggingface-hub в .venv"
            ) from exc
        try:
            source = Path(hf_hub_download(item["hf_repository"], item["checkpoint"]))
        except Exception as exc:
            raise PipelineError(
                "Не удалось скачать SAM 3.1. Запросите доступ к facebook/sam3.1 и выполните hf auth login."
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        with exclusive_lock(target.with_name(target.name + ".setup.lock")):
            if not target.exists():
                partial = target.with_name(target.name + ".part")
                shutil.copyfile(source, partial)
                partial.replace(target)
        source_url = (
            f"https://huggingface.co/{item['hf_repository']}/resolve/main/{item['checkpoint']}"
        )
        record = _fetch_unpinned(
            source_url, target, expected_sha256=item["sha256"], expected_bytes=item["bytes"]
        )
    elif model == "livos":
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "gdown==5.2.0"], check=True
        )
        record = _setup_livos_checkpoint(item, target)
    else:
        record = _fetch_unpinned(
            item["checkpoint_url"],
            target,
            expected_sha256=item.get("checkpoint_sha256"),
            expected_bytes=item.get("checkpoint_bytes"),
        )
    return {
        "status": "COMPLETE",
        "model": model,
        "revision": item["revision"],
        "weights": record,
        "next": f"uv run document-vos infer --model {model} --sample",
    }


def _setup_livos_checkpoint(item: dict, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(target.with_name(target.name + ".setup.lock")):
        partial = target.with_name(target.name + ".part")
        if not target.exists():
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gdown",
                    "--continue",
                    item["checkpoint_url"],
                    "-O",
                    str(partial),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode or not partial.is_file() or partial.stat().st_size < 1024:
                raise PipelineError(f"LiVOS checkpoint не скачан: {proc.stderr[-1000:]}")
            try:
                import torch

                checkpoint = torch.load(partial, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or not checkpoint:
                    raise ValueError("checkpoint должен быть непустым словарём")
            except Exception as exc:
                raise PipelineError(f"LiVOS checkpoint неполон или невалиден: {exc}") from exc
            partial.replace(target)
        actual = sha256(target)
        record = {
            "path": str(target),
            "sha256": actual,
            "bytes": target.stat().st_size,
            "source": item["checkpoint_url"],
            "verification": "first-download-hash; publisher checksum unavailable",
        }
        receipt = target.with_suffix(target.suffix + ".receipt.json")
        if receipt.exists():
            from document_vos.io import read_json

            if read_json(receipt) != record:
                raise PipelineError("LiVOS checkpoint отличается от receipt первого получения")
        else:
            write_json(receipt, record)
        return record


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
