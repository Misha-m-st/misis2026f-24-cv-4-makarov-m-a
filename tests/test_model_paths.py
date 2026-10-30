"""Explicit model locations, canonical hash gates and independent checkout roots."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_vos.config import load, resource
from document_vos.io import PipelineError, sha256
from document_vos.workflows import weight


def test_model_paths_resolve_from_yaml_and_explicit_weights_take_priority(tmp_path):
    folder = tmp_path / "config"
    folder.mkdir()
    settings = folder / "config.yaml"
    settings.write_text(
        "paths:\n  cutie: ../vendor/Cutie\n  external_models: ../cache/models\n"
        "benchmark:\n  models:\n    finetuned:\n      weights: ../training/final.pth\n"
    )
    cfg = load(settings)
    checkpoint = tmp_path / "training/final.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"explicit local checkpoint")
    assert cfg["paths"]["cutie"] == str(tmp_path / "vendor/Cutie")
    assert cfg["paths"]["external_models"] == str(tmp_path / "cache/models")
    assert not (folder / "weights/cutie-document.pth").exists()
    assert weight(cfg, "finetuned") == checkpoint
    override = tmp_path / "override.pth"
    override.write_bytes(b"CLI checkpoint")
    assert weight(cfg, "finetuned", override) == override
    checkpoint.unlink()
    with pytest.raises(PipelineError) as caught:
        weight(cfg, "finetuned")
    assert str(checkpoint) in str(caught.value)
    assert "benchmark.models.finetuned.weights" in str(caught.value)


def test_default_weight_resolution_keeps_catalog_hash_gate(tmp_path, monkeypatch):
    from document_vos import setup

    settings = tmp_path / "config.yaml"
    settings.write_text("{}\n")
    cfg = load(settings)
    checkpoint = tmp_path / "weights/cutie-document.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"catalog checkpoint")
    expected = sha256(checkpoint)
    # Freeze the catalogue independently of subsequent on-disk mutations.
    monkeypatch.setattr(
        setup,
        "models",
        lambda: {"models": {"finetuned": {"filename": checkpoint.name, "sha256": expected}}},
    )
    assert weight(cfg, "finetuned") == checkpoint
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(PipelineError, match="отличаются от каталога"):
        weight(cfg, "finetuned")
    checkpoint.unlink()
    with pytest.raises(PipelineError) as caught:
        weight(cfg, "finetuned")
    assert str(checkpoint) in str(caught.value)
    assert "benchmark.models.finetuned.weights" in str(caught.value)


def test_external_setup_and_adapter_do_not_depend_on_cutie_location(tmp_path, monkeypatch):
    from document_vos import setup
    from document_vos.benchmark import ExternalPredictor

    settings = tmp_path / "config.yaml"
    settings.write_text("paths:\n  cutie: arbitrary/Cutie\n  external_models: external-cache\n")
    cfg = load(settings)
    root = tmp_path / "external-cache/efficienttam"
    root.mkdir(parents=True)
    revision = json.loads(resource("external_models.json").read_text())["efficienttam"]["revision"]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=revision + "\n")

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup, "_fetch_unpinned", lambda *a, **kw: {"sha256": "stub"})
    setup.setup_external(cfg, "efficienttam")
    assert calls[0] == ["git", "-C", str(root), "rev-parse", "HEAD"]
    assert str(root) in calls[1]
    monkeypatch.setattr(ExternalPredictor, "_load", lambda self: None)
    adapter = ExternalPredictor("efficienttam", cfg, Path("unused.pt"), "cpu")
    assert adapter.repo_root == root


def test_missing_cutie_error_identifies_the_config_field(tmp_path):
    from document_vos.cutie import check_checkout

    missing = tmp_path / "missing/Cutie"
    with pytest.raises(PipelineError) as caught:
        check_checkout(missing)
    assert str(missing) in str(caught.value)
    assert "paths.cutie" in str(caught.value)
