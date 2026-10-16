import os

import pytest

import document_vos.data.export as module
from document_vos.data.split import split_manifest
from document_vos.io import PipelineError


def test_export_cache_reuses_only_unchanged_verified_files(tmp_path, manifest, monkeypatch):
    split = tmp_path / "split"
    split_manifest(manifest, split)
    root = tmp_path / "export"
    expected = module.export_training(split / "train.jsonl", split, root, workers=1)
    original = module._validate_export_frame
    calls = []

    def probe(task):
        calls.append(task)
        return original(task)

    monkeypatch.setattr(module, "_validate_export_frame", probe)
    assert module.validate_export(root, workers=1) == expected
    assert len(calls) == expected["frames"]
    calls.clear()
    assert module.validate_export(root, workers=1) == expected
    assert calls == []
    module.validate_export(root, workers=1, refresh=True)
    assert len(calls) == expected["frames"]
    # Same length and restored mtime must not hide a content change.
    path = next((root / "masks").rglob("*.png"))
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(PipelineError, match="экспорт изменён"):
        module.validate_export(root, workers=1)
