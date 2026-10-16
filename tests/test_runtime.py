import sys
import types

import pytest

from document_vos.io import PipelineError
from document_vos.runtime import Runtime, install_loader_options


@pytest.mark.parametrize(
    "values",
    [
        {"workers": 0},
        {"workers": True},
        {"queue_size": -1},
        {"pin_memory": "true"},
        {"device": "specific-gpu"},
        {"gpu_model": "anything"},
        {"dataloader_workers": -1},
    ],
)
def test_invalid_runtime(values):
    with pytest.raises(PipelineError):
        Runtime.parse(values)


def test_zero_workers_loader_bridge(monkeypatch):
    calls = []
    setup = types.SimpleNamespace(DataLoader=lambda *a, **kw: calls.append(kw))
    monkeypatch.setitem(
        sys.modules, "cutie.dataset", types.SimpleNamespace(setup_training_data=setup)
    )
    install_loader_options(Runtime(dataloader_workers=0))
    setup.DataLoader([], num_workers=0, persistent_workers=True)
    assert calls == [{"num_workers": 0, "pin_memory": False}]


def test_loader_settings_are_explicit():
    assert Runtime(dataloader_workers=3, prefetch_factor=1, pin_memory=True).loader_options() == {
        "num_workers": 3,
        "prefetch_factor": 1,
        "pin_memory": True,
        "persistent_workers": True,
        "multiprocessing_context": "spawn",
    }


def test_cli_runtime_overrides_legacy_workers_and_cli_overrides_runtime(tmp_path, monkeypatch):
    import document_vos.runtime as runtime_module
    from document_vos.diagnostics import dispatch, parser

    captured = []
    monkeypatch.setattr(runtime_module, "_current", Runtime())
    monkeypatch.setattr(
        "document_vos.cutie.training.train", lambda *a, **kw: captured.append(kw) or {}
    )
    config = tmp_path / "train.yaml"
    config.write_text(
        "cutie_root: cutie\nweights: weights\nexport: export\noutput: output\nworkers: 7\n"
    )
    profile = tmp_path / "runtime.yaml"
    profile.write_text("runtime:\n  dataloader_workers: 0\n  workers: 2\n")
    arguments = ["train", "--config", str(config), "--runtime-config", str(profile)]
    dispatch(parser().parse_args(arguments))
    assert captured[-1]["workers"] == 0
    dispatch(parser().parse_args([*arguments, "--workers", "3"]))
    assert captured[-1]["workers"] == 3
