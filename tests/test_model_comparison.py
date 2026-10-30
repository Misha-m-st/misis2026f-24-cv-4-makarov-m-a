import pytest

from document_vos.benchmark import _compare_evaluations
from document_vos.config import DEFAULTS, load, merge
from document_vos.io import PipelineError, sha256


def evaluation(*, inputs="same", masks=None, frames=None, protocol=None):
    return {
        "run": {
            "signature": {
                "inputs_sha256": inputs,
                "manifest_sha256": "manifest",
                "initializations": masks
                if masks is not None
                else [{"video": "d/v", "index": 0, "mask_sha256": "mask"}],
            }
        },
        "metric_protocol": protocol or {"version": 1},
        "selection": {"datasets": ["d"], "videos": ["d/v"]},
        "frames": frames if frames is not None else [{"video": "d/v", "frame": "0001"}],
    }


def test_multimodel_comparator_requires_same_inputs_masks_protocol_and_frames():
    _compare_evaluations({"baseline": evaluation(), "efficienttam": evaluation()})
    with pytest.raises(PipelineError, match="входные"):
        _compare_evaluations({"baseline": evaluation(), "livos": evaluation(inputs="other")})
    with pytest.raises(PipelineError, match="инициализации"):
        _compare_evaluations({"baseline": evaluation(), "sam31": evaluation(masks=[])})
    with pytest.raises(PipelineError, match="протокол"):
        _compare_evaluations(
            {"baseline": evaluation(), "livos": evaluation(protocol={"version": 2})}
        )
    with pytest.raises(PipelineError, match="кадров"):
        _compare_evaluations({"baseline": evaluation(), "livos": evaluation(frames=[])})


def test_benchmark_config_is_strict_and_paths_resolve_from_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load()
    assert cfg["benchmark"]["selected"] == [
        "baseline",
        "finetuned",
        "efficienttam",
        "livos",
        "sam31",
    ]
    assert cfg["benchmark"]["models"]["sam31"]["weights"].endswith(
        "weights/sam31/sam3.1_multiplex.pt"
    )
    with pytest.raises(PipelineError, match="Неизвестные параметры"):
        merge(DEFAULTS, {"benchmark": {"models": {"sam31": {"magic": True}}}})
    overrides = merge(
        DEFAULTS, {"benchmark": {"models": {"baseline": {"inference": {"mem_every": 2}}}}}
    )
    assert overrides["benchmark"]["models"]["baseline"]["inference"] == {"mem_every": 2}


def test_unpinned_checkpoint_receipt_is_repeatable_and_detects_mutation(tmp_path):
    from document_vos.setup import _fetch_unpinned

    source = tmp_path / "source.bin"
    source.write_bytes(b"publisher did not publish a checksum")
    target = tmp_path / "weights" / "model.pt"
    record = _fetch_unpinned(source.as_uri(), target)
    assert record["sha256"] == sha256(target)
    assert "first-download-hash" in record["verification"]
    assert _fetch_unpinned(source.as_uri(), target) == record
    target.write_bytes(b"changed")
    with pytest.raises(PipelineError, match="Receipt изменился"):
        _fetch_unpinned(source.as_uri(), target)


def test_external_infer_reports_setup_command_without_network(tmp_path, manifest, monkeypatch):
    import urllib.request

    from document_vos.benchmark import inference

    def no_network(*_args, **_kwargs):
        raise AssertionError("infer must not access the network")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    cfg = load()
    cfg["runtime"]["device"] = "cpu"
    cfg["benchmark"]["models"]["efficienttam"]["weights"] = str(tmp_path / "absent.pt")
    with pytest.raises(PipelineError, match="setup --model efficienttam"):
        inference(cfg, "efficienttam", manifest=manifest, output=tmp_path / "unused")


@pytest.mark.parametrize("compile_model", [False, True])
def test_efficienttam_selects_full_vos_implementation(monkeypatch, tmp_path, compile_model):
    import sys
    from copy import deepcopy
    from types import ModuleType

    from document_vos.benchmark import ExternalPredictor

    calls = []
    builder = ModuleType("efficient_track_anything.build_efficienttam")
    builder.build_efficienttam_video_predictor = lambda *args, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, builder.__name__, builder)
    cfg = deepcopy(DEFAULTS)
    cfg["benchmark"]["models"]["efficienttam"]["compile"] = compile_model
    ExternalPredictor("efficienttam", cfg, tmp_path / "weights.pt", "cpu")
    assert calls[0]["vos_optimized"] is compile_model
    assert calls[0]["hydra_overrides_extra"] == [
        f"++model.compile_image_encoder={str(compile_model).lower()}"
    ]
