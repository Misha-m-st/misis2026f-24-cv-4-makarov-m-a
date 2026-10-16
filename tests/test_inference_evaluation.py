import pytest

from document_vos.cutie.inference import propagate
from document_vos.data import save_manifest
from document_vos.evaluation import evaluate
from document_vos.evaluation.comparison import compare
from document_vos.io import PipelineError, read_json, write_json


class SpyPredictor:
    def __init__(self):
        self.resets = 0
        self.calls = []
        self.last_model_seconds = 0.01

    def reset(self):
        self.resets += 1
        self.mask = None

    def sync(self):
        pass

    def step(self, image, mask=None, end=False):
        self.calls.append((self.resets, mask is not None, end))
        if mask is not None:
            self.mask = mask.copy()
        return self.mask.copy()

    def memory_stats(self):
        return {"status": "NOT_MEASURED", "peak_allocated_bytes": None, "peak_reserved_bytes": None}


def metadata():
    # Только синтетическая проверка протокола, не исследовательский Cutie run.
    return {
        "inference": {"mem_every": 5},
        "seed": 42,
        "scope": "validation",
        "cutie_revision": "synthetic",
        "device": "cpu",
        "versions": {},
        "backend": "official-cutie",
        "weights_sha256": "baseline",
    }


def test_gt_once_reset_missing_gt_and_late_initialization(tmp_path, sequences):
    sequences[0].frames[0].polygon = [[60, 0], [70, 0], [70, 10], [60, 10]]
    sequences[0].frames[2].polygon = None
    manifest = tmp_path / "late.jsonl"
    save_manifest(manifest, sequences)
    spy = SpyPredictor()
    propagate(manifest, tmp_path / "run", spy, metadata())
    assert spy.resets == 6
    assert sum(has_gt for _, has_gt, _ in spy.calls) == 6
    assert sum(end for _, _, end in spy.calls) == 6
    assert len(spy.calls) == 23
    result = evaluate(tmp_path / "run", tmp_path / "evaluation")
    data = read_json(tmp_path / "evaluation/evaluation.json")
    assert result["evaluated_frames"] == 16
    assert data["selection"]["missing_gt"] == 1
    assert data["summary"]["primary_equal_datasets"]["JF"] == 1
    assert data["videos"][0]["adjacent_pairs"] == 0
    assert (tmp_path / "evaluation/report.md").is_file()
    assert (tmp_path / "evaluation/temporal.png").stat().st_size > 1000


def test_missing_prediction_fails_even_without_gt(tmp_path, manifest):
    propagate(manifest, tmp_path / "run", SpyPredictor(), metadata())
    next((tmp_path / "run/predictions").rglob("*.png")).unlink()
    with pytest.raises(PipelineError, match="отсутствует"):
        evaluate(tmp_path / "run", tmp_path / "eval")


def test_serial_parallel_evaluation_identical(tmp_path, sequences):
    sequences[0].frames[0].polygon = [[60, 0], [70, 0], [70, 10], [60, 10]]
    sequences[0].frames[2].polygon = None
    sequences[2].frames[2].polygon = [[5, 3], [32, 3], [32, 26], [5, 26]]
    sequences[3].dataset = "midv2019"
    manifest = tmp_path / "input.jsonl"
    save_manifest(manifest, sequences)
    run = tmp_path / "run"
    propagate(manifest, run, SpyPredictor(), metadata())
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    for workers, output in zip([1, 4], outputs):
        evaluate(run, output, workers=workers)
    assert read_json(outputs[0] / "evaluation.json") == read_json(outputs[1] / "evaluation.json")


def test_threaded_inference_preserves_order_masks_and_signature(tmp_path, manifest, monkeypatch):
    import document_vos.runtime as runtime

    outputs = [tmp_path / "serial", tmp_path / "threaded"]
    for threads, output in zip([1, 4], outputs):
        monkeypatch.setattr(runtime, "_current", runtime.Runtime(io_threads=threads, queue_size=2))
        propagate(manifest, output, SpyPredictor(), metadata())
    assert (
        read_json(outputs[0] / "run.json")["signature"]
        == read_json(outputs[1] / "run.json")["signature"]
    )
    assert (outputs[0] / "predictions.jsonl").read_bytes() == (
        outputs[1] / "predictions.jsonl"
    ).read_bytes()


def test_comparison_pairing_and_incompatible_configs(tmp_path, manifest):
    a = metadata()
    b = {
        **metadata(),
        "weights_sha256": "finetuned",
        "training": {"initial_weights_sha256": "baseline"},
    }
    for label, meta in [("a", a), ("b", b)]:
        propagate(manifest, tmp_path / label, SpyPredictor(), meta)
        evaluate(tmp_path / label, tmp_path / (label + "-eval"))
    left, right = [tmp_path / (label + "-eval") / "evaluation.json" for label in ["a", "b"]]
    result = compare(left, right, tmp_path / "comparison")
    assert result["delta"]["JF"] == 0
    different = {**b, "inference": {"mem_every": 10}}
    propagate(manifest, tmp_path / "different", SpyPredictor(), different)
    evaluate(tmp_path / "different", tmp_path / "different-eval")
    with pytest.raises(PipelineError, match="Несовместимые"):
        compare(left, tmp_path / "different-eval/evaluation.json", tmp_path / "rejected")
    incomplete = read_json(right)
    incomplete["frames"].pop()
    write_json(tmp_path / "incomplete.json", incomplete)
    with pytest.raises(PipelineError, match="неполна"):
        compare(left, tmp_path / "incomplete.json", tmp_path / "incomplete-comparison")
    next((tmp_path / "b/predictions").rglob("*.png")).unlink()
    with pytest.raises(PipelineError, match="отсутствует"):
        compare(left, right, tmp_path / "deleted-comparison")
