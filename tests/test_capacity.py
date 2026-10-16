import pytest

from document_vos.capacity import candidates, should_continue
from document_vos.io import PipelineError


def test_capacity_descends_only_after_cuda_oom():
    assert should_continue("OOM") is True
    assert should_continue("PASS") is False
    for status in ["FAILED", "TIMEOUT", "CPU_OOM", "INTERRUPTED"]:
        with pytest.raises(PipelineError):
            should_continue(status)


def test_capacity_preserves_official_spatial_temporal_before_lowering():
    cases = candidates()
    assert [case["batch_size"] for case in cases[:5]] == [16, 8, 4, 2, 1]
    assert all(case["crop_size"] == [480, 480] for case in cases[:7])
    assert all(case["seq_length"] == 8 and case["num_ref_frames"] == 3 for case in cases[:5])
    assert all(case["num_ref_frames"] < case["seq_length"] for case in cases)
