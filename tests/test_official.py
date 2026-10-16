"""Узкие CPU-проверки официального формата; никаких настоящих весов и обучения."""

import os
from pathlib import Path

import pytest

from document_vos.cutie import (
    activate_checkout,
    check_checkout,
    disable_downloads_and_require_full_weights,
    inference_config,
)
from document_vos.cutie.training import train
from document_vos.data.export import export_training
from document_vos.data.split import split_manifest


@pytest.fixture
def official(monkeypatch):
    root = os.environ.get("CUTIE_ROOT")
    if not root:
        pytest.skip("NOT_RUN: задайте CUTIE_ROOT для проверки официальным loader")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    root = check_checkout(Path(root))
    activate_checkout(root)
    monkeypatch.setenv("LOCAL_RANK", "0")
    return root


@pytest.mark.official
def test_official_vos_merge_dataset(official, manifest, tmp_path):
    from cutie.dataset.vos_dataset import VOSMergeTrainDataset

    split = tmp_path / "split"
    split_manifest(manifest, split)
    exported = tmp_path / "training"
    export_training(split / "train.jsonl", split, exported)
    cfg = {
        "DOCUMENTS": {
            "im_root": str(exported / "images"),
            "gt_root": str(exported / "masks"),
            "max_skip": 5,
            "subset": None,
            "empty_masks": None,
            "multiplier": 1,
        }
    }
    data = VOSMergeTrainDataset(cfg, seq_length=3, max_num_obj=1, size=384, merge_probability=0)
    sample = data[0]
    assert tuple(sample["rgb"].shape) == (3, 3, 384, 384)
    assert sample["info"]["num_objects"] == 1


@pytest.mark.official
def test_official_no_download_and_strict_checkpoint(official, monkeypatch):
    import socket

    from cutie.model.cutie import CUTIE
    from omegaconf import OmegaConf

    def forbidden(*args, **kwargs):
        raise AssertionError("Network must not be used")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    disable_downloads_and_require_full_weights()
    cfg = OmegaConf.create(inference_config(official, {}))
    model = CUTIE(cfg)
    state = model.state_dict()
    model.load_weights(dict(state))
    single = CUTIE(cfg, single_object=True)
    single.load_weights(dict(state))
    for key in ["mask_encoder.conv1.weight", "pixel_fuser.sensory_compress.weight"]:
        assert single.state_dict()[key].shape[1] == state[key].shape[1] - 1
        assert single.state_dict()[key].equal(state[key][:, :-1])
    # Inference использует одну общую multi-object архитектуру и нулевые other-каналы.
    model.load_weights(dict(single.state_dict()), init_as_zero_if_needed=True)
    for key in ["mask_encoder.conv1.weight", "pixel_fuser.sensory_compress.weight"]:
        assert model.state_dict()[key][:, -1].count_nonzero() == 0
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_weights(state)


@pytest.mark.official
def test_official_training_hydra_preflight(official, manifest, tmp_path):
    split = tmp_path / "split"
    split_manifest(manifest, split)
    exported = tmp_path / "training"
    export_training(split / "train.jsonl", split, exported)
    weights = tmp_path / "placeholder.pth"
    weights.write_bytes(b"Only used for Hydra config proof. No training or weight loading.")
    result = train(
        official,
        weights,
        exported,
        tmp_path / "plan",
        {"seq_length": 3, "num_ref_frames": 2},
        execute=False,
    )
    assert result["status"] == "PLANNED" and result["hydra"] == "VERIFIED"
    assert (tmp_path / "plan/resolved.yaml").is_file()


@pytest.mark.official
def test_official_loader_zero_workers_bridge(official, monkeypatch):
    import torch
    from cutie.dataset import setup_training_data

    from document_vos.runtime import Runtime, install_loader_options

    original = setup_training_data.DataLoader
    monkeypatch.setattr(setup_training_data, "DataLoader", original)
    monkeypatch.setattr(
        torch.utils.data.distributed,
        "DistributedSampler",
        lambda dataset, **kw: torch.utils.data.SequentialSampler(dataset),
    )
    install_loader_options(Runtime(dataloader_workers=0))
    dataset = torch.utils.data.TensorDataset(torch.arange(4))
    _, loader = setup_training_data.construct_loader(dataset, 2, 0, 0)
    assert loader.num_workers == 0 and not loader.persistent_workers
    assert list(loader)[0][0].tolist() == [0, 1]
