"""Official loader diagnostics."""

import os
from pathlib import Path

from document_vos.io import PipelineError, read_json, sha256, write_json


def probe_export(root: Path, export: Path, output: Path, crop_size: int = 480) -> dict:
    """Read one real augmented official training sample from every source dataset."""
    from document_vos.cutie import activate_checkout, check_checkout

    # The pinned dataset reads LOCAL_RANK even outside torch.distributed.
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    activate_checkout(check_checkout(root))
    from cutie.dataset.vos_dataset import VOSMergeTrainDataset

    metadata = read_json(export / "export.json")
    dataset_config = {
        "DOCUMENTS": {
            "im_root": str(export / "images"),
            "gt_root": str(export / "masks"),
            "max_skip": 5,
            "subset": None,
            "empty_masks": None,
            "multiplier": 1,
        }
    }
    loader = VOSMergeTrainDataset(
        dataset_config,
        seq_length=metadata["seq_length"],
        max_num_obj=1,
        size=crop_size,
        merge_probability=0,
    )
    video_dataset = {
        r["export_video"]: r["source_video"].split("/", 1)[0] for r in metadata["sequences"]
    }
    examples = {}
    for index, (_, video, _) in enumerate(loader.video_frames):
        dataset = video_dataset[video]
        if dataset not in examples:
            sample = loader[index]
            if sample["info"]["num_objects"] != 1:
                raise PipelineError(f"Официальный loader не нашёл один объект: {dataset}")
            examples[dataset] = {
                "video": video,
                "shape": list(sample["rgb"].shape),
                "num_objects": sample["info"]["num_objects"],
            }
        if len(examples) == len(set(video_dataset.values())):
            break
    result = {
        "status": "COMPLETE",
        "datasets": examples,
        "export_sha256": sha256(export / "export.json"),
    }
    write_json(output, result)
    return result
