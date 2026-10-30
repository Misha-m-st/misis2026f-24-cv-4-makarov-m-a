import csv
import gzip
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from document_vos.cutie import local_weights
from document_vos.cutie.training import training_config, validate_training_config
from document_vos.data import load_manifest
from document_vos.data.adapters import prepare
from document_vos.io import PipelineError, read_json, sha256, write_json, write_jsonl


def image(path, value=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 32, 3), value, np.uint8)).save(path)


QUAD = [[2, 2], [20, 2], [20, 20], [2, 20]]


@pytest.mark.parametrize(
    "dataset",
    ["smartdoc2015", "midv500", "midv2019", "midv2020", "midvlait", "midvholo", "dlc2021", "sidtd"],
)
def test_eight_dataset_published_layouts(tmp_path, dataset):
    root = tmp_path / dataset
    root.mkdir()
    spec = {"root": str(root)}
    if dataset == "smartdoc2015":
        image(root / "background01/letter001/frame_0001.jpeg")
        row = {
            "bg_name": "background01",
            "model_name": "letter001",
            "modeltype_name": "letter",
            "frame_index": 1,
            "image_path": "background01/letter001/frame_0001.jpeg",
        }
        for corner, p in zip(["tl", "bl", "br", "tr"], [QUAD[0], QUAD[3], QUAD[2], QUAD[1]]):
            row.update({corner + "_x": p[0], corner + "_y": p[1]})
        with gzip.open(root / "metadata.csv.gz", "wt") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    elif dataset in {"midv500", "midv2019", "midvlait"}:
        if dataset == "midvlait":
            relative = "afg_id_00/afg_id_00_images/000001.jpg"
            gt = "afg_id_00/afg_id_00_groundtruth/000001.jpg.json"
        else:
            relative = "01_alb_id/images/CA/CA01_01.tif"
            gt = "01_alb_id/ground_truth/CA/CA01_01.json"
        image(root / relative)
        write_json(root / gt, {"quad": QUAD})
    elif dataset in {"midv2020", "dlc2021"}:
        clips = (
            ["00"]
            if dataset == "midv2020"
            else ["00.or0001", "00.cc0001", "00.cg0001", "00.re0001"]
        )
        for n, clip in enumerate(clips):
            image(root / f"images/alb_id/{clip}/000001.jpg", 32 + n)
            write_json(
                root / f"annotations/alb_id/{clip}.json",
                {
                    "_via_img_metadata": {
                        "k": {
                            "filename": "000001.jpg",
                            "regions": [
                                {
                                    "shape_attributes": {
                                        "name": "polygon",
                                        "all_points_x": [p[0] for p in QUAD],
                                        "all_points_y": [p[1] for p in QUAD],
                                    }
                                }
                            ],
                        }
                    }
                },
            )
    elif dataset == "midvholo":
        image(root / "images/origins/passport/psp05_01_02/000001.jpg")
        write_json(root / "markup/origins/passport/psp05_01_02/000001.json", {"quad": QUAD})
        write_json(root / "metadata/A.json", {"iphone12": ["origins/passport/psp05_01_02"]})
    else:
        image(root / "frames/000001.jpg")
        row = {
            "video": "physical-new",
            "document": "sidtd:new",
            "template": "midv-template:alb_id",
            "group": "related:alb_id",
            "provenance": {
                "presentation": "physical",
                "origin": "new capture",
                "source_dataset": "sidtd",
                "annotation_source": "manual verified quad",
            },
            "frames": [{"id": "1", "path": "frames/000001.jpg", "polygon": QUAD}],
        }
        write_jsonl(
            root / "index.jsonl",
            [
                row,
                {
                    **row,
                    "video": "duplicate",
                    "provenance": {**row["provenance"], "source_dataset": "midv2020"},
                },
            ],
        )
        spec["index"] = "index.jsonl"
    output = tmp_path / "prepared.jsonl"
    result = prepare({"selected": [dataset], "datasets": {dataset: spec}}, tmp_path, output)
    assert result["videos"] == (3 if dataset == "dlc2021" else 1)
    seq = load_manifest(output)
    assert seq[0].frames[0].time_s is None
    if dataset == "dlc2021":
        assert {s.provenance["presentation"] for s in seq} == {
            "original",
            "gray_copy",
            "color_copy",
        }
        assert len({s.document for s in seq}) == 1


def test_selected_missing_required_and_optional_not_blocking(tmp_path):
    with pytest.raises(PipelineError, match="каталог не найден"):
        prepare(
            {"selected": ["midv2020"], "datasets": {"midv2020": {"root": "missing"}}},
            tmp_path,
            tmp_path / "m.jsonl",
        )


def test_real_holo_nested_templates_and_related_presentations(tmp_path):
    root = tmp_path / "holo"
    for presentation in ["origins", "fraud/photo_holo_copy"]:
        image(root / f"images/{presentation}/passport/psp05_01_02/img_0001.jpg")
        write_json(
            root / f"markup/{presentation}/passport/psp05_01_02/img_0001.jpg.json",
            {"document": {"templates": {"uto.passport.type5:main": {"template_quad": QUAD}}}},
        )
    write_json(root / "metadata/origins.json", {"A": {"iphone": []}})
    output = tmp_path / "holo.jsonl"
    prepare(
        {"selected": ["midvholo"], "datasets": {"midvholo": {"root": str(root)}}}, tmp_path, output
    )
    sequences = load_manifest(output)
    assert len(sequences) == 2
    assert sequences[0].frames[0].polygon == QUAD
    assert sequences[0].document == sequences[1].document
    assert sequences[0].template == sequences[1].template


def test_full_archive_profile_rejects_incomplete_normalization(tmp_path):
    root = tmp_path / "midv500"
    root.mkdir()
    with pytest.raises(PipelineError, match="полного набора не завершена"):
        prepare(
            {
                "selected": ["midv500"],
                "datasets": {"midv500": {"root": str(root), "require_complete_archives": True}},
            },
            tmp_path,
            tmp_path / "all.jsonl",
        )
    assert not (tmp_path / "all.jsonl").exists()


def test_empty_dlc_annotations_require_opt_in_and_are_audited(tmp_path):
    root = tmp_path / "dlc"
    for clip in ["00.or0001", "00.cc0003"]:
        image(root / f"images/alb_id/{clip}/000001.jpg")
    valid = {
        "_via_img_metadata": {
            "frame": {
                "filename": "000001.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [2, 20, 20, 2],
                            "all_points_y": [2, 2, 20, 20],
                        }
                    }
                ],
            }
        }
    }
    write_json(root / "annotations/alb_id/00.or0001.json", valid)
    empty_path = root / "annotations/alb_id/00.cc0003.json"
    write_json(empty_path, {"_via_img_metadata": {}, "_via_image_id_list": []})
    cfg = {"selected": ["dlc2021"], "datasets": {"dlc2021": {"root": str(root)}}}
    with pytest.raises(PipelineError, match="не найдено VIA"):
        prepare(cfg, tmp_path, tmp_path / "strict.jsonl")
    cfg["datasets"]["dlc2021"]["skip_empty_via"] = True
    output = tmp_path / "admitted.jsonl"
    stats = prepare(cfg, tmp_path, output)
    assert stats["videos"] == 1 and stats["excluded_videos"] == 1
    meta = read_json(output.with_suffix(".meta.json"))
    assert meta["exclusions"] == [
        {
            "video": "dlc2021/alb_id/00.cc0003",
            "reason": "empty_published_via_metadata",
            "frames": 1,
            "annotation": str(empty_path),
            "annotation_sha256": sha256(empty_path),
        }
    ]
    assert empty_path.is_file() and (root / "images/alb_id/00.cc0003/000001.jpg").is_file()
    # A populated project with missing document polygons is still an error.
    empty_path.write_text('{"_via_img_metadata":{"x":{"filename":"000001.jpg","regions":[]}}}')
    with pytest.raises(PipelineError, match="ровно одна"):
        prepare(cfg, tmp_path, tmp_path / "invalid.jsonl")


def test_training_guards_and_explicit_weights(tmp_path):
    with pytest.raises(PipelineError, match="Начальные веса"):
        local_weights(tmp_path / "missing.pth")
    # Композиция официального Hydra из исходников без torch/GPU.
    root = tmp_path / "reference"
    (root / "cutie/config").mkdir(parents=True)
    (root / "cutie/config/train_config.yaml").write_text("""
pre_training:
  enabled: true
main_training:
  enabled: true
data: {}
weights: null
checkpoint: null
seed: 1
num_workers: 16
cudnn_benchmark: true
exp_id: default
save_weights_interval: 10000
save_checkpoint_interval: 10000
""")
    weights = tmp_path / "explicit.pth"
    weights.write_bytes(b"synthetic placeholder; never used as a model")
    export = tmp_path / "export"
    cfg = training_config(root, weights, export, tmp_path / "result", {})
    assert cfg["pre_training"]["enabled"] is False
    assert cfg["main_training"]["num_objects"] == 1
    assert cfg["main_training"]["merge_probability"] == 0
    assert cfg["main_training"]["batch_size"] == 2
    assert cfg["main_training"]["seq_length"] == 8
    assert cfg["main_training"]["crop_size"] == [480, 480]
    assert cfg["main_training"]["num_ref_frames"] == 3
    assert cfg["main_training"]["amp"] is True
    assert cfg["weights"] == str(weights)
    assert cfg["num_workers"] == 6
    selected = training_config(root, weights, export, tmp_path / "result", {}, workers=3)
    assert selected["num_workers"] == 3
    for field, value in [("num_objects", 2), ("merge_probability", 0.5), ("enabled", False)]:
        bad = deepcopy(cfg)
        bad["main_training"][field] = value
        with pytest.raises(PipelineError):
            validate_training_config(bad, export, weights)
    with pytest.raises(PipelineError):
        training_config(root, weights, export, tmp_path / "result", {"merge_probability": 0.5})
    with pytest.raises(PipelineError):
        training_config(root, weights, export, tmp_path / "result", {"crop_size": [384, 256]})
