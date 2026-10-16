"""Synthetic geometry/preparation contracts; no model quality or hardware claims."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image

from document_vos.evaluation import evaluate
from document_vos.inference import propagate
from document_vos.input_folder import select_input_folder
from document_vos.io import read_json, read_jsonl, sha256
from tools.phone_testset.geometry import (
    GeometryOptions,
    equivalent_polygons,
    equivalent_quads,
    rasterize_quad,
    recover_quad,
)
from tools.phone_testset.prepare import prepare
from tools.phone_testset.sam import choose_mask, choose_page_mask, limit_masks


@pytest.fixture
def page():
    visible = np.zeros((160, 220), np.uint8)
    quad = np.array([[30, 20], [190, 35], [175, 140], [20, 120]])
    cv2.fillPoly(visible, [quad], 1)
    return visible, quad


@pytest.mark.parametrize("occlusion", ["none", "hole", "corner", "notch"])
def test_planar_geometry_recovers_supported_hidden_region(page, occlusion):
    full, expected = page
    visible = full.copy()
    if occlusion == "hole":
        visible[60:100, 70:100] = 0
    elif occlusion == "corner":
        visible[:55, 145:] = 0
    elif occlusion == "notch":
        visible[55:95, 160:] = 0
    before = visible.copy()
    recovered, evidence = recover_quad(visible)
    assert np.array_equal(before, visible)
    assert set(np.unique(recovered)) == {0, 1}
    iou = np.count_nonzero(recovered & full) / np.count_nonzero(recovered | full)
    assert iou > 0.995
    actual = np.array(evidence["quad"])
    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 1
    assert evidence["target"] == "planar full-page hypothesis"
    assert evidence["extrapolated_pixels"] > 0 if occlusion != "none" else recovered.sum() > 0
    assert min(evidence["side_support"]) >= GeometryOptions().min_side_support
    # Perspective page sides are not forced to be parallel or perpendicular.
    assert abs((actual[1] - actual[0]) @ (actual[2] - actual[1])) > 1


@pytest.mark.parametrize("kind", ["empty", "circle", "triangle", "border", "small"])
def test_unsupported_geometry_is_rejected(kind):
    mask = np.zeros((160, 220), np.uint8)
    if kind == "circle":
        cv2.circle(mask, (100, 80), 50, 1, -1)
    if kind == "triangle":
        cv2.fillPoly(mask, [np.array([[30, 30], [180, 40], [100, 130]])], 1)
    if kind == "border":
        mask[0:100, 0:100] = 1
    if kind == "small":
        mask[50:53, 50:53] = 1
    with pytest.raises(ValueError):
        recover_quad(mask)


def test_soft_high_resolution_page_boundary_keeps_supported_sides():
    mask = np.zeros((1920, 1080), np.float32)
    quad = np.array([[405, 600], [896, 682], [711, 1328], [72, 1162]], np.int32)
    cv2.fillPoly(mask, [quad], 1)
    visible = (cv2.GaussianBlur(mask, (0, 0), 4.0) >= 0.5).astype(np.uint8)
    recovered, evidence = recover_quad(visible)
    iou = np.count_nonzero(recovered & visible) / np.count_nonzero(recovered | visible)
    assert iou > 0.995
    assert min(evidence["side_support"]) >= GeometryOptions().min_side_support
    assert evidence["visible_containment"] >= GeometryOptions().min_visible_containment


@pytest.mark.parametrize("edge", ["right", "left", "top", "bottom"])
def test_partial_exit_preserves_external_corners_and_clips_polygon(edge):
    shape = (240, 320)
    quad = np.array([[30, 35], [345, 70], [250, 210], [45, 195]], dtype=np.float64)
    if edge == "left":
        quad[:, 0] = shape[1] - 1 - quad[:, 0]
    if edge in {"top", "bottom"}:
        quad = quad[:, ::-1]
        shape = shape[::-1]
        if edge == "top":
            quad[:, 1] = shape[0] - 1 - quad[:, 1]
    # Independent reference: draw in a larger canvas, then crop the image.
    reference = np.zeros((shape[0] + 200, shape[1] + 200), np.uint8)
    cv2.fillPoly(reference, [np.rint(quad + 100).astype(np.int32)], 1)
    visible = reference[100 : 100 + shape[0], 100 : 100 + shape[1]]
    mask, evidence = recover_quad(visible)
    assert evidence["out_of_frame"] is True
    polygon = np.array(evidence["clipped_polygon"])
    assert len(polygon) == 5
    iou = np.count_nonzero(mask & visible) / np.count_nonzero(mask | visible)
    assert iou > 0.995
    corners = np.array(evidence["quad"])
    assert (
        (corners < 0).any()
        or (corners[:, 0] > shape[1] - 1).any()
        or (corners[:, 1] > shape[0] - 1).any()
    )
    naive = quad.copy()
    naive[:, 0] = np.clip(naive[:, 0], 0, shape[1] - 1)
    naive[:, 1] = np.clip(naive[:, 1], 0, shape[0] - 1)
    naive_mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(naive_mask, [np.rint(naive).astype(np.int32)], 1)
    assert np.count_nonzero(naive_mask != visible) > 100


def test_entirely_off_image_polygon_has_empty_intersection_not_clamped_corners():
    quad = np.array([[250, 30], [350, 30], [350, 100], [250, 100]], dtype=float)
    mask, polygon = rasterize_quad(quad, (160, 220))
    assert not mask.any() and len(polygon) == 0


def test_equivalence_compares_only_geometry_inside_image():
    a = np.array([[30, 20], [300, 20], [300, 140], [30, 140]], dtype=float)
    b = np.array([[30, 20], [1000, 20], [1000, 140], [30, 140]], dtype=float)
    _, clipped_a = rasterize_quad(a, (160, 220))
    _, clipped_b = rasterize_quad(b, (160, 220))
    assert equivalent_polygons(clipped_a, clipped_b, 2.5)
    assert not equivalent_polygons(clipped_a, clipped_b + 10, 2.5)


def test_expansion_gate_does_not_silently_relax(page):
    mask, _ = page
    mask[60:100, 70:100] = 0
    with pytest.raises(ValueError, match="no quadrilateral"):
        recover_quad(mask, replace(GeometryOptions(), max_area_ratio=1.01))


def test_instance_selection_requires_nonempty_unambiguous_binary_mask(page):
    mask, _ = page
    selected, info = choose_mask(mask[None, None], np.array([0.9]), mask.shape, 0.5, 0.1)
    assert np.array_equal(selected, mask) and info["instance_score"] == 0.9
    for masks, scores in [
        (np.array([mask, mask]), np.array([0.9, 0.88])),
        (mask[None], np.array([np.nan])),
        (mask[None], np.array([0.1])),
        (np.full_like(mask, 255)[None], np.array([0.9])),
        (np.zeros_like(mask)[None], np.array([0.9])),
        (np.empty((0, *mask.shape)), np.array([])),
    ]:
        with pytest.raises(ValueError):
            choose_mask(masks, scores, mask.shape, 0.5, 0.1)


def test_page_selection_uses_largest_mask_not_highest_confidence(page):
    full, _ = page
    half = full.copy()
    half[:80] = 0
    scrap = np.zeros_like(full)
    scrap[:30, :30] = 1
    selected, info = choose_page_mask(
        np.array([full, half, scrap]),
        np.array([0.8, 0.9, 0.88]),
        full.shape,
        0.5,
        0.1,
        GeometryOptions(),
    )
    assert np.array_equal(selected, full)
    assert info["selected_instance"] == 0 and info["selected_area"] == int(full.sum())
    assert info["retained_instances"] == [0] and info["model_instances"] == 3
    assert info["max_masks"] == 1


def test_page_selection_picks_largest_even_for_distinct_documents():
    masks = np.zeros((2, 300, 500), np.uint8)
    masks[0, 40:260, 30:210] = 1
    masks[1, 40:260, 290:450] = 1
    selected, info = choose_page_mask(
        masks, np.array([0.88, 0.9]), (300, 500), 0.5, 0.1, GeometryOptions(), max_masks=2
    )
    assert np.array_equal(selected, masks[0]) and info["selected_instance"] == 0
    assert info["instances"] == 2
    with pytest.raises(ValueError, match="supported"):
        choose_page_mask(
            np.ones((1, 300, 500), np.uint8),
            np.array([0.9]),
            (300, 500),
            0.5,
            0.1,
            GeometryOptions(),
        )


def test_mask_budget_keeps_largest_confident_masks_in_area_order():
    masks = np.zeros((4, 100, 150), np.uint8)
    masks[0, 60:70, 100:110] = 1
    masks[1, 10:50, 10:50] = 1
    masks[2, 60:80, 10:30] = 1
    masks[3, 10:90, 10:90] = 1
    scores = np.array([0.99, 0.6, 0.8, 0.2])
    kept, kept_scores, indices = limit_masks(masks, scores, (100, 150), max_masks=2)
    assert indices.tolist() == [1, 2]
    assert kept.sum(axis=(1, 2)).tolist() == [1600, 400]
    assert kept_scores.tolist() == [0.6, 0.8]
    assert limit_masks(masks, scores, (100, 150))[2].tolist() == [1]
    for count in [0, -1]:
        with pytest.raises(ValueError, match="max_masks"):
            limit_masks(masks, scores, (100, 150), max_masks=count)
    for bad_masks, bad_scores in [(masks * 255, scores), (masks, np.full(4, np.nan))]:
        with pytest.raises(ValueError):
            limit_masks(bad_masks, bad_scores, (100, 150))


def test_equal_area_overlap_is_deterministic_and_uses_confidence_only_for_ties(page):
    masks = np.stack([page[0]] * 3)
    _, scores, indices = limit_masks(masks, np.array([0.7, 0.9, 0.9]), page[0].shape, max_masks=2)
    assert indices.tolist() == [1] and scores.tolist() == [0.9]


def test_smaller_overlapping_masks_are_removed_before_the_budget():
    masks = np.zeros((4, 100, 150), np.uint8)
    masks[0, 10:50, 10:50] = 1
    masks[1, 30:60, 30:60] = 1  # Partial overlap, not containment.
    masks[2, 15:25, 15:25] = 1  # Fully contained in mask 0.
    masks[3, 70:90, 100:120] = 1  # Smaller independent object.
    scores = np.array([0.6, 0.99, 0.9, 0.8])
    for budget in [2, 10]:
        kept, _, indices = limit_masks(masks, scores, (100, 150), max_masks=budget)
        assert indices.tolist() == [0, 3]
        assert not np.any(kept[0] & kept[1])
        assert np.array_equal(kept[0], masks[0])  # No cropping or merging.


def test_one_shared_pixel_removes_smaller_mask_but_adjacent_masks_survive():
    masks = np.zeros((3, 20, 30), np.uint8)
    masks[0, 1:11, 1:11] = 1
    masks[1, 10:13, 10:13] = 1
    masks[2, 1:6, 11:16] = 1  # Adjacent to mask 0, without a shared pixel.
    _, _, indices = limit_masks(masks, np.full(3, 0.9), (20, 30), max_masks=10)
    assert indices.tolist() == [0, 2]


def test_overlap_with_removed_larger_mask_still_removes_smaller_mask():
    masks = np.zeros((3, 30, 50), np.uint8)
    masks[0, 1:21, 1:21] = 1
    masks[1, 5:20, 15:30] = 1
    masks[2, 10:20, 25:35] = 1
    _, _, indices = limit_masks(masks, np.full(3, 0.9), (30, 50), max_masks=10)
    assert indices.tolist() == [0]


def test_largest_mask_geometry_error_does_not_fall_back_to_smaller_page(page):
    page_mask = page[0]
    circle = np.zeros_like(page_mask)
    cv2.circle(circle, (110, 80), 78, 1, -1)
    assert circle.sum() > page_mask.sum()
    with pytest.raises(ValueError):
        choose_page_mask(
            np.stack([page_mask, circle]),
            np.array([0.9, 0.8]),
            page_mask.shape,
            0.5,
            0.1,
            GeometryOptions(),
            max_masks=2,
        )


def test_quad_equivalence_uses_boundary_tolerance_and_corner_order():
    quad = np.array([[30, 20], [190, 35], [175, 140], [20, 120]], dtype=float)
    assert equivalent_quads(quad, np.roll(quad + 1, 2, axis=0), 2.5)
    assert not equivalent_quads(quad, quad + 3, 2.5)


class SyntheticSAM:
    def __init__(self, visible, fail_at=None):
        self.visible, self.fail_at = visible, fail_at
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic segmentation failure")
        return self.visible.copy(), {"instance_score": 0.9, "instances": 1}


@pytest.fixture
def extracted(tmp_path, monkeypatch, page):
    source = tmp_path / "video-source"
    source.write_bytes(b"synthetic source identity")

    def extract(video, directory, ffmpeg, ffprobe):
        directory.mkdir(parents=True)
        frames = []
        for i in range(23):
            path = directory / f"{i:06d}.png"
            rgb = np.full((*page[0].shape, 3), 100 + i, np.uint8)
            Image.fromarray(rgb).save(path)
            frames.append({"id": str(i), "path": str(path), "time_s": 2.0 + i * 0.1})
        return frames, {
            "time_base": "1/10",
            "source_pts": list(range(20, 43)),
            "rotation_degrees": 0,
        }

    monkeypatch.setattr("tools.phone_testset.prepare.extract", extract)
    return [source]


def test_prepare_and_folder_selection_accept_partial_exit_on_initial_frame(
    tmp_path, extracted, page
):
    quad = page[1].astype(float) + np.array([35, 0])
    visible, _ = rasterize_quad(quad, page[0].shape)
    assert visible[:, -1].any()
    output = tmp_path / "clipped-phone"
    result = prepare(extracted, output, SyntheticSAM(visible))
    assert result["eligible"] == 1
    sequences, selection = select_input_folder(output)
    assert selection["included"] == 1 and selection["excluded"] == 0
    assert len(sequences[0].frames) == 23
    annotations = read_jsonl(output / "annotations.jsonl")
    assert [item["index"] for item in annotations] == [0, 10, 20]
    assert all(item["geometry"]["out_of_frame"] for item in annotations)


def test_prepare_retains_visible_and_full_page_masks_and_no_png_restore(tmp_path, extracted, page):
    visible = page[0].copy()
    visible[:55, 145:] = 0
    model = SyntheticSAM(visible)
    folder = tmp_path / "set"
    result = prepare(extracted, folder, model, identity={"model": "synthetic"})
    assert result["eligible"] == 1 and model.calls == 3
    row = read_jsonl(folder / "manifest.jsonl")[0]
    assert len(row["frames"]) == 23
    assert [i for i, f in enumerate(row["frames"]) if "mask" in f] == [0, 10, 20]
    evidence = read_jsonl(folder / "annotations.jsonl")
    for item in evidence:
        assert item["status"] == "accepted"
        with Image.open(item["visible_mask"]) as im:
            assert np.array_equal(np.asarray(im), visible)
        with Image.open(item["mask"]) as im:
            assert np.asarray(im).sum() > visible.sum()
        assert item["geometry"]["extrapolated_pixels"] > 0
    review = next((folder / "review_initial").glob("*.png"))
    saved = review.read_bytes()
    assets = {p: sha256(p) for p in folder.rglob("*.png") if p != review}
    review.unlink()
    assert not select_input_folder(folder)[0]
    assert (
        prepare(extracted, folder, model, identity={"model": "synthetic"})["status"] == "EXISTING"
    )
    assert model.calls == 3 and not review.exists()
    assert assets == {p: sha256(p) for p in assets}
    review.write_bytes(saved)
    assert len(select_input_folder(folder)[0]) == 1
    with pytest.raises(ValueError, match="inputs changed"):
        prepare(extracted, folder, model, step=7)
    assert prepare(extracted, folder, model, step=7, recreate=True)["eligible"] == 1
    row = read_jsonl(folder / "manifest.jsonl")[0]
    assert [i for i, f in enumerate(row["frames"]) if "mask" in f] == [0, 7, 14, 21]


@pytest.mark.parametrize("failure", ["initial_geometry", "initial_model", "empty"])
def test_failed_initial_annotation_rejects_sequence_without_fake_masks(
    tmp_path, extracted, page, failure
):
    visible = page[0].copy()
    if failure == "initial_geometry":
        visible[:] = 0
        cv2.circle(visible, (100, 80), 50, 1, -1)
    if failure == "empty":
        visible[:] = 0
    folder = tmp_path / "failed"
    model = SyntheticSAM(visible, fail_at=1 if failure == "initial_model" else None)
    assert prepare(extracted, folder, model)["ineligible"] == 1
    assert not list((folder / "review_initial").glob("*.png"))
    assert not select_input_folder(folder)[0]
    row = read_jsonl(folder / "manifest.jsonl")[0]
    reason = row["provenance"]["phone_testset"]["reason"]
    assert "annotation frame" in reason
    assert read_jsonl(folder / "annotations.jsonl")[-1]["status"] == "failed"
    assert not list((folder / "masks").rglob("*.png"))


@pytest.mark.parametrize("failure", ["model", "geometry", "empty"])
def test_failed_later_annotation_keeps_video_and_all_frames_for_cutie(
    tmp_path, extracted, page, failure
):
    class FaultySAM(SyntheticSAM):
        def predict(self, rgb):
            mask, info = super().predict(rgb)
            if self.calls == 2:
                if failure == "model":
                    raise RuntimeError("synthetic segmentation failure")
                mask[:] = 0
                if failure == "geometry":
                    cv2.circle(mask, (100, 80), 50, 1, -1)
            return mask, info

    model = FaultySAM(page[0])
    folder = tmp_path / "sparse"
    result = prepare(extracted, folder, model)
    assert result["eligible"] == 1 and result["failed_annotations"] == 1
    assert model.calls == 3  # Continue to frame 20 after the failure at frame 10.
    row = read_jsonl(folder / "manifest.jsonl")[0]
    assert len(row["frames"]) == 23
    assert [i for i, frame in enumerate(row["frames"]) if "mask" in frame] == [0, 20]
    assert not list((folder / "masks").glob("*/000010.png"))
    info = row["provenance"]["phone_testset"]
    assert info["status"] == "eligible" and info["reason"] is None
    assert info["annotation_coverage"] == {
        "sampled_after_initial": 2,
        "accepted_after_initial": 1,
        "failed_after_initial": 1,
        "fraction": 0.5,
    }
    assert info["annotation_failures"][0]["index"] == 10
    assert [item["status"] for item in read_jsonl(folder / "annotations.jsonl")] == [
        "accepted",
        "failed",
        "accepted",
    ]
    sequences, selection = select_input_folder(folder)
    assert selection["included"] == 1 and selection["annotation_coverage"][0]["fraction"] == 0.5
    from test_inference_evaluation import SpyPredictor, metadata

    spy = SpyPredictor()
    run = tmp_path / "run"
    propagate(
        folder / "manifest.jsonl",
        run,
        spy,
        {**metadata(), "scope": "independent"},
        sequences=sequences,
        selection=selection,
    )
    assert len(spy.calls) == 23 and sum(call[1] for call in spy.calls) == 1
    snapshot = sha256(run / "selected-manifest.jsonl")
    next((folder / "review_initial").glob("*.png")).unlink()
    assert not select_input_folder(folder)[0]
    assert evaluate(run, tmp_path / "eval", workers=1)["evaluated_frames"] == 1
    assert sha256(run / "selected-manifest.jsonl") == snapshot
    evaluation = read_json(tmp_path / "eval/evaluation.json")
    assert evaluation["selection"]["annotation_coverage"][0]["failures"][0]["index"] == 10
    assert "1/2" in (tmp_path / "eval/report.md").read_text()


def test_all_later_annotations_may_fail_without_excluding_video(tmp_path, extracted, page):
    class OnlyInitialSAM(SyntheticSAM):
        def predict(self, rgb):
            mask, info = super().predict(rgb)
            if self.calls > 1:
                raise RuntimeError("no reliable later annotation")
            return mask, info

    folder = tmp_path / "initial-only"
    model = OnlyInitialSAM(page[0])
    assert prepare(extracted, folder, model)["eligible"] == 1
    assert model.calls == 3 and len(select_input_folder(folder)[0]) == 1
    row = read_jsonl(folder / "manifest.jsonl")[0]
    assert [i for i, frame in enumerate(row["frames"]) if "mask" in frame] == [0]
    assert row["provenance"]["phone_testset"]["annotation_coverage"]["fraction"] == 0


def test_geometric_pseudo_label_is_named_in_evaluation(tmp_path, extracted, page):
    folder = tmp_path / "prepared"
    prepare(extracted, folder, SyntheticSAM(page[0]))
    sequences, selection = select_input_folder(folder)
    from test_inference_evaluation import SpyPredictor, metadata

    meta = {**metadata(), "scope": "independent"}
    run = tmp_path / "run"
    propagate(
        folder / "manifest.jsonl",
        run,
        SpyPredictor(),
        meta,
        sequences=sequences,
        selection=selection,
    )
    assert evaluate(run, tmp_path / "evaluation", workers=1)["evaluated_frames"] == 2
    result = read_json(tmp_path / "evaluation/evaluation.json")
    assert result["selection"]["annotation_sources"][0]["kind"] == "geometric_pseudo_labels"
    assert "геометрической псевдоразметкой" in (tmp_path / "evaluation/report.md").read_text()


def test_sam_adapter_local_strict_checkpoint_and_bfloat16_output(tmp_path, monkeypatch, page):
    import sys
    import types

    import torch

    from tools.phone_testset.sam import SAM

    mask, _ = page

    def builder(**kwargs):
        assert kwargs["load_from_HF"] is False and kwargs["checkpoint_path"] is None
        assert kwargs["device"] == "cpu" and kwargs["compile"] is False
        assert kwargs["enable_inst_interactivity"] is False
        return torch.nn.Linear(2, 2)

    class Processor:
        def __init__(self, model, device, confidence_threshold):
            assert device == "cpu" and confidence_threshold == 0.5

        def set_image(self, image):
            assert isinstance(image, Image.Image) and image.size == (220, 160)
            return {}

        def set_text_prompt(self, prompt, state):
            assert prompt == "sheet of paper"
            small = np.zeros_like(mask)
            small[:10, :10] = 1
            return {
                "masks": torch.from_numpy(np.stack([small, mask])[:, None]).bool(),
                "scores": torch.tensor([0.95, 0.9], dtype=torch.bfloat16),
            }

    root, model_module = types.ModuleType("sam3"), types.ModuleType("sam3.model")
    root.__path__, model_module.__path__ = [], []
    build_module, processor_module = (
        types.ModuleType("sam3.model_builder"),
        types.ModuleType("sam3.model.sam3_image_processor"),
    )
    build_module.build_sam3_image_model, processor_module.Sam3Processor = builder, Processor
    for name, module in [
        ("sam3", root),
        ("sam3.model", model_module),
        ("sam3.model_builder", build_module),
        ("sam3.model.sam3_image_processor", processor_module),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("tools.phone_testset.sam.check_installation", lambda: None)
    weights = tmp_path / "local.pt"
    torch.save(
        {
            "model": {
                "detector.weight": torch.ones(2, 2),
                "detector.bias": torch.zeros(2),
                "detector.backbone.vision_backbone.sam2_convs.0.conv_1x1.weight": torch.ones(1),
            }
        },
        weights,
    )
    adapter = SAM(weights, device="cpu")
    prediction, info = adapter.predict(np.zeros((160, 220, 3), np.uint8))
    assert np.array_equal(prediction, mask) and info["instance_score"] > 0.8
    assert info["selected_instance"] == 1 and info["model_instances"] == 2
    assert info["retained_instances"] == [1] and info["instances"] == 1
    adapter.max_masks = 3
    prediction, info = adapter.predict(np.zeros((160, 220, 3), np.uint8))
    assert np.array_equal(prediction, mask) and info["retained_instances"] == [1, 0]
    assert info["instances"] == 2 and info["max_masks"] == 3
    torch.save(
        {
            "detector.weight": torch.ones(2, 2),
            "detector.bias": torch.zeros(2),
            "detector.unexpected_active_weight": torch.ones(1),
        },
        weights,
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        SAM(weights, device="cpu")
    torch.save({"detector.weight": torch.ones(2, 2)}, weights)
    with pytest.raises(RuntimeError, match="Missing key"):
        SAM(weights, device="cpu")
    with pytest.raises(ValueError, match="existing local"):
        SAM(tmp_path / "missing.pt", device="cpu")
