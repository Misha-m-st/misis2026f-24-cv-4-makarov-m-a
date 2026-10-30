"""Pinned, instance-local rectangular adapters; vendor checkouts stay untouched.

Only square geometry expressions are rewritten in selected official methods.
Their complete AST hashes must match the pinned sources before adaptation. Model
weights, layers, attention algorithms and inference scheduling stay upstream.
The transformed methods retain their original module globals and decorators.
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import inspect
import json
import textwrap
from types import MethodType

from document_vos.benchmark_inputs import FrameStore
from document_vos.config import resource
from document_vos.io import PipelineError

ADAPTER_VERSION = "cutie-rectangular/v1"


def _node(code):
    return ast.parse(code, mode="eval").body


def _key(node):
    return ast.dump(node, include_attributes=False)


class _Rectangular(ast.NodeTransformer):
    def __init__(self):
        self.changes = 0
        self.expressions = {
            _key(_node(old)): _node(new)
            for old, new in {
                "(self.image_size, self.image_size)": "self.benchmark_image_shape",
                "(self.sam_image_embedding_size, self.sam_image_embedding_size)": "self.benchmark_embedding_shape",
                "(self.input_mask_size, self.input_mask_size)": "self.benchmark_mask_shape",
                "(input_mask_res, input_mask_res)": "self.tracker.benchmark_mask_shape",
                "mask_H != self.image_size or mask_W != self.image_size": "(mask_H, mask_W) != self.benchmark_image_shape",
                "mask_H != self.input_mask_size or mask_W != self.input_mask_size": "(mask_H, mask_W) != self.benchmark_mask_shape",
                "backbone_features.size(2) == self.sam_image_embedding_size": "backbone_features.size(2) == self.benchmark_embedding_shape[0]",
                "backbone_features.size(3) == self.sam_image_embedding_size": "backbone_features.size(3) == self.benchmark_embedding_shape[1]",
            }.items()
        }

    def visit(self, node):
        replacement = self.expressions.get(_key(node))
        if replacement is not None:
            self.changes += 1
            return ast.copy_location(replacement, node)
        return super().visit(node)

    def visit_Assign(self, node):
        names = tuple(getattr(t, "id", None) for t in node.targets)
        shapes = {
            ("consolidated_H", "consolidated_W"): "self.benchmark_low_res_shape",
            ("H_mask", "W_mask"): "self.tracker.benchmark_low_res_shape",
        }
        if names in shapes:
            self.changes += 1
            return ast.copy_location(
                ast.Assign(
                    targets=[
                        ast.Tuple(
                            elts=[ast.Name(id=name, ctx=ast.Store()) for name in names],
                            ctx=ast.Store(),
                        )
                    ],
                    value=_node(shapes[names]),
                ),
                node,
            )
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in {
            "load_video_frames",
            "load_resource_as_video_frames",
        }:
            self.changes += 1
            node.func.id = "_benchmark_frames"
        elif isinstance(node.func, ast.Name) and node.func.id == "is_image_type":
            self.changes += 1
            return ast.copy_location(ast.Constant(value=False), node)
        return self.generic_visit(node)


def _benchmark_frames(*, video_path=None, resource_path=None, **_kwargs):
    frames = video_path if video_path is not None else resource_path
    if not isinstance(frames, FrameStore):
        raise PipelineError("Rectangular predictor требует bounded FrameStore")
    return frames, *frames.geometry.padded


def _bind(instance, original):
    original = inspect.unwrap(original)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    key = f"{original.__module__}.{original.__qualname__}"
    expected = json.loads(resource("benchmark_geometry_sources.json").read_text())[key]
    actual = hashlib.sha256(_key(tree).encode()).hexdigest()
    if actual != expected:
        raise PipelineError(f"{ADAPTER_VERSION}: pinned API изменился: {key}")
    transform = _Rectangular()
    tree = transform.visit(tree)
    if not transform.changes:
        raise PipelineError(f"{ADAPTER_VERSION}: геометрия API не найдена: {key}")
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, "_benchmark_frames": _benchmark_frames}
    exec(
        compile(
            tree, f"<{ADAPTER_VERSION}:{key}>", "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    method = namespace[original.__name__]
    setattr(instance, original.__name__, MethodType(method, instance))


def adapt_efficienttam(predictor):
    for name in (
        "init_state",
        "add_new_mask",
        "propagate_in_video_preflight",
        "_consolidate_temp_output_across_obj",
        "_forward_sam_heads",
    ):
        _bind(predictor, getattr(type(predictor), name))


def adapt_sam31(predictor):
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTracking

    model = predictor.model
    tracker = model.tracker.model
    # The benchmark bridge supports the single-object bucketized path only.
    if model.tracker.per_obj_inference:
        raise PipelineError("Rectangular SAM 3.1 требует per_obj_inference=false")
    _bind(model, Sam3MultiplexTracking.init_state)
    initialize = model.init_state

    def init_state(*args, **kwargs):
        state = initialize(*args, **kwargs)
        state["action_history"] = []
        return state

    model.init_state = init_state
    for name in (
        "_recondition_masklets",
        "_tracker_add_new_objects",
        "_propogate_tracker_one_frame_local_gpu",
    ):
        _bind(model, getattr(type(model), name))
    for name in (
        "_forward_sam_heads",
        "get_propagation_dense_pe",
        "add_new_masks",
        "_consolidate_temp_output_across_obj",
    ):
        _bind(tracker, getattr(type(tracker), name))


def set_geometry(model, geometry, stride):
    """Set rectangular prompt grids and positional encodings before each sequence."""
    h, w = geometry.padded
    grid = (h // stride, w // stride)
    model.benchmark_image_shape = (h, w)
    model.benchmark_embedding_shape = grid
    model.benchmark_low_res_shape = tuple(n * 4 for n in grid)
    model.benchmark_mask_shape = tuple(n * 16 for n in grid)
    for name in ("sam_prompt_encoder", "interactive_sam_prompt_encoder"):
        prompt = getattr(model, name, None)
        if prompt is not None:
            prompt.image_embedding_size = grid
            prompt.input_image_size = (h, w)
            prompt.mask_input_size = tuple(n * 4 for n in grid)
    for module in model.modules():
        # Populate sine-position caches outside torch.compile/CUDA graphs. A
        # tensor cached by a compiled forward would otherwise outlive its graph
        # generation and can be overwritten on the next video frame.
        if module.__class__.__module__.startswith("efficient_track_anything") and hasattr(
            module, "_pe"
        ):
            module.cache.clear()
            module._pe(1, next(model.parameters()).device, *grid)
        # Memory attention uses stride-sized spatial grids. Initialize the full
        # rectangle so upstream's square sqrt fallback is never entered.
        if hasattr(module, "compute_cis") and hasattr(module, "rope_k_repeat"):
            module.freqs_cis = module.compute_cis(end_x=grid[1], end_y=grid[0]).to(
                next(model.parameters()).device
            )
            if getattr(module, "use_rope_real", False):
                module.freqs_cis_real = module.freqs_cis.real
                module.freqs_cis_imag = module.freqs_cis.imag


def set_sam31_geometry(predictor, geometry):
    model = predictor.model
    tracker = model.tracker.model
    set_geometry(tracker, geometry, 14)
    tracker.maskmem_backbone.mask_downsampler.interpol_size = list(tracker.benchmark_mask_shape)
    model.benchmark_image_shape = geometry.padded
    model.tracker.benchmark_mask_shape = tracker.benchmark_mask_shape
    model.tracker.benchmark_low_res_shape = tracker.benchmark_low_res_shape
    # Global ViT blocks need rectangular RoPE. Window attention keeps its own grid.
    grid = tracker.benchmark_embedding_shape
    for block in model.detector.backbone.vision_backbone.trunk.blocks:
        if block.window_size == 0:
            attn = block.attn
            attn.input_size = grid[::-1]
            attn.rope_tiled = False
            attn._setup_rope_freqs()
            for name in ("freqs_cis", "freqs_cis_real", "freqs_cis_imag"):
                value = getattr(attn, name, None)
                if value is not None:
                    setattr(attn, name, value.to(next(tracker.parameters()).device))
