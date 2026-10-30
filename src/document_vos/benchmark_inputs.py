"""Lossless, bounded frame loading and CUTIE's resize/pad geometry for benchmarks."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from document_vos.io import PipelineError


@dataclass(frozen=True)
class Geometry:
    original: tuple[int, int]
    resized: tuple[int, int]
    padded: tuple[int, int]
    padding: tuple[int, int, int, int]

    @classmethod
    def cutie(cls, shape, max_size, stride=16):
        h, w = shape
        short = min(h, w)
        resized = (
            (int(h / short * max_size), int(w / short * max_size))
            if 0 < max_size < short
            else (h, w)
        )
        rh, rw = resized
        ph, pw = (-rh) % stride, (-rw) % stride
        return cls(
            (h, w), resized, (rh + ph, rw + pw), (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        )

    def image(self, rgb):
        import torch
        import torch.nn.functional as F

        tensor = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float() / 255
        if tuple(tensor.shape[-2:]) != self.original:
            raise PipelineError("Размер кадра изменился внутри последовательности")
        if self.original != self.resized:
            tensor = F.interpolate(
                tensor[None], self.resized, mode="bilinear", align_corners=False
            )[0]
        return F.pad(tensor, self.padding)

    def mask(self, mask):
        import torch
        import torch.nn.functional as F

        tensor = torch.as_tensor(np.array(mask, copy=True)).float()[None, None]
        if tuple(tensor.shape[-2:]) != self.original:
            raise PipelineError("Начальная маска не соответствует исходному кадру")
        tensor = F.interpolate(tensor, self.resized, mode="nearest-exact")
        return F.pad(tensor, self.padding)[0, 0].numpy().astype(np.uint8)

    def crop(self, tensor):
        if tuple(tensor.shape[-2:]) != self.padded:
            raise PipelineError("Модель вернула маску вне согласованной прямоугольной сетки")
        left, _, top, _ = self.padding
        h, w = self.resized
        return tensor[..., top : top + h, left : left + w]

    def restore_binary(self, mask):
        mask = self.crop(mask)
        return np.asarray(
            Image.fromarray(mask).resize(self.original[::-1], Image.Resampling.NEAREST)
        )

    def restore_logits(self, logits):
        import torch
        import torch.nn.functional as F

        scores = torch.from_numpy(self.crop(logits).copy())[None, None]
        return F.interpolate(scores, self.original, mode="bilinear", align_corners=False)[
            0, 0
        ].numpy()

    def to_dict(self):
        return {
            "original_hw": self.original,
            "resized_hw": self.resized,
            "padded_hw": self.padded,
            "padding_lrtb": self.padding,
        }


class FrameStore:
    """Indexable CPU tensors; at most queue_size futures, with bounded lookahead.

    Supports the official predictors' random-access APIs without keeping every
    decoded frame. Only the caller transfers frames to CUDA.
    """

    def __init__(self, frames, geometry, *, mean=(0, 0, 0), std=(1, 1, 1)):
        import torch

        from document_vos.runtime import current_runtime

        self.frames, self.geometry = frames, geometry
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]
        runtime = current_runtime()
        self.capacity = runtime.queue_size
        self.pool = ThreadPoolExecutor(max_workers=max(1, runtime.io_threads // 2))
        self.pending = OrderedDict()
        self.closed = False
        self.peak_pending = 0
        self.device = torch.device("cpu")
        self.shape = (len(frames), 3, *geometry.padded)

    def __len__(self):
        return len(self.frames)

    def _read(self, index):
        with Image.open(self.frames[index].path) as image:
            tensor = self.geometry.image(image.convert("RGB"))
        return (tensor - self.mean) / self.std

    def __getitem__(self, index):
        index = int(index)
        if self.closed or not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.pending:
            while len(self.pending) >= self.capacity:
                _, future = self.pending.popitem(last=False)
                future.cancel()
            self.pending[index] = self.pool.submit(self._read, index)
        tensor = self.pending[index].result()
        self.pending.move_to_end(index)
        # Discard already consumed lookbehind before scheduling more work.
        for old in list(self.pending):
            if old < index:
                self.pending.pop(old).cancel()
        for following in range(index + 1, min(len(self), index + self.capacity)):
            if len(self.pending) >= self.capacity:
                break
            if following not in self.pending:
                self.pending[following] = self.pool.submit(self._read, following)
        self.peak_pending = max(self.peak_pending, len(self.pending))
        return tensor

    def close(self):
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
