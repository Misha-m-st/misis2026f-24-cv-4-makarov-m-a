"""Интеграция зафиксированного официального Cutie без скачивания весов."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from document_vos.io import PipelineError

CUTIE_REVISION = "ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a"
INFERENCE_DEFAULTS = {
    "amp": True,
    "flip_aug": False,
    "max_internal_size": 480,
    "use_long_term": True,
    "mem_every": 5,
    "max_mem_frames": 5,
    "top_k": 30,
    "stagger_updates": 5,
    "chunk_size": -1,
    "save_aux": False,
    "long_term": {
        "count_usage": True,
        "max_mem_frames": 10,
        "min_mem_frames": 5,
        "num_prototypes": 128,
        "max_num_tokens": 10000,
        "buffer_tokens": 2000,
    },
}


def check_checkout(root: Path) -> Path:
    root = root.resolve()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        raise PipelineError(
            f"Не найден Git checkout официального Cutie: {root}. Проверьте paths.cutie "
            "в config.yaml; для установки выполните uv run document-vos setup"
        ) from e
    if revision != CUTIE_REVISION or dirty:
        raise PipelineError(
            f"Нужен чистый Cutie {CUTIE_REVISION}, получен {revision}, dirty={bool(dirty)}"
        )
    if not (root / "cutie/config/train_config.yaml").is_file():
        raise PipelineError("В checkout отсутствует официальный train_config.yaml")
    return root


def local_weights(weights: Path) -> Path:
    weights = weights.expanduser().resolve()
    if not weights.is_file():
        raise PipelineError(
            f"Начальные веса не найдены: {weights}. Укажите явный локальный файл weights."
        )
    return weights


def compose_official(root: Path, name: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "cutie/config")):
        return compose(config_name=name)


def activate_checkout(root: Path) -> None:
    sys.path.insert(0, str(root))
    import cutie

    if {Path(p).resolve() for p in cutie.__path__} != {root / "cutie"}:
        raise PipelineError("Импортирован другой Cutie: используйте отдельное окружение")


def disable_downloads_and_require_full_weights() -> None:
    """Меняем только загрузку инициализации; слои и официальный forward сохраняются."""
    import torch
    from cutie.model.cutie import CUTIE
    from cutie.model.utils import resnet

    if getattr(CUTIE, "_document_vos_offline", False):
        return
    for name in ["resnet18", "resnet50"]:
        original = getattr(resnet, name)

        def offline(*args, _constructor=original, **kwargs):
            kwargs["pretrained"] = False
            return _constructor(*args, **kwargs)

        setattr(resnet, name, offline)

    def blocked(*args, **kwargs):
        raise PipelineError("Сетевая загрузка весов запрещена; нужен полный локальный checkpoint")

    torch.hub.load_state_dict_from_url = blocked
    torch.utils.model_zoo.load_url = blocked
    original_load = CUTIE.load_weights

    def strict_load(self, src_dict, *args, **kwargs):
        if self.single_object:
            # В pinned upstream single-object ветка обрезает только mask_encoder.conv1.
            # В fuser последний вход также относится к "other objects" и равен нулю
            # при одном документе; обрезаем его симметрично обратной upstream-конверсии.
            key = "pixel_fuser.sensory_compress.weight"
            source = src_dict.get(key)
            target = self.state_dict()[key]
            if (
                source is not None
                and source.shape[1] == target.shape[1] + 1
                and source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
            ):
                src_dict[key] = source[:, :-1]
        original_load(self, src_dict, *args, **kwargs)
        # Официальная конверсия single-object каналов уже выполнена.
        self.load_state_dict(src_dict, strict=True)

    CUTIE.load_weights = strict_load
    CUTIE._document_vos_offline = True


def inference_config(root: Path, overrides: dict) -> dict:
    if set(overrides) - INFERENCE_DEFAULTS.keys():
        raise PipelineError(
            f"Неизвестные параметры инференса: {sorted(set(overrides) - INFERENCE_DEFAULTS.keys())}"
        )
    cfg = compose_official(root, "eval_config")
    params = {**INFERENCE_DEFAULTS, **overrides}
    if "long_term" in overrides:
        params["long_term"] = {**INFERENCE_DEFAULTS["long_term"], **overrides["long_term"]}
    if params["mem_every"] < 1 or params["top_k"] < 1 or params["stagger_updates"] < 1:
        raise PipelineError("Параметры памяти должны быть положительными")
    for key, value in params.items():
        cfg[key] = value
    # Пути, exp_id и таблица стандартных VOS не участвуют в нашем инференсе.
    return {"model": OmegaConf.to_container(cfg.model, resolve=True), **params}


class CutiePredictor:
    """Новый InferenceCore для каждого видео полностью сбрасывает object manager и память."""

    def __init__(self, root: Path, weights: Path, cfg: dict, device: str = "cuda"):
        activate_checkout(root)
        disable_downloads_and_require_full_weights()
        import torch
        from cutie.model.cutie import CUTIE

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PipelineError("CUDA недоступна; GPU-инференс не запущен")
        self.cfg = OmegaConf.create(cfg)
        self.model = CUTIE(self.cfg).to(self.device).eval()
        state = torch.load(weights, map_location=self.device, weights_only=True)
        self.model.load_weights(state, init_as_zero_if_needed=True)
        self.processor = None
        self.sync()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def reset(self) -> None:
        from cutie.inference.inference_core import InferenceCore

        self.processor = InferenceCore(self.model, self.cfg)

    def sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def step(
        self, image: np.ndarray, mask: np.ndarray | None = None, end: bool = False
    ) -> np.ndarray:
        from time import perf_counter

        import numpy as np

        torch = self.torch
        tensor = (
            torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).to(self.device).float()
            / 255
        )
        initial = (
            torch.from_numpy(np.array(mask, copy=True)).to(self.device)
            if mask is not None
            else None
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                enabled=bool(self.cfg.amp and self.device.type == "cuda"),
            ),
        ):
            self.sync()
            start = perf_counter()
            prob = self.processor.step(
                tensor, initial, objects=[1] if initial is not None else None, end=end
            )
            self.sync()
            self.last_model_seconds = perf_counter() - start
            result = self.processor.output_prob_to_mask(prob)
        return (result.detach().cpu().numpy() == 1).astype(np.uint8)

    def memory_stats(self) -> dict:
        if self.device.type != "cuda":
            return {
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "status": "NOT_MEASURED",
            }
        return {
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
            "status": "MEASURED",
        }
