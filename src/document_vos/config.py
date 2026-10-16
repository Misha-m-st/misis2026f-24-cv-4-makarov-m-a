"""Strict public configuration. Paths in YAML belong to that YAML, CLI paths to cwd."""

from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

from document_vos.io import PipelineError
from document_vos.runtime import Runtime

DATASETS = ["smartdoc2015", "midv2020", "midv500", "midv2019", "midvholo", "dlc2021"]
DEFAULTS = {
    "paths": {
        "archives": "data/archives",
        "raw": "data/raw",
        "samples": "data/samples",
        "videos": "data/videos",
        "prepared": "data/prepared",
        "weights": "weights",
        "results": "results",
        "cutie": ".cache/Cutie",
    },
    "datasets": {"selected": DATASETS, "roots": {}},
    "samples": {"dataset": "smartdoc2015", "count": 1, "catalog": None},
    "model": "finetuned",
    "preparation": {"seed": 42, "val_fraction": 0.2, "seq_length": 8},
    "training": {
        "batch_size": 2,
        "seq_length": 8,
        "crop_size": [480, 480],
        "amp": True,
        "learning_rate": 1e-5,
        "num_iterations": 30000,
        "num_ref_frames": 3,
        "lr_schedule": "constant",
    },
    "inference": {
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
    },
    "visualization": {"fps": 30.0, "alpha": 0.35},
    "sam": {
        "weights": "weights/sam3/sam3.pt",
        "prompt": "sheet of paper",
        "min_score": 0.5,
        "max_masks": 1,
        "step": None,
    },
    "runtime": Runtime().to_dict(),
}


def merge(default: dict, supplied: dict, prefix: str = "") -> dict:
    if not isinstance(supplied, dict):
        raise PipelineError(f"{prefix or 'config'}: требуется словарь")
    unknown = set(supplied) - set(default)
    if unknown:
        raise PipelineError(f"Неизвестные параметры {prefix}: {sorted(unknown)}")
    result = deepcopy(default)
    for key, val in supplied.items():
        name = f"{prefix}.{key}" if prefix else key
        if key == "roots":
            if not isinstance(val, dict) or set(val) - set(DATASETS):
                raise PipelineError("datasets.roots: неизвестный датасет")
            result[key] = dict(val)
        elif isinstance(default[key], dict):
            result[key] = merge(default[key], val, name)
        else:
            expected = default[key]
            if expected is not None:
                valid = (
                    type(val) in (int, float)
                    if type(expected) is float
                    else type(val) is type(expected)
                )
                if not valid:
                    raise PipelineError(f"{name}: неверный тип значения")
            result[key] = val
    return result


def validate(cfg: dict) -> None:
    import math

    Runtime.parse(cfg["runtime"])
    if cfg["model"] not in {"baseline", "finetuned"}:
        raise PipelineError("model: baseline или finetuned")
    selected = cfg["datasets"]["selected"]
    if (
        not selected
        or any(x not in DATASETS for x in selected)
        or len(set(selected)) != len(selected)
    ):
        raise PipelineError("datasets.selected: нужны уникальные известные датасеты")
    if cfg["samples"]["dataset"] not in DATASETS or not 1 <= cfg["samples"]["count"] <= 2:
        raise PipelineError("samples: известный датасет, count от 1 до 2")
    p, t, i = cfg["preparation"], cfg["training"], cfg["inference"]
    if p["seed"] < 0 or not 0 < p["val_fraction"] < 1 or p["seq_length"] < 2:
        raise PipelineError("preparation: неверный seed/val_fraction/seq_length")
    if (
        t["batch_size"] < 1
        or t["seq_length"] < 2
        or t["num_iterations"] < 1
        or not 0 < t["learning_rate"] < 1
        or not 1 <= t["num_ref_frames"] < t["seq_length"]
        or t["lr_schedule"] != "constant"
    ):
        raise PipelineError("training: неверные параметры обучения")
    crop = t["crop_size"]
    if (
        len(crop) != 2
        or any(type(n) is not int for n in crop)
        or crop[0] != crop[1]
        or crop[0] < 16
        or crop[0] % 16
    ):
        raise PipelineError("training.crop_size: квадрат, кратный 16")
    if any(i[k] < 1 for k in ("mem_every", "top_k", "stagger_updates", "max_mem_frames")):
        raise PipelineError("inference: параметры памяти должны быть положительными")
    if i["max_internal_size"] != -1 and i["max_internal_size"] < 16:
        raise PipelineError("inference.max_internal_size: -1 либо >=16")
    lt = i["long_term"]
    if (
        any(lt[k] < 1 for k in lt if k != "count_usage")
        or lt["min_mem_frames"] >= lt["max_mem_frames"]
        or lt["buffer_tokens"] >= lt["max_num_tokens"]
    ):
        raise PipelineError("inference.long_term: неверные границы памяти")
    v, s = cfg["visualization"], cfg["sam"]
    if not math.isfinite(v["fps"]) or v["fps"] <= 0 or not 0 <= v["alpha"] <= 1:
        raise PipelineError("visualization: неверный fps/alpha")
    if (
        not s["prompt"].strip()
        or not 0 <= s["min_score"] <= 1
        or s["max_masks"] < 1
        or (s["step"] is not None and (type(s["step"]) is not int or s["step"] < 1))
    ):
        raise PipelineError("sam: неверные параметры")
    for path in (
        list(cfg["paths"].values()) + list(cfg["datasets"]["roots"].values()) + [s["weights"]]
    ):
        if not isinstance(path, str) or not path.strip():
            raise PipelineError("Путь должен быть непустой строкой")


def load(path: Path | None = None) -> dict:
    explicit = path is not None
    path = path if explicit else Path("config.yaml")
    if path.is_file():
        supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
    elif not explicit:
        supplied, base = {}, Path.cwd()
    else:
        raise PipelineError(f"Конфиг не найден: {path}")
    cfg = merge(DEFAULTS, supplied)
    validate(cfg)
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str((base / Path(val).expanduser()).resolve())
    cfg["datasets"]["roots"] = {
        key: str((base / Path(val).expanduser()).resolve())
        for key, val in cfg["datasets"]["roots"].items()
    }
    cfg["sam"]["weights"] = str((base / Path(cfg["sam"]["weights"]).expanduser()).resolve())
    if cfg["samples"]["catalog"] is not None:
        val = cfg["samples"]["catalog"]
        if not isinstance(val, str):
            raise PipelineError("samples.catalog: строка пути или null")
        cfg["samples"]["catalog"] = str((base / Path(val).expanduser()).resolve())
    return cfg


def resource(name: str):
    return files("document_vos.resources").joinpath(name)
