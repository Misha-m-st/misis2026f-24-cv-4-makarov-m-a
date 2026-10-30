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
        "cutie": "external/Cutie",
        "external_models": ".cache/external",
    },
    "datasets": {"selected": DATASETS, "roots": {}},
    "samples": {"dataset": "smartdoc2015", "count": 1, "catalog": None},
    "model": "finetuned",
    "benchmark": {
        "selected": ["baseline", "finetuned", "efficienttam", "livos", "sam31"],
        "seed": 42,
        "resolution": "cutie",
        "manifest": None,
        "split_dir": None,
        "models": {
            "baseline": {"weights": None, "inference": {}},
            "finetuned": {"weights": None, "inference": {}},
            "efficienttam": {
                "weights": "weights/efficienttam/efficienttam_s.pt",
                "amp": True,
                "dtype": "bfloat16",
                "compile": True,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            },
            "livos": {
                "weights": "weights/livos/livos-wmose-480p.pth",
                "max_internal_size": 480,
                "amp": False,
            },
            "sam31": {
                "weights": "weights/sam31/sam3.1_multiplex.pt",
                "amp": True,
                "dtype": "bfloat16",
                "compile": False,
                "use_fa3": False,
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": False,
            },
        },
    },
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
    if prefix in {
        "benchmark.models.baseline.inference",
        "benchmark.models.finetuned.inference",
    }:
        # These are sparse overrides applied after the shared inference block.
        merge(DEFAULTS["inference"], supplied)
        return deepcopy(supplied)
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
    if cfg["model"] not in {"baseline", "finetuned", "efficienttam", "livos", "sam31"}:
        raise PipelineError("model: baseline, finetuned, efficienttam, livos или sam31")
    bench = cfg["benchmark"]
    known_models = set(bench["models"])
    if (
        not isinstance(bench["selected"], list)
        or not bench["selected"]
        or any(not isinstance(model, str) for model in bench["selected"])
        or len(set(bench["selected"])) != len(bench["selected"])
        or set(bench["selected"]) - known_models
    ):
        raise PipelineError("benchmark.selected: нужны уникальные поддерживаемые модели")
    if type(bench["seed"]) is not int or bench["seed"] < 0:
        raise PipelineError("benchmark.seed: требуется целое число >=0")
    if bench["resolution"] not in {"cutie", "native"}:
        raise PipelineError("benchmark.resolution: cutie или native")
    for model, options in bench["models"].items():
        if model in {"baseline", "finetuned"}:
            if options["weights"] is not None and (
                not isinstance(options["weights"], str) or not options["weights"].strip()
            ):
                raise PipelineError(f"benchmark.models.{model}.weights: путь или null")
        elif options["weights"] is not None and (
            not isinstance(options["weights"], str) or not options["weights"].strip()
        ):
            raise PipelineError(f"benchmark.models.{model}.weights: требуется путь")
        if model in {"efficienttam", "sam31"} and options.get("dtype") not in {
            "float16",
            "bfloat16",
        }:
            raise PipelineError(f"benchmark.models.{model}.dtype: float16 или bfloat16")
    for key in ("manifest", "split_dir"):
        if bench[key] is not None and (not isinstance(bench[key], str) or not bench[key].strip()):
            raise PipelineError(f"benchmark.{key}: путь или null")
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
    for key in ("manifest", "split_dir"):
        value = cfg["benchmark"][key]
        if value is not None:
            cfg["benchmark"][key] = str((base / Path(value).expanduser()).resolve())
    for options in cfg["benchmark"]["models"].values():
        value = options.get("weights")
        if value is not None:
            options["weights"] = str((base / Path(value).expanduser()).resolve())
    if cfg["samples"]["catalog"] is not None:
        val = cfg["samples"]["catalog"]
        if not isinstance(val, str):
            raise PipelineError("samples.catalog: строка пути или null")
        cfg["samples"]["catalog"] = str((base / Path(val).expanduser()).resolve())
    return cfg


def resource(name: str):
    return files("document_vos.resources").joinpath(name)
