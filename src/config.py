"""Configuration loading.

Configs are plain YAML files that are deep-merged on top of ``DEFAULTS``.
Any value can also be overridden from the command line with
``--set train.lr=0.0003 data.batch_size=32`` (dotted keys).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "task": "classification",
    "run_name": None,
    "output_dir": "outputs/runs",
    "data": {
        "root": "data/processed",
        "image_size": 224,
        "batch_size": 64,
        "num_workers": 4,
        "normalization": "imagenet",  # imagenet | cub
        "augmentation": "standard",  # none | light | standard | strong | randaugment | trivialaugment
        "crop": "none",  # none | bbox  (crop every image to the bird bounding box)
        "bbox_margin": 0.15,  # padding around the box, as a fraction of its size
        "train_on_val": False,  # final retrain on train + val
    },
    "model": {
        "encoder": "resnet50",
        "pretrained": True,
        "freeze_encoder": False,
        "dropout": 0.2,
        "encoder_kwargs": {},
        "cosine_scale": 16.0,  # only used with train.loss=cosine
    },
    "train": {
        "epochs": 30,
        "optimizer": "adamw",  # adamw | sgd
        "lr": 0.001,
        "backbone_lr_mult": 0.1,  # backbone LR = lr * mult (pretrained models only)
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "scheduler": "cosine",  # cosine | none
        "warmup_epochs": 1,
        "label_smoothing": 0.1,
        "amp": True,
        "grad_clip": None,
        "early_stopping_patience": 10,
        "max_steps_per_epoch": None,  # for quick smoke tests
        "evaluate_test": True,
        "allow_cpu": False,
        "loss": "ce",  # ce | cosine
        "cosine_xent_weight": 0.1,
        "mixup_alpha": 0.0,  # > 0 enables Mixup
        "cutmix_alpha": 0.0,  # > 0 enables CutMix (both > 0: one of them per batch)
        "mix_prob": 1.0,  # fraction of batches that get mixed
        "select": "best_val",  # best_val | last (use "last" when training on train + val)
        "tta": True,  # also report test metrics with horizontal-flip test-time augmentation  # refuse to train on CPU by accident (smoke tests: set true)
    },
    "tracking": {
        "backends": ["local", "wandb"],
        "project": "cub-pixels-to-language",
        "entity": "23p0094-ain-shams-university",
        "mode": "online",  # online | offline | disabled
        "group": None,
        "tags": [],
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(raw: str) -> Any:
    # yaml handles ints, floats, bools, null and lists ("[a, b]")
    try:
        val = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    # PyYAML reads "1e-3" as a string; fix that common case
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_by_dotted_key(cfg: dict, dotted: str, default: Any = None) -> Any:
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, raw = item.split("=", 1)
        set_by_dotted_key(cfg, key.strip(), _parse_value(raw.strip()))
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_merge(cfg, _read_with_bases(path))
    cfg = apply_overrides(cfg, overrides)
    _coerce_numbers(cfg)
    return cfg


def _read_with_bases(path) -> dict:
    """Read a YAML config; ``extends: other.yaml`` (relative to this file) is loaded first,
    so an experiment file only needs the keys it changes."""
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    user_cfg.pop("base_config", None)  # used by tuning configs, not a model config
    parent = user_cfg.pop("extends", None)
    if parent:
        return deep_merge(_read_with_bases(Path(path).parent / parent), user_cfg)
    return user_cfg


def _coerce_numbers(cfg: dict) -> None:
    """Make sure numeric hyper-parameters written as strings ("1e-3") become floats."""
    for section in ("train",):
        for k, v in cfg.get(section, {}).items():
            if isinstance(v, str):
                try:
                    cfg[section][k] = float(v)
                except ValueError:
                    pass


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
