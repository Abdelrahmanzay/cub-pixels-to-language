"""Image transforms and the augmentation strategy.

Design choice (justified in the EDA): colour is one of the strongest cues for
telling bird species apart (e.g. Scarlet vs Summer Tanager), so we never use
hue jitter or random grayscale. Augmentation targets the real nuisance factors
of CUB instead: scale/position of the bird in the frame (RandomResizedCrop),
left/right pose (horizontal flip), small camera tilt (rotation), lighting
(brightness/contrast) and partial occlusion by branches (RandomErasing).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalization(name: str, data_root: str | Path) -> tuple[tuple, tuple]:
    if name == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if name == "cub":
        stats_path = Path(data_root) / "stats.json"
        with open(stats_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        return tuple(s["mean"]), tuple(s["std"])
    raise ValueError(f"Unknown normalization: {name}")


def _to_tensor(mean, std):
    return [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]


def build_train_transform(level: str, size: int, mean, std) -> T.Compose:
    if level == "none":
        return build_eval_transform(size, mean, std)
    if level == "light":
        aug = [T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
               T.RandomHorizontalFlip()]
        return T.Compose(aug + _to_tensor(mean, std))
    if level == "standard":
        aug = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
               T.RandomHorizontalFlip(),
               T.RandomRotation(15),
               T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)]
        return T.Compose(aug + _to_tensor(mean, std) + [T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    if level == "strong":
        aug = [T.RandomResizedCrop(size, scale=(0.25, 1.0), antialias=True),
               T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
               T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0)]
        return T.Compose(aug + _to_tensor(mean, std) + [T.RandomErasing(p=0.4, scale=(0.02, 0.2))])
    if level == "randaugment":  # automated policy (Cubuk et al., 2020); note: it includes colour ops
        aug = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip(),
               T.RandAugment(num_ops=2, magnitude=9)]
        return T.Compose(aug + _to_tensor(mean, std) + [T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    if level == "trivialaugment":  # parameter-free policy (Mueller & Hutter, 2021)
        aug = [T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), T.RandomHorizontalFlip(),
               T.TrivialAugmentWide()]
        return T.Compose(aug + _to_tensor(mean, std) + [T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
    raise ValueError(f"Unknown augmentation level: {level}")


def build_eval_transform(size: int, mean, std) -> T.Compose:
    return T.Compose([T.Resize(int(round(size * 256 / 224)), antialias=True),
                      T.CenterCrop(size)] + _to_tensor(mean, std))


def denormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device).view(-1, 1, 1)
    s = torch.tensor(std, device=x.device).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)
