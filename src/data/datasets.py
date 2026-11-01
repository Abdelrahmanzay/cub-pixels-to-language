"""PyTorch datasets and dataloader construction (reads the prepared copy)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import build_eval_transform, build_train_transform, get_normalization


def load_metadata(root: str | Path) -> pd.DataFrame:
    path = Path(root) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.cli prepare` first.")
    return pd.read_csv(path)


def load_class_names(root: str | Path) -> list[str]:
    with open(Path(root) / "class_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


def crop_to_bird(img: Image.Image, row, margin: float = 0.15) -> Image.Image:
    """Crop the image to the (square, padded) bird bounding box.

    The box is given in original-image pixels, so it is rescaled to the cached image size.
    It is expanded by ``margin`` on every side and made square (so a later centre crop does not
    cut off long tails or wings), then clipped to the image."""
    if pd.isna(row.get("bbox_x", float("nan"))):
        return img
    s = img.width / float(row["orig_width"])
    x, y, w, h = (float(row[k]) * s for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + 2 * margin)
    left, top = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
    right, bottom = min(float(img.width), cx + side / 2), min(float(img.height), cy + side / 2)
    return img.crop((int(left), int(top), int(round(right)), int(round(bottom))))


class CubClassificationDataset(Dataset):
    """Returns (image_tensor, label, row_index). ``crop="bbox"`` crops every image to the bird."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform, crop: str = "none",
                 bbox_margin: float = 0.15):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.crop = crop
        self.bbox_margin = bbox_margin

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.crop == "bbox":
            img = crop_to_bird(img, row, self.bbox_margin)
        return self.transform(img), int(row["label"]), i


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    mean: tuple
    std: tuple
    frames: dict
    crop: dict


def build_classification_data(cfg: dict) -> DataBundle:
    dcfg = cfg["data"]
    root = dcfg["root"]
    df = load_metadata(root)
    mean, std = get_normalization(dcfg["normalization"], root)
    size = dcfg["image_size"]
    train_tf = build_train_transform(dcfg["augmentation"], size, mean, std)
    eval_tf = build_eval_transform(size, mean, std)

    if "drop" in df.columns:  # duplicates / leaks found by `prepare` (never in test)
        df = df[~df["drop"].astype(bool)]
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if dcfg.get("train_on_val", False):  # final retrain: use train + val for the gradient updates
        frames["train"] = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    crop = dict(crop=dcfg.get("crop", "none"), bbox_margin=float(dcfg.get("bbox_margin", 0.15)))
    nw = int(dcfg["num_workers"])
    common = dict(batch_size=int(dcfg["batch_size"]), num_workers=nw,
                  pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    train = DataLoader(CubClassificationDataset(frames["train"], root, train_tf, **crop),
                       shuffle=True, drop_last=True, **common)
    val = DataLoader(CubClassificationDataset(frames["val"], root, eval_tf, **crop), shuffle=False, **common)
    test = DataLoader(CubClassificationDataset(frames["test"], root, eval_tf, **crop), shuffle=False, **common)
    return DataBundle(train, val, test, load_class_names(root), mean, std, frames, crop)
