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


class CubClassificationDataset(Dataset):
    """Returns (image_tensor, label, row_index)."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
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
    nw = int(dcfg["num_workers"])
    common = dict(batch_size=int(dcfg["batch_size"]), num_workers=nw,
                  pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0)
    train = DataLoader(CubClassificationDataset(frames["train"], root, train_tf),
                       shuffle=True, drop_last=True, **common)
    val = DataLoader(CubClassificationDataset(frames["val"], root, eval_tf), shuffle=False, **common)
    test = DataLoader(CubClassificationDataset(frames["test"], root, eval_tf), shuffle=False, **common)
    return DataBundle(train, val, test, load_class_names(root), mean, std, frames)
