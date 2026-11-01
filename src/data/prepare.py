"""Download CUB-200-2011 (with captions) from Hugging Face and build a local,
fast-to-read copy:

    data/processed/
        images/train/00000.jpg ...   (shorter side resized to 256 px)
        images/test/00000.jpg ...
        metadata.csv                 (one row per image, see COLUMNS)
        class_names.json
        stats.json                   (per-channel mean/std of the train split)

Why: decoding the original JPEGs from the HF arrow files and resizing them on
every epoch is the main bottleneck on small GPUs. Doing it once makes every
epoch several times faster. A fixed, seeded, stratified validation split is
stored in ``metadata.csv`` so every machine trains on exactly the same split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.text import split_description

HF_TRAIN = "Multimodal-Fatima/CUB_train"
HF_TEST = "Multimodal-Fatima/CUB_test"
SHORT_SIDE = 256
BBOX_FILE = Path(__file__).resolve().parents[2] / "resources" / "cub_bounding_boxes.csv"


def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    if s >= 1:  # never upsample
        return img
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC)


def prepare_dataset(out_root: str = "data/processed", val_fraction: float = 0.1,
                    seed: int = 42, limit: int | None = None, short_side: int = SHORT_SIDE) -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily (heavy)

    out = Path(out_root)
    rows = []
    class_names = None
    for split_name, hf_name in (("train", HF_TRAIN), ("test", HF_TEST)):
        ds = load_dataset(hf_name, split=split_name)
        class_names = ds.features["label"].names
        img_dir = out / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        n = len(ds) if limit is None else min(limit, len(ds))
        for i in tqdm(range(n), desc=f"prepare {split_name}"):
            rec = ds[i]
            img: Image.Image = rec["image"]
            orig_mode = img.mode
            orig_w, orig_h = img.size
            raw_bytes_md5 = hashlib.md5(np.asarray(img).tobytes()).hexdigest()
            rgb = img.convert("RGB")
            # an RGB file can still be grayscale (R == G == B everywhere)
            arr = np.asarray(rgb.resize((64, 64)), dtype=np.int16)
            is_gray_content = bool(np.abs(arr[..., 0] - arr[..., 1]).max() <= 2 and
                                   np.abs(arr[..., 1] - arr[..., 2]).max() <= 2)
            small = _resize_short_side(rgb, short_side)
            rel = Path("images") / split_name / f"{i:05d}.jpg"
            small.save(out / rel, quality=95)
            caps = split_description(rec["description"])
            rows.append({
                "split": split_name,
                "index": i,
                "id": rec["id"],
                "label": int(rec["label"]),
                "class_name": class_names[int(rec["label"])],
                "file_name": rec["file_name"],
                "path": rel.as_posix(),
                "orig_width": orig_w,
                "orig_height": orig_h,
                "orig_mode": orig_mode,
                "is_grayscale": orig_mode in ("L", "LA", "1") or is_gray_content,
                "pixel_md5": raw_bytes_md5,
                "raw_description": rec["description"],
                "captions": json.dumps(caps),
                "n_captions": len(caps),
            })

    df = pd.DataFrame(rows)
    df = add_bounding_boxes(df)
    df = mark_duplicates(df, out)
    # stratified validation split carved from the official train split
    train_idx = df.index[df["split"] == "train"]
    labels = df.loc[train_idx, "label"]
    stratify = labels if labels.value_counts().min() >= 2 else None
    _, val_idx = train_test_split(train_idx, test_size=val_fraction,
                                  random_state=seed, stratify=stratify)
    df.loc[val_idx, "split"] = "val"
    df.to_csv(out / "metadata.csv", index=False)
    print(f"Duplicates dropped from train/val: {int(df['drop'].sum())} -> "
          f"{df.loc[df['drop'], 'file_name'].tolist()}")
    with open(out / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=1)

    stats = compute_channel_stats(out, df[df["split"] == "train"])
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {len(df)} images to {out} | split sizes: "
          f"{df['split'].value_counts().to_dict()} | stats: {stats}")
    return df


def _hamming_matrix_pairs(hashes: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for s in range(0, len(hashes), 512):
        x = np.bitwise_xor(hashes[s:s + 512, None], hashes[None, :])
        d = pop[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        for i, j in zip(*np.where(d <= threshold)):
            if i + s < j:
                pairs.append((int(i + s), int(j), int(d[i, j])))
    return pairs


def add_bounding_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the official CUB-200-2011 bird bounding boxes (x, y, w, h in ORIGINAL pixel
    coordinates). They are not part of the Hugging Face version; we extracted them once from the
    original release (bounding_boxes.txt + images.txt) into resources/cub_bounding_boxes.csv."""
    if not BBOX_FILE.exists():
        print(f"warning: {BBOX_FILE} missing -> no bounding boxes")
        return df
    boxes = pd.read_csv(BBOX_FILE)
    out = df.merge(boxes, on="file_name", how="left")
    print(f"bounding boxes matched for {out['bbox_x'].notna().sum()}/{len(out)} images")
    return out


DUPLICATE_HAMMING = 2  # pHash distance <= 2 -> same photo (checked visually in the EDA)


def mark_duplicates(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Perceptual-hash every image and record its duplicates.

    Policy (the official TEST split is never modified, so results stay comparable
    with published numbers):
    * train/val image duplicated by a test image  -> dropped (leakage)
    * duplicate pair inside train/val              -> keep the first, drop the second
    (val is carved out of train afterwards, so val-vs-test leaks are covered too.)
    """
    import imagehash

    df = df.copy()
    hashes = [str(imagehash.phash(Image.open(root / p))) for p in tqdm(df["path"], desc="pHash")]
    df["phash"] = hashes
    h = np.array([int(x, 16) for x in hashes], dtype=np.uint64)
    df["drop"] = False
    df["duplicate_of"] = ""
    for i, j, _ in _hamming_matrix_pairs(h, DUPLICATE_HAMMING):
        a, b = df.index[i], df.index[j]
        sa, sb = df.at[a, "split"], df.at[b, "split"]
        if sa == "test" and sb == "test":
            df.at[a, "duplicate_of"] = df.at[b, "file_name"]
            df.at[b, "duplicate_of"] = df.at[a, "file_name"]
            continue
        victim, other = (a, b) if sb == "test" else (b, a)
        df.at[victim, "drop"] = True
        df.at[victim, "duplicate_of"] = df.at[other, "file_name"]
    return df


def compute_channel_stats(root: Path, df: pd.DataFrame) -> dict:
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for p in tqdm(df["path"], desc="channel stats"):
        a = np.asarray(Image.open(root / p).convert("RGB"), dtype=np.float64) / 255.0
        a = a.reshape(-1, 3)
        s += a.sum(0)
        s2 += (a ** 2).sum(0)
        n += a.shape[0]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return {"mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
