"""Exploratory data analysis (Task 1.1).

Reads the prepared copy (``python -m src.cli prepare``) and writes figures
plus ``eda_summary.json`` / ``eda_summary.md`` to ``outputs/eda``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.text import clean_caption, raw_caption_lines, tokenize, word_counts  # noqa: E402
from src.data.transforms import build_train_transform, denormalize  # noqa: E402
from src.utils import save_json  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})
SPLIT_COLORS = {"train": "#4f81bd", "val": "#9bbb59", "test": "#c0504d"}

COLOR_WORDS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
               "grey", "gray", "tan", "buff", "cream", "olive", "rust", "golden", "gold", "maroon",
               "crimson", "scarlet", "beige", "teal", "turquoise", "chestnut", "rufous", "silver", "dark",
               "light", "bright", "colorful", "multicolored", "iridescent", "speckled", "spotted", "striped"}
PART_WORDS = {"bill", "beak", "wing", "wings", "crown", "breast", "belly", "back", "tail", "head", "throat",
              "nape", "eye", "eyes", "eyebrow", "eyering", "superciliary", "feet", "legs", "tarsus",
              "rectrices", "coverts", "secondaries", "primaries", "abdomen", "chest", "neck", "cheek",
              "forehead", "mandible", "body", "feathers", "plumage", "crest", "flank", "flanks", "rump"}


def _save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def sample_grid(df, root, out, n=16, seed=0):
    rows = df[df.split == "train"].sample(n, random_state=seed)
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        ax.imshow(Image.open(root / r.path))
        cap = json.loads(r.captions)[0]
        ax.set_title(f"{r.class_name}\n“{cap[:45]}{'…' if len(cap) > 45 else ''}”", fontsize=8)
        ax.axis("off")
    fig.suptitle("Random training samples: label + first caption")
    return _save(fig, out, "01_sample_grid.png")


def class_distribution(df, out):
    counts = df.groupby(["split", "label"]).size().unstack(0).fillna(0)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7))
    order = counts["test"].sort_values().index
    bottom = np.zeros(len(order))
    for s in ("train", "val", "test"):
        ax[0].bar(range(len(order)), counts.loc[order, s], bottom=bottom, color=SPLIT_COLORS[s], label=s, width=1)
        bottom += counts.loc[order, s].values
    ax[0].set(xlabel="class (sorted by test count)", ylabel="# images", title="Images per class and split")
    ax[0].legend()
    for s in ("train", "test"):
        ax[1].hist(counts[s], bins=range(0, 33), alpha=0.6, color=SPLIT_COLORS[s], label=s)
    ax[1].set(xlabel="# images in a class", ylabel="# classes", title="Distribution of class sizes")
    ax[1].legend()
    fig.tight_layout()
    stats = {s: {"min": int(counts[s].min()), "max": int(counts[s].max()), "mean": float(counts[s].mean()),
                 "imbalance_ratio_max_over_min": float(counts[s].max() / max(1, counts[s].min()))}
             for s in ("train", "val", "test")}
    smallest_test = counts["test"].sort_values().head(8)
    stats["smallest_test_classes"] = {df[df.label == k].class_name.iloc[0]: int(v) for k, v in smallest_test.items()}
    return _save(fig, out, "02_class_distribution.png"), stats


def family_distribution(df, out):
    """Many classes share a family word (25 warblers, 21 sparrows...): the fine-grained challenge."""
    names = pd.Series(sorted(df.class_name.unique()))
    fam = names.str.split().str[-1].value_counts()
    top = fam[fam >= 3]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(top.index, top.values, color="#8064a2")
    ax.set(ylabel="# species (classes)", title="Species per family word — visually similar groups")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _save(fig, out, "03_family_groups.png"), {k: int(v) for k, v in top.items()}


def image_sizes(df, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].scatter(df.orig_width, df.orig_height, s=3, alpha=0.3)
    ax[0].set(xlabel="width (px)", ylabel="height (px)", title="Original image sizes")
    ar = df.orig_width / df.orig_height
    ax[1].hist(ar, bins=60, color="#4f81bd")
    ax[1].set(xlabel="aspect ratio (w/h)", ylabel="# images", title="Aspect ratios")
    short = np.minimum(df.orig_width, df.orig_height)
    ax[2].hist(short, bins=60, color="#4f81bd")
    ax[2].axvline(224, color="r", ls="--", label="model input 224")
    ax[2].set(xlabel="shorter side (px)", ylabel="# images", title="Shorter side")
    ax[2].legend()
    fig.tight_layout()
    stats = {"width": df.orig_width.describe().round(1).to_dict(), "height": df.orig_height.describe().round(1).to_dict(),
             "aspect_ratio": ar.describe().round(3).to_dict(), "n_short_side_below_224": int((short < 224).sum()),
             "n_unique_sizes": int(df.groupby(["orig_width", "orig_height"]).ngroups),
             "modes": df.orig_mode.value_counts().to_dict()}
    return _save(fig, out, "04_image_sizes.png"), stats


def image_statistics(df, root, out):
    """Brightness, contrast, colourfulness (Hasler & Suesstrunk), sharpness (Laplacian variance)."""
    recs = []
    for r in tqdm(df.itertuples(), total=len(df), desc="image stats"):
        a = np.asarray(Image.open(root / r.path).convert("RGB").resize((128, 128)), dtype=np.float32)
        gray = a.mean(2)
        rg = a[..., 0] - a[..., 1]
        yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
        colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        lap = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
        recs.append({"brightness": gray.mean(), "contrast": gray.std(), "colorfulness": colorful,
                     "sharpness": lap.var(), "r": a[..., 0].mean(), "g": a[..., 1].mean(), "b": a[..., 2].mean()})
    st = pd.DataFrame(recs, index=df.index)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, col, title in zip(ax, ("brightness", "contrast", "colorfulness", "sharpness"),
                             ("Brightness (mean gray)", "Contrast (std gray)", "Colourfulness", "Sharpness (Laplacian var, log)")):
        for s in ("train", "test"):
            v = st.loc[df.split == s, col]
            a.hist(np.log10(v + 1) if col == "sharpness" else v, bins=50, alpha=0.55, density=True,
                   color=SPLIT_COLORS[s], label=s)
        a.set(title=title)
        a.legend()
    fig.tight_layout()
    f1 = _save(fig, out, "05_image_statistics.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for c, col in zip(("r", "g", "b"), ("#c0504d", "#9bbb59", "#4f81bd")):
        ax.hist(st[c], bins=50, alpha=0.5, color=col, label=c.upper())
    ax.set(xlabel="mean channel value (0-255)", ylabel="# images", title="Per-image channel means")
    ax.legend()
    f2 = _save(fig, out, "06_channel_means.png")

    # extremes: darkest, brightest, blurriest, least colourful (potential quality issues)
    picks = {"darkest": st.brightness.nsmallest(4).index, "brightest": st.brightness.nlargest(4).index,
             "blurriest": st.sharpness.nsmallest(4).index, "least colourful": st.colorfulness.nsmallest(4).index}
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for row, (k, idx) in enumerate(picks.items()):
        for col, i in enumerate(idx):
            axes[row, col].imshow(Image.open(root / df.loc[i, "path"]))
            axes[row, col].set_title(f"{k}\n{df.loc[i, 'class_name']}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Quality extremes")
    f3 = _save(fig, out, "07_quality_extremes.png")
    stats = {c: st[c].describe().round(2).to_dict() for c in ("brightness", "contrast", "colorfulness", "sharpness")}
    return [f1, f2, f3], stats, st


def grayscale_images(df, root, out):
    g = df[df.is_grayscale]
    if g.empty:
        return None, []
    fig, axes = plt.subplots(1, len(g), figsize=(2.6 * len(g), 3))
    for a, (_, r) in zip(np.atleast_1d(axes), g.iterrows()):
        a.imshow(Image.open(root / r.path))
        a.set_title(f"{r.split}\n{r.class_name}", fontsize=8)
        a.axis("off")
    fig.suptitle("Grayscale images inside the dataset (no colour target for colourization)")
    return _save(fig, out, "08_grayscale_images.png"), g[["split", "class_name", "file_name", "orig_mode"]].to_dict("records")


def duplicates(df, root, out, threshold: int = 4):
    import imagehash
    exact = df[df.pixel_md5.duplicated(keep=False)].sort_values("pixel_md5")
    hashes = []
    for p in tqdm(df.path, desc="perceptual hash"):
        hashes.append(int(str(imagehash.phash(Image.open(root / p))), 16))
    h = np.array(hashes, dtype=np.uint64)
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for s in range(0, len(h), 512):
        x = np.bitwise_xor(h[s:s + 512, None], h[None, :])
        d = pop[x.view(np.uint8)].reshape(x.shape[0], x.shape[1], 8).sum(-1)
        ii, jj = np.where(d <= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
    near = pd.DataFrame(pairs, columns=["i", "j", "hamming"])
    rec = []
    for r in near.itertuples():
        a, b = df.iloc[r.i], df.iloc[r.j]
        rec.append({"a": a.file_name, "a_split": a.split, "b": b.file_name, "b_split": b.split,
                    "same_class": bool(a.label == b.label), "hamming": r.hamming})
    near_df = pd.DataFrame(rec)
    fname = None
    if len(near_df):
        n = len(near)
        rows_n = (n + 1) // 2
        fig, axes = plt.subplots(rows_n, 4, figsize=(12, 2.6 * rows_n))
        axes = np.atleast_2d(axes)
        for k, r in enumerate(near.itertuples()):
            for c, i in enumerate((r.i, r.j)):
                a = axes[k // 2, (k % 2) * 2 + c]
                a.imshow(Image.open(root / df.iloc[i].path))
                a.set_title(f"{df.iloc[i].split}: {df.iloc[i].file_name[:28]}\n(pHash dist {r.hamming})",
                            fontsize=7, color="red" if r.hamming <= 2 else "gray")
        for a in axes.ravel():
            a.axis("off")
        fig.suptitle(f"Near-duplicate candidates (pHash distance <= {threshold}); red = true duplicate (<= 2)")
        fname = _save(fig, out, "09_near_duplicates.png")
    cross = near_df[(near_df.a_split != near_df.b_split)] if len(near_df) else near_df
    stats = {"exact_duplicate_images": int(len(exact)),
             "exact_duplicate_files": exact[["split", "file_name", "class_name"]].to_dict("records"),
             "candidate_pairs_hamming_le_4": int(len(near_df)),
             "true_duplicate_pairs_hamming_le_2": int((near_df.hamming <= 2).sum()) if len(near_df) else 0,
             "dropped_by_prepare": df.loc[df["drop"].astype(bool), ["split", "file_name", "duplicate_of"]].to_dict("records")
             if "drop" in df.columns else [],
             "pairs_across_splits": int(len(cross)),
             "true_duplicates_different_class": int(((~near_df.same_class) & (near_df.hamming <= 2)).sum()) if len(near_df) else 0,
             "near_duplicates": near_df.to_dict("records")[:30]}
    return fname, stats


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
def caption_analysis(df, out):
    raw_lines = df.raw_description.fillna("").apply(raw_caption_lines)
    n_raw = raw_lines.apply(len)
    n_empty = raw_lines.apply(lambda ls: sum(1 for l in ls if not l.strip()))
    n_nonempty = raw_lines.apply(lambda ls: sum(1 for l in ls if l.strip()))
    caps = df.captions.apply(json.loads)
    n_dups = n_nonempty - caps.apply(len)
    raw_text = "\n".join(df.raw_description.fillna(""))
    non_ascii = Counter(ch for ch in raw_text if ord(ch) > 127)
    upper_lines = sum(1 for ls in raw_lines for l in ls if any(c.isupper() for c in l))

    train_caps = [c for cs, s in zip(caps, df.split) if s != "test" for c in cs]
    all_caps = [c for cs in caps for c in cs]
    lens = np.array([len(tokenize(c)) for c in all_caps])
    counts = word_counts(train_caps)
    total_tokens = sum(counts.values())

    # 1) caption length + captions per image
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].hist(lens, bins=range(0, lens.max() + 2), color="#4f81bd")
    ax[0].axvline(np.percentile(lens, 95), color="r", ls="--", label=f"95th pct = {np.percentile(lens, 95):.0f}")
    ax[0].set(xlabel="words per caption", ylabel="# captions", title="Caption length")
    ax[0].legend()
    vc = caps.apply(len).value_counts().sort_index()
    ax[1].bar(vc.index.astype(str), vc.values, color="#9bbb59")
    ax[1].set(xlabel="unique non-empty captions per image", ylabel="# images", title="Captions per image (after cleaning)")
    for i, v in enumerate(vc.values):
        ax[1].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    # vocab size vs min frequency, and token coverage
    thr = [1, 2, 3, 5, 10, 20, 50]
    vs = [sum(1 for c in counts.values() if c >= t) for t in thr]
    cov = [sum(c for c in counts.values() if c >= t) / total_tokens for t in thr]
    ax[2].plot(thr, vs, marker="o", color="#8064a2")
    ax[2].set(xscale="log", xlabel="min word frequency", ylabel="vocabulary size", title="Vocabulary vs frequency cut-off")
    a2 = ax[2].twinx()
    a2.plot(thr, np.array(cov) * 100, marker="s", color="#f79646")
    a2.set_ylabel("% of tokens covered", color="#f79646")
    fig.tight_layout()
    f1 = _save(fig, out, "10_caption_lengths_vocab.png")

    # 2) top words (excluding stop words) + colour / part words
    stop = {"a", "the", "and", "is", "has", "with", "this", "bird", "of", "it", "its", "that", "on", "an", "are",
            "in", "to", "small", "has", "very", "while", "but", "as", "which", "there", "be"}
    top = [(w, c) for w, c in counts.most_common(200) if w not in stop][:30]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar([w for w, _ in top], [c for _, c in top], color=["#c0504d" if w in COLOR_WORDS else
                                                            "#4f81bd" if w in PART_WORDS else "#7f7f7f" for w, _ in top])
    ax.set(ylabel="count (train+val captions)", title="Top content words (red = colour words, blue = body parts)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    f2 = _save(fig, out, "11_top_words.png")

    # 3) typos / rare words
    try:
        from spellchecker import SpellChecker
        sp = SpellChecker()
        rare = [w for w, c in counts.items() if c <= 2 and w.isalpha()]
        unknown = sp.unknown(rare)
        typo_examples = sorted(unknown)[:400]
        corrections = {w: sp.correction(w) for w in typo_examples[:60]}
        n_typo_tokens = sum(counts[w] for w in unknown)
    except ImportError:
        unknown, corrections, n_typo_tokens = set(), {}, 0

    # 4) noisy / uninformative captions
    def informative(c):
        t = set(tokenize(c))
        return bool(t & COLOR_WORDS) or bool(t & PART_WORDS)
    uninformative = [c for c in all_caps if not informative(c)]
    shortest = sorted(set(all_caps), key=lambda c: len(tokenize(c)))[:10]
    longest = sorted(set(all_caps), key=lambda c: -len(tokenize(c)))[:3]
    caption_dup_across_images = pd.Series(all_caps).value_counts()
    common_caps = caption_dup_across_images.head(10).to_dict()

    # 5) words that only appear in test (OOV) -> matters for captioning
    test_vocab = word_counts([c for cs, s in zip(caps, df.split) if s == "test" for c in cs])
    oov = {w: c for w, c in test_vocab.items() if w not in counts}

    stats = {
        "raw_lines_per_description": n_raw.value_counts().to_dict(),
        "empty_lines_total": int(n_empty.sum()),
        "images_with_fewer_than_10_nonempty_lines": int((n_nonempty < 10).sum()),
        "duplicate_captions_within_image_removed": int(n_dups.sum()),
        "images_with_duplicate_captions": int((n_dups > 0).sum()),
        "captions_after_cleaning": len(all_caps),
        "captions_per_image": caps.apply(len).value_counts().sort_index().to_dict(),
        "lines_with_uppercase": int(upper_lines),
        "non_ascii_characters": {repr(k): v for k, v in non_ascii.most_common(20)},
        "words_per_caption": {k: round(float(v), 2) for k, v in pd.Series(lens).describe().items()},
        "words_per_caption_p95": float(np.percentile(lens, 95)),
        "words_per_caption_p99": float(np.percentile(lens, 99)),
        "vocab_size_train_val": len(counts),
        "vocab_size_by_min_freq": dict(zip(map(str, thr), vs)),
        "token_coverage_by_min_freq": {str(t): round(c, 4) for t, c in zip(thr, cov)},
        "singleton_words": sum(1 for c in counts.values() if c == 1),
        "likely_typos_rare_unknown_words": len(unknown),
        "likely_typo_token_share": round(n_typo_tokens / total_tokens, 5),
        "typo_corrections_examples": corrections,
        "uninformative_captions_no_colour_or_part": len(uninformative),
        "uninformative_examples": uninformative[:15],
        "shortest_captions": shortest,
        "longest_captions": longest,
        "most_repeated_captions_across_images": common_caps,
        "test_words_not_in_train_vocab": len(oov),
        "test_oov_examples": sorted(oov, key=lambda w: -oov[w])[:30],
        "colour_word_caption_share": round(np.mean([bool(set(tokenize(c)) & COLOR_WORDS) for c in all_caps]), 4),
        "top_words": top,
    }
    return [f1, f2], stats


def caption_cleaning_examples(df):
    ex = []
    for d in df.raw_description.dropna().head(400):
        for l in d.split("\n"):
            c = clean_caption(l)
            if l.strip() and c != l.strip():
                ex.append({"raw": l, "clean": c})
        if len(ex) >= 12:
            break
    return ex


# --------------------------------------------------------------------------- #
# Augmentation + colourization preview
# --------------------------------------------------------------------------- #
def augmentation_examples(df, root, out, seed=3):
    import torch
    torch.manual_seed(seed)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    rows = df[df.split == "train"].sample(3, random_state=seed)
    levels = ["light", "standard", "strong"]
    fig, axes = plt.subplots(len(rows) * len(levels), 6, figsize=(13, 2.3 * len(rows) * len(levels)))
    k = 0
    for _, r in rows.iterrows():
        img = Image.open(root / r.path).convert("RGB")
        for lvl in levels:
            tf = build_train_transform(lvl, 224, mean, std)
            axes[k, 0].imshow(img)
            axes[k, 0].set_title(f"original\n({lvl})", fontsize=8)
            for c in range(1, 6):
                axes[k, c].imshow(denormalize(tf(img), mean, std).permute(1, 2, 0).numpy())
            k += 1
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Augmentation levels (rows: light / standard / strong); no hue jitter — colour identifies species")
    return _save(fig, out, "12_augmentation_examples.png")


def colorization_pairs(df, root, out, seed=1):
    rows = df[(df.split == "train") & (~df.is_grayscale)].sample(5, random_state=seed)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
    for c, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(root / r.path).convert("RGB")
        axes[0, c].imshow(img.convert("L"), cmap="gray")
        axes[1, c].imshow(img)
        axes[0, c].set_title("input (grayscale)", fontsize=8)
        axes[1, c].set_title("target (colour)", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Colourization pairs built from the data (Milestone 2 preview)")
    return _save(fig, out, "13_colorization_pairs.png")


def bird_crop_analysis(df, root, out, seed=4):
    """How much of each image is bird? + examples of the bounding-box crop used as preprocessing."""
    from src.data.datasets import crop_to_bird
    if "bbox_w" not in df.columns:
        return None, {}
    frac = (df.bbox_w * df.bbox_h) / (df.orig_width * df.orig_height)
    rows = df[df.split == "train"].sample(5, random_state=seed)
    fig = plt.figure(figsize=(15, 7.5))
    ax0 = fig.add_subplot(2, 6, (1, 7))
    ax0.hist(frac, bins=50, color="#4f81bd")
    ax0.axvline(frac.median(), color="r", ls="--", label=f"median {frac.median():.2f}")
    ax0.set(xlabel="bird box area / image area", ylabel="# images", title="How much of the image is bird?")
    ax0.legend()
    for k, (_, r) in enumerate(rows.iterrows()):
        img = Image.open(root / r.path).convert("RGB")
        s = img.width / r.orig_width
        a = fig.add_subplot(2, 6, 2 + k)
        a.imshow(img)
        a.add_patch(plt.Rectangle((r.bbox_x * s, r.bbox_y * s), r.bbox_w * s, r.bbox_h * s, fill=False, ec="red", lw=2))
        a.set_title(r.class_name[:22], fontsize=8)
        a.axis("off")
        b = fig.add_subplot(2, 6, 8 + k)
        b.imshow(crop_to_bird(img, r))
        b.set_title("bird crop (15% margin)", fontsize=8)
        b.axis("off")
    fig.suptitle("Bounding-box crop preprocessing: most of every image is background")
    stats = {"bird_area_fraction": frac.describe().round(3).to_dict(),
             "images_bird_under_20pct": int((frac < 0.2).sum())}
    return _save(fig, out, "14_bird_crop.png"), stats


# --------------------------------------------------------------------------- #
def run_eda(data_root: str = "data/processed", out_dir: str = "outputs/eda") -> dict:
    root, out = Path(data_root), Path(out_dir)
    fig_dir = out / "figures"
    df = pd.read_csv(root / "metadata.csv")
    summary = {"n_images": len(df), "split_sizes": df.split.value_counts().to_dict(),
               "n_classes": int(df.label.nunique()),
               "train_channel_stats": json.load(open(root / "stats.json"))}
    figs = [sample_grid(df, root, fig_dir)]
    f, summary["class_distribution"] = class_distribution(df, fig_dir); figs.append(f)
    f, summary["family_groups"] = family_distribution(df, fig_dir); figs.append(f)
    f, summary["image_sizes"] = image_sizes(df, fig_dir); figs.append(f)
    fs, summary["image_statistics"], _ = image_statistics(df, root, fig_dir); figs += fs
    f, summary["grayscale_images"] = grayscale_images(df, root, fig_dir); figs.append(f)
    f, summary["duplicates"] = duplicates(df, root, fig_dir); figs.append(f)
    fs, summary["captions"] = caption_analysis(df, fig_dir); figs += fs
    summary["caption_cleaning_examples"] = caption_cleaning_examples(df)
    figs.append(augmentation_examples(df, root, fig_dir))
    figs.append(colorization_pairs(df, root, fig_dir))
    f, summary["bird_crop"] = bird_crop_analysis(df, root, fig_dir); figs.append(f)
    summary["figures"] = [f for f in figs if f]
    save_json(summary, out / "eda_summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("captions", "duplicates")}, indent=1, default=str)[:3000])
    print(f"EDA done -> {out}")
    return summary
