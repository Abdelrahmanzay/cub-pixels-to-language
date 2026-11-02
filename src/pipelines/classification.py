"""High-level classification workflows used by the CLI:
train -> evaluate -> plots, standalone evaluation, image/folder prediction,
Grad-CAM, and run comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import save_config
from src.data.datasets import CubClassificationDataset, build_classification_data, load_metadata
from src.data.transforms import build_eval_transform, get_normalization
from src.evaluation.metrics import (classification_metrics, confusion, most_confused_pairs,
                                    per_class_accuracy, predict_loader)
from src.models.classifier import build_classifier, load_classifier
from src.tracking.trackers import build_tracker
from src.training.trainer import Trainer
from src.utils import write_markdown as _write_markdown
from src.utils import count_parameters, describe_device, get_device, load_json, save_json, seed_everything
from src.visualization import plots
from src.visualization.gradcam import GradCAM, gradcam_grid


def make_run_dir(cfg: dict) -> tuple[Path, str]:
    name = cfg.get("run_name") or f"{cfg['model']['encoder']}"
    name = f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg["output_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, name


def train_classification(cfg: dict) -> dict:
    seed_everything(int(cfg["seed"]))
    device = get_device()
    if device.type == "cpu" and not cfg["train"].get("allow_cpu", False):
        raise SystemExit(
            "No CUDA GPU visible to PyTorch -> refusing to train on CPU.\n"
            "Fix: run `python -m src.cli doctor`, and install the CUDA build of PyTorch, e.g.\n"
            "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126\n"
            "(For a quick CPU smoke test add: --set train.allow_cpu=true)")
    run_dir, run_name = make_run_dir(cfg)
    save_config(cfg, run_dir / "config.yaml")
    data = build_classification_data(cfg)
    model = build_classifier(cfg, len(data.class_names))
    total, trainable = count_parameters(model)
    print(f"Run: {run_name} | device: {describe_device(device)} | params: {total / 1e6:.2f}M "
          f"(trainable {trainable / 1e6:.2f}M) | train/val/test: "
          f"{len(data.train.dataset)}/{len(data.val.dataset)}/{len(data.test.dataset)}")
    tracker = build_tracker(cfg, run_dir, run_name)
    tracker.log_summary({"params_total": total, "params_trainable": trainable, "device": describe_device(device)})
    trainer = Trainer(model, data, cfg, tracker, run_dir, device)
    result = trainer.fit()
    plots.plot_training_curves(result["history"], run_dir / "curves.png", title=run_name)
    tracker.log_image("curves", run_dir / "curves.png")
    save_json(result, run_dir / "train_result.json")

    if cfg["train"].get("evaluate_test", True):
        test = evaluate_checkpoint(trainer.best_ckpt, split="test", out_dir=run_dir, data=data)
        tracker.log_summary({f"test_{k}": v for k, v in test["metrics"].items()})
        if cfg["train"].get("tta", True):
            tta = evaluate_checkpoint(trainer.best_ckpt, split="test", out_dir=run_dir, data=data, tta=True)
            tracker.log_summary({f"test_tta_{k}": v for k, v in tta["metrics"].items()})
            result["test_metrics_tta"] = tta["metrics"]
        for img in ("confusion_matrix.png", "confused_pairs.png", "per_class_accuracy.png"):
            if (run_dir / img).exists():
                tracker.log_image(img.replace(".png", ""), run_dir / img)
        result["test_metrics"] = test["metrics"]
    tracker.finish()
    print(f"Done. Outputs in {run_dir}")
    result["run_dir"] = str(run_dir)
    return result


def evaluate_checkpoint(ckpt_path, split: str = "test", out_dir=None, data=None, batch_size: int = 64,
                        num_workers: int = 2, tta: bool = False) -> dict:
    device = get_device()
    model, ckpt = load_classifier(ckpt_path, device)
    cfg = ckpt["config"]
    class_names = ckpt["class_names"]
    if data is None:
        cfg = {**cfg, "data": {**cfg["data"], "batch_size": batch_size, "num_workers": num_workers}}
        data = build_classification_data(cfg)
    loader = {"train": data.train, "val": data.val, "test": data.test}[split]
    if split == "train":  # evaluate train images WITHOUT augmentation
        loader = torch.utils.data.DataLoader(
            CubClassificationDataset(data.frames["train"], cfg["data"]["root"],
                                     build_eval_transform(cfg["data"]["image_size"], data.mean, data.std),
                                     **data.crop),
            batch_size=loader.batch_size, num_workers=num_workers)
    logits, labels, idx = predict_loader(model, loader, device, tta=tta)
    metrics = classification_metrics(logits, labels, len(class_names))
    if tta:  # only the headline metrics; the plain run already wrote the detailed files
        out_dir = Path(out_dir or Path(ckpt_path).parent)
        save_json({"split": split, "checkpoint": str(ckpt_path), "tta": "hflip", "metrics": metrics},
                  out_dir / f"{split}_metrics_tta.json")
        print(f"[{split} +TTA] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return {"metrics": metrics}
    cm = confusion(logits, labels, len(class_names))
    pairs = most_confused_pairs(cm, class_names)
    per_class = per_class_accuracy(cm, class_names)
    out_dir = Path(out_dir or Path(ckpt_path).parent)
    probs = logits.softmax(1)
    frame = data.frames[split]
    pd.DataFrame({"row": idx.numpy(), "path": frame["path"].values[idx.numpy()],
                  "label": labels.numpy(), "pred": probs.argmax(1).numpy(),
                  "confidence": probs.max(1).values.numpy()}).to_csv(out_dir / f"predictions_{split}.csv", index=False)
    save_json({"split": split, "checkpoint": str(ckpt_path), "metrics": metrics,
               "most_confused_pairs": pairs, "per_class_accuracy": per_class},
              out_dir / f"{split}_metrics.json")
    np.save(out_dir / f"confusion_{split}.npy", cm)
    if split == "test":
        plots.plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
        plots.plot_confused_pairs(pairs, out_dir / "confused_pairs.png")
        plots.plot_per_class_accuracy(per_class, out_dir / "per_class_accuracy.png")
    print(f"[{split}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return {"metrics": metrics, "most_confused_pairs": pairs}


@torch.no_grad()
def predict_image(ckpt_path, image_path, topk: int = 5) -> list[dict]:
    device = get_device()
    model, ckpt = load_classifier(ckpt_path, device)
    cfg = ckpt["config"]
    mean, std = get_normalization(cfg["data"]["normalization"], cfg["data"]["root"])
    tf = build_eval_transform(cfg["data"]["image_size"], mean, std)
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0].cpu()
    top = probs.topk(topk)
    out = [{"class": ckpt["class_names"][i], "probability": float(p)} for p, i in zip(top.values, top.indices)]
    for r in out:
        print(f"{r['probability']:.3f}  {r['class']}")
    return out


@torch.no_grad()
def predict_folder(ckpt_path, folder, out_csv, topk: int = 5) -> pd.DataFrame:
    """Predict direct JPG/JPEG/PNG children in filename order, in batches of 32.

    CSV columns are file, top1_class, top1_probability, and topk_classes (a JSON
    list of class names ranked by probability). Empty folders write a header.
    As with predict_image, preprocessing uses the checkpoint's normalization and
    evaluation transform. CUB normalization requires its saved stats.json.
    """
    if topk < 1:
        raise ValueError("topk must be at least 1")
    files = sorted(path for path in Path(folder).iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    device = get_device()
    model, ckpt = load_classifier(ckpt_path, device)
    cfg = ckpt["config"]
    class_names = ckpt["class_names"]
    mean, std = get_normalization(cfg["data"]["normalization"], cfg["data"]["root"])
    tf = build_eval_transform(cfg["data"]["image_size"], mean, std)
    k = min(topk, len(class_names))

    rows = []
    for start in range(0, len(files), 32):
        batch_files = files[start:start + 32]
        images = []
        for path in batch_files:
            with Image.open(path) as image:
                images.append(tf(image.convert("RGB")))
        x = torch.stack(images).to(device)
        top = model(x).softmax(1).topk(k, dim=1)
        for path, indices, probabilities in zip(batch_files, top.indices.cpu(), top.values.cpu()):
            names = [class_names[i] for i in indices.tolist()]
            rows.append({"file": path.name, "top1_class": names[0],
                         "top1_probability": float(probabilities[0]),
                         "topk_classes": json.dumps(names, ensure_ascii=False)})

    frame = pd.DataFrame(rows, columns=["file", "top1_class", "top1_probability", "topk_classes"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"saved {len(frame)} predictions to {out_csv}")
    return frame


def run_gradcam(ckpt_paths: list[str], out_dir: str, n_correct: int = 6, n_wrong: int = 6,
                seed: int = 0, names: list[str] | None = None) -> list[Path]:
    """Grad-CAM for the same test images across several models. Images are chosen
    from the FIRST model's predictions: some it gets right and some it gets wrong."""
    device = get_device()
    loaded = [load_classifier(p, device) for p in ckpt_paths]
    names = names or [Path(p).parent.name.split("_2")[0] for p in ckpt_paths]
    cfg0 = loaded[0][1]["config"]
    class_names = loaded[0][1]["class_names"]
    root = cfg0["data"]["root"]
    df = load_metadata(root)
    if "drop" in df.columns:
        df = df[~df["drop"].astype(bool)]
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    mean, std = get_normalization("imagenet", root)  # display only

    # choose images using the first model's saved predictions if present, else predict
    pred_file = Path(ckpt_paths[0]).parent / "predictions_test.csv"
    if not pred_file.exists():
        evaluate_checkpoint(ckpt_paths[0], "test")
    preds = pd.read_csv(pred_file)
    rng = np.random.default_rng(seed)
    correct = preds[preds.label == preds.pred]
    wrong = preds[preds.label != preds.pred]
    sel = {"correct": correct.iloc[rng.permutation(len(correct))[:n_correct]],
           "wrong": wrong.sort_values("confidence", ascending=False).head(n_wrong)}  # confident mistakes

    outputs = []
    for kind, rows in sel.items():
        if len(rows) == 0:
            continue
        model_outs, display = [], None
        for (model, ckpt), name in zip(loaded, names):
            c = ckpt["config"]
            m_mean, m_std = get_normalization(c["data"]["normalization"], root)
            tf = build_eval_transform(c["data"]["image_size"], m_mean, m_std)
            crop = dict(crop=c["data"].get("crop", "none"), bbox_margin=float(c["data"].get("bbox_margin", 0.15)))
            ds = CubClassificationDataset(test_df.iloc[rows["row"].values], root, tf, **crop)
            x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
            cam = GradCAM(model, model.encoder.gradcam_layer)
            cams, logits = cam(x)
            cam.remove()
            probs = logits.softmax(1)
            model_outs.append({"name": name, "cams": cams, "preds": probs.argmax(1).tolist(),
                               "conf": probs.max(1).values.tolist()})
            if display is None:
                dtf = build_eval_transform(c["data"]["image_size"], mean, std)
                dds = CubClassificationDataset(test_df.iloc[rows["row"].values], root, dtf, **crop)
                display = torch.stack([dds[i][0] for i in range(len(dds))])
        path = Path(out_dir) / f"gradcam_{kind}.png"
        gradcam_grid(display, rows["label"].tolist(), model_outs, class_names, mean, std, path,
                     title=f"Grad-CAM — images the first model got {kind.upper()}")
        outputs.append(path)
        print(f"saved {path}")
    return outputs


def compare_runs(run_dirs: list[str], out_dir: str) -> pd.DataFrame:
    rows, histories = [], {}
    for rd in run_dirs:
        rd = Path(rd)
        if not (rd / "test_metrics.json").exists():
            print(f"skip {rd} (no test_metrics.json)")
            continue
        m = load_json(rd / "test_metrics.json")["metrics"]
        tr = load_json(rd / "train_result.json")
        summ = load_json(rd / "summary.json") if (rd / "summary.json").exists() else {}
        name = rd.name.rsplit("_", 1)[0]
        tta_acc = load_json(rd / "test_metrics_tta.json")["metrics"]["accuracy"] if (rd / "test_metrics_tta.json").exists() else None
        rows.append({"run": name, **m, "accuracy_tta": tta_acc, "best_val_acc": tr["best_val_acc"], "epochs": tr["epochs_ran"],
                     "train_time_min": tr["train_time_s"] / 60, "params_M": summ.get("params_total", 0) / 1e6})
        histories[name] = tr["history"]
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    _write_markdown(df, out / "comparison.md")
    plots.plot_model_comparison(rows, out / "comparison.png")
    plots.plot_val_curves_overlay(histories, out / "val_curves.png")
    print(df.to_string(index=False))
    return df
