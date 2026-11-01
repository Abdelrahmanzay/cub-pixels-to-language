"""Generic supervised training loop for classification.

The trainer does not know which encoder, dataset or tracker it is given; it only
needs a model, dataloaders and a ``Tracker`` (so new models/backends plug in
without edits here). Features: AdamW/SGD, warmup + cosine LR per step, mixed
precision (AMP) on CUDA, label smoothing, gradient clipping, early stopping on
validation accuracy and best-checkpoint saving.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.evaluation.metrics import classification_metrics, predict_loader
from src.models.classifier import Classifier, save_checkpoint
from src.tracking.trackers import Tracker


def build_optimizer(model: Classifier, tcfg: dict) -> torch.optim.Optimizer:
    groups = model.param_groups(float(tcfg["lr"]), float(tcfg["backbone_lr_mult"]),
                                float(tcfg["weight_decay"]))
    if tcfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups)
    if tcfg["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=float(tcfg["momentum"]), nesterov=True)
    raise ValueError(f"Unknown optimizer {tcfg['optimizer']}")


def build_scheduler(optimizer, tcfg: dict, steps_per_epoch: int):
    total = max(1, int(tcfg["epochs"]) * steps_per_epoch)
    warmup = int(tcfg.get("warmup_epochs", 0)) * steps_per_epoch
    if tcfg.get("scheduler", "cosine") == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model: Classifier, data, cfg: dict, tracker: Tracker, run_dir: str | Path,
                 device: torch.device):
        self.model = model.to(device)
        if device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.data = data
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.tracker = tracker
        self.run_dir = Path(run_dir)
        self.device = device
        self.use_amp = bool(self.tcfg.get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(self.tcfg.get("label_smoothing", 0.0)))
        steps = len(data.train)
        if self.tcfg.get("max_steps_per_epoch"):
            steps = min(steps, int(self.tcfg["max_steps_per_epoch"]))
        self.steps_per_epoch = steps
        self.optimizer = build_optimizer(self.model, self.tcfg)
        self.scheduler = build_scheduler(self.optimizer, self.tcfg, steps)
        self.num_classes = len(data.class_names)
        self.best_ckpt = self.run_dir / "best.pt"

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        pbar = tqdm(self.data.train, total=self.steps_per_epoch, desc=f"epoch {epoch}", leave=False)
        for step, (x, y, _) in enumerate(pbar):
            if step >= self.steps_per_epoch:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.tcfg.get("grad_clip"):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.tcfg["grad_clip"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
            pbar.set_postfix(loss=f"{total_loss / seen:.3f}", acc=f"{correct / seen:.3f}")
        return {"train_loss": total_loss / max(1, seen), "train_acc": correct / max(1, seen)}

    def _evaluate(self, loader) -> dict:
        logits, labels, _ = predict_loader(self.model, loader, self.device, self.use_amp)
        loss = nn.functional.cross_entropy(logits, labels).item()
        m = classification_metrics(logits, labels, self.num_classes)
        return {"loss": loss, **m}

    def fit(self) -> dict:
        best_acc, best_epoch, patience = -1.0, -1, int(self.tcfg.get("early_stopping_patience") or 10**9)
        history = []
        t0 = time.perf_counter()
        for epoch in range(1, int(self.tcfg["epochs"]) + 1):
            te = time.perf_counter()
            train_m = self._train_one_epoch(epoch)
            val_m = self._evaluate(self.data.val)
            rec = {"epoch": epoch, **train_m,
                   "val_loss": val_m["loss"], "val_acc": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
                   "val_top5": val_m["top5_accuracy"],
                   "lr": self.optimizer.param_groups[0]["lr"], "epoch_time_s": time.perf_counter() - te}
            history.append(rec)
            self.tracker.log_metrics(rec, step=epoch)
            improved = val_m["accuracy"] > best_acc
            if improved:
                best_acc, best_epoch = val_m["accuracy"], epoch
                save_checkpoint(self.best_ckpt, self.model, self.cfg, self.data.class_names,
                                {"epoch": epoch, "val_acc": best_acc})
            print(f"[epoch {epoch:3d}] train_loss={rec['train_loss']:.3f} train_acc={rec['train_acc']:.3f} "
                  f"val_loss={rec['val_loss']:.3f} val_acc={rec['val_acc']:.3f} "
                  f"({rec['epoch_time_s']:.1f}s){' *' if improved else ''}")
            if epoch - best_epoch >= patience:
                print(f"Early stopping: no val improvement for {patience} epochs.")
                break
        summary = {"best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_ran": len(history),
                   "train_time_s": time.perf_counter() - t0}
        self.tracker.log_summary(summary)
        self.tracker.log_artifact(self.best_ckpt, "best_checkpoint")
        return {"history": history, **summary}
