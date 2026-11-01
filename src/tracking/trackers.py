"""Experiment tracking.

``Tracker`` is a tiny interface; the training code only talks to it, so the
backend (W&B, local files, both) is chosen in the config (``tracking.backends``)
without touching the training loop (dependency inversion).

* ``LocalTracker``  always on: writes config + per-epoch metrics to the run folder,
  so every plot in the report can be regenerated offline.
* ``WandbTracker``  logs the same things to Weights & Biases. If W&B is not
  installed or not logged in, it falls back to offline mode instead of crashing.
"""
from __future__ import annotations

import json
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path


class Tracker(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int | None = None) -> None: ...

    def log_summary(self, summary: dict) -> None: ...

    def log_image(self, key: str, path: str | Path) -> None: ...

    def log_artifact(self, path: str | Path, name: str, kind: str = "model") -> None: ...

    def finish(self) -> None: ...


class LocalTracker(Tracker):
    def __init__(self, run_dir: str | Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"
        self.history_path.write_text("")
        self.summary: dict = {}

    def log_metrics(self, metrics, step=None):
        rec = dict(metrics)
        if step is not None:
            rec["step"] = step
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def log_summary(self, summary):
        self.summary.update(summary)
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)


class WandbTracker(Tracker):
    def __init__(self, run_dir: str | Path, config: dict, run_name: str):
        tcfg = config["tracking"]
        mode = tcfg.get("mode", "online")
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb not installed -> W&B tracking disabled")
            self.run = None
            return
        if mode == "online" and not (os.environ.get("WANDB_API_KEY") or _wandb_logged_in()):
            warnings.warn("Not logged in to W&B (run `wandb login`). Logging OFFLINE; "
                          "upload later with `wandb sync <run_dir>/wandb/offline-run-*`.")
            mode = "offline"
        self.wandb = wandb
        self.run = wandb.init(project=tcfg.get("project"), entity=tcfg.get("entity"),
                              name=run_name, group=tcfg.get("group"), tags=tcfg.get("tags") or None,
                              config=config, mode=mode, dir=str(run_dir), reinit=True)

    def log_metrics(self, metrics, step=None):
        if self.run:
            self.run.log(metrics, step=step)

    def log_summary(self, summary):
        if self.run:
            for k, v in summary.items():
                self.run.summary[k] = v

    def log_image(self, key, path):
        if self.run:
            self.run.log({key: self.wandb.Image(str(path))})

    def log_artifact(self, path, name, kind="model"):
        # we log a *reference* (path) to the checkpoint, not the file itself,
        # to stay inside the free W&B storage quota
        if self.run:
            self.run.summary[f"{name}_path"] = str(path)

    def finish(self):
        if self.run:
            self.run.finish()


def _wandb_logged_in() -> bool:
    try:
        import wandb
        return bool(wandb.api.api_key)
    except Exception:
        return False


class CompositeTracker(Tracker):
    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    def log_metrics(self, metrics, step=None):
        for t in self.trackers:
            t.log_metrics(metrics, step)

    def log_summary(self, summary):
        for t in self.trackers:
            t.log_summary(summary)

    def log_image(self, key, path):
        for t in self.trackers:
            t.log_image(key, path)

    def log_artifact(self, path, name, kind="model"):
        for t in self.trackers:
            t.log_artifact(path, name, kind)

    def finish(self):
        for t in self.trackers:
            t.finish()


def build_tracker(config: dict, run_dir: str | Path, run_name: str) -> Tracker:
    tcfg = config["tracking"]
    backends = tcfg.get("backends", ["local"])
    trackers: list[Tracker] = [LocalTracker(run_dir, config)]
    if "wandb" in backends and tcfg.get("mode", "online") != "disabled":
        trackers.append(WandbTracker(run_dir, config, run_name))
    return CompositeTracker(trackers)
