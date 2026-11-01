"""Hyper-parameter search: Grid Search, Random Search and (optional) Optuna TPE.

A tuning config looks like::

    base_config: configs/cls_resnet50_ft.yaml
    trials: 9                  # budget for random / optuna (grid = all combinations)
    overrides: [train.epochs=8, train.evaluate_test=false]
    space:
      train.lr:           {type: loguniform, low: 0.0001, high: 0.003, grid: [0.0003, 0.001, 0.003]}
      train.weight_decay: {type: loguniform, low: 0.00001, high: 0.01, grid: [0.00001, 0.0001, 0.001]}

Each trial is a normal short training run (tracked in W&B under group
``tune-<method>``). Selection uses VALIDATION accuracy only; the test set is
never looked at during tuning. Results: ``outputs/tuning/<method>_<time>/``.
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import apply_overrides, load_config, set_by_dotted_key
from src.utils import write_markdown as _write_markdown
from src.utils import save_json


def _grid(space: dict) -> list[dict]:
    keys = list(space)
    values = [space[k]["grid"] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _sample(spec: dict, rng: np.random.Generator):
    t = spec["type"]
    if t == "loguniform":
        return float(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    if t == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if t == "int":
        return int(rng.integers(spec["low"], spec["high"] + 1))
    if t == "choice":
        return spec["values"][int(rng.integers(len(spec["values"])))]
    raise ValueError(f"Unknown space type {t}")


def _random(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: _sample(s, rng) for k, s in space.items()} for _ in range(n)]


def run_search(tune_cfg_path: str, method: str, extra_overrides: list[str] | None = None,
               trials: int | None = None) -> pd.DataFrame:
    from src.pipelines.classification import train_classification  # avoid import cycle

    with open(tune_cfg_path, "r", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)
    space = tcfg["space"]
    n_trials = int(trials or tcfg.get("trials", 9))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path("outputs/tuning") / f"{method}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_overrides = list(tcfg.get("overrides", [])) + list(extra_overrides or [])

    def objective(params: dict, trial_no: int) -> dict:
        cfg = load_config(tcfg["base_config"], base_overrides)
        for k, v in params.items():
            set_by_dotted_key(cfg, k, v)
        cfg["run_name"] = f"tune-{method}-t{trial_no:02d}"
        cfg["output_dir"] = str(out_dir / "runs")
        cfg["tracking"]["group"] = f"tune-{method}-{stamp}"
        cfg["tracking"]["tags"] = ["tuning", method]
        t0 = time.perf_counter()
        res = train_classification(cfg)
        return {"trial": trial_no, **params, "best_val_acc": res["best_val_acc"],
                "best_epoch": res["best_epoch"], "time_s": time.perf_counter() - t0}

    records = []
    t_start = time.perf_counter()
    if method in ("grid", "random"):
        candidates = _grid(space) if method == "grid" else _random(space, n_trials, seed=int(tcfg.get("seed", 0)))
        print(f"{method} search: {len(candidates)} trials")
        for i, params in enumerate(candidates):
            print(f"\n=== {method} trial {i + 1}/{len(candidates)}: {params}")
            records.append(objective(params, i))
            pd.DataFrame(records).to_csv(out_dir / "results.csv", index=False)  # save as we go
    elif method == "optuna":
        import optuna

        def opt_obj(trial):
            params = {}
            for k, s in space.items():
                if s["type"] == "loguniform":
                    params[k] = trial.suggest_float(k, s["low"], s["high"], log=True)
                elif s["type"] == "uniform":
                    params[k] = trial.suggest_float(k, s["low"], s["high"])
                elif s["type"] == "int":
                    params[k] = trial.suggest_int(k, s["low"], s["high"])
                else:
                    params[k] = trial.suggest_categorical(k, s["values"])
            rec = objective(params, trial.number)
            records.append(rec)
            pd.DataFrame(records).to_csv(out_dir / "results.csv", index=False)
            return rec["best_val_acc"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(opt_obj, n_trials=n_trials)
    else:
        raise ValueError("method must be grid | random | optuna")

    df = pd.DataFrame(records)
    df["elapsed_s"] = df["time_s"].cumsum()
    df["best_so_far"] = df["best_val_acc"].cummax()
    df.to_csv(out_dir / "results.csv", index=False)
    best = df.loc[df["best_val_acc"].idxmax()]
    summary = {"method": method, "n_trials": len(df), "best_val_acc": float(best["best_val_acc"]),
               "best_params": {k: float(best[k]) for k in space}, "total_time_s": time.perf_counter() - t_start,
               "time_to_best_s": float(df.loc[df["best_val_acc"].idxmax(), "elapsed_s"]),
               "trials_to_best": int(df["best_val_acc"].idxmax()) + 1, "space": space}
    save_json(summary, out_dir / "summary.json")
    print(f"\nBest ({method}): val_acc={summary['best_val_acc']:.4f} with {summary['best_params']}")
    print(f"Saved to {out_dir}")
    return df


def compare_searches(dirs: list[str], out_dir: str) -> pd.DataFrame:
    """Compare tuning methods: best score, #trials, total time, time-to-best + plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils import load_json

    rows, curves = [], {}
    for d in dirs:
        s = load_json(Path(d) / "summary.json")
        r = pd.read_csv(Path(d) / "results.csv")
        rows.append({k: s[k] for k in ("method", "n_trials", "best_val_acc", "total_time_s",
                                       "time_to_best_s", "trials_to_best")} | {"best_params": s["best_params"]})
        curves[s["method"]] = r
    table = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tuning_comparison.csv", index=False)
    _write_markdown(table, out / "tuning_comparison.md")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, r in curves.items():
        ax[0].plot(range(1, len(r) + 1), r["best_so_far"], marker="o", label=m)
        ax[1].plot(r["elapsed_s"] / 60, r["best_so_far"], marker="o", label=m)
    ax[0].set(xlabel="trial", ylabel="best validation accuracy so far", title="Search efficiency per trial")
    ax[1].set(xlabel="wall-clock time (min)", ylabel="best validation accuracy so far", title="Search efficiency over time")
    for a in ax:
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tuning_progress.png", dpi=120)
    plt.close(fig)

    # where each method sampled in the 2 most important dims (log scale)
    keys = [k for k in curves[next(iter(curves))].columns if k.startswith(("train.", "model.", "data."))][:2]
    if len(keys) == 2:
        fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.5), squeeze=False)
        for a, (m, r) in zip(axes[0], curves.items()):
            sc = a.scatter(r[keys[0]], r[keys[1]], c=r["best_val_acc"], cmap="viridis", s=90, edgecolor="k")
            a.set(xscale="log", yscale="log", xlabel=keys[0], ylabel=keys[1], title=f"{m}: sampled points")
            fig.colorbar(sc, ax=a, label="val acc")
        fig.tight_layout()
        fig.savefig(out / "tuning_samples.png", dpi=120)
        plt.close(fig)
    print(table.to_string(index=False))
    return table
