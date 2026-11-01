"""Command line entry point.

    python -m src.cli prepare
    python -m src.cli eda
    python -m src.cli train    --task classification --config configs/cls_resnet50_ft.yaml [--set train.lr=0.0003 ...]
    python -m src.cli evaluate --task classification --checkpoint outputs/runs/<run>/best.pt [--split test]
    python -m src.cli predict  --task classification --checkpoint <best.pt> --image path/to/bird.jpg
    python -m src.cli tune     --config configs/tune_resnet50.yaml --method grid|random|optuna
    python -m src.cli compare-tuning --dirs outputs/tuning/grid_* outputs/tuning/random_*
    python -m src.cli gradcam  --checkpoints <scratch best.pt> <transfer best.pt> --names scratch resnet50
    python -m src.cli compare  --runs outputs/runs/*
"""
from __future__ import annotations

import argparse
import sys

TASKS = ("classification", "colorization", "caption")


def _milestone2(task: str):
    sys.exit(f"Task '{task}' is part of Milestone 2 and is not implemented yet.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.cli", description="CUB: pixels to language")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="download CUB from Hugging Face and build the local cache + splits")
    s.add_argument("--out", default="data/processed")
    s.add_argument("--val-fraction", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--limit", type=int, default=None, help="only N images per split (debugging)")

    s = sub.add_parser("eda", help="exploratory data analysis (figures + summary)")
    s.add_argument("--data", default="data/processed")
    s.add_argument("--out", default="outputs/eda")

    s = sub.add_parser("train", help="train a model")
    s.add_argument("--task", choices=TASKS, default="classification")
    s.add_argument("--config", required=True)
    s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")

    s = sub.add_parser("evaluate", help="evaluate a checkpoint on a split")
    s.add_argument("--task", choices=TASKS, default="classification")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--split", choices=("train", "val", "test"), default="test")
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--num-workers", type=int, default=2)

    s = sub.add_parser("predict", help="run inference on one image")
    s.add_argument("--task", choices=TASKS, default="classification")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--topk", type=int, default=5)

    s = sub.add_parser("tune", help="hyper-parameter search")
    s.add_argument("--config", required=True)
    s.add_argument("--method", choices=("grid", "random", "optuna"), required=True)
    s.add_argument("--trials", type=int, default=None)
    s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

    s = sub.add_parser("compare-tuning", help="compare tuning methods")
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--out", default="outputs/tuning/comparison")

    s = sub.add_parser("gradcam", help="Grad-CAM on correct and wrong test predictions")
    s.add_argument("--checkpoints", nargs="+", required=True)
    s.add_argument("--names", nargs="*", default=None)
    s.add_argument("--n-correct", type=int, default=6)
    s.add_argument("--n-wrong", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="outputs/gradcam")

    s = sub.add_parser("compare", help="compare finished training runs (table + plots)")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--out", default="outputs/comparison")

    a = p.parse_args(argv)

    if a.command == "prepare":
        from src.data.prepare import prepare_dataset
        prepare_dataset(a.out, a.val_fraction, a.seed, a.limit)
    elif a.command == "eda":
        from src.data.eda import run_eda
        run_eda(a.data, a.out)
    elif a.command == "train":
        if a.task != "classification":
            _milestone2(a.task)
        from src.config import load_config
        from src.pipelines.classification import train_classification
        train_classification(load_config(a.config, a.set))
    elif a.command == "evaluate":
        if a.task != "classification":
            _milestone2(a.task)
        from src.pipelines.classification import evaluate_checkpoint
        evaluate_checkpoint(a.checkpoint, a.split, batch_size=a.batch_size, num_workers=a.num_workers)
    elif a.command == "predict":
        if a.task != "classification":
            _milestone2(a.task)
        from src.pipelines.classification import predict_image
        predict_image(a.checkpoint, a.image, a.topk)
    elif a.command == "tune":
        from src.training.tuning import run_search
        run_search(a.config, a.method, a.set, a.trials)
    elif a.command == "compare-tuning":
        from src.training.tuning import compare_searches
        compare_searches(a.dirs, a.out)
    elif a.command == "gradcam":
        from src.pipelines.classification import run_gradcam
        run_gradcam(a.checkpoints, a.out, a.n_correct, a.n_wrong, a.seed, a.names)
    elif a.command == "compare":
        from src.pipelines.classification import compare_runs
        compare_runs(a.runs, a.out)


if __name__ == "__main__":
    main()
