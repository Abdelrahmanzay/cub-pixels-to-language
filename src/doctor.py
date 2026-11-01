"""`python -m src.cli doctor`: verify a machine is ready before starting long runs."""
from __future__ import annotations

import platform
import sys
from pathlib import Path


def run_doctor(data_root: str = "data/processed") -> bool:
    ok = True
    print(f"python      : {sys.version.split()[0]} ({platform.system()})")
    try:
        import torch
        print(f"torch       : {torch.__version__} (built for CUDA {torch.version.cuda})")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            vram = p.total_memory / 1024 ** 3
            print(f"GPU         : OK  {p.name}, {vram:.1f} GB VRAM")
            print(f"batch size  : transfer configs use 32; "
                  f"{'OK' if vram >= 5.5 else 'use --set data.batch_size=16'}")
        else:
            ok = False
            print("GPU         : NOT FOUND  -> training will refuse to start.")
            if torch.version.cuda is None:
                print("              You have the CPU-only PyTorch. Install the CUDA build:")
                print("              pip install --force-reinstall torch torchvision "
                      "--index-url https://download.pytorch.org/whl/cu126")
            else:
                print("              CUDA PyTorch is installed but no GPU is visible: update the NVIDIA "
                      "driver / on Kaggle enable GPU in Settings -> Accelerator.")
    except ImportError:
        ok = False
        print("torch       : NOT INSTALLED")
    try:
        import wandb  # noqa: F401
        from src.utils import wandb_has_credentials
        logged = wandb_has_credentials()
        print(f"wandb       : {'logged in' if logged else 'NOT logged in -> run `wandb login` (runs would be offline)'}")
    except Exception:
        print("wandb       : not installed (pip install wandb)")
    meta = Path(data_root) / "metadata.csv"
    print(f"data        : {'prepared' if meta.exists() else 'NOT prepared -> run `python -m src.cli prepare`'}")
    if not meta.exists():
        ok = False
    print("READY" if ok else "NOT READY (see above)")
    return ok
