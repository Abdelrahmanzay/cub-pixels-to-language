"""Small shared helpers: seeding, device selection, timing."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda ({props.name}, {props.total_memory / 1024**3:.1f} GB)"
    return device.type


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    yield result
    result["seconds"] = time.perf_counter() - start


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def wandb_has_credentials() -> bool:
    """True if a W&B API key is available (env var or ~/.netrc from `wandb login`)."""
    import os
    from pathlib import Path as _P
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = _P(os.environ.get("NETRC", _P.home() / ("_netrc" if os.name == "nt" else ".netrc")))
    return netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore")
