"""Print and save CPU encoder measurements without downloading data or weights.

Run from the repository root: python scripts/model_summary.py
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoders import ENCODERS, build_encoder


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def summarize(warmup: int, repeats: int, threads: int) -> str:
    """Benchmark one image at a time, with initialization outside the timed region."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    rows = []
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(42)
            inputs = torch.randn(1, 3, 224, 224)
            for name in sorted(ENCODERS):
                encoder = build_encoder(name, pretrained=False).cpu().eval()
                total = sum(p.numel() for p in encoder.parameters())
                trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                for _ in range(warmup):
                    encoder.forward_features(inputs)
                elapsed_ms = []
                for _ in range(repeats):
                    started = perf_counter()
                    encoder.forward_features(inputs)
                    elapsed_ms.append((perf_counter() - started) * 1000)
                channels = ", ".join(str(c) for c in encoder.out_channels)
                rows.append(f"| {name} | {total:,} | {trainable:,} | {channels} | "
                            f"{median(elapsed_ms):.3f} |")
                del encoder
    finally:
        torch.set_num_threads(previous_threads)

    environment = (f"{platform.system()} {platform.release()}, {platform.machine()}; "
                   f"CPU: {platform.processor() or 'not reported'}; "
                   f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
                   f"torchvision {torchvision.__version__}")
    return "\n".join([
        "# Encoder model summary", "",
        "Generated with `python scripts/model_summary.py`.", "",
        f"Environment: {environment}.", "",
        f"Method: CPU, {threads} intra-op thread(s), eval mode, inference mode, "
        f"one synthetic RGB image of shape `(1, 3, 224, 224)`, {warmup} warm-up "
        f"passes and the median of {repeats} timed `forward_features` passes.", "",
        "Every encoder uses `pretrained=False`; no dataset or weights are downloaded. "
        "Initialization, image generation and disk I/O are excluded from timing. "
        "Latency depends on the machine and system load; it is not training time.", "",
        "Counts cover the encoder only, without a classifier head. Eval/inference mode "
        "does not change `requires_grad`, so all parameters remain trainable here. "
        "A frozen linear probe would instead have zero trainable encoder parameters.", "",
        "| Name | Total params | Trainable params | Channels per stage | CPU forward (ms/image) |",
        "|---|---:|---:|---|---:|",
        *rows, "",
        "All four maps have spatial sizes 56x56, 28x28, 14x14 and 7x7 for this input. "
        "See [models.md](models.md) for architecture and interface explanations.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=10)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model_summary.md")
    args = parser.parse_args()
    report = summarize(args.warmup, args.repeats, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
