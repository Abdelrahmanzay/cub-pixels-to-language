# From Pixels to Language — CUB-200-2011

CSE485/CSE488 Deep Learning final project (Ain Shams University).
One visual encoder, three tasks on Caltech-UCSD Birds-200-2011 with captions:

| Task | Milestone | Status |
|---|---|---|
| 1. Fine-grained classification (200 species) — from scratch vs transfer learning | M1 | ✅ implemented |
| 2. Image colorization (grayscale → colour) using the encoder in a U-Net | M2 | planned |
| 3. Image captioning (frozen encoder → autoregressive decoder) | M2 | planned |

## 1. Setup

Python 3.10–3.12.

```bash
git clone https://github.com/Abdelrahmanzay/cub-pixels-to-language.git
cd cub-pixels-to-language
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# 1) PyTorch with CUDA (pick the line for your system from pytorch.org); for CUDA 12.6 (any recent NVIDIA driver):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2) everything else
pip install -r requirements.txt

# 3) experiment tracking (Weights & Biases) — once per machine
wandb login            # paste the API key from https://wandb.ai/authorize
```

If you do not log in, runs are still saved locally and W&B logs **offline**
(upload later with `wandb sync outputs/runs/<run>/wandb/offline-run-*`).
To disable W&B completely add `--set tracking.mode=disabled`.

On **Kaggle** (GPU + Internet enabled in the notebook settings):
```bash
!git clone https://github.com/Abdelrahmanzay/cub-pixels-to-language.git
%cd cub-pixels-to-language
!pip install -q -r requirements.txt
import os; os.environ["WANDB_API_KEY"] = "<your key>"   # or use Kaggle Secrets
!python -m src.cli prepare
```

## 2. Data

```bash
python -m src.cli doctor      # checks GPU / CUDA PyTorch / W&B / data
python -m src.cli prepare     # ~2 min: downloads from Hugging Face, builds data/processed/
python -m src.cli eda         # ~1 min: figures + summary in outputs/eda/
```

`prepare` downloads `Multimodal-Fatima/CUB_train` / `CUB_test`, resizes every image
once (shorter side 256 px — makes training several times faster), splits the
`description` field into clean captions, carves a **stratified 10 % validation
split** (3 images/class, seed 42) from the official train split, finds duplicate
images with perceptual hashing, and drops train/val copies of duplicates
(the official test split is never changed).

| split | images | use |
|---|---|---|
| train | 5,392 | training |
| val | 598 | early stopping, model selection, hyper-parameter tuning |
| test | 5,794 | final evaluation only (official CUB test split) |

## 3. Training and inference (CLI)

All commands: `python -m src.cli --help`. Every run writes to `outputs/runs/<name>_<time>/`
(config, `best.pt`, curves, metrics, confusion matrix) and to W&B.

```bash
# from-scratch baseline (our own residual CNN)
python -m src.cli train --task classification --config configs/cls_scratch.yaml
# ablation: same CNN without residual connections
python -m src.cli train --task classification --config configs/cls_scratch_plain.yaml
# transfer learning
python -m src.cli train --task classification --config configs/cls_resnet50_probe.yaml   # frozen backbone
python -m src.cli train --task classification --config configs/cls_resnet50_ft.yaml      # full fine-tuning
python -m src.cli train --task classification --config configs/cls_efficientnet_b0_ft.yaml
python -m src.cli train --task classification --config configs/cls_convnext_tiny_ft.yaml

# override any config value from the command line
python -m src.cli train --config configs/cls_resnet50_ft.yaml --set data.batch_size=16 train.lr=0.0005

# hyper-parameter tuning (same 9-trial budget for both methods) + comparison
python -m src.cli tune --config configs/tune_resnet50.yaml --method grid
python -m src.cli tune --config configs/tune_resnet50.yaml --method random
python -m src.cli tune --config configs/tune_resnet50.yaml --method optuna        # optional
python -m src.cli compare-tuning --dirs outputs/tuning/grid_* outputs/tuning/random_*

# evaluation, inference, explainability, comparison
python -m src.cli evaluate --task classification --checkpoint outputs/runs/<run>/best.pt --split test
python -m src.cli predict  --task classification --checkpoint outputs/runs/<run>/best.pt --image path/to/bird.jpg
python -m src.cli gradcam  --checkpoints outputs/runs/<scratch>/best.pt outputs/runs/<resnet50_ft>/best.pt --names scratch resnet50
python -m src.cli compare  --runs outputs/runs/*
```

Trained checkpoints: Google Drive link — *to be added*.

## 4. Milestone 1 run plan (3 GPUs in parallel)

Do this on **every** machine first:

```bash
python -m src.cli doctor      # must print "GPU : OK" and end with READY
python -m src.cli prepare     # once per machine (~2 min)
```

Training refuses to start on CPU (so nobody wastes hours by accident).
All runs go to the shared W&B project `cub-pixels-to-language`.

| Machine | Run, in this order |
|---|---|
| A: RTX 4060 (8 GB) | `train --config configs/cls_scratch.yaml` → `train --config configs/cls_scratch_plain.yaml` |
| B: Kaggle GPU | `tune --config configs/tune_resnet50.yaml --method grid` → `tune --config configs/tune_resnet50.yaml --method random` |
| C: RTX 3060 laptop (6 GB) | `cls_resnet50_probe.yaml` → `cls_resnet50_ft.yaml` → `cls_efficientnet_b0_ft.yaml` → `cls_convnext_tiny_ft.yaml` |

(prefix every command with `python -m src.cli`; machine C: `train --config configs/<file>`)

Then, on one machine with all run folders copied into `outputs/`:

```bash
python -m src.cli compare --runs outputs/runs/*
python -m src.cli compare-tuning --dirs outputs/tuning/grid_* outputs/tuning/random_*
python -m src.cli gradcam --checkpoints outputs/runs/scratch_cnn_*/best.pt outputs/runs/resnet50_ft_*/best.pt --names scratch resnet50-ft
```

## 5. Project structure

```
configs/              experiment + tuning configs (YAML, overridable with --set)
src/
  cli.py              command line entry point
  config.py           config loading / merging / overrides
  data/               prepare (download, cache, split, dedup), text (caption cleaning),
                      transforms (augmentation), datasets, eda
  models/             encoders.py (BaseEncoder interface + registry: scratch CNN, ResNet,
                      EfficientNet, ConvNeXt), classifier.py
  training/           trainer.py (generic loop), tuning.py (grid / random / optuna)
  evaluation/         metrics.py
  visualization/      plots.py, gradcam.py
  tracking/           trackers.py (W&B + local, behind one interface)
  pipelines/          classification.py (train/evaluate/predict/gradcam/compare workflows)
notebooks/            experiments only (not part of the final code)
outputs/              figures and logs (checkpoints are not committed)
```

Design: every encoder implements `BaseEncoder` (`forward_features` → multi-scale
feature maps, `out_channels`, `gradcam_layer`), and is created through a registry,
so the classifier, the M2 U-Net and the M2 caption decoder can swap encoders from the
config alone. The trainer depends only on the model, dataloaders and a `Tracker`
interface, so tracking backends can change without touching the training loop.

## 6. Reproducibility

Fixed seeds (`seed: 42`), a stored validation split (`data/processed/metadata.csv`),
the exact config saved inside every run folder and checkpoint, and the same batch
size (32) for all transfer runs on every machine.
