# From Pixels to Language — CUB-200-2011

CSE485/CSE488 Deep Learning final project, Ain Shams University — **Team 16**.
One visual encoder, three tasks on Caltech-UCSD Birds-200-2011 with captions:

| Task | Milestone | Status |
|---|---|---|
| 1. Fine-grained classification (200 species): from scratch vs transfer learning | M1 | ✅ done |
| 2. Image colorization (grayscale → colour) using the encoder in a U-Net | M2 | planned |
| 3. Image captioning (frozen encoder → autoregressive decoder) | M2 | planned |

**Milestone 1 results** (official CUB test split, 5,794 images, selected on the validation set):

| Model | Test accuracy | + flip TTA |
|---|---|---|
| Our CNN from scratch (baseline) | 52.5% | — |
| Our CNN from scratch (bird crop + cosine loss + Mixup/CutMix, 250 epochs) | 66.8% | 67.5% |
| ConvNeXt-Tiny fine-tuned (baseline, 224 px) | 84.6% | — |
| **ConvNeXt-Tiny fine-tuned — final model** (384 px, Mixup/CutMix, tuned) | **87.0%** | **87.3%** |

Full analysis: [`report/Milestone1_Report.pdf`](report/Milestone1_Report.pdf).
Result tables and figures: [`outputs/results/`](outputs/results).
Experiment tracking: W&B project `cub-pixels-to-language`.

## 1. Setup

Python 3.10–3.13.

```bash
git clone https://github.com/Abdelrahmanzay/cub-pixels-to-language.git
cd cub-pixels-to-language
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate

# 1) PyTorch with CUDA (pick the line for your system from pytorch.org), e.g. CUDA 12.6:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 2) everything else
pip install -r requirements.txt
# 3) experiment tracking (Weights & Biases), once per machine
wandb login            # paste the API key from https://wandb.ai/authorize

python -m src.cli doctor   # checks GPU, CUDA PyTorch, W&B login and prepared data
```

Training **refuses to run on CPU** unless you add `--set train.allow_cpu=true` (for quick tests).
Without a W&B login, runs are still saved locally and W&B logs offline; add
`--set tracking.mode=disabled` to switch W&B off.

On **Kaggle** (GPU + Internet enabled): `!git clone ...`, `%cd cub-pixels-to-language`,
`!pip install -q -r requirements.txt`, set `WANDB_API_KEY` from Kaggle Secrets, then run the commands below.

## 2. Data

```bash
python -m src.cli prepare                                         # 256 px cache   -> data/processed/      (baseline configs)
python -m src.cli prepare --out data/processed_hr --short-side 512   # full-res cache -> data/processed_hr/   (configs/improve/*)
python -m src.cli eda                                             # figures + summary -> outputs/eda/
```

`prepare` downloads `Multimodal-Fatima/CUB_train` / `CUB_test` from Hugging Face, resizes every
image once, splits the `description` field into cleaned captions, attaches the official CUB
bounding boxes (`resources/cub_bounding_boxes.csv`), carves a **stratified 10% validation split**
(3 images/class, seed 42) from the official train split, finds duplicate images with perceptual
hashing and drops their train/val copies. The official test split is never modified.

| split | images | used for |
|---|---|---|
| train | 5,392 | training |
| val | 598 | early stopping, model selection, hyper-parameter tuning |
| test | 5,794 | final evaluation only |

## 3. Training and inference (CLI)

All commands: `python -m src.cli --help`. Every run writes config, `best.pt`, curves, metrics,
confusion matrix and test results (plain and with flip test-time augmentation) to
`outputs/runs/<name>_<time>/`, and logs to W&B.

**Final model** (needs the full-resolution cache from section 2):
```bash
python -m src.cli train --task classification --config configs/improve/final_full_ce.yaml
```

**Baseline experiments:**
```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml          # our CNN from scratch
python -m src.cli train --task classification --config configs/cls_scratch_plain.yaml    # ablation: no residual connections
python -m src.cli train --task classification --config configs/cls_resnet50_probe.yaml   # frozen backbone (linear probe)
python -m src.cli train --task classification --config configs/cls_resnet50_ft.yaml      # full fine-tuning
python -m src.cli train --task classification --config configs/cls_efficientnet_b0_ft.yaml
python -m src.cli train --task classification --config configs/cls_convnext_tiny_ft.yaml
```

**Improvement experiments** (`configs/improve/`, one change at a time):

| Config | What it changes |
|---|---|
| `convnext_bbox.yaml` | crop every image to the bird (ground-truth box) |
| `convnext_384.yaml`, `convnext_bbox384.yaml` | 384 px input |
| `convnext_bbox_cosine.yaml` | cosine loss instead of cross-entropy |
| `convnext_bbox_mix.yaml` | Mixup/CutMix |
| `resnet50_aug.yaml` + `--set data.augmentation=none\|light\|standard\|strong\|randaugment\|trivialaugment` | augmentation ablation |
| `scratch_bbox*.yaml`, `scratch_final.yaml` | scratch CNN improvements |
| `final_bbox_ce.yaml`, `final_bbox_cosine.yaml`, `final_full_ce.yaml` | final combined runs |

Any config value can be overridden: `--set data.batch_size=16 train.lr=0.0005`.

**Hyper-parameter tuning:**
```bash
python -m src.cli tune --config configs/tune_resnet50.yaml --method grid      # tuning v1: grid ...
python -m src.cli tune --config configs/tune_resnet50.yaml --method random    # ... vs random, same 9-trial budget
python -m src.cli compare-tuning --dirs outputs/tuning/grid_* outputs/tuning/random_*
python -m src.cli tune --config configs/tune_convnext.yaml --method random    # tuning v2 (15-epoch proxy, 4 parameters)
```

**Evaluation, inference, explainability, comparison:**
```bash
python -m src.cli evaluate --task classification --checkpoint <run>/best.pt --split test --tta
python -m src.cli predict  --task classification --checkpoint <run>/best.pt --image path/to/bird.jpg
python -m src.cli predict  --task classification --checkpoint <run>/best.pt --folder path/to/images --out preds.csv --topk 5
python -m src.cli gradcam  --checkpoints <runA>/best.pt <runB>/best.pt --names modelA modelB
python -m src.cli compare  --runs outputs/runs/*
```

Folder inference reads the `.jpg/.jpeg/.png` files of a folder and writes `file`, `top1_class`,
`top1_probability` and `topk_classes` to a CSV. See [docs/evaluation.md](docs/evaluation.md).

**Trained checkpoints:** [Google Drive folder](https://drive.google.com/drive/folders/1unNqlhEprTFObBeeLp3oaC3JL68MlUSe?usp=sharing) — one zip per Kaggle job; each run folder holds
`best.pt`, its config, metrics and curves. Pass a `best.pt` with `--checkpoint`; its config is stored inside the file.

## 4. Tests

```bash
python -m pytest -q tests     # 54 tests, CPU only, no dataset or downloads needed
```

## 5. Project structure

```
configs/              experiment and tuning configs (YAML; `extends:` inheritance; --set overrides)
  improve/            improvement and final experiments
src/
  cli.py              command line entry point
  config.py           config loading / merging / overrides
  doctor.py           environment check
  data/               prepare (download, cache, split, boxes, dedup), text (captions),
                      transforms (augmentation), datasets (incl. bird crop), eda
  models/             encoders.py (BaseEncoder + registry: scratch CNN, ResNet, EfficientNet,
                      ConvNeXt), classifier.py (linear or cosine head)
  training/           trainer.py, losses.py (cross-entropy, cosine), tuning.py (grid/random/optuna)
  evaluation/         metrics.py (incl. test-time augmentation)
  visualization/      plots.py, gradcam.py
  tracking/           trackers.py (W&B + local files behind one interface)
  pipelines/          classification.py (train / evaluate / predict / gradcam / compare)
tests/                unit tests (pytest)
docs/                 module documentation (data, models, evaluation)
scripts/              model_summary.py (encoder size and speed table)
resources/            official CUB bounding boxes
outputs/eda/          EDA figures and summary
outputs/results/      result tables, curves, Grad-CAM figures of all runs (checkpoints not committed)
report/               Milestone 1 report (PDF + LaTeX sources)
notebooks/            experiments only (not part of the final code)
```

Design: every encoder implements `BaseEncoder` (`forward_features` returns four feature maps
at strides 4/8/16/32, plus `out_channels` and `gradcam_layer`) and is created through a registry,
so the classifier, the Milestone-2 U-Net and the caption decoder can swap encoders from the config
alone. The trainer only depends on the model, the dataloaders and a `Tracker` interface.

## 6. Reproducibility

Fixed seed (42), a stored validation split (`metadata.csv`), the exact configuration saved inside
every run folder and checkpoint, and the same batch size for all transfer runs.
