---
title: "From Pixels to Language --- Milestone 1 Discussion Guide"
subtitle: "How the pipeline works, every decision and why, and how to defend it"
geometry: margin=2cm
fontsize: 11pt
colorlinks: true
---

This guide follows the course's AI-usage workflow:
**Think → Plan → Challenge with AI → Refine → Define → Build Incrementally → Review → Validate.**
For each step it says what was decided and why, so that you can explain and defend the decisions
yourself. Read it together with the code: the file names in `code` are in the repo.

# 1. Think: what is the problem?

**The task.** One system, three tasks, one shared visual encoder:

1. **Classification** of 200 bird species (Milestone 1). This trains the encoder.
2. **Colorization** (grayscale → colour) using that encoder inside an image-to-image network (Milestone 2).
3. **Captioning** (image → sentence) with the frozen encoder and a text decoder (Milestone 2).

So the classifier is not the final goal. It is how we *teach the encoder to see birds*, and its
features will be reused twice.

**Why it is hard.** CUB is *fine-grained*: the classes are species of the same family (25 warblers,
21 sparrows, 8 gulls...). They differ in small details (a wing bar, the colour of the bill), and
there are only about 30 training images per class. Two consequences drive almost every decision:

* **Little data per class** means overfitting risk. Transfer learning and regularisation matter more
  than architecture tricks.
* **Small, local differences** mean the model must look at the *bird*, at enough resolution.
  This is why cropping to the bird and higher resolution help.

**What "done" means for M1** (from the PDF): EDA + cleaning + augmentation; a CNN trained from
scratch *and* transfer learning, compared; hyper-parameter tuning with Grid *and* Random search,
compared; experiment tracking; Grad-CAM on correct and wrong predictions; ≥2 metrics; benchmarking;
a CLI; modular code. **There is no required accuracy.** The PDF says scoring lower than published work
is fine if the comparison is fair and explained.

# 2. Plan: the pipeline

The whole project runs through one command line entry point, `python -m src.cli <command>`:

| Step | Command | File(s) | What happens |
|---|---|---|---|
| 0 | `doctor` | `src/doctor.py` | checks GPU, CUDA PyTorch, W&B login, data |
| 1 | `prepare` | `src/data/prepare.py` | download, resize cache, captions, val split, bounding boxes, deduplication |
| 2 | `eda` | `src/data/eda.py` | all data figures + `eda_summary.json` |
| 3 | `train` | `src/pipelines/classification.py`, `src/training/trainer.py` | config → data → model → train → test → plots |
| 4 | `tune` | `src/training/tuning.py` | grid / random / optuna search over hyper-parameters |
| 5 | `evaluate`, `predict` | `src/pipelines/classification.py` | metrics on a split, or top-5 for one image |
| 6 | `gradcam` | `src/visualization/gradcam.py` | heat-maps for correct / wrong predictions |
| 7 | `compare`, `compare-tuning` | pipelines, tuning | tables and comparison plots |

**What happens inside `train`** (worth being able to draw on a whiteboard):

1. `load_config` merges defaults ← YAML file (← its `extends:` parent) ← `--set key=value`.
2. `build_classification_data` reads `metadata.csv`, drops duplicates, builds train/val/test
   `DataLoader`s with the transforms (and the bird crop if `data.crop: bbox`).
3. `build_classifier` = encoder (from the registry) + global-average-pool + dropout + linear head
   (or the cosine head).
4. `Trainer.fit`: for every epoch, the training loop runs Mixup/CutMix if enabled, a forward pass in
   mixed precision, the loss, backward, an optimizer step and a scheduler step. It then evaluates on
   val, logs to W&B, saves `best.pt` when val accuracy improves, and stops early after `patience`
   epochs without improvement.
5. It reloads the **best** checkpoint, evaluates on **test** (plain and with flip test-time
   augmentation), and draws the curves, confusion matrix, most-confused pairs and per-class accuracy.

# 3. Challenge with AI → Refine: the decisions, and the alternatives we rejected

This is the most important part for the discussion. For each decision: **what**, **why**, and
**what we did not do**.

## 3.1 Data and splits

* **A validation split carved from train (10%, stratified, 3 images per class, seed 42).**
  CUB has only train/test. Tuning or early-stopping on the test set would leak the answer and make
  the test score optimistic. *Rejected:* using the test set for model selection.
* **The test set is never modified.** It keeps our numbers comparable with published results
  (same 5,794 images).
* **Duplicate detection with perceptual hashing (pHash).** An exact hash only finds byte-identical
  files. pHash finds the same photo after resizing or re-encoding. We found 8 true duplicate pairs:
  2 val↔test leaks (dropped from val), 2 inside train (one copy dropped), 1 photo labelled as two
  species (in test, reported only), and 3 inside test. The threshold (distance ≤ 2) was chosen by
  *looking* at the candidates: pairs at distance 3--4 were different photos.
* **Resize once and cache** (shorter side 256, or 512 for high resolution). Decoding full JPEGs every
  epoch was the bottleneck on small GPUs.
* **Grayscale images (8)** are kept for classification and flagged for removal in colorization.

## 3.2 What the EDA told us, and what we did with it

| EDA finding | Decision |
|---|---|
| Test set imbalanced (11--30 images/class) | report **macro-F1 / balanced accuracy** too |
| ~27 training images per class | transfer learning + regularisation (label smoothing, weight decay, dropout, augmentation) |
| colour words in 98.7% of captions | **no hue jitter / no random grayscale**: colour identifies species |
| bird covers only **32% of the image** (median; 5% at minimum) | **crop to the bird** (bounding box) |
| fine details | higher resolution helps (224 → 384) |
| one photo labelled as two nightjar species | explains the top confused pair |
| corrupted characters, duplicate/templated captions | caption cleaning; template issue noted for M2 |

**Caption cleaning** (`src/data/text.py`): split on `\n` → normalise unicode, lowercase → replace any
character outside `[a-z0-9 ,.'-]` (including the corrupted U+FFFD) by a space → fix spacing around
punctuation → drop empty lines and duplicates. *Rejected:* automatic spell-correction, because it
"corrects" bird words like *tarsus* or *rectrices*. Rare typos fall out through the vocabulary cut-off
(min frequency 5 keeps 1,243 words and covers 99.5% of tokens).

## 3.3 Augmentation

*Standard* = RandomResizedCrop (scale/position), horizontal flip (pose), rotation ±15° (camera tilt),
brightness/contrast (lighting), RandomErasing (occlusion by branches). **No hue changes**, because they
would create images whose label is wrong. We also implemented *strong*, *RandAugment*,
*TrivialAugment* and *Mixup/CutMix*, and compare them in an ablation (see the results section once
the runs finish). *Strong* sometimes crops the bird out entirely, which is why it is not the default.

## 3.4 Models

* **Scratch CNN (ours, 11M params)**: a ResNet-18-style network. Stem → 4 stages × 2 residual blocks,
  channels 64→128→256→512, each stage halves the resolution. Kaiming initialisation. The last BN of
  each residual branch starts at zero, so every block starts as an identity mapping, which makes early
  training stable.
* **Ablation without skip connections** (same size): it can't even fit the training set (44% train
  acc). This is the *degradation problem* that residual connections solve: gradients get a short
  path through the network.
* **Transfer**: ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, all pretrained on ImageNet.
  * **Linear probe** = freeze the backbone and train only the last layer: 67.6%.
  * **Full fine-tuning** with *discriminative learning rates* (backbone LR = 0.1 × head LR): 82.4%.
    The small backbone LR adapts the ImageNet features gently instead of destroying them.
* **One encoder interface** (`BaseEncoder`): every encoder returns 4 feature maps (strides 4, 8, 16,
  32). *Why:* the M2 U-Net needs these as skip connections, and the M2 caption decoder attends over
  the last 7×7 grid. Encoders are picked by name from a **registry** (factory pattern), so new ones
  need no change to the trainer.

## 3.5 Training choices

* **AdamW**: Adam with decoupled weight decay, a robust default for fine-tuning.
* **Warm-up + cosine schedule**: a small LR at the start protects the pretrained weights from the
  random head's big early gradients; cosine decay ends with small, precise steps.
* **Label smoothing 0.1**: targets are 0.9 / 0.1÷199 instead of 1 / 0, so the model is discouraged
  from becoming over-confident on 27 examples.
* **Mixed precision (FP16)**: about 2× faster on T4 tensor cores, with half the memory.
* **Early stopping on validation accuracy + best checkpoint**: never on test.
* **Batch size 32 for all transfer runs**, so results are comparable across the 6 GB / 8 GB / 16 GB GPUs.

## 3.6 Loss: cross-entropy vs cosine loss (your friend's suggestion)

* **Cross-entropy** pushes the true-class logit up without limit. On tiny datasets the network can
  lower the loss just by making logits *bigger*, which is a form of overfitting.
* **Cosine loss** (Barz & Denzler, WACV 2020, proposed exactly for *small datasets from scratch*):
  normalise the output vector to length 1 and maximise its cosine similarity with the one-hot vector
  of the true class. Only the *direction* matters, so making outputs bigger doesn't help. We use their
  combined version: cosine loss + 0.1 × cross-entropy on the normalised vector (`src/training/losses.py`).

## 3.7 Tuning

* **Grid vs Random, same budget (9 trials), same ranges, 8-epoch proxy runs, selection on val only.**
* **Random won (80.4% vs 79.3%)**: with only one important hyper-parameter (the learning rate), a
  3×3 grid tests just 3 LR values, while random search tests 9 (Bergstra & Bengio 2012). The grid's
  best LR was also on the edge of its range.
* **But the tuned config didn't beat the default at 30 epochs** (82.40% vs 82.38%). Short proxies
  reward settings that learn *fast*, not settings that end *best*. **Refinement:** tuning v2 uses a
  15-epoch proxy, the final model family (ConvNeXt + bird crop) and 4 hyper-parameters (LR, weight
  decay, backbone LR multiplier, label smoothing). It is running now.

## 3.8 Explainability (Grad-CAM)

Grad-CAM weights each channel of the last conv stage by the average gradient of the class score,
sums them, applies ReLU and upsamples: bright = pixels that pushed the score up. What we learned:

* The pretrained models look at the bird (head and breast), with tight maps.
* The scratch CNN looked at the **red feeder** instead of the hummingbird: a *shortcut*. This is also
  a direct argument for **cropping to the bird**.
* Mistakes shared by all models (crow vs raven: separated by size and bill thickness; red-headed vs
  red-bellied woodpecker, where the model looked at the shared barred back) show what the model
  cannot see in one photo.

# 4. Define: the interfaces (what each part promises)

* **Config**: one nested dict. `DEFAULTS` ← YAML ← `--set`. The exact config is saved in every run
  folder and inside every checkpoint, which is what makes results reproducible.
* **Data**: a dataset item is `(image_tensor[3,H,W], label:int, row_index:int)`; the row index lets us
  map predictions back to file names (Grad-CAM, prediction CSVs).
* **Encoder** (`BaseEncoder`): `forward_features(x) → [4 feature maps]`, `out_channels`,
  `gradcam_layer`, `is_pretrained`.
* **Classifier**: `forward(x) → logits[B,200]`; `param_groups()` gives the head and backbone their
  learning rates.
* **Tracker**: `log_metrics`, `log_summary`, `log_image`, `finish`. W&B and local files implement
  the same interface, so the trainer never imports W&B (dependency inversion).
* **Loss**: `criterion(logits, target)` where target is either an int label or soft labels
  (Mixup/CutMix).

# 5. Build incrementally (what the commit history shows)

The commits follow the pipeline order: skeleton/config → data pipeline → EDA → models → training,
tracking and tuning → evaluation, Grad-CAM and CLI → doctor and CPU guard → results → bounding boxes
→ augmentations → cosine loss and Mixup → TTA → experiment configs. Each step was smoke-tested
(1--2 training steps on CPU) before the next.

# 6. Review: how the work was checked

* **Smoke tests** of every command and option on CPU before any GPU time was spent.
* **Every figure was looked at**: this is how we caught that the *strong* augmentation can remove the
  bird, and that pHash distance 3--4 pairs are false positives.
* **A fresh clone from GitHub** was run end to end, to prove the README works.
* **The GPU guard**: the first Kaggle runs silently got no GPU (account not verified). The `doctor`
  check and the "refuse to train on CPU" rule caught it before hours were wasted.

# 7. Validate: are the results trustworthy?

* Test set used **only** for final numbers; all selection on val.
* **Benchmarking on the same split and metric** as published work. At 224 px, our ResNet-50 (82.4%)
  is 2--3 points under published 448 px ResNet-50 baselines (84.5--85.3%). Our ConvNeXt-T at 224 px
  (84.6%) matches them. The state of the art (TransFG, 91.7%) uses ViT-B/16 pretrained on ImageNet-21k,
  448 px input and part-selection.
* **Noise level**: 598 val images, so one image = 0.17% and the standard error is about 1.6 points.
  On the test set (5,794 images) it is about 0.5 points. Don't over-interpret differences smaller than that.
* **Single seed**, which is a stated limitation.

# 8. Numbers to remember (Milestone 1 baseline runs)

| Model | Test acc. | Note |
|---|---|---|
| Scratch, no skips | 36.5% | can't optimise (44% train acc) |
| Scratch CNN | 52.5% | overfits: 96% train vs 51% val |
| ResNet-50 linear probe | 67.6% | frozen ImageNet features |
| EfficientNet-B0 | 79.8% | only 4.3M params |
| ResNet-50 fine-tune | 82.4% | tuned version also 82.4% |
| **ConvNeXt-T fine-tune** | **84.6%** | selected; top-5 96.0% |

The improvement runs (bird crop, 384 px, cosine loss, Mixup/CutMix, augmentation ablation, tuning v2,
TTA) are running; their results will be added here and to the report.

# 9. Likely discussion questions, with short answers

1. **Why a validation split?** To choose models and hyper-parameters without looking at test.
   Otherwise the test score would be optimistic.
2. **Why is transfer learning so much better than scratch?** 27 images per class are not enough to
   learn edges, textures and parts from zero. ImageNet already learned them from 1.2M images.
3. **Linear probe vs fine-tuning?** ImageNet features separate "bird vs dog", not "California vs
   Western gull". Fine-tuning adapts the deep layers to fine-grained cues (+14.8 points).
4. **Why a smaller LR for the backbone?** It is already good. A large LR would destroy the
   pretrained features while the random head is still wrong.
5. **What does the gap between train and val curves mean?** Overfitting: the scratch CNN memorises
   the training set (96%) but generalises to 51%.
6. **Why did removing skip connections hurt, and is it overfitting?** No, it's *under*fitting (44%
   train). Deep plain networks are hard to optimise; skip connections give gradients a short path.
7. **Why random search over grid search?** It tests more distinct values of the one parameter that
   matters (LR) with the same budget.
8. **Why didn't tuning improve the final model?** The 8-epoch proxy favoured fast-converging
   settings; with the full 30-epoch cosine schedule the default caught up. Fix: longer proxy (tuning v2).
9. **Why macro-F1?** The test set is imbalanced; macro averaging gives every species equal weight.
10. **Why is top-5 (96%) so much higher than top-1 (84.6%)?** Most errors are between sibling species,
    so the right answer is usually among the top few.
11. **Why no hue jitter?** Colour is a species feature. Changing hue can turn one species into
    another's colours, which means training on wrong labels.
12. **What is Grad-CAM, and what did it show?** It shows the regions that increase the class score. The
    scratch CNN used the red feeder (a shortcut), while the pretrained models looked at the bird.
13. **Why crop to the bird?** The bird is only about a third of the image; the rest is background that
    can create shortcuts. Cropping gives the model more pixels on the bird at the same input size.
14. **Is using bounding boxes at test time fair?** It uses an annotation that a real user wouldn't
    have. We report it separately, labelled as "with ground-truth boxes", next to the full-image
    results. In M2 / in a deployed system a bird detector would provide the box.
15. **What is the cosine loss, and why might it help?** It only cares about the *direction* of the
    output vector, so the network can't lower the loss by inflating logits. It was designed for small
    datasets trained from scratch.
16. **What is test-time augmentation?** Predict on the image and its mirror, then average the
    probabilities. Cheap, and usually +0.3 to 1 point.
17. **Why did you choose ConvNeXt-T?** Best validation accuracy (that is how it was *selected*),
    best on every test metric, acceptable speed, and its 4 feature maps fit the M2 U-Net.
18. **How would you get higher accuracy?** Higher resolution (448), bird crops, stronger pretraining
    (ImageNet-21k), longer training with Mixup/CutMix, several seeds, part-based attention.

# 10. How to study the code (suggested order, about 2 hours)

1. `src/cli.py`: every command, and where it goes.
2. `src/config.py` + one config file in `configs/`.
3. `src/data/prepare.py`, `src/data/datasets.py` (incl. `crop_to_bird`), `src/data/transforms.py`, `src/data/text.py`.
4. `src/models/encoders.py` (`BaseEncoder`, `ScratchCNN`, registry), `src/models/classifier.py`.
5. `src/training/trainer.py` (`fit`, `_train_one_epoch`), `src/training/losses.py`, `src/training/tuning.py`.
6. `src/evaluation/metrics.py`, `src/visualization/gradcam.py`.
7. `src/pipelines/classification.py`: how everything is glued together.

Then open W&B, pick two runs (scratch vs ConvNeXt) and explain their curves out loud.
That is the best preparation for the discussion.
