# Data pipeline

This describes what `python -m src.cli prepare` builds, how the raw captions
are cleaned, and why the training augmentations were chosen the way they were.
Numbers below come from `outputs/eda/eda_summary.json`.

## What `prepare` does, and why

CUB-200-2011 (11,788 bird photos across 200 species, with 10 captions per
image) is downloaded from Hugging Face as two splits: `train` (5,994 images)
and the official `test` split (5,794 images). Reading and decoding the
original arrow/JPEG files on every training step would be the main
bottleneck on a small GPU, so `prepare` does the expensive work once and
writes a fast-to-read local copy to `data/processed/`:

- **Resize cache**: every image is resized so its shorter side is 256 px
  (never upsampled) and saved as a JPEG. Training then only needs a cheap
  `RandomResizedCrop`/`CenterCrop` to the model's input size instead of
  decoding and resizing a full-resolution image every epoch.
- **Metadata table**: one row per image (`metadata.csv`) records the split,
  class label, file paths, original size/mode, the official bounding box
  (merged in from `resources/cub_bounding_boxes.csv`), a perceptual hash, and
  the cleaned captions.
- **Per-channel stats**: `stats.json` stores the mean/std of the train split's
  pixels (measured: mean `[0.4849, 0.4988, 0.4324]`, std `[0.2295, 0.225,
  0.2627]`), for an alternative to ImageNet normalization.

### Stratified validation split

CUB-200-2011 ships with only train/test; there is no validation split. `prepare`
carves one out of the official train split using a **seeded, stratified**
`train_test_split` (10% of train, `seed=42`), so every class is represented
proportionally in train and val, and every machine that runs `prepare` gets
the exact same split. The result: 5,394 train images, 600 val images, 5,794
test images, with 26-27 images per class in train and exactly 3 per class in
val. The test split is never touched by this step, and its sizes range 11-30
images per class (e.g. "Least Auklet" has only 11) — this imbalance is why
evaluation reports macro-averaged metrics alongside accuracy.

### Duplicate removal (never touches the test set)

Perceptual hashing (`imagehash.phash`) finds near-identical photos (Hamming
distance <= 2, visually confirmed in the EDA) that would otherwise leak
across splits or inflate apparent performance. The policy is asymmetric by
design:

- A train/val image that duplicates a **test** image is **dropped** from
  train/val — keeping it would leak test images into training.
- A duplicate pair entirely inside train/val keeps the first occurrence and
  drops the second.
- The official test split itself is **never modified**, so test-set results
  stay comparable to published CUB benchmarks.

In practice this drops 4 images (e.g. `Pigeon_Guillemot_0081_40339.jpg`,
`Swainson_Warbler_0049_794880.jpg`) out of 2 exact duplicates and 8 true
near-duplicate pairs found overall. Dropped rows aren't deleted from
`metadata.csv`; they're flagged with a `drop` column so the exact exclusion
is auditable, and `build_classification_data` filters them out when loading.

## Caption cleaning

Each image's `description` field is one string holding ~10 raw captions
separated by newlines. `split_description` (in `src/data/text.py`) turns this
into a clean list:

1. **Split on newlines** — each image has exactly 11 raw lines, the last
   always empty (hence `images_with_fewer_than_10_nonempty_lines: 0`).
2. **Unicode-normalize, lowercase, strip whitespace.** The raw captions
   contain a corrupted character (`�`, the Unicode replacement
   character) 1,700 times — a legacy encoding issue in the source data — and
   anything else outside `[a-z0-9\s,.'-]` is also mapped to a space rather
   than silently merging words together (e.g. `"this��bird"` becomes
   `"this bird"`, not `"thisbird"`).
3. **Normalize punctuation spacing.** Captioners wrote commas inconsistently
   — sometimes `"word,word"` with no space, sometimes `"word ,"` with a
   stray leading space. Both are normalized to `"word, word"`
   (`"breast,vent"` -> `"breast, vent"`).
4. **Drop empty lines and exact duplicate captions**, keeping the first
   occurrence's order. 1,306 duplicate captions across 1,183 images are
   removed this way, leaving 116,574 cleaned captions (7-10 per image; a
   small number of images lose captions entirely to deduplication).

Typos are intentionally **not** auto-corrected in the cleaned text: `prepare`
keeps the EDA's typo list (`likely_typos_rare_unknown_words`, 679 words,
only ~0.09% of all tokens) purely as diagnostic information, not a rewrite
step, so the pipeline doesn't risk mangling a correct rare word.

`tokenize` and `Vocabulary` (used by the Milestone 2 captioning model) build
on the cleaned text: words are lowercased word/number tokens plus kept
contractions (`"it's"`), special tokens `<pad>/<bos>/<eos>/<unk>` occupy ids
0-3, and `min_freq` controls which words make the vocabulary (at `min_freq=5`
the vocabulary has 1,243 words and still covers 99.52% of all tokens, so
rare words have little effect on captioning quality).

## Augmentation choices

`build_train_transform` (in `src/data/transforms.py`) offers several levels
(`none`, `light`, `standard`, `strong`, plus `randaugment`/`trivialaugment`
policies) built from `RandomResizedCrop`, horizontal flip, rotation/affine,
brightness/contrast jitter, and `RandomErasing`, scaled up in strength from
`light` to `strong`.

**Why there is no hue jitter (and no random grayscale):** for CUB-200-2011,
colour is one of the strongest cues for telling similar species apart (e.g.
Scarlet vs. Summer Tanager) — `colour_word_caption_share` shows 98.71% of
captions mention a colour, and the top tokens by frequency are overwhelmingly
colour words (`black`, `white`, `brown`, `yellow`, `grey`, `orange`, `red`,
`blue`...). Randomly shifting hue would train the model to ignore the exact
signal captioners (and the classifier) rely on most, so `ColorJitter` is
always called with `hue=0.0`. `brightness`/`contrast`/`saturation` jitter is
kept, since lighting conditions vary across photos but don't change a bird's
actual plumage colour.

Augmentation otherwise targets the real sources of nuisance variation in the
photos rather than appearance: scale and position of the bird in the frame
(`RandomResizedCrop`, since birds occupy very different fractions of the
500x(120-500)px source images), left/right pose (horizontal flip), small
camera tilt (rotation/affine), and partial occlusion by branches or leaves
(`RandomErasing`). The automated policies (`RandAugment`, `TrivialAugment`)
are offered as alternatives for experimentation, but are noted as including
colour operations, so they're used deliberately rather than by default.
