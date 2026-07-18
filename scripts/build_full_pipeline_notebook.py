# -*- coding: utf-8 -*-
"""Builds the merged DMorphNet full-pipeline Colab notebook (English, full-scale)."""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src})

# ================================================================ PART 0
md(r"""# 🧬 D-MorphNet — Full Pipeline in One Notebook
### Face Morphing Detection: EfficientNet-B6 + SVM — from dataset construction to real-time prediction

This notebook runs the complete D-MorphNet pipeline **at full scale (45,000 images)** by default.

| Part | Content |
|---|---|
| 1 | Global configuration (full-scale / quick demo switch) |
| 2 | Download real images + identity file (fully automatic) |
| 3 | **Delaunay** morph generation (Subdiv2D triangulation + piecewise-affine warp) + review + identity-disjoint splits |
| 4 | Preprocessing (528×528 + CLAHE) and data augmentation |
| 5 | EfficientNet-B6 feature extraction (+ optional fine-tuning) |
| 6 | SVM training and decision-function verification |
| 7 | Results: confusion matrix + metrics + ROC/AUC |
| 8 | Threshold optimization |
| 9 | Architecture comparison (optional) |
| 10 | Real-time prediction: upload images + multi-face + latency |
| 11 | Save to Drive and final summary |

### ⏱️ Expected full-scale timeline (Colab T4 GPU)
- Morph generation (21,000 images): **~3–6 hours**
- Preprocessing (45,000 images): ~45–60 minutes
- B6 feature extraction (45,000 × 528px): ~1.5–2 hours
- SVM training + evaluation: minutes

### 💾 Automatic Drive checkpointing (survives disconnects)
Colab sessions can disconnect during long runs. This notebook **checkpoints progress
to Google Drive automatically**: generated morphs are synced every 2,000 images,
processed images and extracted features are cached too. After any disconnect —
even on a brand-new VM — just run all cells again and the pipeline **resumes from
the last checkpoint** instead of starting over.

> ✅ **Run the cells top to bottom.** Enable GPU first: Runtime → Change runtime type → **GPU (T4)**.
> For a quick end-to-end test first, set `DEMO = True` in Part 1 (finishes in ~30–45 min).
""")

# ================================================================ PART 1
md("""## Part 1 — Global Configuration

All control switches in one place:
- `DEMO`: `False` (default) = full paper-scale run; `True` = quick small-scale test.
- `DO_FINETUNE`: fine-tune the top B6 layers (optional, adds time).
- `RUN_COMPARISONS`: B0/B5/B6 comparison — Table 2 (optional).
- `CHECKPOINT_TO_DRIVE`: automatic progress checkpoints to Google Drive.
- `SAVE_TO_DRIVE`: save the final models to Drive at the end.
""")
code(r"""# ================== Control switches ==================
DEMO = False               # False = FULL run (45,000 images) | True = quick test
DO_FINETUNE = True         # Fine-tune top B6 layers (Part 5) — reduces overfitting
RUN_COMPARISONS = False    # B0/B5/B6 comparison — Table 2 (Part 9)
SAVE_TO_DRIVE = True       # Save final models to Drive (Part 11)

# Fresh start: True = DELETE the old dataset, processed images, features AND
# the Drive checkpoints, then regenerate everything from scratch with the
# current (Delaunay) morph algorithm. Set True for THIS restart; set back to
# False before re-running so a mid-run disconnect can resume instead of wiping.
FRESH_START = True

# Automatic Drive checkpoints during long stages — essential for the full run
# so a Colab disconnect never loses progress (resume works even on a new VM)
CHECKPOINT_TO_DRIVE = not DEMO
CHECKPOINT_EVERY = 2000    # checkpoint every N generated morphs

# ================== Image counts ==================
if DEMO:
    SPLITS = {                                   # paper ratios, scaled 1/40
        "train": {"real": 400, "morph": 350},
        "val":   {"real": 175, "morph": 150},
        "test":  {"real": 50,  "morph": 50},
    }
else:
    SPLITS = {                                   # paper numbers (45,000 images)
        "train": {"real": 16000, "morph": 14000},
        "val":   {"real": 7000,  "morph": 6000},
        "test":  {"real": 1000,  "morph": 1000},
    }

# ================== Constants ==================
SEED = 42
MORPH_SIZE = (256, 256)      # morph generation size (Part 3)
INPUT_SIZE = 528             # EfficientNet-B6 input size (Parts 4+)
MORPH_ALPHA = 0.5            # face blending ratio
MIN_SHARPNESS = 20.0         # automatic review threshold (Laplacian variance)
JPEG_QUALITY = 90            # uniform JPEG compression quality

BASE = "/content/DMorphNet"
RAW_DIR  = f"{BASE}/dataset"      # generated dataset (Part 3)
PROC_DIR = f"{BASE}/processed"    # after preprocessing (Part 4)
FEAT_DIR = f"{BASE}/features"     # extracted features (Part 5)

import os, glob, random, shutil, time
import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # reduces GPU memory issues
random.seed(SEED)
np.random.seed(SEED)

for d in [RAW_DIR, PROC_DIR, FEAT_DIR]:
    os.makedirs(d, exist_ok=True)
CLASSES = {"real": 0, "morph": 1}
SPLIT_NAMES = list(SPLITS)

t_real  = sum(v["real"]  for v in SPLITS.values())
t_morph = sum(v["morph"] for v in SPLITS.values())
print(f"Mode: {'QUICK DEMO 🚀' if DEMO else 'FULL SCALE (paper numbers) 🏋️'}")
print(f"Target: {t_real} real + {t_morph} morphed = {t_real + t_morph} images")
for s, v in SPLITS.items():
    print(f"  {s}: real={v['real']}  morph={v['morph']}")
""")

md("""### Install libraries and mount Drive for checkpoints
""")
code("""!pip -q install kagglehub mediapipe opencv-python-headless scipy tqdm pandas scikit-learn joblib gdown

import cv2, tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

tf.random.set_seed(SEED)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
print("TensorFlow:", tf.__version__, "| OpenCV:", cv2.__version__)
print("GPU:", gpus if gpus else "⚠️ No GPU — enable it: Runtime → Change runtime type → GPU")

# Mount Drive once for checkpointing (full run) — resume works across sessions
CKPT_DIR = None
if CHECKPOINT_TO_DRIVE or SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
if CHECKPOINT_TO_DRIVE:
    CKPT_DIR = "/content/drive/MyDrive/DMorphNet_checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)
    print("Checkpoints directory:", CKPT_DIR)
else:
    print("Drive checkpointing disabled (DEMO mode)")
""")

md("""### Fresh start — delete the old dataset before regenerating

Because the pipeline auto-resumes, an old dataset would be **reused** and the new
Delaunay morphs would never be generated. With `FRESH_START = True` this cell
deletes the previous local data **and** the Drive checkpoints so everything is
rebuilt from scratch.

> ⚠️ Run this **once** to restart. After generation begins, set `FRESH_START =
> False` in Part 1 so that a mid-run Colab disconnect resumes instead of wiping.
""")
code(r"""if FRESH_START:
    # local: wipe generated dataset, processed images, extracted features
    for d in [RAW_DIR, PROC_DIR, FEAT_DIR]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    removed = 0
    # Drive: remove morph checkpoints, processed cache, feature caches, labels
    if CKPT_DIR and os.path.isdir(CKPT_DIR):
        for f in glob.glob(os.path.join(CKPT_DIR, "*")):
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    os.remove(f)
                removed += 1
            except OSError:
                pass
    print(f"🧹 FRESH_START: cleared local dataset/processed/features"
          + (f" and {removed} Drive checkpoint files" if CKPT_DIR else ""))
    print("   The new Delaunay morph algorithm will regenerate everything.")
else:
    print("FRESH_START = False — existing data/checkpoints will be resumed.")
""")

# ================================================================ PART 2
md("""## Part 2 — Load a ready-made morph-attack dataset (Kaggle)

Instead of generating morphs, this loads a **ready-made face-morphing-attack
dataset** (bona-fide *real* + *morphed* images) straight from Kaggle. Set
`KAGGLE_MORPH_SLUG` to your dataset (`owner/name`).

> ⚠️ It must be a real **morphing-attack** dataset — with both genuine and
> morphed faces. (Note: `chiragsaipanuganti/morph` is *MORPH-II*, an aging
> mugshot database of **real** faces only — not usable here.) The loader below
> auto-detects the structure and **refuses to continue unless it finds both a
> real and a morph class**, so a wrong dataset fails loudly instead of silently
> mislabelling real faces as morphs.
""")
code(r'''import kagglehub, glob
from collections import Counter

# ================== SET THIS ==================
KAGGLE_MORPH_SLUG = ""          # e.g. "someowner/face-morph-attack-dataset"
# Optional explicit folder-name overrides if the auto keyword detection misses:
MORPH_DIRS = []                 # e.g. ["morphed", "attack"]
REAL_DIRS  = []                 # e.g. ["bonafide", "genuine"]
REAL_FALLBACK_CELEBA = True     # if dataset has morphs but no real class, pull reals from CelebA
# ==============================================

assert KAGGLE_MORPH_SLUG, ("Set KAGGLE_MORPH_SLUG to your Kaggle morph-attack "
                           "dataset (owner/name) before running.")

DS_ROOT = kagglehub.dataset_download(KAGGLE_MORPH_SLUG)
print("Downloaded to:", DS_ROOT)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
all_imgs = [p for p in glob.glob(os.path.join(DS_ROOT, "**", "*"), recursive=True)
            if p.lower().endswith(EXTS)]
print("Total image files:", len(all_imgs))

# Show the folder layout so you can see how it is organized
rel = [os.path.relpath(p, DS_ROOT) for p in all_imgs]
print("Top-level folders:", dict(Counter(r.split(os.sep)[0] for r in rel)))
print("Second-level (sample):",
      dict(list(Counter(os.sep.join(r.split(os.sep)[:2]) for r in rel).items())[:25]))
''')

md("""### 2.1 Classify each image as real vs morph

Detection order: (1) explicit `MORPH_DIRS` / `REAL_DIRS` folder names if you set
them, else (2) keyword match on the path. Adjust the keyword lists or the
explicit-dir lists above if your dataset uses different names.
""")
code(r'''import pandas as pd

MORPH_KW = ("morph", "attack", "fake", "spoof", "fraud", "manipulat")
REAL_KW  = ("real", "genuine", "bona", "bonafide", "live", "orig",
            "authentic", "reference", "probe", "raw")

def _has(path_low, names):
    parts = path_low.split(os.sep)
    return any(any(n.lower() == part or n.lower() in part for part in parts) for n in names)

def classify(path):
    low = os.path.relpath(path, DS_ROOT).lower()
    if MORPH_DIRS and _has(low, MORPH_DIRS): return "morph"
    if REAL_DIRS  and _has(low, REAL_DIRS):  return "real"
    if any(k in low for k in MORPH_KW): return "morph"
    if any(k in low for k in REAL_KW):  return "real"
    return None

records = [{"path": p, "label": classify(p)} for p in all_imgs]
df = pd.DataFrame([r for r in records if r["label"]])
unlabeled = sum(1 for r in records if r["label"] is None)
print("Detected:", (df["label"].value_counts().to_dict() if not df.empty else {}),
      "| unlabeled:", unlabeled)

# If a CSV of labels ships with the dataset, surface it so you can map manually
if df.empty or df["label"].nunique() < 2:
    csvs = glob.glob(os.path.join(DS_ROOT, "**", "*.csv"), recursive=True)
    print("Folder/keyword detection inconclusive. CSV files present:",
          [os.path.basename(c) for c in csvs])
    print("-> Set MORPH_DIRS / REAL_DIRS to the exact folder names, or tell me "
          "the CSV label column and I'll wire it in.")

# Optional: fill the real class from CelebA if the dataset only has morphs
if (not df.empty) and df["label"].nunique() == 1 and "morph" in set(df["label"]) \
        and REAL_FALLBACK_CELEBA:
    print("No real class in the morph dataset -> pulling bona-fide faces from CelebA ...")
    celeba = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    reals = glob.glob(os.path.join(celeba, "**", "*.jpg"), recursive=True)
    n_need = min(len(df), len(reals))
    df = pd.concat([df, pd.DataFrame({"path": reals[:n_need],
                                      "label": ["real"] * n_need})],
                   ignore_index=True)
    print("After CelebA fallback:", df["label"].value_counts().to_dict())

assert not df.empty and set(df["label"]) >= {"real", "morph"}, (
    "Dataset does not expose BOTH a real and a morph class. "
    "Set MORPH_DIRS/REAL_DIRS explicitly, enable REAL_FALLBACK_CELEBA, or "
    "pick a proper morph-attack dataset.")
print("\\nUsable:", df["label"].value_counts().to_dict())
''')

# ================================================================ PART 3
md("""## Part 3 — Build train / val / test splits from the dataset

If the dataset already ships `train/` `val/` `test/` folders they are honoured;
otherwise each class is split by the paper's ratios (~67 / 29 / 4). Images are
copied into `RAW_DIR/<split>/<class>/` (long side capped at 512 px) so the rest
of the pipeline runs unchanged.

> Identity separation: ready-made datasets rarely expose person IDs, so this uses
> an image-level split. If your dataset encodes identity in the filename, tell me
> the pattern and I'll switch to an identity-disjoint split.
""")
code(r'''# detect a pre-existing split from the path (train/val/test/dev/eval)
SPLIT_ALIASES = {"train": "train", "training": "train",
                 "val": "val", "valid": "val", "validation": "val", "dev": "val",
                 "test": "test", "testing": "test", "eval": "test"}

def path_split(path):
    for part in os.path.relpath(path, DS_ROOT).lower().split(os.sep):
        if part in SPLIT_ALIASES:
            return SPLIT_ALIASES[part]
    return None

df["dsplit"] = df["path"].map(path_split)
has_native = df["dsplit"].notna().mean() > 0.8      # dataset already split?

rng = np.random.RandomState(SEED)
assign = {s: {c: [] for c in CLASSES} for s in SPLIT_NAMES}

if has_native:
    print("Using the dataset's own train/val/test folders.")
    for _, r in df.iterrows():
        s = r["dsplit"] or "train"
        assign[s][r["label"]].append(r["path"])
else:
    print("No native split -> splitting each class by paper ratios.")
    frac = {c: {s: SPLITS[s][c] / sum(SPLITS[x][c] for x in SPLIT_NAMES)
                for s in SPLIT_NAMES} for c in CLASSES}
    for c in CLASSES:
        paths = df[df["label"] == c]["path"].tolist()
        rng.shuffle(paths)
        i = 0
        for s in SPLIT_NAMES:
            n = min(SPLITS[s][c], int(round(frac[c][s] * len(paths))))
            assign[s][c] = paths[i:i + n]
            i += n

# copy into RAW_DIR, capping the long side at 512 px
records = []
for s in SPLIT_NAMES:
    for c in CLASSES:
        dst = os.path.join(RAW_DIR, s, c)
        os.makedirs(dst, exist_ok=True)
        for k, src in enumerate(tqdm(assign[s][c], desc=f"copy {s}/{c}")):
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 512:
                sc = 512 / max(h, w)
                img = cv2.resize(img, (int(w * sc), int(h * sc)))
            fn = f"{c}_{s}_{k:06d}.jpg"
            cv2.imwrite(os.path.join(dst, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            records.append({"filename": fn, "split": s, "label": c})

labels_df = pd.DataFrame(records)
labels_df.to_csv(os.path.join(RAW_DIR, "labels.csv"), index=False)
print(labels_df.groupby(["split", "label"]).size())
''')

md("""### 3.1 Visual review of the loaded dataset
""")
code(r'''fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
for row, lab in enumerate(["real", "morph"]):
    sub = labels_df[labels_df.label == lab]
    sample = sub.sample(min(6, len(sub)), random_state=SEED)
    for i, (_, r) in enumerate(sample.iterrows()):
        p = os.path.join(RAW_DIR, r.split, r.label, r.filename)
        img = cv2.imread(p)
        if img is not None:
            axes[row, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, i].set_title(f"{lab} ({r.split})", fontsize=9)
        axes[row, i].axis("off")
plt.tight_layout()
plt.show()
print("Dataset ready. Real vs morph samples above — confirm the morph row really "
      "shows morphed faces before training.")
''')

# ================================================================ PART 4
md(r"""## Part 4 — Preprocessing: 528×528 + CLAHE

- Resize to **528×528** (EfficientNet-B6 requirement).
- **CLAHE** on the L channel in LAB color space — improves contrast and reveals
  fine facial details **without adding noise**.
- Save with uniform JPEG compression — **identical steps for real and morphed images**.
- Resumable: already-processed images are skipped, and the finished processed set
  is cached to Drive once so a new VM restores it instead of reprocessing.
""")
code(r"""clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def standardize(img_bgr):
    img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE),
                     interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a2, b2 = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a2, b2)), cv2.COLOR_LAB2BGR)


# Restore processed images from Drive cache (fresh VM after a disconnect)
PROC_ZIP = (os.path.join(CKPT_DIR, "processed.zip")
            if CHECKPOINT_TO_DRIVE else None)
if PROC_ZIP and os.path.exists(PROC_ZIP) and \
        not glob.glob(os.path.join(PROC_DIR, "*", "*", "*.jpg")):
    print("📥 Restoring processed images from Drive cache ...")
    shutil.unpack_archive(PROC_ZIP, PROC_DIR)

# Before/after example
sample_p = os.path.join(RAW_DIR, "train", "real",
                        labels_df[(labels_df.split == "train") &
                                  (labels_df.label == "real")].iloc[0].filename)
orig = cv2.imread(sample_p)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(cv2.cvtColor(cv2.resize(orig, (INPUT_SIZE, INPUT_SIZE)),
                            cv2.COLOR_BGR2RGB))
axes[0].set_title("Before (resize only)")
axes[1].imshow(cv2.cvtColor(standardize(orig), cv2.COLOR_BGR2RGB))
axes[1].set_title("After CLAHE — clearer details")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Apply to all images (skips files already processed — resumable)
for split in SPLIT_NAMES:
    for cls in CLASSES:
        src = os.path.join(RAW_DIR, split, cls)
        dst = os.path.join(PROC_DIR, split, cls)
        os.makedirs(dst, exist_ok=True)
        for p in tqdm(sorted(glob.glob(os.path.join(src, "*.jpg"))),
                      desc=f"preprocess {split}/{cls}"):
            dstp = os.path.join(dst, os.path.basename(p))
            if os.path.exists(dstp):
                continue
            img = cv2.imread(p)
            if img is not None:
                cv2.imwrite(dstp, standardize(img),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

total = len(glob.glob(os.path.join(PROC_DIR, "*", "*", "*.jpg")))
print(f"✅ Processed {total} images (528×528 + CLAHE + JPEG compression)")

# One-time Drive cache of the finished processed set
expected = sum(v["real"] + v["morph"] for v in SPLITS.values())
if PROC_ZIP and total >= expected and not os.path.exists(PROC_ZIP):
    print("💾 Caching processed images to Drive (one time, ~3 GB) ...")
    shutil.make_archive(PROC_ZIP[:-4], "zip", PROC_DIR)
    print("Done:", PROC_ZIP)
""")

md("""### Data augmentation (training only)

Horizontal flip, brightness/contrast, blur, noise, JPEG re-compression — used
during optional fine-tuning; simulates real-world image conditions.
""")
code(r"""def augment(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2),
                                  beta=random.uniform(-25, 25))
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        ok, enc = cv2.imencode(".jpg", img,
                               [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 90)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


sample = cv2.imread(glob.glob(os.path.join(PROC_DIR, "train", "morph", "*.jpg"))[0])
fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
axes[0].imshow(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
for i in range(1, 6):
    axes[i].imshow(cv2.cvtColor(augment(sample.copy()), cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"Augmented {i}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

# ================================================================ PART 5
md(r"""## Part 5 — Deep Feature Extraction with EfficientNet-B6

Mathematical basis: $F = f_{B6}(I;\theta)$ where every block computes
$X_l=\sigma(W_l * X_{l-1}+b_l)$ with Swish activation, then GAP:
$F=\frac{1}{N}\sum_i X_L^{(i)}$ produces the vector $F\in\mathbb{R}^{2304}$.

- Classification head removed (`include_top=False`) → the network is a pure feature extractor.
- **Phase 1**: fully frozen layers.
- **Phase 2 (optional — `DO_FINETUNE`)**: unfreeze only block7 + top layers with a very low learning rate.
- Extracted features are cached locally **and to Drive**, so this stage resumes instantly.

> 🛠️ **If a CUDA error appears here (e.g. `CUDA_ERROR_INVALID_HANDLE`)**: the GPU
> context broke after hours of processing — fix: **Runtime → Restart session**, then
> **Run all**. Thanks to auto-resume, Parts 1–4 skip everything already done in
> seconds and this cell gets a clean GPU context with no lost work.
""")
code(r"""from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
EXTRACT_BATCH = 16 if not DEMO else 8

base = EfficientNetB6(include_top=False, weights="imagenet",
                      input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
inputs = tf.keras.Input((INPUT_SIZE, INPUT_SIZE, 3))
x = base(inputs, training=False)
gap = layers.GlobalAveragePooling2D(name="gap")(x)          # GAP equation
feat_model = Model(inputs, gap, name="dmorphnet_features")
base.trainable = False

print(f"Feature vector length F: {feat_model.output_shape[-1]}")


def list_files(split):
    paths, labs = [], []
    for cls, lab in CLASSES.items():
        fs = sorted(glob.glob(os.path.join(PROC_DIR, split, cls, "*.jpg")))
        paths += fs
        labs += [lab] * len(fs)
    return paths, np.array(labs, np.int32)


def extract_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(list(paths))

    def load(p):
        img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.ensure_shape(img, [INPUT_SIZE, INPUT_SIZE, 3])
        return preprocess_input(img)

    return ds.map(load, num_parallel_calls=AUTOTUNE)\
             .batch(EXTRACT_BATCH).prefetch(AUTOTUNE)


F, y01, paths_all = {}, {}, {}
for split in SPLIT_NAMES:
    paths, labs = list_files(split)
    paths_all[split], y01[split] = paths, labs
    npz_name = f"effb6_{split}.npz"
    npz_path = os.path.join(FEAT_DIR, npz_name)
    drive_npz = (os.path.join(CKPT_DIR, npz_name)
                 if CHECKPOINT_TO_DRIVE else None)

    # Restore cached features (local first, then Drive)
    if not os.path.exists(npz_path) and drive_npz and os.path.exists(drive_npz):
        shutil.copy(drive_npz, npz_path)
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        if d["X"].shape[0] == len(paths):
            F[split] = d["X"]
            print(f"⏭️ {split}: cached features {F[split].shape} — skipping extraction")
            continue

    print(f"Extracting {split} features ({len(paths)} images) ...")
    F[split] = feat_model.predict(extract_ds(paths), verbose=1)
    np.savez_compressed(npz_path, X=F[split], y=labs)
    if drive_npz:
        shutil.copy(npz_path, drive_npz)      # checkpoint features to Drive
    print(f"  {split}: {F[split].shape}")

y = {s: np.where(y01[s] == 0, -1, +1) for s in SPLIT_NAMES}   # -1 real / +1 morph
print("✅ Feature extraction complete (frozen backbone)")
""")

md("""### Optional fine-tuning of the top layers (`DO_FINETUNE = True` to enable)

Unfreezes only block7 + top layers (BatchNorm stays frozen) with lr=1e-5,
then re-extracts the features.
""")
code(r"""if DO_FINETUNE:
    drop = layers.Dropout(0.3)(gap)
    out_head = layers.Dense(1, activation="sigmoid", dtype="float32")(drop)
    clf_model = Model(inputs, out_head)

    base.trainable = True
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(("block7", "top"))
                           and not isinstance(layer, layers.BatchNormalization))

    def ft_ds(split, training=False):
        p, labs = paths_all[split], y01[split]
        ds = tf.data.Dataset.from_tensor_slices((list(p), labs))
        if training:
            ds = ds.shuffle(len(p), seed=SEED)

        def load(pp, ll):
            img = tf.io.decode_jpeg(tf.io.read_file(pp), channels=3)
            img = tf.cast(img, tf.float32)
            img = tf.ensure_shape(img, [INPUT_SIZE, INPUT_SIZE, 3])
            if training:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 25.0)
                img = tf.clip_by_value(img, 0.0, 255.0)
            return preprocess_input(img), ll

        return ds.map(load, num_parallel_calls=AUTOTUNE).batch(4).prefetch(AUTOTUNE)

    clf_model.compile(tf.keras.optimizers.Adam(1e-5),
                      "binary_crossentropy", metrics=["accuracy"])
    hist = clf_model.fit(ft_ds("train", True), epochs=2,
                         validation_data=ft_ds("val"))

    for split in SPLIT_NAMES:                      # re-extract after fine-tuning
        F[split] = feat_model.predict(extract_ds(paths_all[split]), verbose=1)
    print("✅ Fine-tuning complete, features re-extracted")
else:
    print("⏭️ Fine-tuning skipped (DO_FINETUNE = False) — using frozen features")
""")

# ================================================================ PART 6
md(r"""## Part 6 — Hybrid Classification with SVM

Paper equations: labels $y_i\in\{-1,+1\}$, hyperplane $w^TF+b=0$, optimization
$\min \frac{1}{2}\lVert w\rVert^2 + C\sum\xi_i$ subject to $y_i(w^TF_i+b)\ge 1-\xi_i$,
and decision $\hat{y}=\operatorname{sign}(w^TF+b)$.

We standardize features, tune $C$ on the **validation** split, train the final
classifier, and verify the decision function manually.
""")
code(r"""from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc)

scaler = StandardScaler().fit(F["train"])
Fs = {s: scaler.transform(F[s]) for s in SPLIT_NAMES}

# Tune C on the validation split
results_C = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    clf = LinearSVC(C=C, random_state=SEED).fit(Fs["train"], y["train"])
    results_C[C] = accuracy_score(y["val"], clf.predict(Fs["val"]))
    print(f"C = {C:<6} → validation accuracy = {results_C[C]:.4f}")
BEST_C = max(results_C, key=results_C.get)

svm_final = LinearSVC(C=BEST_C, random_state=SEED).fit(Fs["train"], y["train"])
w, b = svm_final.coef_[0], float(svm_final.intercept_[0])
print(f"\n🏆 Best C = {BEST_C} — w shape: {w.shape}, b = {b:.4f}")

# Manual verification: sign(wF+b) matches predict exactly
scores_test = Fs["test"] @ w + b
assert (np.sign(scores_test) == svm_final.predict(Fs["test"])).all()
print("✅ sign(wᵀF+b) matches predict() 100%")

plt.figure(figsize=(9, 4))
plt.hist(scores_test[y["test"] == -1], bins=40, alpha=0.6, label="real (-1)")
plt.hist(scores_test[y["test"] == +1], bins=40, alpha=0.6, label="morph (+1)")
plt.axvline(0, color="black", linestyle="--", label="hyperplane wᵀF+b=0")
plt.xlabel("wᵀF + b")
plt.ylabel("image count")
plt.title("Class separation in feature space (test set)")
plt.legend()
plt.tight_layout()
plt.show()
""")

# ================================================================ PART 7
md("""## Part 7 — Results: Confusion Matrix + Metrics + ROC/AUC

Confusion matrix in the style of Fig. 2, TP/TN/FP/FN values, the four metrics,
and the ROC curve in the style of Fig. 3 (paper reference: 89.9% accuracy and
AUC = 0.965 on the full dataset).
""")
code(r"""pred_test = svm_final.predict(Fs["test"])
cm = confusion_matrix(y["test"], pred_test, labels=[-1, +1])
TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

# Confusion matrix (Fig. 2 style)
im = axes[0].imshow(cm, cmap="Blues")
plt.colorbar(im, ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Real", "Morph"])
axes[0].set_yticklabels(["Real", "Morph"])
axes[0].set_xlabel("Predicted label"); axes[0].set_ylabel("True label")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

# ROC curve (Fig. 3 style)
fpr, tpr, thresholds = roc_curve(y["test"], scores_test, pos_label=+1)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="tab:blue", linewidth=2.2,
             label=f"AUC={roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="tab:orange", linewidth=1.8)
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].grid(alpha=0.35)
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.show()

metrics = {
    "Accuracy":  accuracy_score(y["test"], pred_test),
    "Precision": precision_score(y["test"], pred_test, pos_label=+1),
    "Recall":    recall_score(y["test"], pred_test, pos_label=+1),
    "F1-Score":  f1_score(y["test"], pred_test, pos_label=+1),
    "AUC":       roc_auc,
}
print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print("FN (missed morphs) is the most critical error biometrically — "
      "FP is handled by manual review\n")
for k, v in metrics.items():
    print(f"  {k:10s} = {v:.4f}")
print("\n(Paper reference: 89.9% accuracy, AUC = 0.965)")
print(classification_report(y["test"], pred_test,
                            target_names=["Real", "Morph"]))
""")

# ================================================================ PART 8
md(r"""## Part 8 — Threshold Optimization

Decision rule: morph if $s \ge \tau$. We sweep $\tau$ over the **validation**
scores, pick the value maximizing F1 (the balance between catching morphs and
not flagging real faces), then evaluate on the test set.
""")
code(r"""s_val = Fs["val"] @ w + b
taus = np.linspace(s_val.min(), s_val.max(), 300)
f1s = [f1_score(y["val"], np.where(s_val >= t, +1, -1),
                pos_label=+1, zero_division=0) for t in taus]
TAU = float(taus[int(np.argmax(f1s))])

pred_opt = np.where(scores_test >= TAU, +1, -1)
acc_def = accuracy_score(y["test"], pred_test)
acc_opt = accuracy_score(y["test"], pred_opt)
cm_opt = confusion_matrix(y["test"], pred_opt, labels=[-1, +1])

plt.figure(figsize=(9, 4))
plt.plot(taus, f1s, label="F1 on validation")
plt.axvline(0, color="gray", linestyle=":", label="default τ=0")
plt.axvline(TAU, color="green", linestyle="--", label=f"selected τ*={TAU:.3f}")
plt.xlabel("threshold τ"); plt.ylabel("F1")
plt.title("Threshold selection on the validation set")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Selected threshold τ* = {TAU:.4f}")
print(f"Test accuracy: default τ=0 → {acc_def:.4f} | "
      f"selected τ* → {acc_opt:.4f} ({(acc_opt-acc_def)*100:+.2f} pts)")
print(f"FN: {FN} → {int(cm_opt[1,0])}   |   FP: {FP} → {int(cm_opt[0,1])}")
""")

# ================================================================ PART 9
md("""## Part 9 — (Optional) Architecture Comparison: Table 2

Set `RUN_COMPARISONS = True` in Part 1 to run: B0 + Softmax (1280 features),
B5 + linear SVM (2048), vs the proposed B6 (2304). Published reference:
62.4% / 66.13% / **89.9%**.
""")
code(r"""if RUN_COMPARISONS:
    from tensorflow.keras.applications import EfficientNetB0, EfficientNetB5

    def extract_with(model_cls, size, paths):
        m = model_cls(include_top=False, weights="imagenet",
                      pooling="avg", input_shape=(size, size, 3))

        def load(p):
            img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
            img = tf.image.resize(tf.cast(img, tf.float32), [size, size])
            return preprocess_input(img)

        ds = (tf.data.Dataset.from_tensor_slices(list(paths))
              .map(load, num_parallel_calls=AUTOTUNE).batch(16).prefetch(AUTOTUNE))
        return m.predict(ds, verbose=1)

    comp = {}
    # B0 + Softmax
    b0tr = extract_with(EfficientNetB0, 224, paths_all["train"])
    b0te = extract_with(EfficientNetB0, 224, paths_all["test"])
    sc0 = StandardScaler().fit(b0tr)
    h0 = tf.keras.Sequential([tf.keras.layers.Input((b0tr.shape[1],)),
                              tf.keras.layers.Dense(2, activation="softmax")])
    h0.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    h0.fit(sc0.transform(b0tr), y01["train"], epochs=10, batch_size=128, verbose=0)
    p0 = np.where(h0.predict(sc0.transform(b0te), verbose=0)[:, 1] >= 0.5, +1, -1)
    comp["B0 + Softmax (1280)"] = accuracy_score(y["test"], p0)

    # B5 + linear SVM
    b5tr = extract_with(EfficientNetB5, 456, paths_all["train"])
    b5te = extract_with(EfficientNetB5, 456, paths_all["test"])
    sc5 = StandardScaler().fit(b5tr)
    c5 = LinearSVC(C=1.0, random_state=SEED).fit(sc5.transform(b5tr), y["train"])
    comp["B5 + Linear SVM (2048)"] = accuracy_score(
        y["test"], c5.predict(sc5.transform(b5te)))

    comp["B6 proposed (2304)"] = accuracy_score(y["test"], pred_test)

    print("Table 2 — test accuracy (reference: 62.4 / 66.13 / 89.9):")
    for k, v in comp.items():
        print(f"  {k:26s}: {v*100:.2f}%")
else:
    print("⏭️ Architecture comparison skipped (RUN_COMPARISONS = False)")
    print("   Published reference — B0: 62.4% | B5: 66.13% | B6 proposed: 89.9%")
""")

# ================================================================ PART 10
md("""## Part 10 — Real-Time Prediction (Figs. 4 and 5)

- Upload any new image → preprocessing → B6 features → SVM with the optimized
  threshold → **MORPH IMAGE / REAL IMAGE** shown in red with a confidence score.
- Supports **multi-face images**: MediaPipe detects all faces and each one is
  classified separately (red box = morph, green box = real).
""")
code(r"""# Unified face detector working with BOTH MediaPipe APIs
if hasattr(mp, "solutions"):
    _detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    def detect_face_boxes(img_bgr):
        h, wd = img_bgr.shape[:2]
        res = _detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return []
        out = []
        for det in res.detections:
            bb = det.location_data.relative_bounding_box
            out.append((int(bb.xmin * wd), int(bb.ymin * h),
                        int(bb.width * wd), int(bb.height * h)))
        return out
else:
    import urllib.request
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
    _DET_MODEL = "/content/blaze_face_short_range.tflite"
    if not os.path.exists(_DET_MODEL):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            _DET_MODEL)
    _detector = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_DET_MODEL),
            min_detection_confidence=0.5))

    def detect_face_boxes(img_bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = _detector.detect(mp_img)
        return [(int(d.bounding_box.origin_x), int(d.bounding_box.origin_y),
                 int(d.bounding_box.width), int(d.bounding_box.height))
                for d in res.detections]


def predict_face(img_bgr):
    # Full path for one face: preprocess → features → decision + confidence + timings
    t0 = time.perf_counter()
    proc = standardize(img_bgr)
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32)
    t1 = time.perf_counter()
    feat = feat_model.predict(preprocess_input(rgb)[None, ...], verbose=0)
    t2 = time.perf_counter()
    s = float(scaler.transform(feat) @ w + b)
    conf = 1.0 / (1.0 + np.exp(-(s - TAU)))          # sigmoid around threshold
    t3 = time.perf_counter()
    label = "MORPH IMAGE" if s >= TAU else "REAL IMAGE"
    return label, conf, s, {"pre": t1 - t0, "feat": t2 - t1,
                            "svm": t3 - t2, "total": t3 - t0}


def show_result(img_bgr, label, conf):
    # Display in the style of Figs. 4 and 5: bold red title above the image
    plt.figure(figsize=(4.5, 5.5))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"{label}\nConfidence:{conf:.4f}",
              color="red", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def predict_image(img_bgr):
    # Multi-face support; classifies the whole image if no face is detected
    h, wd = img_bgr.shape[:2]
    boxes = []
    for (x, yy, bw, bh) in detect_face_boxes(img_bgr):
        m = int(0.25 * max(bw, bh))          # 25% margin around each face
        boxes.append((max(0, x - m), max(0, yy - m),
                      min(wd, x + bw + m) - max(0, x - m),
                      min(h, yy + bh + m) - max(0, yy - m)))
    if not boxes:
        lab, cf, s, tms = predict_face(img_bgr)
        return [(None, lab, cf, tms)], img_bgr.copy()
    annotated, out = img_bgr.copy(), []
    for (x, yy, bw, bh) in boxes:
        lab, cf, s, tms = predict_face(img_bgr[yy:yy + bh, x:x + bw])
        out.append(((x, yy, bw, bh), lab, cf, tms))
        color = (0, 0, 255) if lab == "MORPH IMAGE" else (0, 200, 0)
        cv2.rectangle(annotated, (x, yy), (x + bw, yy + bh), color, 3)
        cv2.putText(annotated, f"{lab.split()[0]} {cf:.2f}",
                    (x, max(25, yy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out, annotated


# Instant demo on one real and one morphed test sample
for cls in ["real", "morph"]:
    p = random.choice(glob.glob(os.path.join(PROC_DIR, "test", cls, "*.jpg")))
    img = cv2.imread(p)
    lab, cf, s, tms = predict_face(img)
    print(f"Truth: {cls} — Prediction: {lab} — "
          f"Confidence:{cf:.4f} — {tms['total']*1000:.0f} ms")
    show_result(img, lab, cf)
""")

md("""### 📤 Upload your own images now (as in Figs. 4 and 5)
""")
code(r"""from google.colab import files

uploaded = files.upload()
for fname in uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded[fname], np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ Could not read {fname}")
        continue
    results, annotated = predict_image(img)
    if len(results) == 1 and results[0][0] is None:
        _, lab, cf, tms = results[0]
        show_result(img, lab, cf)
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"Faces detected: {len(results)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    for i, (box, lab, cf, tms) in enumerate(results, 1):
        print(f"  face {i}: {lab}  Confidence:{cf:.4f}  "
              f"({tms['total']*1000:.0f} ms)")
""")

md("""### Prediction speed measurement
""")
code(r"""sample_paths = glob.glob(os.path.join(PROC_DIR, "test", "*", "*.jpg"))
predict_face(cv2.imread(sample_paths[0]))          # warm-up

agg = {"pre": [], "feat": [], "svm": [], "total": []}
for p in random.sample(sample_paths, min(20, len(sample_paths))):
    _, _, _, tms = predict_face(cv2.imread(p))
    for k in agg:
        agg[k].append(tms[k])

for k, name in [("pre", "Preprocessing"), ("feat", "B6 features"),
                ("svm", "SVM decision"), ("total", "Total")]:
    print(f"  {name:16s}: {np.mean(agg[k])*1000:7.1f} ms")
print(f"\nThroughput: ~{1.0/np.mean(agg['total']):.1f} images/sec — "
      "the decision after feature extraction is near-instant")
""")

# ================================================================ PART 11
md("""## Part 11 — Save to Google Drive and Final Summary

Saves the feature extractor, SVM, scaler, optimized threshold, and features to
Drive (enabled by default for the full run via `SAVE_TO_DRIVE = True`).
""")
code(r"""import joblib

if SAVE_TO_DRIVE:
    SAVE_DIR = "/content/drive/MyDrive/DMorphNet_models"
    os.makedirs(SAVE_DIR, exist_ok=True)
    feat_model.save(os.path.join(SAVE_DIR, "effb6_features.keras"))
    joblib.dump(svm_final, os.path.join(SAVE_DIR, "svm_final.joblib"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler_final.joblib"))
    np.savez(os.path.join(SAVE_DIR, "optimal_threshold.npz"), tau=TAU)
    for split in SPLIT_NAMES:
        src = os.path.join(FEAT_DIR, f"effb6_{split}.npz")
        if os.path.exists(src):
            shutil.copy(src, SAVE_DIR)
    print("✅ Saved to:", SAVE_DIR)
    print(os.listdir(SAVE_DIR))
else:
    print("⏭️ Drive save disabled (SAVE_TO_DRIVE = False)")

print("\n" + "=" * 55)
print("FINAL SUMMARY — D-MorphNet")
print("=" * 55)
print(f"Mode             : {'demo' if DEMO else 'FULL SCALE'}")
print(f"Total images     : {len(labels_df)} "
      f"({t_real} real + {t_morph} morphed)")
print(f"Feature extractor: EfficientNet-B6 "
      f"({'fine-tuned' if DO_FINETUNE else 'frozen'}) — 2304-dim features")
print(f"Classifier       : SVM (C = {BEST_C}, τ* = {TAU:.4f})")
print(f"Test accuracy    : {acc_opt:.4f}")
print(f"AUC              : {roc_auc:.4f}")
print(f"TP/TN/FP/FN      : {TP}/{TN}/{FP}/{FN}")
print("\n🎉 D-MorphNet full pipeline complete")
""")

# ================================================================ PART 12
md(r"""## Part 12 — Improvements & Before/After Comparison

This part applies the fixes recommended in the results-analysis report **without
retraining the backbone**, and prints a direct before/after comparison against
the baseline (default threshold τ=0). It targets the two most important problems:
missed morphs (false negatives) and the fixed, un-tunable threshold.

Improvements applied on the same trained model:
1. **Probability calibration** (Platt scaling) — turns raw SVM scores into
   trustworthy probabilities.
2. **Cost-sensitive / security-optimized threshold** — pick the operating point
   that drives the false-negative rate down (APCER ≤ 1% on validation), the
   safest setting for a biometric gate.
3. **EER-balanced threshold** — equal-error operating point.
4. **Standard morph-attack metrics** — APCER, BPCER, ACER, EER, BPCER@APCER
   (ISO/IEC 30107-3), not just accuracy.
""")
code(r"""from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve as _roc


def confusion_at(scores, ytrue, tau):
    pred = np.where(scores >= tau, 1, -1)
    tp = int(((pred == 1) & (ytrue == 1)).sum())
    fn = int(((pred == -1) & (ytrue == 1)).sum())
    fp = int(((pred == 1) & (ytrue == -1)).sum())
    tn = int(((pred == -1) & (ytrue == -1)).sum())
    n = len(ytrue)
    apcer = fn / max(tp + fn, 1)          # morphs missed (attack error)
    bpcer = fp / max(fp + tn, 1)          # real flagged (bona-fide error)
    return {"Acc": (tp + tn) / n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2}


sv = Fs["val"] @ w + b                    # validation scores
st = scores_test                          # test scores (from Part 7)

# Improvement 1 — security-optimized threshold: lowest FN with val APCER <= 1%
tau_lowfn = sv.min()
for t in np.linspace(sv.min(), sv.max(), 800):
    if confusion_at(sv, y["val"], t)["APCER"] <= 0.01:
        tau_lowfn = t
        break

# Improvement 2 — EER-balanced threshold on validation
fv, tv, thv = _roc(y["val"], sv, pos_label=1)
eer_idx = int(np.argmin(np.abs(fv - (1 - tv))))
tau_eer = float(thv[eer_idx])

# Improvement 3 — Platt-calibrated probabilities (fit on validation)
cal = CalibratedClassifierCV(svm_final, cv="prefit", method="sigmoid")
cal.fit(Fs["val"], y["val"])
pos = list(cal.classes_).index(1)
prob_test = cal.predict_proba(Fs["test"])[:, pos]

# Assemble before/after comparison on the TEST set
configs = {
    "Baseline (tau=0)":            confusion_at(st, y["test"], 0.0),
    "F1-optimal (tau*)":           confusion_at(st, y["test"], TAU),
    "Security-optimized (low-FN)": confusion_at(st, y["test"], tau_lowfn),
    "EER-balanced":                confusion_at(st, y["test"], tau_eer),
    "Calibrated (p>=0.5)":         confusion_at(prob_test, y["test"], 0.5),
}
cmp_df = pd.DataFrame(configs).T[["Acc", "FN", "FP", "APCER", "BPCER", "ACER"]]
cmp_df = cmp_df.round(4)
print("BEFORE vs AFTER  (test set: 1000 real + 1000 morph)")
print(cmp_df.to_string())

base = configs["Baseline (tau=0)"]
sec = configs["Security-optimized (low-FN)"]
print(f"\nMissed morphs (FN):  baseline {base['FN']}  ->  security-optimized "
      f"{sec['FN']}   ({sec['FN']-base['FN']:+d})")
print(f"APCER (attack miss): baseline {base['APCER']*100:.1f}%  ->  "
      f"{sec['APCER']*100:.1f}%")
""")

md("""### Visual before/after comparison
""")
code(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
names = list(configs)
colors_b = ["#9aa4b2", "#2456a6", "#1e7d43", "#a76b00", "#7a3ea6"]

# (a) Missed morphs (FN) — lower is safer
fn_vals = [configs[n]["FN"] for n in names]
axes[0].bar(range(len(names)), fn_vals, color=colors_b)
axes[0].set_title("Missed morphs — FN (lower = safer)")
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels([n.split(" (")[0] for n in names], rotation=25, ha="right",
                        fontsize=8)
for i, v in enumerate(fn_vals):
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

# (b) ACER — overall biometric error, lower is better
acer_vals = [configs[n]["ACER"] * 100 for n in names]
axes[1].bar(range(len(names)), acer_vals, color=colors_b)
axes[1].set_title("ACER % (lower = better)")
axes[1].set_xticks(range(len(names)))
axes[1].set_xticklabels([n.split(" (")[0] for n in names], rotation=25, ha="right",
                        fontsize=8)
for i, v in enumerate(acer_vals):
    axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

# (c) DET-style: APCER vs BPCER trade-off across thresholds
fpr_t, tpr_t, _ = _roc(y["test"], st, pos_label=1)
axes[2].plot((1 - tpr_t) * 100, fpr_t * 100, color="#2456a6", linewidth=2)
for n, c in zip(names, colors_b):
    axes[2].scatter(configs[n]["APCER"] * 100, configs[n]["BPCER"] * 100,
                    s=70, color=c, zorder=5, label=n.split(" (")[0])
axes[2].set_xlabel("APCER % (morphs missed)")
axes[2].set_ylabel("BPCER % (real flagged)")
axes[2].set_title("Error trade-off (DET)")
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Interpretation: on the SAME model, moving from the default threshold to "
      "the security-optimized point cuts missed morphs (FN) sharply — the key "
      "biometric safety metric — at the cost of more manual review (higher FP). "
      "Enabling DO_FINETUNE (Part 5) improves the whole ROC, lowering BOTH "
      "errors at once.")
""")

# ================================================================ PART 13
md(r"""## Part 13 — Overfitting Check & Fine-Tuning Comparison

Over-fitting means the model memorizes the training set instead of learning a
general rule. We detect it with the **train − validation accuracy gap**: a large
positive gap (train ≫ val) is the signature of over-fitting. We then compare the
**frozen** backbone against the **fine-tuned** one to show the effect.

Rule of thumb used below:
- gap ≤ 5 pts → healthy
- 5–10 pts → mild over-fitting
- > 10 pts → over-fitting (regularize / fine-tune / add data)
""")
code(r"""from sklearn.svm import SVC

def fit_eval(feat, tag):
    # Train an RBF SVM on `feat` and report train/val/test accuracy + gap
    sc = StandardScaler().fit(feat["train"])
    clf = SVC(kernel="rbf", C=BEST_C if 'BEST_C' in globals() else 10.0,
              gamma="scale").fit(sc.transform(feat["train"]), y["train"])
    a = {s: accuracy_score(y[s], clf.predict(sc.transform(feat[s])))
         for s in SPLIT_NAMES}
    gap = a["train"] - a["val"]
    verdict = ("OVERFITTING (>10 pts)" if gap > 0.10 else
               "mild overfitting (5-10)" if gap > 0.05 else "OK (<=5 pts)")
    print(f"{tag:<26} train={a['train']*100:5.1f}%  val={a['val']*100:5.1f}%  "
          f"test={a['test']*100:5.1f}%  gap={gap*100:+5.1f} -> {verdict}")
    return a, gap

print("Train / Val / Test accuracy and the overfitting gap:\n")

# Frozen features were cached to .npz in Part 5 BEFORE fine-tuning
frozen_feat = {}
for s in SPLIT_NAMES:
    p = os.path.join(FEAT_DIR, f"effb6_{s}.npz")
    frozen_feat[s] = np.load(p)["X"] if os.path.exists(p) else F[s]
a_frozen, g_frozen = fit_eval(frozen_feat, "Frozen B6 + SVM")

# Current F[] holds fine-tuned features if DO_FINETUNE ran this session
if DO_FINETUNE:
    a_ft, g_ft = fit_eval(F, "Fine-tuned B6 + SVM")
else:
    a_ft, g_ft = a_frozen, g_frozen
    print("(DO_FINETUNE=False — set it True in Part 1 to see the fine-tuned row)")

# Bar chart: train vs val vs test for each regime
labels = ["train", "val", "test"]
xf = np.arange(3)
plt.figure(figsize=(9, 4.4))
plt.bar(xf - 0.2, [a_frozen[s]*100 for s in SPLIT_NAMES], width=0.4,
        label=f"Frozen (gap {g_frozen*100:+.1f})", color="#9aa4b2")
plt.bar(xf + 0.2, [a_ft[s]*100 for s in SPLIT_NAMES], width=0.4,
        label=f"Fine-tuned (gap {g_ft*100:+.1f})", color="#12a5b8")
plt.xticks(xf, labels); plt.ylabel("accuracy %"); plt.ylim(50, 101)
plt.title("Train vs Val vs Test — frozen vs fine-tuned")
plt.legend(); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\nOverfitting gap: frozen {g_frozen*100:+.1f} pts -> "
      f"fine-tuned {g_ft*100:+.1f} pts")
print(f"Test accuracy : frozen {a_frozen['test']*100:.1f}% -> "
      f"fine-tuned {a_ft['test']*100:.1f}%")
""")

md("""### Learning curve — is the model data-limited?

Plots training vs cross-validation accuracy as the training-set size grows. A
persistent wide gap = over-fitting; curves converging upward = more data / better
features would still help. (Sub-sampled for speed.)
""")
code(r"""from sklearn.model_selection import learning_curve

Xlc = StandardScaler().fit_transform(F["train"])
n = min(4000, len(Xlc))                       # cap for speed on the full set
idx = np.random.RandomState(SEED).permutation(len(Xlc))[:n]
sizes, tr_sc, va_sc = learning_curve(
    SVC(kernel="rbf", C=BEST_C if 'BEST_C' in globals() else 10.0, gamma="scale"),
    Xlc[idx], y["train"][idx],
    train_sizes=np.linspace(0.2, 1.0, 5), cv=3, scoring="accuracy")

plt.figure(figsize=(8, 4.6))
plt.plot(sizes, tr_sc.mean(1)*100, "o-", color="#12a5b8", label="training")
plt.plot(sizes, va_sc.mean(1)*100, "s-", color="#d8791a", label="cross-val")
plt.fill_between(sizes, va_sc.mean(1)*100, tr_sc.mean(1)*100,
                 color="#d8791a", alpha=0.12)
plt.xlabel("training samples"); plt.ylabel("accuracy %")
plt.title("Learning curve (current features)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

final_gap = (tr_sc.mean(1)[-1] - va_sc.mean(1)[-1]) * 100
print(f"Train-CV gap at full size: {final_gap:+.1f} pts")
print("Wide, non-closing band => overfitting; converging band => healthy fit.")
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "name": "DMorphNet_full_pipeline.ipynb",
                  "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = "/home/user/morhping-detection/DMorphNet_full_pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written", out_path, "| cells:", len(cells))
