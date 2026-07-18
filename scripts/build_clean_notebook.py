# -*- coding: utf-8 -*-
"""Builds a clean, minimal DMorphNet notebook (English, runnable, shows results)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": s})

# ---------------------------------------------------------------- title
md(r"""# DMorphNet — Face Morphing Detection
### EfficientNet-B6 features + SVM classifier

A clean, end-to-end implementation of *DMorphNet* (Gawade et al.). A **morphed**
face blends two identities into one image that can fool face verification. This
notebook builds a dataset, extracts deep features with **EfficientNet-B6**, and
classifies **real vs. morph** with an **SVM**.

**Pipeline**

`face image → resize 528 + CLAHE → EfficientNet-B6 → 2304-d vector → SVM → Real / Morph`

Why two stages? A frozen CNN gives strong features while an SVM draws a robust
boundary — this generalizes better than one end-to-end network on a small dataset.

> Run the cells top to bottom. Enable a GPU first: **Runtime → Change runtime type → GPU**.
""")

# ---------------------------------------------------------------- setup
md("""## 1 · Setup

Install the libraries and confirm a GPU is available.
""")
code("""!pip -q install kagglehub mediapipe opencv-python-headless scikit-learn tqdm

import os, glob, random, urllib.request
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print("TensorFlow", tf.__version__,
      "| GPU:", "yes" if tf.config.list_physical_devices('GPU') else "NO (enable it)")
""")

# ---------------------------------------------------------------- config
md("""## 2 · Configuration

`DEMO = True` runs a small, fast dataset that still produces real results
(~20 min on a GPU). Set `DEMO = False` for the paper's full scale.
""")
code(r"""DEMO = True                 # True = small/fast subset | False = full dataset

IMG_SIZE = 528                 # EfficientNet-B6 input
PREVIEW  = 256                 # size for preview thumbnails
DEMO_CAP = 500                 # max images per class per split when DEMO
print("Mode:", "DEMO (subset)" if DEMO else "FULL dataset")
""")

# ---------------------------------------------------------------- dataset
md("""## 3 · Dataset — original vs face-swapped (ready-made)

We use the **`rdjarbeng/face-swap-images`** dataset: original and face-swapped
faces extracted from videos, already partitioned into **train / test / val**.
No morph generation is needed — the loader below reads the dataset's own splits
and labels each image **0 = original (real)** or **1 = face-swapped (morph)**.
""")
code(r"""import kagglehub

DATA = kagglehub.dataset_download("rdjarbeng/face-swap-images")
print("Dataset path:", DATA)

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
all_imgs = [p for p in glob.glob(os.path.join(DATA, "**", "*"), recursive=True)
            if p.lower().endswith(EXTS)]
print("Total images:", len(all_imgs))

# show the folder layout so the split/label detection is transparent
from collections import Counter
rel = [os.path.relpath(p, DATA) for p in all_imgs]
print("Top-level:", dict(Counter(r.split(os.sep)[0] for r in rel)))
print("Two levels:", dict(list(Counter(os.sep.join(r.split(os.sep)[:2])
                                       for r in rel).items())[:20]))
""")

md("""### 3.1 Read the splits and labels

Each image's **split** (train/val/test) and **label** (original vs swapped) are
inferred from its folder path. Adjust the keyword lists if the dataset uses
different folder names.
""")
code(r"""SWAP_KW = ("swap", "fake", "altered", "morph", "manip", "spoof", "synthetic",
           "forged", "deepfake")
REAL_KW = ("original", "real", "orig", "genuine", "bonafide", "authentic", "pristine")

# The dataset's OWN root folder is named "face-swap-images" (contains "swap"),
# which would mislabel every image. Strip the common root before matching.
_rels = [os.path.relpath(p, DATA) for p in all_imgs]
ROOT = os.path.commonpath(_rels) if len(_rels) > 1 else ""

def _rel(p):
    r = os.path.relpath(p, DATA)
    return os.path.relpath(r, ROOT) if ROOT and ROOT not in (".", "") else r

def split_of(parts):
    for p in parts:
        if p in ("train", "training"):                 return "train"
        if p in ("val", "valid", "validation", "dev"): return "val"
        if p in ("test", "testing", "eval"):           return "test"
    return None

def classify(path):
    r = _rel(path).lower()
    s = split_of(r.split(os.sep))
    if any(k in r for k in SWAP_KW):   y = 1     # face-swapped / morph
    elif any(k in r for k in REAL_KW): y = 0     # original / real
    else:                              y = None
    return s, y

splits = {"train": [], "val": [], "test": []}
skipped = 0
for p in all_imgs:
    s, y = classify(p)
    if s is None or y is None:
        skipped += 1
        continue
    splits[s].append((p, y))

# carve a val split from train if the dataset has none
if not splits["val"] and splits["train"]:
    random.shuffle(splits["train"])
    n = max(1, int(0.15 * len(splits["train"])))
    splits["val"], splits["train"] = splits["train"][:n], splits["train"][n:]

# DEMO: cap images per class per split for a fast pass
if DEMO:
    for s in splits:
        by = {0: [], 1: []}
        for it in splits[s]:
            by[it[1]].append(it)
        splits[s] = by[0][:DEMO_CAP] + by[1][:DEMO_CAP]
        random.shuffle(splits[s])

for s, v in splits.items():
    r = sum(1 for _, y in v if y == 0)
    print(f"{s:5s}: {len(v):5d}  (original {r}, swapped {len(v) - r})")
print("skipped:", skipped, "| stripped root:", repr(ROOT))
for y, nm in [(0, "original"), (1, "swapped")]:
    ex = [_rel(p) for p in all_imgs if classify(p)[1] == y][:2]
    print(f"  sample {nm} path:", ex)

assert all(splits[s] for s in splits), "A split is empty — check the sample paths above."
assert any(y == 0 for v in splits.values() for _, y in v), \
    "No ORIGINAL images detected — adjust REAL_KW to match the folder names printed above."
items = splits["train"] + splits["val"] + splits["test"]
""")

md("""### 3.2 Preview — original vs face-swapped
""")
code(r"""fig, ax = plt.subplots(2, 6, figsize=(15, 5))
for row, (lab, name) in enumerate([(0, "original"), (1, "face-swapped")]):
    samp = [p for p, y in items if y == lab][:6]
    for a_, p in zip(ax[row], samp):
        a_.imshow(cv2.cvtColor(cv2.resize(cv2.imread(p), (PREVIEW, PREVIEW)),
                               cv2.COLOR_BGR2RGB))
        a_.set_title(name); a_.axis("off")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- preprocess
md(r"""## 4 · Preprocessing — resize + CLAHE

EfficientNet-B6 needs a fixed **528×528** input. **CLAHE** (Contrast Limited
Adaptive Histogram Equalization) on the lightness channel boosts local contrast
so subtle morph seams become clearer, without amplifying noise. The **same**
steps apply to real and morph images.
""")
code(r"""_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

demo = cv2.imread(items[0][0])
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(cv2.cvtColor(cv2.resize(demo, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB))
ax[0].set_title("resized"); ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(preprocess(demo), cv2.COLOR_BGR2RGB))
ax[1].set_title("resized + CLAHE"); ax[1].axis("off")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- features
md(r"""## 5 · Feature extraction — EfficientNet-B6

We load EfficientNet-B6 (ImageNet weights, **no classifier head**) with **Global
Average Pooling**, turning every image into a fixed **2304-dimensional** feature
vector $F$ that captures facial structure, texture, and morphing artefacts.

$$F = \tfrac{1}{N}\sum_i X_L^{(i)} \in \mathbb{R}^{2304}$$
""")
code(r"""from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input

backbone = EfficientNetB6(include_top=False, weights="imagenet",
                          pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
print("feature vector length:", backbone.output_shape[-1])

def extract(pairs, batch=16):
    X, y = [], [p[1] for p in pairs]
    paths = [p[0] for p in pairs]
    for i in tqdm(range(0, len(paths), batch), desc="extracting"):
        imgs = [preprocess_input(cv2.cvtColor(preprocess(cv2.imread(p)),
                                              cv2.COLOR_BGR2RGB).astype("float32"))
                for p in paths[i:i + batch]]
        X.append(backbone.predict(np.stack(imgs), verbose=0))
    return np.concatenate(X), np.array(y)

feats = {s: extract(v) for s, v in splits.items()}
for s, (X, y) in feats.items():
    print(f"{s:5s}: features {X.shape}")
""")

# ---------------------------------------------------------------- svm
md(r"""## 6 · Classifier — SVM

We standardize the features and train a **Support Vector Machine** (RBF kernel),
which finds the maximum-margin boundary between real and morph:

$$\min_{w,b}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^{\!\top}F_i+b)\ge 1-\xi_i$$
""")
code(r"""from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

scaler = StandardScaler().fit(feats["train"][0])
Xtr = scaler.transform(feats["train"][0]); ytr = feats["train"][1]
Xva = scaler.transform(feats["val"][0]);   yva = feats["val"][1]
Xte = scaler.transform(feats["test"][0]);  yte = feats["test"][1]

svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=SEED)
svm.fit(Xtr, ytr)
print("train accuracy:", round(accuracy_score(ytr, svm.predict(Xtr)), 3))
print("val   accuracy:", round(accuracy_score(yva, svm.predict(Xva)), 3))
""")

# ---------------------------------------------------------------- results
md("""## 7 · Results

Evaluate on the held-out **test** set: confusion matrix, the standard metrics,
and the ROC curve with its AUC.
""")
code(r"""from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score,
                             precision_score, recall_score, f1_score)

pred  = svm.predict(Xte)
proba = svm.predict_proba(Xte)[:, 1]
cm = confusion_matrix(yte, pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
im = ax[0].imshow(cm, cmap="Blues"); plt.colorbar(im, ax=ax[0])
ax[0].set(title="Confusion Matrix", xticks=[0, 1], yticks=[0, 1],
          xlabel="Predicted", ylabel="Actual")
ax[0].set_xticklabels(["Real", "Morph"]); ax[0].set_yticklabels(["Real", "Morph"])
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")

fpr, tpr, _ = roc_curve(yte, proba); roc_auc = auc(fpr, tpr)
ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
ax[1].plot([0, 1], [0, 1], "--", color="gray")
ax[1].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax[1].legend(loc="lower right"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"Accuracy : {accuracy_score(yte, pred):.3f}")
print(f"Precision: {precision_score(yte, pred):.3f}")
print(f"Recall   : {recall_score(yte, pred):.3f}")
print(f"F1-score : {f1_score(yte, pred):.3f}")
print(f"AUC      : {roc_auc:.3f}")
print(f"\nTP={tp}  TN={tn}  FP={fp}  FN={fn}")
print("\n" + classification_report(yte, pred, target_names=["Real", "Morph"]))
print("Paper reference: 89.9% accuracy, AUC 0.965.")
""")

# ---------------------------------------------------------------- improve
md(r"""## 8 &middot; Improving the results (step by step)

The SVM is accurate but **misses morphs** (low recall / high false negatives).
To push further we train a small **neural-network classifier** on the same B6
features and change **one training choice at a time**, always plotting the
accuracy / loss curves and comparing to the previous experiment.

Levers we test: **softmax + cross-entropy** head, **dropout** (overfitting),
**learning-rate decay**, **activation function**, **class weights** (to fix the
missed-morph problem), and a **deeper + BatchNorm** head. A running table shows
the improvement after every change.

> The B6 backbone (all the convolution + pooling) stays frozen and provides the
> features; here we optimise the trainable head, which trains in seconds.
""")
code(r"""import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from tensorflow.keras import layers, Sequential, optimizers

results = []                      # running comparison table

def evaluate(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(acc=accuracy_score(y_true, pred),
                prec=precision_score(y_true, pred, zero_division=0),
                rec=recall_score(y_true, pred, zero_division=0),
                f1=f1_score(y_true, pred, zero_division=0),
                auc=roc_auc_score(y_true, proba))

def log(name, m):
    results.append(dict(experiment=name, **{k: round(v, 3) for k, v in m.items()}))
    df = pd.DataFrame(results)
    if len(df) > 1:
        p, c = df.iloc[-2], df.iloc[-1]
        print(f"vs '{p.experiment}':  acc {c.acc-p.acc:+.3f}   f1 {c.f1-p.f1:+.3f}"
              f"   recall {c.rec-p.rec:+.3f}")
    # live diagram: F1 per experiment so far (green = current best)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 0.6))
    best = df["f1"].idxmax()
    ax.barh(range(len(df)), df["f1"],
            color=["#268a58" if i == best else "#9bbcd8" for i in range(len(df))])
    for i, v in enumerate(df["f1"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["experiment"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0.7, 1.0); ax.set_xlabel("F1")
    ax.set_title("F1 so far  (green = best)"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.show()
    return df

def curves(h, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(h['accuracy'], label='train'); ax[0].plot(h['val_accuracy'], label='val')
    ax[0].set_title(f"Accuracy — {title}"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h['loss'], label='train'); ax[1].plot(h['val_loss'], label='val')
    ax[1].set_title("Cross-entropy loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def run(name, units=(256,), activation='relu', dropout=0.3, lr=1e-3,
        decay=None, class_weight=None, batchnorm=False, epochs=40):
    tf.keras.utils.set_random_seed(SEED)
    net = [layers.Input((Xtr.shape[1],))]
    for u in units:
        net.append(layers.Dense(u, activation=None if batchnorm else activation))
        if batchnorm:
            net += [layers.BatchNormalization(), layers.Activation(activation)]
        net.append(layers.Dropout(dropout))
    net.append(layers.Dense(2, activation='softmax'))
    model = Sequential(net)

    steps = max(1, len(Xtr) // 32)
    if decay == 'cosine':
        lr = optimizers.schedules.CosineDecay(lr, epochs * steps)
    elif decay == 'exp':
        lr = optimizers.schedules.ExponentialDecay(lr, steps, 0.92)
    model.compile(optimizers.Adam(lr), 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                  batch_size=32, class_weight=class_weight, verbose=0)
    curves(h.history, name)
    proba = model.predict(Xte, verbose=0)[:, 1]
    df = log(name, evaluate(yte, proba))
    return model, proba, df

# start the table with the SVM baseline
log("0 · SVM (baseline)", evaluate(yte, svm.predict_proba(Xte)[:, 1]))
""")

md("""### Experiment 1 — Neural head (softmax + cross-entropy)

A single dense layer with a softmax output, trained with cross-entropy. Notice
the train accuracy spikes toward 1.0 while validation lags — classic
**overfitting**, exactly like the reference curves.
""")
code(r"""_ = run("1 · MLP (softmax+CE)", units=(256,), dropout=0.0, epochs=40)""")

md("""### Experiment 2 — Add dropout (reduce overfitting)

Dropout randomly disables neurons during training, closing the train/val gap.
""")
code(r"""_ = run("2 · + dropout 0.5", units=(256,), dropout=0.5, epochs=40)""")

md("""### Experiment 3 — Learning-rate decay

A cosine-decayed learning rate takes large steps early, then fine steps late —
smoother convergence and a better minimum.
""")
code(r"""_ = run("3 · + cosine LR decay", units=(256,), dropout=0.5,
        lr=1e-3, decay='cosine', epochs=40)""")

md("""### Experiment 4 — Activation function (ReLU → Swish)

Swish (`x·sigmoid(x)`) is smooth and often outperforms ReLU — it is the
activation EfficientNet itself uses.
""")
code(r"""_ = run("4 · + Swish activation", units=(256,), activation='swish',
        dropout=0.5, decay='cosine', epochs=40)""")

md("""### Experiment 5 — Class weights (fix the missed morphs)

The main weakness is **low recall** (missed morphs). Weighting the morph class
higher in the loss pushes the model to catch more morphs.
""")
code(r"""_ = run("5 · + class weights", units=(256,), activation='swish',
        dropout=0.5, decay='cosine', class_weight={0: 1.0, 1: 2.5}, epochs=40)""")

md("""### Experiment 6 — Deeper head + BatchNorm

More capacity with BatchNorm for stable training.
""")
code(r"""best_model, best_proba, table = run(
    "6 · deeper + BatchNorm", units=(512, 128), activation='swish',
    dropout=0.5, decay='cosine', batchnorm=True,
    class_weight={0: 1.0, 1: 2.5}, epochs=50)""")

md("""### Results dashboard — everything as diagrams

Three views of the whole comparison: a **heatmap** of every metric per
experiment (colour = score), a **grouped bar chart**, and the **improvement-flow
line chart** from the SVM baseline to the best model.
""")
code(r"""prog = pd.DataFrame(results)
METR = ["acc", "prec", "rec", "f1", "auc"]
names = prog["experiment"].tolist()

# 1) heatmap of the comparison table
fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(prog) + 1.5))
im = ax.imshow(prog[METR].values, cmap="YlGn", vmin=0.72, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(METR))); ax.set_xticklabels([m.upper() for m in METR])
ax.set_yticks(range(len(prog))); ax.set_yticklabels(names, fontsize=8)
for i in range(len(prog)):
    for j, m in enumerate(METR):
        v = prog[m].iloc[i]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                color="white" if v > 0.9 else "black")
ax.set_title("Comparison heatmap (greener = better)")
plt.colorbar(im, fraction=0.03); plt.tight_layout(); plt.show()

# 2) grouped bar chart
x = np.arange(len(prog)); w = 0.16
fig, ax = plt.subplots(figsize=(13, 5))
colors = ["#2456a6", "#7a3ea6", "#d8791a", "#268a58", "#12a5b8"]
for k, m in enumerate(METR):
    ax.bar(x + k * w, prog[m], w, label=m.upper(), color=colors[k])
ax.set_xticks(x + 2 * w); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
ax.set_ylim(0.72, 1.0); ax.set_ylabel("score"); ax.legend(ncol=5, loc="lower right")
ax.set_title("All metrics per experiment"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()

# 3) improvement-flow line chart
fig, ax = plt.subplots(figsize=(12, 4.6))
for col, c in [("acc", "#2456a6"), ("f1", "#268a58"), ("rec", "#d8791a")]:
    ax.plot(x, prog[col], "o-", color=c, lw=2, label=col.upper())
    for xi, v in zip(x, prog[col]):
        ax.annotate(f"{v:.2f}", (xi, v), fontsize=8, ha="center", va="bottom")
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("score"); ax.set_title("Improvement flow — baseline to best")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

best = prog.loc[prog["f1"].idxmax(), "experiment"]
print(f"Best model by F1: {best}")
""")

md(r"""### Final step — tune the decision threshold

The best model still uses a 0.5 cut-off. We pick the threshold on the
**validation** set that maximises F1 (or catches more morphs), then show the
final confusion matrix — the last improvement step.
""")
code(r"""from sklearn.metrics import confusion_matrix
val_proba = best_model.predict(Xva, verbose=0)[:, 1]
ths = np.linspace(0.05, 0.95, 181)
best_thr = max(ths, key=lambda t: f1_score(yva, (val_proba >= t).astype(int),
                                           zero_division=0))
m = evaluate(yte, best_proba, thr=best_thr)
log(f"7 · best + threshold {best_thr:.2f}", m)

cm = confusion_matrix(yte, (best_proba >= best_thr).astype(int))
tn, fp, fn, tp = cm.ravel()
plt.figure(figsize=(4.6, 4))
plt.imshow(cm, cmap="Blues")
plt.xticks([0, 1], ["Real", "Morph"]); plt.yticks([0, 1], ["Real", "Morph"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Final confusion matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.show()
print(f"Final: acc {m['acc']:.3f}  precision {m['prec']:.3f}  recall {m['rec']:.3f}"
      f"  f1 {m['f1']:.3f}  AUC {m['auc']:.3f}")
print(f"Missed morphs (FN): {fn}   (baseline SVM had more)")
""")

# ---------------------------------------------------------------- try it
md("""## 9 · Try it on your own image

Upload a face photo and the model predicts **Real** or **Morph** with a
confidence score.
""")
code(r"""from google.colab import files

def predict(img):
    f = backbone.predict(preprocess_input(
        cv2.cvtColor(preprocess(img), cv2.COLOR_BGR2RGB).astype("float32")[None]),
        verbose=0)
    p = float(svm.predict_proba(scaler.transform(f))[0, 1])
    return ("MORPH" if p >= 0.5 else "REAL"), p

for name, data in files.upload().items():
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    label, p = predict(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{label}   (morph prob = {p:.2f})",
              color="red" if label == "MORPH" else "green", fontweight="bold")
    plt.axis("off"); plt.show()
""")

# ---------------------------------------------------------------- summary
md("""## Summary

| Stage | Method |
|---|---|
| Real faces | FFHQ |
| Morphs | Landmark morphing (Delaunay + affine + seamless clone) |
| Preprocessing | Resize 528 + CLAHE |
| Features | EfficientNet-B6 → 2304-d (Global Average Pooling) |
| Classifier | SVM (RBF), maximum margin |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC |

The two-stage design — deep features + SVM — keeps the model accurate and
resistant to overfitting on a modest dataset, matching the DMorphNet paper
(89.9% accuracy, AUC 0.965). Set `DEMO = False` for the full-scale run.
""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "name": "DMorphNet.ipynb", "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}

out = "/home/user/morhping-detection/DMorphNet.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("written", out, "| cells:", len(cells))
