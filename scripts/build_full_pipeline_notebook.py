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
| 2-3 | FFHQ real faces + **Delaunay** morph generation (Subdiv2D + piecewise-affine warp) |
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
md("""## Part 2 — Real faces from FFHQ (Flickr-Faces-HQ)

The **real / bona-fide** class is FFHQ — 70,000 high-quality real faces. Morphs
are then generated from FFHQ pairs (Part 3), which is exactly how the FFHQ-Morphs
benchmark is built.

We use a **256px Kaggle mirror** (Colab-feasible). The official NVIDIA 1024px set
is 89 GB over Google Drive and will not finish on Colab; images are resized to
528 for EfficientNet-B6 anyway, so 256px is plenty. To use full-res instead,
download it with NVIDIA's `download_ffhq.py` and point `FFHQ_DIR` at it.
""")
code(r'''import kagglehub, os, glob

FFHQ_SLUG = "xhlulu/flickrfaceshq-dataset-nvidia-resized-256px"   # 256px mirror
FFHQ_DIR  = ""     # optional: set to a local FFHQ folder to skip the download

DATA_ROOT = FFHQ_DIR if FFHQ_DIR else kagglehub.dataset_download(FFHQ_SLUG)
all_jpgs = [p for p in glob.glob(os.path.join(DATA_ROOT, "**", "*"), recursive=True)
            if p.lower().endswith((".png", ".jpg", ".jpeg"))]
# basename -> full path (FFHQ names are unique, even across subfolders)
img2path = {os.path.basename(p): p for p in all_jpgs}
IMG_DIR = os.path.dirname(all_jpgs[0])
print("FFHQ real faces:", len(all_jpgs), "| folder:", IMG_DIR)
''')

md("""### 2.1 Identities (FFHQ has distinct people)

FFHQ contains distinct individuals with no repeated identity, so **each image is
its own identity**. This makes the train/val/test split identity-clean by
construction and guarantees every generated morph pair is two different people.
""")
code(r'''import pandas as pd

names = sorted(os.path.basename(p) for p in all_jpgs)
ident = pd.DataFrame({"image_id": names, "identity": np.arange(len(names))})
print("images:", len(ident), "| distinct identities:", ident["identity"].nunique())
ident.head()
''')

# ================================================================ PART 3
md("""## Part 3 — Dataset Construction and Morph Generation

### 3.1 Identity split and real image selection

Every person goes with ALL of their images into exactly one split (train OR val OR
test) — morphs are later generated only from people inside the same split, so no
identity ever leaks between splits.
""")
code(r"""by_id = ident.groupby("identity")["image_id"].apply(list).to_dict()
img2id = dict(zip(ident["image_id"], ident["identity"]))

available = {os.path.basename(p) for p in all_jpgs}
by_id = {k: [f for f in v if f in available] for k, v in by_id.items()}
by_id = {k: v for k, v in by_id.items() if v}

identities = list(by_id)
random.shuffle(identities)

selected, id_iter = {}, iter(identities)
for split, tgt in SPLITS.items():
    images, sids = [], set()
    while len(images) < tgt["real"]:
        pid = next(id_iter, None)
        if pid is None:                     # guard: identities exhausted
            raise RuntimeError("Not enough identities for the requested targets")
        sids.add(pid)
        images.extend(by_id[pid])
    selected[split] = {"identities": sids, "images": images[: tgt["real"]]}
    print(f"{split}: {len(selected[split]['images'])} real images "
          f"from {len(sids)} identities")

for a in SPLIT_NAMES:
    for b in SPLIT_NAMES:
        if a < b:
            assert not (selected[a]["identities"] & selected[b]["identities"])
print("✅ Identities are fully disjoint across the three splits")
""")

md("""### 3.2 Morph generation — Delaunay triangulation + piecewise-affine warp

The dataset morphs are built with the classic **Beier-Neely / Delaunay landmark
morphing** used in the face-morphing-attack literature (e.g. Ferrara et al., 2014):

1. **Dense landmarks** — a MediaPipe FaceLandmarker 478-point mesh per face, plus
   8 fixed frame-boundary points so the whole image (not just the face)
   participates in the warp.
2. **Delaunay triangulation** — triangulate the *averaged* landmark set with
   `cv2.Subdiv2D`, giving a shared triangle mesh indexed identically on both faces.
3. **Piecewise-affine warp** — every triangle of face A and of face B is
   affine-warped to the morph shape `(1-α)pA + α pB`.
4. **Cross-dissolve** — the two warped faces are blended `(1-α)WA + α WB`.
5. **Face-only composite** — because hair and background have no landmarks, a raw
   full-frame cross-dissolve would ghost them; instead the blended **face** is
   pasted (convex-hull mask) onto **one** subject's warped photo, and
   **Poisson `seamlessClone`** harmonizes the seam. This is how a real morph
   attack is assembled: the photo passes as one subject, with the blended
   identity confined to the face.

α = 0.5 gives an equal blend of both identities.
""")
code(r'''import cv2, os, urllib.request
import numpy as np
import mediapipe as mp

# ---- MediaPipe FaceLandmarker (Tasks API) -> 478 dense landmarks ----
_LM_MODEL = "/content/face_landmarker.task"
if not os.path.exists(_LM_MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task", _LM_MODEL)
from mediapipe.tasks.python import vision as _mpv
from mediapipe.tasks import python as _mpp
_landmarker = _mpv.FaceLandmarker.create_from_options(
    _mpv.FaceLandmarkerOptions(
        base_options=_mpp.BaseOptions(model_asset_path=_LM_MODEL),
        running_mode=_mpv.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

MORPH_WORK = 384   # common canvas so both faces share one coordinate frame


def _boundary_points(w, h):
    # 8 fixed frame points so the whole image participates in the warp
    return np.array([[0, 0], [w // 2, 0], [w - 1, 0],
                     [0, h // 2], [w - 1, h // 2],
                     [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]],
                    dtype=np.float64)


def get_landmarks(img_bgr):
    # 478 face-mesh points + 8 boundary points; returns (pts, img) or (None, None)
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    fp = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]],
                  dtype=np.float64)
    pts = np.vstack([fp, _boundary_points(w, h)])
    return pts, img_bgr


def delaunay_triangulation(points, size):
    # Delaunay via cv2.Subdiv2D; returns triangles as index-triplets into points
    w, h = size
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    point_index = {(round(p[0], 1), round(p[1], 1)): i for i, p in enumerate(points)}
    triangles = []
    for t in subdiv.getTriangleList():
        tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if not all(0 <= x < w and 0 <= y < h for x, y in tri_pts):
            continue
        idx = []
        for x, y in tri_pts:
            key = (round(x, 1), round(y, 1))
            if key not in point_index:
                d = np.hypot(points[:, 0] - x, points[:, 1] - y)
                idx.append(int(np.argmin(d)))
            else:
                idx.append(point_index[key])
        if len(set(idx)) == 3:
            triangles.append(tuple(idx))
    return triangles


def _warp_triangle(src_img, dst_img, tri_src, tri_dst):
    # affine-warp one triangular patch from src into dst (border-safe)
    r_src = cv2.boundingRect(np.float32([tri_src]))
    r_dst = cv2.boundingRect(np.float32([tri_dst]))
    tri_src_rect = [(p[0] - r_src[0], p[1] - r_src[1]) for p in tri_src]
    tri_dst_rect = [(p[0] - r_dst[0], p[1] - r_dst[1]) for p in tri_dst]
    src_patch = src_img[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
    if src_patch.size == 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
        return
    mat = cv2.getAffineTransform(np.float32(tri_src_rect), np.float32(tri_dst_rect))
    warped = cv2.warpAffine(src_patch, mat, (r_dst[2], r_dst[3]), None,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), (1.0, 1.0, 1.0), cv2.LINE_AA)
    H, W = dst_img.shape[:2]
    x0, y0, w0, h0 = r_dst
    x1c, y1c = max(x0, 0), max(y0, 0)
    x2c, y2c = min(x0 + w0, W), min(y0 + h0, H)
    if x2c <= x1c or y2c <= y1c:
        return
    ox0, oy0 = x1c - x0, y1c - y0
    ox1, oy1 = ox0 + (x2c - x1c), oy0 + (y2c - y1c)
    warped_c = warped[oy0:oy1, ox0:ox1]
    mask_c = mask[oy0:oy1, ox0:ox1]
    dst_slice = dst_img[y1c:y2c, x1c:x2c]
    dst_slice[:] = dst_slice * (1 - mask_c) + warped_c * mask_c


def morph_images(img1, pts1, img2, pts2, alpha, triangles=None):
    # full-frame cross-dissolve morph (alpha=0 -> img1, alpha=1 -> img2)
    h, w = img1.shape[:2]
    pts1 = np.asarray(pts1, np.float64)
    pts2 = np.asarray(pts2, np.float64)
    pts_morph = (1 - alpha) * pts1 + alpha * pts2
    if triangles is None:
        triangles = delaunay_triangulation(pts_morph, (w, h))
    img1f = img1.astype(np.float32)
    img2f = img2.astype(np.float32)
    warped1 = np.zeros_like(img1f)
    warped2 = np.zeros_like(img2f)
    for (i, j, k) in triangles:
        tri_m = [tuple(pts_morph[i]), tuple(pts_morph[j]), tuple(pts_morph[k])]
        _warp_triangle(img1f, warped1,
                       [tuple(pts1[i]), tuple(pts1[j]), tuple(pts1[k])], tri_m)
        _warp_triangle(img2f, warped2,
                       [tuple(pts2[i]), tuple(pts2[j]), tuple(pts2[k])], tri_m)
    morphed = (1 - alpha) * warped1 + alpha * warped2
    return np.clip(morphed, 0, 255).astype(np.uint8), triangles


def _warp_to_shape(img, pts_src, pts_dst, triangles):
    imgf = img.astype(np.float32)
    out = np.zeros_like(imgf)
    for (i, j, k) in triangles:
        _warp_triangle(imgf, out,
                       [tuple(pts_src[i]), tuple(pts_src[j]), tuple(pts_src[k])],
                       [tuple(pts_dst[i]), tuple(pts_dst[j]), tuple(pts_dst[k])])
    return out


def render_morph(up_a, pts_a, up_b, pts_b, alpha, frame_from, triangles=None):
    # frame_from None -> raw full-frame cross-dissolve;
    # frame_from A/B -> blend face only, take hair+background from that subject
    if frame_from is None:
        return morph_images(up_a, pts_a, up_b, pts_b, alpha, triangles=triangles)
    h, w = up_a.shape[:2]
    pts_m = (1 - alpha) * pts_a + alpha * pts_b
    if triangles is None:
        triangles = delaunay_triangulation(pts_m, (w, h))
    warped_a = _warp_to_shape(up_a, pts_a, pts_m, triangles)
    warped_b = _warp_to_shape(up_b, pts_b, pts_m, triangles)
    blend = (1 - alpha) * warped_a + alpha * warped_b
    frame = warped_a if frame_from == "A" else warped_b
    n_face = len(pts_m) - 8
    hull = cv2.convexHull(pts_m[:n_face].astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    blend8 = np.clip(blend, 0, 255).astype(np.uint8)
    frame8 = np.clip(frame, 0, 255).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(hull)
    try:
        out = cv2.seamlessClone(blend8, frame8, mask,
                                (x + bw // 2, y + bh // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        soft = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), np.uint8)),
                                (31, 31), 0)
        m3 = (soft.astype(np.float32) / 255.0)[..., None]
        out = np.clip(blend * m3 + frame * (1 - m3), 0, 255).astype(np.uint8)
    return out, triangles


def morph_faces(img1, img2, alpha=MORPH_ALPHA):
    # Dataset wrapper: put both faces on a shared MORPH_WORK canvas, landmark,
    # morph, composite face-only (hair/bg from a random subject), resize to output.
    a = cv2.resize(img1, (MORPH_WORK, MORPH_WORK))
    b = cv2.resize(img2, (MORPH_WORK, MORPH_WORK))
    pts_a, up_a = get_landmarks(a)
    pts_b, up_b = get_landmarks(b)
    if pts_a is None or pts_b is None:
        return None
    out, _ = render_morph(up_a, pts_a, up_b, pts_b, alpha,
                          frame_from=random.choice(["A", "B"]))
    return cv2.resize(out, MORPH_SIZE)


def sharpness(img_bgr):
    return cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var()

# Quick smoke test: show the Delaunay morph next to its two source faces
a, b = selected["train"]["images"][:2]
src1 = cv2.imread(img2path[a])
src2 = cv2.imread(img2path[b])
test_m = morph_faces(src1, src2)
print("Delaunay morph works - output shape:",
      None if test_m is None else test_m.shape)
if test_m is not None:
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    for a_, im, t in zip(ax, [src1, src2, test_m],
                         ["source 1", "source 2", "Delaunay morph"]):
        a_.imshow(cv2.cvtColor(cv2.resize(im, MORPH_SIZE), cv2.COLOR_BGR2RGB))
        a_.set_title(t); a_.axis("off")
    plt.tight_layout(); plt.show()
''')

md(r"""> #### (Optional) GAN-based morphs — the hardest attacks
>
> The classical pipeline above produces seamless landmark morphs. The very hardest
> attacks are **GAN latent-space morphs**: encode both faces into a StyleGAN latent
> space (e.g. with an e4e/pSp encoder), interpolate the latent codes, and generate
> a brand-new synthetic face. These have no warping seams at all.
>
> They are **not enabled by default** because they need heavy pretrained models
> (StyleGAN2 + encoder, several GB) and are slow/fragile to run for 21,000 images
> inside a single Colab session. To use them, generate the GAN morphs offline,
> drop them into `RAW_DIR/<split>/morph/`, and skip Part 3.3. Mixing ~10–20% GAN
> morphs with the classical ones is a good way to make the detector robust to both
> attack families without a full StyleGAN run.
""")

md("""### 3.3 Generate the morphed images (with automatic review and Drive checkpoints)

Pick two images of **different people from the same split** → blend → automatic
review (clear face + sufficient sharpness) → save and record.

**Checkpointing / resume**: progress is saved to Drive every `CHECKPOINT_EVERY`
morphs (zip of the split's morph folder + a records CSV). If the session
disconnects, running this cell again — even on a new VM — restores from Drive and
continues from where it stopped.
""")
code(r"""LABELS_CSV = os.path.join(RAW_DIR, "labels.csv")


def _sync_split_to_drive(split, out_dir):
    # Checkpoint: zip this split's morphs + its records CSV to Drive
    if not CHECKPOINT_TO_DRIVE:
        return
    shutil.make_archive(os.path.join(CKPT_DIR, f"morph_{split}"), "zip", out_dir)
    csvp = os.path.join(RAW_DIR, f"morph_records_{split}.csv")
    if os.path.exists(csvp):
        shutil.copy(csvp, CKPT_DIR)


# Restore from Drive checkpoints when starting on a fresh VM
if CHECKPOINT_TO_DRIVE:
    for split in SPLIT_NAMES:
        out_dir = os.path.join(RAW_DIR, split, "morph")
        os.makedirs(out_dir, exist_ok=True)
        zpath = os.path.join(CKPT_DIR, f"morph_{split}.zip")
        if os.path.exists(zpath) and not glob.glob(os.path.join(out_dir, "*.jpg")):
            print(f"📥 Restoring {split} morphs from Drive checkpoint ...")
            shutil.unpack_archive(zpath, out_dir)
        csvd = os.path.join(CKPT_DIR, f"morph_records_{split}.csv")
        csvl = os.path.join(RAW_DIR, f"morph_records_{split}.csv")
        if os.path.exists(csvd) and not os.path.exists(csvl):
            shutil.copy(csvd, csvl)
    lcz = os.path.join(CKPT_DIR, "labels.csv")
    if os.path.exists(lcz) and not os.path.exists(LABELS_CSV):
        shutil.copy(lcz, LABELS_CSV)

# Full shortcut: if the entire dataset is already complete, skip generation
RESUME = False
if os.path.exists(LABELS_CSV):
    prev = pd.read_csv(LABELS_CSV)
    counts_ok = all(
        (prev[(prev.split == s) & (prev.label == l)].shape[0] >= SPLITS[s][l])
        for s in SPLITS for l in ["real", "morph"])
    files_ok = all(
        len(glob.glob(os.path.join(RAW_DIR, s, l, "*.jpg"))) >= SPLITS[s][l]
        for s in SPLITS for l in ["real", "morph"])
    if counts_ok and files_ok:
        RESUME = True
        print("⏭️ Dataset already fully generated — skipping (auto-resume)")

records, rejected = [], 0

for split, tgt in ([] if RESUME else list(SPLITS.items())):
    pool = selected[split]["images"]
    out_dir = os.path.join(RAW_DIR, split, "morph")
    os.makedirs(out_dir, exist_ok=True)

    # Partial resume inside the split: continue from where we stopped
    ckpt_csv = os.path.join(RAW_DIR, f"morph_records_{split}.csv")
    prev_recs = (pd.read_csv(ckpt_csv).to_dict("records")
                 if os.path.exists(ckpt_csv) else [])
    n_files = len(glob.glob(os.path.join(out_dir, "morph_*.jpg")))
    done = min(len(prev_recs), n_files, tgt["morph"])
    split_records = list(prev_recs[:done])
    if done:
        print(f"📥 {split}: resuming from {done}/{tgt['morph']} morphs")

    attempts = 0
    max_attempts = (tgt["morph"] - done) * 30 + 1000   # infinite-loop guard
    pbar = tqdm(total=tgt["morph"], initial=done, desc=f"morphs {split}")
    while done < tgt["morph"] and attempts < max_attempts:
        attempts += 1
        a, b = random.sample(pool, 2)
        if img2id[a] == img2id[b]:          # must be two different people
            continue
        i1 = cv2.imread(img2path[a])
        i2 = cv2.imread(img2path[b])
        if i1 is None or i2 is None:
            continue
        m = morph_faces(i1, i2)
        # --- automatic review (reject unrealistic results) ---
        if m is None or get_landmarks(m)[0] is None or sharpness(m) < MIN_SHARPNESS:
            rejected += 1
            continue
        fn = f"morph_{split}_{done:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fn), m,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        split_records.append({"filename": fn, "split": split, "label": "morph",
                              "src1": a, "src2": b,
                              "id1": img2id[a], "id2": img2id[b]})
        done += 1
        pbar.update(1)
        # Periodic Drive checkpoint so a disconnect never loses progress
        if done % CHECKPOINT_EVERY == 0:
            pd.DataFrame(split_records).to_csv(ckpt_csv, index=False)
            _sync_split_to_drive(split, out_dir)
            pbar.set_postfix_str("💾 checkpoint saved")
    pbar.close()
    pd.DataFrame(split_records).to_csv(ckpt_csv, index=False)
    _sync_split_to_drive(split, out_dir)
    records.extend(split_records)

if not RESUME:
    n_morph = sum(1 for r in records if r["label"] == "morph")
    print(f"✅ Generated {n_morph} morphs — {rejected} rejected by automatic review")
""")

md("""### 3.4 Copy real images + labels + visual review and verification
""")
code(r"""for split, tgt in ([] if RESUME else list(SPLITS.items())):
    out_dir = os.path.join(RAW_DIR, split, "real")
    os.makedirs(out_dir, exist_ok=True)
    for fn in tqdm(selected[split]["images"], desc=f"copy real {split}"):
        img = cv2.imread(img2path[fn])
        if img is None:
            continue
        out_name = f"real_{fn}"
        cv2.imwrite(os.path.join(out_dir, out_name),
                    cv2.resize(img, MORPH_SIZE),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({"filename": out_name, "split": split, "label": "real",
                        "src1": fn, "src2": None,
                        "id1": img2id[fn], "id2": None})

if RESUME:
    labels_df = pd.read_csv(LABELS_CSV)          # resume from saved labels
else:
    labels_df = pd.DataFrame(records)
    labels_df.to_csv(LABELS_CSV, index=False)
    if CHECKPOINT_TO_DRIVE:
        shutil.copy(LABELS_CSV, os.path.join(CKPT_DIR, "labels.csv"))
print(labels_df.groupby(["split", "label"]).size())

# Verification: counts + identity disjointness (including morph sources)
def ids_of(s):
    sub = labels_df[labels_df.split == s]
    return set(sub.id1.dropna()) | set(sub.id2.dropna())

for i, a in enumerate(SPLIT_NAMES):
    for b in SPLIT_NAMES[i + 1:]:
        assert not (ids_of(a) & ids_of(b)), f"Identity overlap between {a} and {b}!"
print("✅ No identity is shared between splits (morph sources included)")

# Visual review: one row of real images, one row of morphs
fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
for row, lab in enumerate(["real", "morph"]):
    sample = labels_df[labels_df.label == lab].sample(6, random_state=SEED)
    for i, (_, r) in enumerate(sample.iterrows()):
        p = os.path.join(RAW_DIR, r.split, r.label, r.filename)
        axes[row, i].imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        axes[row, i].set_title(f"{lab} ({r.split})", fontsize=9)
        axes[row, i].axis("off")
plt.tight_layout()
plt.show()
""")

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
