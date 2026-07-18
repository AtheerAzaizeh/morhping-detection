# Face Morphing & Morphing-Attack Detection (D-MorphNet)

Two-part course project:

1. **Part 1 — Morph generation** (`notebooks/Part1_Face_Morphing.ipynb`)
   Landmark-based face morphing with OpenCV + MediaPipe: 478-point face mesh → Delaunay
   triangulation on the interpolated shape → piece-wise affine warping → cross-dissolve.
   Deliverables: 6 pairs of originals with their morphs (`results/morph_pairs/`), a morph
   α-sequence, and a labelled real/morph dataset (`data/dataset/`) with an
   **identity-disjoint** train/test split.

2. **Part 2 — Morph detection** (`notebooks/Part2_Morph_Detection_DMorphNet.ipynb`)
   Implementation of *“DMorphNet: Face Morphing Detection Using GANs and EfficientNet-B6”*
   (Gawade et al., COMPUTATIA 2026): CLAHE + 528×528 standardization, augmentation
   (JPEG/blur/noise/brightness/flip), **EfficientNet-B6** deep features (2304-d GAP),
   transfer-learning head (training curves), **SVM-RBF** classification, confusion matrix,
   ROC-AUC, threshold optimization, EfficientNet B0/B5/B6 comparison, network-internals
   visualizations (filters, activation maps, PCA/t-SNE) and a real-time prediction demo.

## Improvement rounds (professional engineering passes)

- `notebooks/Part3_Professional_Improvements.ipynb` — code review of Part 2 + one measured
  fix per step: leakage-free `GroupKFold` selection, expanded balanced data, fusion trial,
  calibration, out-of-fold thresholds, MAD metrics (DET/EER/APCER/BPCER, bootstrap CIs) and
  the cross-generator probe. Report: `reports/Part3_Improvements_Report.md`.
- `notebooks/Part2_Improved_DMorphNet.ipynb` — **the article-faithful refinement**: stays
  strictly inside the paper's EfficientNet-B6 + SVM-RBF recipe and adds what Part 2 skipped —
  a true §3.4 fine-tune of B6's top layers (block7, via model surgery on CPU), the full data
  recipe, and §4.4 threshold optimization done on out-of-fold predictions. Step-by-step
  results in the notebook; exports the deployed `app/model/pipeline.joblib`.
- `notebooks/Part4_Advanced_Improvements.ipynb` — exploratory round beyond the paper:
  classifier zoo, frequency features, TTA, multi-generator training (closes the spliced-morph
  hole: APCER 0.97 → 0.50), fine-tuned B0 comparison. Report:
  `reports/Part4_Advanced_Improvements_Report.md`.

The pipeline itself lives in **`src/dmorphnet/`** (unit-tested: `pytest tests/`).

## Try the model — web demo

A small Flask app lets you upload any photo and get a live **real / morph** verdict
(face detection → 512² crop → CLAHE → EfficientNet-B6 features → SVM):

```bash
pip install -r requirements.txt
python app/export_model.py     # one-time: builds app/model/ from the cached B6 features
python app/server.py           # open http://localhost:7860
```

The page shows the detected face crop, the verdict, P(morph) as a meter, and the CPU
latency (~0.5–2 s per image after warm-up).

## Reports

- `reports/Part1_Morphing_Report.md` — method, landmark alignment, interpolation, observations & limitations.
- `reports/Part2_SDLC_Report.md` — step-by-step SDLC ML-engineering report with code excerpts and final metrics.

## Repository layout

```
notebooks/   the two executed notebooks (all outputs embedded)
data/        raw downloads → face crops → labelled dataset (+ manifest.csv)
models/      MediaPipe face detector / landmarker model files
results/     all figures, morph pairs, cached CNN features, metrics json
reports/     written reports
```

## Reproducing

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/Part1_Face_Morphing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/Part2_Morph_Detection_DMorphNet.ipynb
```

Both notebooks are idempotent: downloads, face crops, the morph dataset and the CNN feature
caches are reused when present. Part 2 takes ~25 min on a modern CPU (EfficientNet-B6 at
528×528 is ~1 s/image).
