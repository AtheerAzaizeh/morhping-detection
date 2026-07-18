# Part 2 — Morphing-Attack Detection (D-MorphNet): SDLC Engineering Report

*Notebook: `notebooks/Part2_Morph_Detection_DMorphNet.ipynb` · Reference paper: Gawade et al., “DMorphNet: Face Morphing Detection Using Generative Adversarial Networks and EfficientNet-B6”, COMPUTATIA 2026.*

This report walks through the project as an ML engineer would run it through the **Software/ML Development Life Cycle (SDLC)**: requirements → data engineering → design → implementation → training → evaluation → deployment → maintenance. Each phase shows the actual code used in the notebook and explains what it does.

---

## Phase 1 — Requirements analysis

**Problem.** Face-recognition systems (border control, passports, e-KYC) can be fooled by *morphing attacks*: a single image blended from two people matches both. The system must classify a face image as **real** or **morph**.

**Functional requirements** (from the paper): CLAHE + 528×528 standardization, realistic augmentation, EfficientNet-B6 deep-feature extraction (2304-d GAP vector), SVM (RBF) classification, evaluation with accuracy/precision/recall/F1, confusion matrix, ROC-AUC, threshold optimization, comparison of EfficientNet variants, and near-real-time single-image prediction.

**Non-functional requirements**: identity-disjoint evaluation (no identity in both train and test), reproducibility (fixed seeds, cached features), CPU-only operation.

## Phase 2 — Data engineering

The dataset comes from **Part 1**: 50 real face crops (512×512) and 170 landmark-morphs, organized as `data/dataset/{train|test}/{real|morph}` with a `manifest.csv`. Crops are grouped **by source photograph**, and whole groups go to one side of the split only — the paper's *"the model doesn't see the same identities in both training and testing"* protocol. A validation set is carved out of the training bases the same way:

```python
files = base['train'][cls][:]
rng.shuffle(files)
n_val = max(4, int(0.2 * len(files)))
val[cls], train[cls] = files[:n_val], files[n_val:]
```

**Standardization (paper §3.2)** — CLAHE on the L channel of LAB (contrast enhancement without colour distortion), then resize to the B6 input size:

```python
def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def standardize(img_bgr, size=528):
    return cv2.resize(apply_clahe(img_bgr), (size, size), interpolation=cv2.INTER_AREA)
```

**Augmentation (paper §3.2)** — flips, brightness/contrast, Gaussian blur, sensor noise and JPEG re-compression simulate real acquisition conditions; each split/class is expanded to a fixed target (train 160, val 40, test 44 per class) so classes stay balanced:

```python
if rng.random() < 0.7:                      # JPEG re-compression
    q = rng.randint(45, 90)
    _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
```

## Phase 3 — System design

The paper's key design decision is a **hybrid two-stage architecture** — deep *feature extraction* separated from *classification* — because an end-to-end deep model overfits a small dataset:

```
image → CLAHE+resize → EfficientNet-B6 (frozen, ImageNet) → GAP → F ∈ R²³⁰⁴
      → StandardScaler → SVM (RBF)  → real / morph
```

EfficientNet-B6 was chosen for its accuracy/efficiency balance (compound scaling of depth, width, resolution); the SVM for robustness in high-dimensional feature spaces on small data.

## Phase 4 — Implementation: feature extraction

The backbone is instantiated without its classification top; `pooling='avg'` performs the paper's Global-Average-Pooling step `F = (1/N) Σᵢ X_L⁽ⁱ⁾`, producing one 2304-d vector per image. Features are cached to `.npz` so re-runs are cheap:

```python
model = keras.applications.EfficientNetB6(include_top=False, weights='imagenet',
                                          pooling='avg', input_shape=(528, 528, 3))
for i in range(0, len(X), batch):
    out.append(model.predict(X[i:i+batch].astype(np.float32), verbose=0))
```

The notebook also *opens the network up* (Section 5): parameters per stage, learned stem filters, activation maps from shallow to deep layers for a real vs a morphed face, the raw 2304-d feature vectors, and PCA/t-SNE projections of the feature space — real and morph separate partially (many small per-base-image clusters) before any classifier is trained, consistent with the ~0.93 test AUC.

## Phase 5 — Model training

**(a) Transfer-learning head (deep-learning stage).** Reproducing the paper's "freeze the backbone, train the top" setting, a small dense head is trained on the frozen features; this stage supplies the **training/validation loss and accuracy curves** (`results/part2_training_curves.png`):

```python
head = keras.Sequential([
    keras.layers.Input(shape=(2304,)),
    keras.layers.Dense(256, activation='swish'), keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='swish'),  keras.layers.Dropout(0.25),
    keras.layers.Dense(1, activation='sigmoid')])
head.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
hist = head.fit(Ztr, ytr, validation_data=(Zva, yva), epochs=60, batch_size=32)
```

**(b) SVM classifier (paper §3.5).** Features are standardized, then C and γ are selected by 3-fold grid search; a learning curve shows accuracy vs training-set size:

```python
grid = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=SEED),
                    {'C': [1, 10, 100], 'gamma': ['scale', 1e-3, 1e-4]},
                    cv=3, scoring='accuracy')
grid.fit(Ztr, ytr)
```

Selected hyper-parameters: **{'C': 10, 'gamma': 0.0001}** (CV accuracy 0.981 — optimistic, because augmented copies of one base image can land in different CV folds; the honest number is the identity-disjoint test score below).

## Phase 6 — Evaluation (identity-disjoint test set)

| Metric | Value |
|---|---|
| Accuracy | **0.852** |
| Precision (morph) | 0.897 |
| Recall (morph) | 0.795 |
| F1 (morph) | 0.843 |
| ROC-AUC | **0.935** |
| Confusion matrix | TN=40 FP=4 FN=9 TP=35 |
| Neural head (test acc.) | 0.875 |

**Threshold optimization (paper §4.4).** The decision threshold on P(morph) is swept on the *validation* set; the F1-optimal value 0.95 applied to the test set gives accuracy 0.773 — *worse* than the default 0.5 (0.852). With only 80 validation images the tuned threshold overfits, so the default operating point is kept; this is a useful negative result showing threshold calibration needs a large calibration set. In deployment the threshold would instead cap the false-negative (missed-morph) rate, since undetected morphs are the costly error in border control.

**Comparative analysis (paper §4.2).** The pipeline was repeated with EfficientNet-B0 (1280-d, softmax baseline) and B5 (2048-d, linear SVM):

| Model | Feature dim | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| EfficientNet-B0 (softmax) | 1280 | 0.920 | 0.97 | 0.86 | 0.92 |
| EfficientNet-B5 (SVM linear) | 2048 | 0.909 | 0.93 | 0.89 | 0.91 |
| EfficientNet-B6 (SVM RBF) — proposed | 2304 | 0.852 | 0.90 | 0.80 | 0.84 |

Honest reading: on this small test set (44+44 images) the smaller backbones matched or beat B6+RBF — a few images swing accuracy by several points, and our landmark-based morphs carry strong low-level artifacts that shallow features already catch. The paper's ordering (B0 < B5 < B6) was obtained on a much larger GAN-morph dataset where subtler artifacts dominate; reproducing it would require that scale and type of data.

## Phase 7 — Deployment: real-time prediction

The full inference path is packaged as one function and demonstrated on unseen identities (`results/part2_realtime_demo.png`), reporting per-image latency (≈1–2 s on CPU, dominated by the B6 forward pass):

```python
def predict_image(img_bgr):
    x = standardize(img_bgr).astype(np.float32)[None]
    f = rt_model.predict(x, verbose=0)
    p = svm.predict_proba(scaler.transform(f))[0, 1]
    return ('MORPH' if p > best_t else 'REAL'), p
```

## Phase 8 — Maintenance & known limitations

- **Cross-generator generalization is untested**: all morphs come from one landmark-based generator; the paper's GAN/AI-FaceSwap morphs would require re-training or at least re-calibration. New morphing tools should trigger dataset refresh + re-evaluation (model monitoring).
- **Small, celebrity-biased dataset** (50 real bases): metrics carry high variance; confidence intervals would need a larger pool.
- **Compute**: B6 at 528×528 costs ~1 s/image on CPU; for high-throughput gates a distilled or smaller backbone (with the same hybrid design) is the natural optimization.
- Retraining is cheap by design: features are cached, and only the scaler + SVM (+ threshold) need refitting when data changes.
