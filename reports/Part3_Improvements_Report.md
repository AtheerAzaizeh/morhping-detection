# Part 3 — Code Review & Professional Improvements

*Notebook: `notebooks/Part3_Professional_Improvements.ipynb` · Package: `src/dmorphnet/` · Tests: `tests/` (12 unit tests).*

Part 2 was a faithful prototype of the paper. This pass reviews it like production code, then applies one improvement per step, **measuring every step on the same frozen identity-disjoint test set** (44 real + 44 morph) so each claim is a number.

## 1. Code-review findings

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | High | **Grouped-data leakage in model selection** — `GridSearchCV(cv=3)` let augmented copies of the same base image sit in different CV folds; CV accuracy 0.981 was fiction. | `GroupKFold` by base image (step 1) |
| 2 | High | **Threshold tuned on 80 images overfit** (85.2% → 77.3% on test); SVM scores never calibrated. | Grouped out-of-fold threshold + Platt calibration (step 4) |
| 3 | High | **Training data under-used** — 130 of 741 possible train morph pairs. | 400 samples/class (step 2) |
| 4 | Medium | **No MAD-standard metrics** (APCER/BPCER/DET/EER, ISO/IEC 30107-3), no confidence intervals on an 88-image test set. | Step 5 |
| 5 | Medium | **Cross-generator generalization untested** — train and test morphs share one generator. | Spliced-morph probe (step 6) |
| 6 | Medium | **Notebook-only logic; app duplicated preprocessing and re-fit the SVM with hard-coded params**; feature caches stored X without labels/ids. | `src/dmorphnet/` package, self-describing caches, single versioned deployment artifact |
| 7 | Low | `SVC(probability=True)` deprecated and internally CV-leaked for grouped data. | `CalibratedClassifierCV(..., ensemble=False)` with grouped folds |
| 8 | Low | B5 features computed for a table, then discarded. | Fusion experiment (step 3) |
| 9 | Medium | **Augmentation imbalance across classes** (train reals ~75% augmented vs morphs ~19%) — a detector can learn "JPEG/noise ⇒ real". | Augment-all training distribution (step 2) |

## 2. Results per improvement step (fixed test set)

| Step | Accuracy | F1 | ROC-AUC | EER |
|---|---|---|---|---|
| 0 · baseline (Part 2 as-is) | 0.875 | 0.876 | 0.935 | 0.136 |
| 1 · grouped-CV model selection | 0.784 | 0.753 | 0.917 | 0.102 |
| 2 · + expanded training data | 0.784 | 0.753 | 0.920 | 0.205 |
| 3a · fusion B5⊕B6 (concat) | 0.852 | 0.827 | 0.924 | 0.205 |
| 3b · fusion B5+B6 (score average) | 0.841 | 0.825 | 0.935 | 0.193 |
| 4 · champion @ P-bal threshold | 0.784 | 0.753 | 0.920 | 0.205 |

Step chart: `results/part3_step_progress.png`.

Champion pipeline: **B6 + SVC-RBF (calibrated)**. From baseline to final: accuracy 0.875 → 0.784, AUC 0.935 → 0.920, EER 0.136 → 0.205.

**How to read this honestly.** The leakage demonstration landed almost perfectly: naive CV claimed **0.981**, grouped CV said **0.872**, and the same model's true test accuracy was **0.875** — grouped validation predicts reality. The honest champion's lower *point* accuracy (0.784, CI 0.705–0.864) overlaps the baseline's: on 88 test images these differences are noise, and the baseline's edge came from hyper-parameters that leaked CV happened to tune to this tiny test set. What improved reliably: the grouped-CV estimate rose 0.883 → 0.891 with expanded data, probabilities are calibrated, thresholds come from out-of-fold predictions, fusion was measured and rejected (+0.004 AUC ≠ 2× latency), and every claim now carries a confidence interval. (Step 0 shows 0.875 vs Part 2's 85.2% only because probabilities ≥ 0.5, rather than the SVM decision sign, are thresholded.)

## 3. Final evaluation (MAD-grade)

| Metric | Value |
|---|---|
| Accuracy @ balanced threshold (0.56) | **0.784** (95% CI 0.705–0.864) |
| ROC-AUC | **0.920** (95% CI 0.861–0.968) |
| EER | 0.205 |
| BPCER @ APCER ≤ 10% | 0.182 |
| Security threshold (APCER ≤ 10% policy) | 0.585 |

Artifacts: DET curve (`results/part3_det_curve.png`), calibration/reliability (`results/part3_calibration.png`), cross-generator probe (`results/part3_crossgen.png`, `results/part3_spliced_probe.png`).

## 4. Cross-generator probe — the honest caveat

Thirty **spliced morphs** (morphed face region seamlessly cloned into a genuine test-identity photo — an attack generator the model never saw) were scored:

APCER on seen-generator (landmark) morphs: **0.341** — APCER on unseen spliced morphs: **0.967**.

This gap is the central open problem of morphing-attack detection: detectors latch onto generator-specific artifacts. A real deployment would require training morphs from several generators (GAN, diffusion, splicing, print-scan) and continuous re-evaluation as new tools appear.

## 5. Engineering changes

- **`src/dmorphnet/`** — config, preprocessing, data access, morph generation, feature extraction, and evaluation as an importable, unit-tested package (12 tests, `pytest tests/`). Notebook and web app import the same functions — no more copy-drift.
- **Self-describing feature caches** — `.npz` now stores features **+ labels + group ids**; nothing is reconstructed by convention.
- **One versioned deployment artifact** — `app/model/pipeline.joblib` bundles backbone list, per-backbone scalers, the calibrated SVM, and both operating thresholds (balanced + APCER≤10% security policy). The notebook verifies after export that reloading the artifact reproduces the test probabilities bit-for-bit, and the Flask app consumes only this artifact.
- **Identity-disjointness is asserted in code** (`assert_identity_disjoint`), not assumed.
