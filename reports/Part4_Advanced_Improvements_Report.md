# Part 4 — Advanced Improvements Report

*Notebook: `notebooks/Part4_Advanced_Improvements.ipynb`.*

Second improvement round on top of Part 3's honest protocol. All selection decisions were made
on grouped CV / out-of-fold estimates; the frozen identity-disjoint test set only reports.

## Results per step (frozen test set: 44 real + 44 landmark morphs)

| Step | Accuracy | F1 | ROC-AUC | EER |
|---|---|---|---|---|
| A1 · SVC-RBF | 0.784 | 0.753 | 0.920 | 0.205 |
| A1 · LogisticRegression | 0.818 | 0.778 | 0.923 | 0.182 |
| A1 · LinearSVC | 0.773 | 0.737 | 0.905 | 0.216 |
| A2 · champion features (freq rejected by CV) | 0.773 | 0.737 | 0.905 | 0.216 |
| A3 · + test-time augmentation | 0.784 | 0.753 | 0.904 | 0.205 |
| A4 · multi-generator training | 0.784 | 0.796 | 0.890 | 0.227 |
| A5 · fine-tuned B0 (end-to-end) | 0.864 | 0.872 | 0.964 | 0.114 |
| A5b · ensemble (champion + B0-ft) | 0.807 | 0.821 | 0.941 | 0.148 |

- **A1 classifier zoo winner** (by grouped CV): **LinearSVC**
- **Frequency-feature fusion adopted:** False
- **Test-time augmentation adopted:** False
- **Multi-generator grouped-CV balanced accuracy:** 0.737
- Operating thresholds (from grouped out-of-fold predictions): balanced 0.590,
  security (APCER ≤ 10%) 0.271

## The headline: cross-generator robustness (step A4)

Part 3 measured **APCER 0.97** on spliced morphs — a single-generator detector misses almost
every attack from an unseen tool. After adding 100 spliced morphs from *training identities*
to the training set (with matching real augmentations to keep classes balanced):

| Attack generator | APCER (missed rate) |
|---|---|
| landmark (gen #1, seen) | 0.205 |
| spliced (gen #2, now trained) | 0.500 |
| naive blend (gen #3, unseen) | 0.200 |

BPCER on test reals at the balanced threshold: 0.227.
Chart: `results/part4_multigen_apcer.png`.

The naive-blend row is the new *honest* unseen-generator probe (generator #3, never trained on).

## End-to-end fine-tuned B0 (step A5)

Training curves: `results/part4_b0_finetune_curves.png`. Scores are in the step table; the
fine-tuned CNN is a comparison point, not the deployed model — its robustness rests on the same
multi-generator data, and the sklearn artifact machinery (calibration, dual thresholds,
reload-verification) stays the deployment path.

## Deployment artifact v3

`app/model/pipeline.joblib` now carries: backbone list, per-backbone + frequency scalers,
the calibrated LinearSVC classifier trained on **landmark + spliced morphs**, both
operating thresholds, and the per-generator APCER measurements. The web app reads the
`use_freq` / `use_tta` flags and applies the same transforms at inference — no duplicated logic.

Overall progress chart across both rounds: `results/part4_all_steps.png`.
