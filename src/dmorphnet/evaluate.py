"""Evaluation: standard classification metrics + ISO 30107-3 style MAD metrics.

Convention: morph = attack = positive class (1), real = bona fide (0).
APCER(t) = share of morphs accepted as real   = FN rate of attacks
BPCER(t) = share of reals flagged as morphs   = FP rate on bona fide
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def basic_metrics(y_true, proba, threshold=0.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, proba),
    }


def det_points(y_true, proba):
    """(thresholds, APCER, BPCER) across the score range."""
    fpr, tpr, th = roc_curve(y_true, proba)
    return th, 1 - tpr, fpr  # APCER = 1-TPR, BPCER = FPR


def eer(y_true, proba):
    _, apcer, bpcer = det_points(y_true, proba)
    i = int(np.argmin(np.abs(apcer - bpcer)))
    return float((apcer[i] + bpcer[i]) / 2)


def bpcer_at_apcer(y_true, proba, apcer_max=0.10):
    """Operational point: lowest BPCER with APCER <= apcer_max, and its threshold."""
    th, apcer, bpcer = det_points(y_true, proba)
    ok = apcer <= apcer_max
    if not ok.any():
        return 1.0, 0.0
    i = int(np.argmin(np.where(ok, bpcer, np.inf)))
    return float(bpcer[i]), float(th[i])


def apcer_bpcer(y_true, proba, threshold):
    y_true, pred = np.asarray(y_true), (np.asarray(proba) >= threshold).astype(int)
    morph, real = y_true == 1, y_true == 0
    return {
        "apcer": float((pred[morph] == 0).mean()) if morph.any() else np.nan,
        "bpcer": float((pred[real] == 1).mean()) if real.any() else np.nan,
    }


def bootstrap_ci(y_true, proba, metric_fn, n=2000, seed=42, level=0.95):
    """Percentile bootstrap CI over test samples for any metric(y, proba)."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        vals.append(metric_fn(y_true[b], proba[b]))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
