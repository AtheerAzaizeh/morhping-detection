"""Refit the D-MorphNet classifier from the cached EfficientNet-B6 features
and save the deployable artifacts (scaler + SVM + config) for the web app.

Run from the repo root after executing the Part 2 notebook:
    python app/export_model.py
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FEATS = ROOT / "results" / "features_b6.npz"
OUT = Path(__file__).resolve().parent / "model"
OUT.mkdir(exist_ok=True)

# Split targets used in the Part 2 notebook: per split, features are stacked
# [real x N, morph x N] with N = TARGETS[split].
TARGETS = {"train": 160, "val": 40, "test": 44}
BEST_PARAMS = {"C": 10, "gamma": 1e-4}  # grid-search result from the notebook

z = np.load(FEATS)
X_train = z["X_train"]
y_train = np.array([0] * TARGETS["train"] + [1] * TARGETS["train"])
assert len(X_train) == len(y_train), (len(X_train), len(y_train))

scaler = StandardScaler().fit(X_train)
svm = SVC(kernel="rbf", probability=True, random_state=42, **BEST_PARAMS)
svm.fit(scaler.transform(X_train), y_train)

# sanity check against the notebook's reported test accuracy
X_test = z["X_test"]
y_test = np.array([0] * TARGETS["test"] + [1] * TARGETS["test"])
acc = (svm.predict(scaler.transform(X_test)) == y_test).mean()
print(f"reconstructed test accuracy: {acc:.3f} (notebook reported 0.852)")

joblib.dump(scaler, OUT / "scaler.joblib")
joblib.dump(svm, OUT / "svm.joblib")
joblib.dump({"threshold": 0.5, "input_size": 528, "feature_dim": X_train.shape[1]},
            OUT / "config.joblib")
print("artifacts written to", OUT)
