"""Dataset access: manifest, identity-disjoint grouping, augmented expansion."""
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np

from .config import DATASET
from .preprocess import augment, standardize

_GROUP_RE = re.compile(r"face_\d+_(.+?)(?:_t\d+x\d+)?$")


def group_of_face(path):
    """Source-photograph group of a real face crop (identity/scene unit)."""
    return _GROUP_RE.match(Path(path).stem).group(1)


def load_manifest(dataset_dir=DATASET):
    with open(Path(dataset_dir) / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    base = {"train": {"real": [], "morph": []}, "test": {"real": [], "morph": []}}
    for row in rows:
        base[row["split"]][row["label"]].append(Path(dataset_dir) / row["path"])
    for split in base:
        for cls in base[split]:
            base[split][cls] = sorted(base[split][cls])
    return base


def expand(files, target, seed, size=528, augment_all=False):
    """Base images + augmented copies up to `target` samples.

    Returns (X uint8 [N,size,size,3], ids) where ids[i] is the base filename —
    the grouping key that keeps augmented copies of one image in one CV fold.

    With augment_all=True every sample (including the first copy of each base)
    goes through the augmentation chain, so classes with different base counts
    end up with the *same* degradation distribution — otherwise the class with
    fewer bases carries more augmentation artifacts and the classifier can learn
    "JPEG/noise present => class X" instead of morphing cues.
    """
    rng = random.Random(seed)
    files = list(files)
    names = [Path(f).name for f in files]
    if augment_all:
        X, ids = [], []
    else:
        X = [standardize(cv2.imread(str(f)), size) for f in files]
        ids = names[:]
    k = 0
    while len(X) < target:
        i = k % len(files)
        X.append(standardize(augment(cv2.imread(str(files[i])), rng), size))
        ids.append(names[i])
        k += 1
    return np.stack(X[:target]).astype(np.uint8), ids[:target]


def build_split(files_by_class, target_per_class, seed, size=528, augment_all=False):
    """Balanced (X, y, ids) for one split; y: 0=real, 1=morph."""
    Xr, idr = expand(files_by_class["real"], target_per_class, seed, size, augment_all)
    Xm, idm = expand(files_by_class["morph"], target_per_class, seed + 1, size, augment_all)
    X = np.concatenate([Xr, Xm])
    y = np.array([0] * len(Xr) + [1] * len(Xm))
    return X, y, idr + idm


def assert_identity_disjoint(base):
    """The train/test contract: no source-photo group on both sides."""
    tr = {group_of_face(f) for f in base["train"]["real"]}
    te = {group_of_face(f) for f in base["test"]["real"]}
    overlap = tr & te
    if overlap:
        raise AssertionError(f"identity groups leak across splits: {overlap}")
    return True
