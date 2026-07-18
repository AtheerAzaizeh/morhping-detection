"""Fast unit tests for the dmorphnet package (no TensorFlow / MediaPipe needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dmorphnet import data, evaluate, preprocess  # noqa: E402
from dmorphnet.config import DATASET  # noqa: E402


def _img(seed=0, size=64):
    return np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)


class TestPreprocess:
    def test_standardize_shape_dtype(self):
        out = preprocess.standardize(_img(), size=96)
        assert out.shape == (96, 96, 3) and out.dtype == np.uint8

    def test_clahe_preserves_shape(self):
        img = _img(1, 80)
        assert preprocess.apply_clahe(img).shape == img.shape

    def test_augment_valid_image(self):
        import random
        out = preprocess.augment(_img(2, 80), random.Random(0))
        assert out.shape == (80, 80, 3) and out.dtype == np.uint8

    def test_augment_deterministic_given_seed(self):
        import random
        a = preprocess.augment(_img(3, 64), random.Random(7))
        b = preprocess.augment(_img(3, 64), random.Random(7))
        assert np.array_equal(a, b)


class TestData:
    def test_group_parsing(self):
        assert data.group_of_face("face_012_dlib_bald_guys_t324x462.png") == "dlib_bald_guys"
        assert data.group_of_face("face_019_fr_obama.png") == "fr_obama"

    @pytest.mark.skipif(not (DATASET / "manifest.csv").exists(),
                        reason="dataset not generated yet (run Part 1)")
    def test_manifest_identity_disjoint(self):
        base = data.load_manifest()
        assert data.assert_identity_disjoint(base)
        for split in ("train", "test"):
            for cls in ("real", "morph"):
                assert len(base[split][cls]) > 0

    @pytest.mark.skipif(not (DATASET / "manifest.csv").exists(),
                        reason="dataset not generated yet (run Part 1)")
    def test_expand_balances_and_groups(self):
        base = data.load_manifest()
        X, y, ids = data.build_split(base["test"], 10, seed=1, size=64)
        assert X.shape == (20, 64, 64, 3)
        assert (y == 0).sum() == (y == 1).sum() == 10
        assert len(ids) == 20
        # every augmented copy keeps its base id, so ids repeat but never invent
        assert set(ids[:10]) <= {Path(f).name for f in base["test"]["real"]}


class TestEvaluate:
    def test_perfect_separation(self):
        y = np.array([0] * 50 + [1] * 50)
        p = np.concatenate([np.random.default_rng(0).uniform(0, 0.4, 50),
                            np.random.default_rng(1).uniform(0.6, 1.0, 50)])
        m = evaluate.basic_metrics(y, p)
        assert m["accuracy"] == 1.0 and m["auc"] == 1.0
        assert evaluate.eer(y, p) == 0.0

    def test_chance_level_eer(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1] * 200)
        p = rng.uniform(size=400)
        assert 0.35 < evaluate.eer(y, p) < 0.65

    def test_apcer_bpcer_definitions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.9, 0.2, 0.8])   # 1 FP, 1 FN at t=0.5
        m = evaluate.apcer_bpcer(y, p, 0.5)
        assert m["apcer"] == 0.5 and m["bpcer"] == 0.5

    def test_bpcer_at_apcer_monotonic(self):
        y = np.array([0] * 100 + [1] * 100)
        rng = np.random.default_rng(3)
        p = np.concatenate([rng.normal(0.3, 0.15, 100), rng.normal(0.7, 0.15, 100)])
        p = np.clip(p, 0, 1)
        b10, _ = evaluate.bpcer_at_apcer(y, p, 0.10)
        b20, _ = evaluate.bpcer_at_apcer(y, p, 0.20)
        assert 0 <= b20 <= b10 <= 1

    def test_bootstrap_ci_contains_point(self):
        y = np.array([0] * 60 + [1] * 60)
        rng = np.random.default_rng(4)
        p = np.concatenate([rng.uniform(0, 0.45, 60), rng.uniform(0.55, 1, 60)])
        from sklearn.metrics import roc_auc_score
        lo, hi = evaluate.bootstrap_ci(y, p, roc_auc_score, n=300)
        assert lo <= roc_auc_score(y, p) <= hi


class TestFreqFeat:
    def test_dim_and_dtype(self):
        from dmorphnet import freqfeat
        f = freqfeat.freq_features(_img(5, 128))
        assert f.shape == (freqfeat.DIM,) and f.dtype == np.float32

    def test_blur_reduces_highpass_energy(self):
        import cv2
        from dmorphnet import freqfeat
        img = _img(6, 256)
        sharp = freqfeat.freq_features(img)[:64].mean()
        blurred = freqfeat.freq_features(cv2.GaussianBlur(img, (7, 7), 0))[:64].mean()
        assert blurred < sharp
