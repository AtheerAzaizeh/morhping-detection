"""Deterministic standardization and stochastic augmentation (paper §3.2)."""
import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def standardize(img_bgr, size=528):
    return cv2.resize(apply_clahe(img_bgr), (size, size), interpolation=cv2.INTER_AREA)


def augment(img, rng):
    """One random degradation chain: flip / brightness-contrast / blur / noise / JPEG."""
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.8:
        a, b = rng.uniform(0.85, 1.15), rng.uniform(-18, 18)
        out = cv2.convertScaleAbs(out, alpha=a, beta=b)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (rng.choice([3, 5]),) * 2, 0)
    if rng.random() < 0.4:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(4, 10), out.shape)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        q = rng.randint(45, 90)
        _, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out
