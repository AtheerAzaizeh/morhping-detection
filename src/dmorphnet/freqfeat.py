"""Hand-crafted frequency/residual features complementary to CNN features.

Morphing's warp + cross-dissolve suppresses and re-shapes high-frequency
content; splicing leaves blending seams. Two cheap descriptors capture this:

* high-pass residual energy on an 8x8 spatial grid (64-d) — where fine detail
  was smoothed away or duplicated;
* radially-averaged log power spectrum in 24 frequency bands (24-d) — the
  global frequency fingerprint that resampling/blending disturbs.
"""
import cv2
import numpy as np

GRID = 8
BANDS = 24
DIM = GRID * GRID + BANDS


def freq_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # --- high-pass residual grid energy ---
    residual = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
    h, w = residual.shape
    gh, gw = h // GRID, w // GRID
    grid = residual[: gh * GRID, : gw * GRID].reshape(GRID, gh, GRID, gw)
    grid_energy = grid.mean(axis=(1, 3)).ravel()
    # --- radial log power spectrum ---
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    logp = np.log1p(f)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    bands = np.empty(BANDS, dtype=np.float32)
    edges = np.linspace(0, r_max, BANDS + 1)
    for i in range(BANDS):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        bands[i] = logp[sel].mean() if sel.any() else 0.0
    return np.concatenate([grid_energy, bands]).astype(np.float32)


def freq_features_batch(X):
    """Feature matrix [N, DIM] for a uint8 image stack."""
    return np.stack([freq_features(x) for x in X])
