"""Morph generation: landmark morphs (training generator) and spliced morphs
(a *different* generator used only as an unseen-attack robustness probe)."""
import numpy as np
import cv2
from scipy.spatial import Delaunay

from .config import MODELS

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        _landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS / "face_landmarker.task")),
                num_faces=1, min_face_detection_confidence=0.5))
    return _landmarker


def get_landmarks(img_bgr):
    import mediapipe as mp
    res = _get_landmarker().detect(mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return None
    h, w = img_bgr.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)
    return np.clip(pts, 0, [w - 1, h - 1])


def _with_boundary(pts, w, h):
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]],
                       dtype=np.float64)
    return np.vstack([pts, corners])


def _warp_triangle(src, dst, t_src, t_dst):
    r1, r2 = cv2.boundingRect(np.float32([t_src])), cv2.boundingRect(np.float32([t_dst]))
    if r2[2] == 0 or r2[3] == 0:
        return
    t1 = [(t_src[i][0] - r1[0], t_src[i][1] - r1[1]) for i in range(3)]
    t2 = [(t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]) for i in range(3)]
    patch = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    if patch.size == 0:
        return
    M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(patch, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1.0, 1.0, 1.0), 16, 0)
    roi = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    roi[:] = roi * (1 - mask) + warped.astype(np.float32) * mask


def _warp_to_shape(img, pts_src, pts_dst, simplices):
    out = np.zeros_like(img, dtype=np.float32)
    for tri in simplices:
        _warp_triangle(img, out, pts_src[tri].astype(np.float32),
                       pts_dst[tri].astype(np.float32))
    return out


def morph_faces(imgA, imgB, alpha=0.5):
    """Landmark morph: shape interpolation + piece-wise affine warp + cross-dissolve."""
    h, w = imgA.shape[:2]
    ptsA, ptsB = get_landmarks(imgA), get_landmarks(imgB)
    if ptsA is None or ptsB is None:
        return None
    ptsA, ptsB = _with_boundary(ptsA, w, h), _with_boundary(ptsB, w, h)
    ptsM = (1 - alpha) * ptsA + alpha * ptsB
    simplices = Delaunay(ptsM).simplices
    warpA = _warp_to_shape(imgA, ptsA, ptsM, simplices)
    warpB = _warp_to_shape(imgB, ptsB, ptsM, simplices)
    return np.clip((1 - alpha) * warpA + alpha * warpB, 0, 255).astype(np.uint8)


def spliced_morph(imgA, imgB, alpha=0.5):
    """Unseen-generator attack: morph only the *face region* and seamlessly clone
    it back into donor A's genuine photo (classic passport-splicing attack).

    The background, hair and clothing stay 100% genuine, so the global morphing
    artifacts the detector learned from full-frame morphs largely disappear.
    """
    full = morph_faces(imgA, imgB, alpha)
    if full is None:
        return None
    pts = get_landmarks(full)
    if pts is None:
        return None
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros(imgA.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    x, y, w, h = cv2.boundingRect(hull)
    center = (x + w // 2, y + h // 2)
    try:
        return cv2.seamlessClone(full, imgA, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return None
