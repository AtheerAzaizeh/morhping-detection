"""D-MorphNet demo server — upload a face photo, get a real/morph verdict.

Pipeline per request: BlazeFace face detection -> square 512x512 crop ->
CLAHE + resize -> EfficientNet backbone features (B6, plus B5 if the deployed
artifact uses fusion) -> per-backbone StandardScaler -> calibrated SVM (RBF)
-> probability + verdict.

The model comes from ONE versioned artifact (app/model/pipeline.joblib),
exported and verified by notebooks/Part3_Professional_Improvements.ipynb —
the server never re-fits or hard-codes hyper-parameters.

Run from the repo root:
    python app/server.py            # http://localhost:7860
"""
import base64
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dmorphnet.config import MODELS  # noqa: E402
from dmorphnet.features import extract as extract_features  # noqa: E402
from dmorphnet.preprocess import standardize  # noqa: E402

print("loading pipeline artifact …")
ART_PATH = APP_DIR / "model" / "pipeline.joblib"
if ART_PATH.exists():
    art = joblib.load(ART_PATH)
else:  # legacy fallback (pre-Part 3 artifacts)
    art = {
        "version": 1,
        "backbones": ["b6"],
        "scalers": {"b6": joblib.load(APP_DIR / "model" / "scaler.joblib")},
        "model": joblib.load(APP_DIR / "model" / "svm.joblib"),
        "threshold": joblib.load(APP_DIR / "model" / "config.joblib")["threshold"],
    }
USE_FREQ = bool(art.get("use_freq", False))
USE_TTA = bool(art.get("use_tta", False))
if USE_FREQ:
    from dmorphnet.freqfeat import freq_features_batch
print(f"artifact v{art['version']} | backbones: {art['backbones']} "
      f"| freq: {USE_FREQ} | tta: {USE_TTA} | threshold: {art['threshold']:.2f}")

detector = mp_vision.FaceDetector.create_from_options(mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
    min_detection_confidence=0.5))

print("loading backbone(s) …")
from tensorflow import keras  # deferred: slow import

_CTOR = {"b0": keras.applications.EfficientNetB0,
         "b5": keras.applications.EfficientNetB5,
         "b6": keras.applications.EfficientNetB6}
_SIZE = {"b0": 224, "b5": 456, "b6": 528}
backbones = {name: _CTOR[name](include_top=False, weights="imagenet", pooling="avg",
                               input_shape=(_SIZE[name], _SIZE[name], 3))
             for name in art["backbones"]}


def crop_face(img):
    """Largest detected face as a 512x512 square crop; None if no face."""
    h, w = img.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dets = detector.detect(mp_img).detections
    if not dets:
        return None
    bb = max(dets, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
    cx, cy = bb.origin_x + bb.width / 2, bb.origin_y + bb.height / 2
    s = max(bb.width, bb.height) * 1.9
    x0, y0, x1, y1 = int(cx - s/2), int(cy - s/2), int(cx + s/2), int(cy + s/2)
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    crop = cv2.copyMakeBorder(img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)],
                              *pad, cv2.BORDER_REFLECT)
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)


def _features(x):
    """Stacked scaled feature row(s) for a uint8 image batch, artifact-driven."""
    Z = np.hstack([art["scalers"][b].transform(
                       extract_features(b, x, model=backbones[b]))
                   for b in art["backbones"]])
    if USE_FREQ:
        Z = np.hstack([Z, art["scalers"]["freq"].transform(freq_features_batch(x))])
    return Z


def predict(img):
    t0 = time.time()
    face = crop_face(img)
    if face is None:
        return {"error": "No face detected in the image. Upload a clear frontal face photo."}
    x = standardize(face)[None]
    p_morph = float(art["model"].predict_proba(_features(x))[0, 1])
    if USE_TTA:
        p_flip = float(art["model"].predict_proba(_features(x[:, :, ::-1, :].copy()))[0, 1])
        p_morph = (p_morph + p_flip) / 2
    verdict = "MORPH" if p_morph >= art["threshold"] else "REAL"
    ok, buf = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "verdict": verdict,
        "p_morph": round(p_morph, 4),
        "threshold": round(float(art["threshold"]), 3),
        "latency_s": round(time.time() - t0, 2),
        "face_crop": "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None,
    }


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(APP_DIR / "static", "index.html")


@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "Send the photo as multipart field 'image'."}), 400
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode the file as an image."}), 400
    result = predict(img)
    return jsonify(result), (200 if "error" not in result else 422)


if __name__ == "__main__":
    print("ready — open http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
