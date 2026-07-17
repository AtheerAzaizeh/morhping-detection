"""D-MorphNet demo server — upload a face photo, get a real/morph verdict.

Pipeline per request: BlazeFace face detection -> square 512x512 crop ->
CLAHE + resize 528 -> EfficientNet-B6 GAP features (2304-d) -> StandardScaler
-> SVM (RBF) -> probability + verdict.

Run from the repo root (after `python app/export_model.py`):
    python app/server.py            # http://localhost:7860
"""
import os
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
MODELS = ROOT / "models"

print("loading artifacts …")
scaler = joblib.load(APP_DIR / "model" / "scaler.joblib")
svm = joblib.load(APP_DIR / "model" / "svm.joblib")
cfg = joblib.load(APP_DIR / "model" / "config.joblib")

detector = mp_vision.FaceDetector.create_from_options(mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
    min_detection_confidence=0.5))

print("loading EfficientNet-B6 backbone …")
from tensorflow import keras  # deferred: slow import
backbone = keras.applications.EfficientNetB6(
    include_top=False, weights="imagenet", pooling="avg",
    input_shape=(cfg["input_size"], cfg["input_size"], 3))
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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


def predict(img):
    t0 = time.time()
    face = crop_face(img)
    if face is None:
        return {"error": "No face detected in the image. Upload a clear frontal face photo."}
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    x = cv2.resize(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR),
                   (cfg["input_size"],) * 2, interpolation=cv2.INTER_AREA)
    feats = backbone.predict(x.astype(np.float32)[None], verbose=0)
    p_morph = float(svm.predict_proba(scaler.transform(feats))[0, 1])
    verdict = "MORPH" if p_morph > cfg["threshold"] else "REAL"
    ok, buf = cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 88])
    import base64
    return {
        "verdict": verdict,
        "p_morph": round(p_morph, 4),
        "threshold": cfg["threshold"],
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
