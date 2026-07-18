"""Backbone feature extraction with self-describing caches (X + y + ids)."""
import cv2
import numpy as np

from .config import IMG_SIZE, RESULTS


def _backbone(name):
    from tensorflow import keras
    ctor = {"b0": keras.applications.EfficientNetB0,
            "b5": keras.applications.EfficientNetB5,
            "b6": keras.applications.EfficientNetB6}[name]
    size = IMG_SIZE[name]
    return ctor(include_top=False, weights="imagenet", pooling="avg",
                input_shape=(size, size, 3))


def extract(name, X, batch=8, model=None):
    """GAP feature matrix for a uint8 image stack, resizing to the backbone size."""
    size = IMG_SIZE[name]
    own = model is None
    if own:
        model = _backbone(name)
    if X.shape[1] != size:
        X = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in X])
    out = []
    for i in range(0, len(X), batch):
        out.append(model.predict(X[i:i + batch].astype(np.float32), verbose=0))
    feats = np.concatenate(out)
    if own:
        import gc
        from tensorflow import keras
        del model
        gc.collect()
        keras.backend.clear_session()
    return feats


def extract_cached(cache_name, backbone, splits, results_dir=RESULTS):
    """Extract features for {split: (X, y, ids)} with a labelled .npz cache."""
    cache = results_dir / f"{cache_name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {s: {"F": z[f"F_{s}"], "y": z[f"y_{s}"], "ids": list(z[f"ids_{s}"])}
                for s in splits}
    model = _backbone(backbone)
    out, payload = {}, {}
    for s, (X, y, ids) in splits.items():
        F = extract(backbone, X, model=model)
        out[s] = {"F": F, "y": np.asarray(y), "ids": list(ids)}
        payload[f"F_{s}"], payload[f"y_{s}"] = F, np.asarray(y)
        payload[f"ids_{s}"] = np.array(ids)
    np.savez_compressed(cache, **payload)
    import gc
    from tensorflow import keras
    del model
    gc.collect()
    keras.backend.clear_session()
    return out
