from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATASET = DATA / "dataset"
FACES = DATA / "faces"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
APP_MODEL = ROOT / "app" / "model"

SEED = 42
IMG_SIZE = {"b0": 224, "b5": 456, "b6": 528}
FEATURE_DIM = {"b0": 1280, "b5": 2048, "b6": 2304}

# per-class sample targets after augmentation (test kept identical to Part 2
# so every improvement step stays comparable to the 85.2%/0.935 baseline)
TARGETS = {"train": 160, "val": 40, "test": 44}
EXPANDED_TRAIN_TARGET = 400  # per class, used from improvement step 2 onwards
