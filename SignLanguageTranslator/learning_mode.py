"""
Learning Mode Module
====================
Provides functionality to:
  1. Record hand-landmark data from MediaPipe and store it as CSV.
  2. Train a lightweight sklearn RandomForestClassifier on the recorded data.
  3. Save / load the trained model via joblib.
  4. Predict gestures using the trained model.

Data format (gesture_data.csv):
    label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20   (63 features)
"""

from pathlib import Path
import csv
import os

import numpy as np

# ── Paths (relative to this file's directory) ────────────────────────
_BASE_DIR = Path(__file__).parent
DATA_PATH = _BASE_DIR / "gesture_data.csv"
MODEL_PATH = _BASE_DIR / "gesture_model.pkl"

# Number of MediaPipe hand landmarks and coordinates per landmark
_NUM_LANDMARKS = 21
_COORDS_PER_LM = 3  # x, y, z
_NUM_FEATURES = _NUM_LANDMARKS * _COORDS_PER_LM  # 63


# ═══════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════

def landmarks_to_features(landmarks) -> list[float]:
    """
    Flatten a list of 21 MediaPipe NormalizedLandmark objects into a
    63-element feature vector [x0, y0, z0, x1, y1, z1, ...].
    """
    features: list[float] = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def save_landmarks(label: str, landmarks_batch: list[list[float]]) -> int:
    """
    Append a batch of landmark feature vectors to the CSV dataset.

    Parameters
    ----------
    label : str
        The gesture name to associate with each row.
    landmarks_batch : list[list[float]]
        Each element is a 63-float feature vector from one frame.

    Returns
    -------
    int
        Total number of rows in the dataset after saving.
    """
    file_exists = DATA_PATH.exists()

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header only if the file is new
        if not file_exists:
            header = ["label"] + [
                f"{c}{i}" for i in range(_NUM_LANDMARKS) for c in ("x", "y", "z")
            ]
            writer.writerow(header)
        for features in landmarks_batch:
            writer.writerow([label] + features)

    return get_dataset_info()["total_rows"]


def get_dataset_info() -> dict:
    """
    Return a summary of the current dataset.

    Returns
    -------
    dict
        Keys: total_rows (int), labels (dict mapping label → count)
    """
    info: dict = {"total_rows": 0, "labels": {}}
    if not DATA_PATH.exists():
        return info

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            label = row[0]
            info["total_rows"] += 1
            info["labels"][label] = info["labels"].get(label, 0) + 1

    return info


# ═══════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the CSV dataset into numpy arrays.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 63)
    y : np.ndarray of shape (n_samples,) — string labels
    """
    import pandas as pd  # deferred import — only needed for training

    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values
    return X, y

    if len(y) == 0:
        return {"error": "No data available for training"}
def train_model() -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib
    import numpy as np

    X, y = load_dataset()

    # 🔥 FIX: ensure proper format
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=str)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    # 🔥 SAFE CROSS VALIDATION
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()

    n_folds = min(3, min_samples)

    if n_folds >= 2:
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring="accuracy")
        accuracy = float(scores.mean())
    else:
        accuracy = 0.0

    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return {
        "accuracy": accuracy,
        "n_samples": len(y),
        "n_classes": len(set(y)),
        "labels": sorted(set(y)),
    }



def load_model():
    """
    Load the trained model from disk.

    Returns
    -------
    sklearn classifier or None if the file doesn't exist.
    """
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


# ═══════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def predict_gesture(model, landmarks) -> tuple[str, float]:
    """
    Predict a gesture from raw MediaPipe landmarks using the trained model.

    Parameters
    ----------
    model : sklearn classifier
        The trained model loaded via load_model().
    landmarks : list
        21 MediaPipe NormalizedLandmark objects.

    Returns
    -------
    (label, confidence) : tuple[str, float]
    """
    features = landmarks_to_features(landmarks)
    X = np.array(features).reshape(1, -1)
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba.max())
    return label, confidence
