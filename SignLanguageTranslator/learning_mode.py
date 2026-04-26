"""
Learning Mode Module (Improved Detection + Model Management)
=============================================================
Features:
  • Rich feature engineering: normalized xyz + distances + angles + ratios
  • StandardScaler for feature normalization
  • Data augmentation (Gaussian noise) for robustness
  • Temporal smoothing via sliding-window voting
  • Multi-model management: save, load, list, delete models
  • Reset model & data
"""

from pathlib import Path
import csv
import math
import numpy as np

# ── Paths ─────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
DATA_PATH = _BASE_DIR / "gesture_data.csv"
MODEL_PATH = _BASE_DIR / "gesture_model.pkl"
SCALER_PATH = _BASE_DIR / "gesture_scaler.pkl"
MODELS_DIR = _BASE_DIR / "saved_models"

_NUM_LANDMARKS = 21
_COORDS_PER_LM = 3
_RAW_FEATURES = _NUM_LANDMARKS * _COORDS_PER_LM  # 63

# ── Landmark indices ──────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def _dist(a, b):
    """Euclidean distance between two landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a, b, c):
    """Angle at point b formed by points a-b-c (in degrees)."""
    ba = [a.x - b.x, a.y - b.y, a.z - b.z]
    bc = [c.x - b.x, c.y - b.y, c.z - b.z]

    dot = sum(x * y for x, y in zip(ba, bc))
    mag_ba = math.sqrt(sum(x ** 2 for x in ba)) + 1e-8
    mag_bc = math.sqrt(sum(x ** 2 for x in bc)) + 1e-8

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def landmarks_to_features(landmarks):
    """
    Convert 21 hand landmarks → rich feature vector.

    Returns a flat list of features:
      1. Normalized xyz (relative to wrist) — 63 features
      2. Fingertip-to-wrist distances — 5 features
      3. Fingertip-to-palm-center distances — 5 features
      4. Finger curl angles (PIP joint angles) — 5 features
      5. Finger MCP joint angles — 5 features
      6. Inter-fingertip distances (pairs) — 10 features
      7. Finger extension ratios (tip-wrist / mcp-wrist) — 5 features

    Total: 63 + 5 + 5 + 5 + 5 + 10 + 5 = 98 features
    """
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    # ── 1. Normalized xyz coordinates (63 features) ──
    norm_xyz = []
    for lm in landmarks:
        norm_xyz.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z - base_z,
        ])

    # ── 2. Fingertip-to-wrist distances (5 features) ──
    tip_wrist_dists = [_dist(landmarks[t], landmarks[WRIST]) for t in FINGER_TIPS]

    # ── 3. Fingertip-to-palm-center distances (5 features) ──
    # Palm center approximated as average of wrist + all MCP joints
    palm_indices = [WRIST] + list(FINGER_MCPS)
    palm_x = sum(landmarks[i].x for i in palm_indices) / len(palm_indices)
    palm_y = sum(landmarks[i].y for i in palm_indices) / len(palm_indices)
    palm_z = sum(landmarks[i].z for i in palm_indices) / len(palm_indices)

    class _PalmCenter:
        pass

    palm = _PalmCenter()
    palm.x, palm.y, palm.z = palm_x, palm_y, palm_z

    tip_palm_dists = [_dist(landmarks[t], palm) for t in FINGER_TIPS]

    # ── 4. PIP joint curl angles (5 features) ──
    pip_angles = []
    finger_chains = [
        (THUMB_MCP, THUMB_IP, THUMB_TIP),
        (INDEX_MCP, INDEX_PIP, INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP, RING_PIP, RING_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP),
    ]
    for a, b, c in finger_chains:
        pip_angles.append(_angle(landmarks[a], landmarks[b], landmarks[c]))

    # ── 5. MCP joint angles (5 features) ──
    mcp_chains = [
        (WRIST, THUMB_CMC, THUMB_MCP),
        (WRIST, INDEX_MCP, INDEX_PIP),
        (WRIST, MIDDLE_MCP, MIDDLE_PIP),
        (WRIST, RING_MCP, RING_PIP),
        (WRIST, PINKY_MCP, PINKY_PIP),
    ]
    mcp_angles = []
    for a, b, c in mcp_chains:
        mcp_angles.append(_angle(landmarks[a], landmarks[b], landmarks[c]))

    # ── 6. Inter-fingertip distances — all 10 pairs (10 features) ──
    inter_tip_dists = []
    for i in range(len(FINGER_TIPS)):
        for j in range(i + 1, len(FINGER_TIPS)):
            inter_tip_dists.append(
                _dist(landmarks[FINGER_TIPS[i]], landmarks[FINGER_TIPS[j]])
            )

    # ── 7. Finger extension ratios (5 features) ──
    ext_ratios = []
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        tip_dist = _dist(landmarks[tip_idx], landmarks[WRIST])
        mcp_dist = _dist(landmarks[mcp_idx], landmarks[WRIST]) + 1e-8
        ext_ratios.append(tip_dist / mcp_dist)

    # ── Combine all features ──
    features = (
        norm_xyz
        + tip_wrist_dists
        + tip_palm_dists
        + pip_angles
        + mcp_angles
        + inter_tip_dists
        + ext_ratios
    )

    return features


# Total features count
_NUM_FEATURES = 98  # 63 + 5 + 5 + 5 + 5 + 10 + 5


# ═══════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════

def _get_csv_header():
    """Generate the CSV header row."""
    header = ["label"]
    # 63 normalized xyz
    header += [f"{c}{i}" for i in range(_NUM_LANDMARKS) for c in ("x", "y", "z")]
    # 5 tip-wrist distances
    header += [f"tw_d{i}" for i in range(5)]
    # 5 tip-palm distances
    header += [f"tp_d{i}" for i in range(5)]
    # 5 PIP angles
    header += [f"pip_a{i}" for i in range(5)]
    # 5 MCP angles
    header += [f"mcp_a{i}" for i in range(5)]
    # 10 inter-tip distances
    header += [f"it_d{i}" for i in range(10)]
    # 5 extension ratios
    header += [f"ext_r{i}" for i in range(5)]
    return header


def save_landmarks(label, landmarks_batch):
    """Save a batch of feature vectors to the CSV file."""
    file_exists = DATA_PATH.exists()

    with open(DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(_get_csv_header())

        for features in landmarks_batch:
            if len(features) == _NUM_FEATURES:
                writer.writerow([label] + features)

    return get_dataset_info()["total_rows"]


def get_dataset_info():
    """Return info about the current dataset."""
    info = {"total_rows": 0, "labels": {}}

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


# ═══════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════

def _augment_data(X, y, factor=2, noise_level=0.005):
    """
    Augment training data by adding Gaussian noise.

    Parameters
    ----------
    X : np.ndarray — shape (n_samples, n_features)
    y : np.ndarray — shape (n_samples,)
    factor : int — how many augmented copies per original sample
    noise_level : float — std of Gaussian noise added

    Returns
    -------
    X_aug, y_aug with original + augmented samples
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(factor):
        noise = np.random.normal(0, noise_level, X.shape).astype(np.float32)
        augmented_X.append(X + noise)
        augmented_y.append(y)

    return np.vstack(augmented_X), np.concatenate(augmented_y)


# ═══════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════

def load_dataset():
    """Load dataset from CSV file."""
    import pandas as pd

    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty")

    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y = df.iloc[:, 0].to_numpy(dtype=str)

    if len(y) == 0:
        raise ValueError("No data available")

    return X, y


def train_model():
    """
    Train a RandomForest model with improved settings.

    Returns a dict with training results or error.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import joblib

    try:
        X, y = load_dataset()
    except Exception as e:
        return {"error": str(e)}

    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()

    if min_samples < 2:
        return {"error": "Not enough samples per class (need ≥ 2)"}

    if len(unique) < 2:
        return {"error": "Need at least 2 different gesture classes"}

    # ── Data augmentation ──
    X_aug, y_aug = _augment_data(X, y, factor=2, noise_level=0.005)

    # ── Feature scaling ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    # ── Model ──
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # ── Cross-validation (on original data only, not augmented) ──
    X_orig_scaled = scaler.transform(X)
    n_folds = min(5, min_samples)
    if n_folds >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_orig_scaled, y, cv=skf, scoring="accuracy")
        accuracy = float(scores.mean())
    else:
        accuracy = 0.0

    # ── Fit on full augmented data ──
    clf.fit(X_scaled, y_aug)

    # ── Save model + scaler ──
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Training Done")
    print(f"  Samples (original): {len(y)}")
    print(f"  Samples (augmented): {len(y_aug)}")
    print(f"  Classes: {len(unique)}")
    print(f"  Labels: {unique}")
    print(f"  CV Accuracy: {accuracy:.2%}")

    return {
        "accuracy": accuracy,
        "n_samples": len(y),
        "n_augmented": len(y_aug),
        "n_classes": len(unique),
        "labels": sorted(unique.tolist()),
    }


def load_model():
    """Load the current model and scaler. Returns (model, scaler) or (None, None)."""
    if not MODEL_PATH.exists():
        return None, None

    import joblib

    model = joblib.load(MODEL_PATH)
    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ═══════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════

def predict_gesture(model, scaler, landmarks):
    """
    Predict gesture from landmarks using trained model.

    Parameters
    ----------
    model : sklearn classifier
    scaler : StandardScaler or None
    landmarks : list of NormalizedLandmark

    Returns
    -------
    label : str
    confidence : float
    """
    if model is None:
        return "No model", 0.0

    try:
        features = landmarks_to_features(landmarks)

        if len(features) != _NUM_FEATURES:
            return "Invalid input", 0.0

        X = np.array(features, dtype=np.float32).reshape(1, -1)

        # Apply scaler if available
        if scaler is not None:
            X = scaler.transform(X)

        label = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = float(proba.max())

        # Low confidence filter
        if confidence < 0.45:
            return "Unknown", confidence

        return label, confidence

    except Exception:
        return "Error", 0.0


# ═══════════════════════════════════════════════════════
# TEMPORAL SMOOTHING
# ═══════════════════════════════════════════════════════

class PredictionSmoother:
    """
    Sliding window majority-vote smoother for predictions.

    Collects the last `window_size` predictions and returns
    the most common one — reduces jitter / flicker.
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self._buffer: list[tuple[str, float]] = []

    def add(self, label: str, confidence: float):
        """Add a new prediction to the buffer."""
        self._buffer.append((label, confidence))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def get_smoothed(self) -> tuple[str, float]:
        """Return the majority-voted label and average confidence."""
        if not self._buffer:
            return "Unknown", 0.0

        # Count votes (ignore "Unknown" / "Error" / "No model")
        votes: dict[str, list[float]] = {}
        for label, conf in self._buffer:
            if label in ("Unknown", "Error", "No model", "Invalid input"):
                continue
            if label not in votes:
                votes[label] = []
            votes[label].append(conf)

        if not votes:
            return "Unknown", 0.0

        # Find label with most votes
        best_label = max(votes, key=lambda k: len(votes[k]))
        avg_conf = sum(votes[best_label]) / len(votes[best_label])

        # Require at least 40% of window to agree
        vote_ratio = len(votes[best_label]) / len(self._buffer)
        if vote_ratio < 0.4:
            return "Unknown", avg_conf

        return best_label, avg_conf

    def clear(self):
        """Reset the buffer."""
        self._buffer.clear()


# ═══════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════

def _ensure_models_dir():
    """Create the saved_models directory if it doesn't exist."""
    MODELS_DIR.mkdir(exist_ok=True)


def list_saved_models() -> list[str]:
    """Return a list of saved model names."""
    _ensure_models_dir()
    return sorted([
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "model.pkl").exists()
    ])


def save_model_as(name: str) -> bool:
    """
    Save the current model + scaler under a given name.

    Returns True if successful.
    """
    _ensure_models_dir()
    import shutil

    if not MODEL_PATH.exists():
        return False

    model_dir = MODELS_DIR / name
    model_dir.mkdir(exist_ok=True)

    shutil.copy2(MODEL_PATH, model_dir / "model.pkl")
    if SCALER_PATH.exists():
        shutil.copy2(SCALER_PATH, model_dir / "scaler.pkl")
    if DATA_PATH.exists():
        shutil.copy2(DATA_PATH, model_dir / "data.csv")

    return True


def load_model_by_name(name: str):
    """
    Load a saved model by name.

    Returns (model, scaler) or (None, None).
    """
    import joblib

    model_dir = MODELS_DIR / name
    model_file = model_dir / "model.pkl"

    if not model_file.exists():
        return None, None

    model = joblib.load(model_file)
    scaler = None
    scaler_file = model_dir / "scaler.pkl"
    if scaler_file.exists():
        scaler = joblib.load(scaler_file)

    return model, scaler


def delete_saved_model(name: str) -> bool:
    """Delete a saved model by name. Returns True if deleted."""
    import shutil

    model_dir = MODELS_DIR / name
    if model_dir.exists():
        shutil.rmtree(model_dir)
        return True
    return False


def reset_current_model():
    """Delete the current active model, scaler, and training data."""
    for p in [MODEL_PATH, SCALER_PATH, DATA_PATH]:
        if p.exists():
            p.unlink()