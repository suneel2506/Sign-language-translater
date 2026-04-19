"""
Motion Model Module (LSTM-Based Action Recognition — PyTorch)
==============================================================
Builds, trains, and loads an LSTM model for temporal gesture/action
classification using sequences of hand-landmark frames.

Input:  (batch, 30, 63)  — 30 frames × 63 features per frame
Output: (batch, n_classes) — softmax probability over action labels
"""

from pathlib import Path
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_BASE_DIR = Path(__file__).parent
ACTION_MODEL_PATH = _BASE_DIR / "action_model.pth"
ACTION_LABELS_PATH = _BASE_DIR / "action_labels.pkl"

SEQUENCE_LENGTH = 15   # shorter = faster response (was 30)
NUM_FEATURES = 63  # 21 landmarks × 3 coords

# Use GPU if available, else CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════

class ActionLSTM(nn.Module):
    """
    LSTM-based action recognition model.

    Architecture (lightweight for fast inference):
        LSTM(64)  → LSTM(64)
        ↓ (last hidden state)
        Dense(32) → ReLU → Dropout(0.3)
        Dense(n_classes) → Softmax
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()

        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Take the last time step's output
        x = x[:, -1, :]  # (batch, 64)

        x = self.classifier(x)
        return x  # raw logits; use CrossEntropyLoss (includes softmax)


def build_lstm_model(n_classes: int, n_features: int = NUM_FEATURES):
    """Build an ActionLSTM model and move to device."""
    model = ActionLSTM(n_features, n_classes).to(_DEVICE)
    return model


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_lstm(dataset_path: Path | str | None = None) -> dict:
    """
    Full training pipeline: load data → encode → split → train → save.

    Parameters
    ----------
    dataset_path : Path or str or None
        Root of the dataset folder. If None, uses the default from
        action_recorder.

    Returns
    -------
    dict with keys:
        'accuracy'   : float — validation accuracy
        'n_samples'  : int   — total sequences
        'n_classes'  : int   — number of actions
        'labels'     : list[str] — sorted action names
        'epochs_run' : int   — actual epochs before early stop
    Or {'error': str} on failure.
    """
    from sklearn.model_selection import train_test_split
    from action_recorder import load_all_sequences, DEFAULT_DATASET_DIR

    ds_dir = Path(dataset_path) if dataset_path else DEFAULT_DATASET_DIR

    try:
        X, y, actions = load_all_sequences(ds_dir)
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    if len(X) == 0:
        return {"error": "No sequences found. Record some actions first."}

    if len(actions) < 2:
        return {"error": "Need at least 2 different actions to train."}

    n_samples = len(X)
    n_classes = len(actions)

    # ── Label encoding ──
    label_to_idx = {label: idx for idx, label in enumerate(actions)}
    y_encoded = np.array([label_to_idx[label] for label in y])

    # ── Train / validation split ──
    min_per_class = min(np.bincount(y_encoded))
    if min_per_class < 2:
        return {"error": "Need at least 2 sequences per action class."}

    # We need at least n_classes samples in the validation set for
    # stratified splitting.  Compute the minimum fraction that gives us
    # that, and fall back to training-only when the dataset is very small.
    min_val_samples = n_classes          # 1 per class at minimum
    use_validation = n_samples > min_val_samples and min_per_class >= 2

    if use_validation:
        # Ensure test set has at least n_classes samples
        test_size = max(n_classes / n_samples, 0.15)
        test_size = min(test_size, 0.4)  # never more than 40 %
        # Also ensure every class keeps ≥1 sample in train
        max_test = (min_per_class - 1) / min_per_class
        test_size = min(test_size, max_test)

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded,
                test_size=test_size,
                random_state=42,
                stratify=y_encoded,
            )
        except ValueError:
            # Fallback: dataset too small for stratified split
            use_validation = False

    if not use_validation:
        # Train on everything; validation = training set (small-data mode)
        X_train, y_train = X, y_encoded
        X_val, y_val = X, y_encoded

    # ── Convert to PyTorch tensors ──
    X_train_t = torch.FloatTensor(X_train).to(_DEVICE)
    y_train_t = torch.LongTensor(y_train).to(_DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(_DEVICE)
    y_val_t = torch.LongTensor(y_val).to(_DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    batch_size = min(32, len(X_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ── Build model ──
    model = build_lstm_model(n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop with early stopping ──
    best_val_acc = 0.0
    best_model_state = None
    patience = 15
    patience_counter = 0
    epochs_run = 0

    for epoch in range(100):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        scheduler.step(val_loss)
        epochs_run = epoch + 1

        # Early stopping
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ── Save model + label map ──
    save_data = {
        "model_state": model.state_dict(),
        "n_classes": n_classes,
        "n_features": NUM_FEATURES,
    }
    torch.save(save_data, ACTION_MODEL_PATH)

    label_map = {idx: label for label, idx in label_to_idx.items()}
    with open(ACTION_LABELS_PATH, 'wb') as f:
        pickle.dump(label_map, f)

    print(f"LSTM Training Done")
    print(f"  Samples: {n_samples}")
    print(f"  Classes: {n_classes} — {actions}")
    print(f"  Val Accuracy: {best_val_acc:.2%}")
    print(f"  Epochs: {epochs_run}")

    return {
        "accuracy": float(best_val_acc),
        "n_samples": n_samples,
        "n_classes": n_classes,
        "labels": actions,
        "epochs_run": epochs_run,
    }


# ═══════════════════════════════════════════════════════════════════════
#  LOADING & PREDICTION
# ═══════════════════════════════════════════════════════════════════════

_cached_model = None
_cached_label_map = None


def load_action_model():
    """
    Load the trained LSTM model and label map.

    Returns (model, label_map) or (None, None) if not found.
    Caches the model after first load for performance.
    """
    global _cached_model, _cached_label_map

    if _cached_model is not None:
        return _cached_model, _cached_label_map

    if not ACTION_MODEL_PATH.exists() or not ACTION_LABELS_PATH.exists():
        return None, None

    try:
        save_data = torch.load(ACTION_MODEL_PATH, map_location=_DEVICE, weights_only=True)
        model = ActionLSTM(save_data["n_features"], save_data["n_classes"]).to(_DEVICE)
        model.load_state_dict(save_data["model_state"])
        model.eval()
        _cached_model = model

        with open(ACTION_LABELS_PATH, 'rb') as f:
            _cached_label_map = pickle.load(f)

        return _cached_model, _cached_label_map
    except Exception as e:
        print(f"[MotionModel] Failed to load: {e}")
        return None, None


def clear_model_cache():
    """Clear the cached model (call after retraining)."""
    global _cached_model, _cached_label_map
    _cached_model = None
    _cached_label_map = None


def predict_action(model, label_map: dict, sequence: np.ndarray):
    """
    Predict action from a single sequence.

    Parameters
    ----------
    model : ActionLSTM
    label_map : dict[int, str]
    sequence : np.ndarray — shape (SEQUENCE_LENGTH, NUM_FEATURES)

    Returns
    -------
    label : str — predicted action name
    confidence : float — prediction confidence (0–1)
    """
    if model is None or label_map is None:
        return "No model", 0.0

    try:
        X = torch.FloatTensor(sequence).unsqueeze(0).to(_DEVICE)

        model.eval()
        with torch.no_grad():
            logits = model(X)
            proba = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(proba))
            confidence = float(proba[idx])

        label = label_map.get(idx, f"Class_{idx}")
        return label, confidence
    except Exception as e:
        print(f"[MotionModel] Prediction error: {e}")
        return "Error", 0.0


def reset_action_model():
    """Delete the saved LSTM model and label map."""
    clear_model_cache()
    for p in [ACTION_MODEL_PATH, ACTION_LABELS_PATH]:
        if p.exists():
            p.unlink()


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENCE BUFFER (Sliding Window)
# ═══════════════════════════════════════════════════════════════════════

class SequenceBuffer:
    """
    Sliding-window frame buffer for real-time action prediction.

    Collects frames until full (SEQUENCE_LENGTH), then allows prediction.
    After a prediction, slides by `slide_step` frames (keeps the rest).
    """

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH,
                 slide_step: int = 3):
        self.sequence_length = sequence_length
        self.slide_step = slide_step
        self._buffer: list[list[float]] = []
        self._ready_for_prediction = False

    def add_frame(self, frame_features: list[float]):
        """Add a single frame's features to the buffer."""
        self._buffer.append(frame_features)

        if len(self._buffer) >= self.sequence_length:
            self._ready_for_prediction = True

    def is_ready(self) -> bool:
        """Check if enough frames are buffered for prediction."""
        return self._ready_for_prediction

    def get_sequence(self) -> np.ndarray:
        """
        Get the current sequence for prediction and slide the window.

        Returns shape (SEQUENCE_LENGTH, NUM_FEATURES).
        """
        if not self.is_ready():
            return np.array([])

        # Take last SEQUENCE_LENGTH frames
        seq = self._buffer[-self.sequence_length:]
        arr = np.array(seq, dtype=np.float32)

        # Slide the window
        self._buffer = self._buffer[self.slide_step:]
        self._ready_for_prediction = len(self._buffer) >= self.sequence_length

        return arr

    def clear(self):
        """Reset the buffer."""
        self._buffer.clear()
        self._ready_for_prediction = False

    @property
    def frame_count(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._buffer)
