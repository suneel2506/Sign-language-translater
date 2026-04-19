"""
Action Recorder Module
======================
Manages folder-based dataset storage for motion/action sequences.

Each action is stored as:
    dataset/<action_name>/sequence_0.npy
    dataset/<action_name>/sequence_1.npy
    ...

Each .npy file contains shape (SEQUENCE_LENGTH, NUM_FEATURES):
    30 frames × 63 features (21 landmarks × 3 coords, wrist-normalized)
"""

from pathlib import Path
import shutil
import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
DEFAULT_DATASET_DIR = _BASE_DIR / "dataset"

SEQUENCE_LENGTH = 15       # frames per sequence (shorter = faster response)
NUM_LANDMARKS = 21
COORDS_PER_LM = 3
NUM_FEATURES = NUM_LANDMARKS * COORDS_PER_LM  # 63


# ═══════════════════════════════════════════════════════════════════════
#  DATASET PATH MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

_current_dataset_dir: Path = DEFAULT_DATASET_DIR


def set_dataset_path(path: str | Path) -> Path:
    """
    Set the root dataset directory.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path for the dataset root.

    Returns
    -------
    Path — the resolved dataset directory.
    """
    global _current_dataset_dir
    _current_dataset_dir = Path(path).resolve()
    _current_dataset_dir.mkdir(parents=True, exist_ok=True)
    return _current_dataset_dir


def get_dataset_path() -> Path:
    """Return the current dataset root directory."""
    return _current_dataset_dir


# ═══════════════════════════════════════════════════════════════════════
#  ACTION FOLDER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def create_action_folder(action_name: str) -> Path:
    """
    Create a folder for the given action inside the dataset directory.

    Returns the path to the folder.
    """
    folder = _current_dataset_dir / action_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_actions() -> list[str]:
    """Return sorted list of action names (subdirectory names)."""
    if not _current_dataset_dir.exists():
        return []
    return sorted([
        d.name for d in _current_dataset_dir.iterdir()
        if d.is_dir() and any(d.glob("*.npy"))
    ])


def get_action_info() -> dict[str, int]:
    """
    Return a dict mapping action_name → number of recorded sequences.

    Example: {"Hello": 15, "Good": 12, "Stop": 18}
    """
    info = {}
    if not _current_dataset_dir.exists():
        return info

    for d in sorted(_current_dataset_dir.iterdir()):
        if d.is_dir():
            n_sequences = len(list(d.glob("*.npy")))
            if n_sequences > 0:
                info[d.name] = n_sequences

    return info


def get_next_sequence_index(action_name: str) -> int:
    """Return the next available sequence index for the given action."""
    folder = _current_dataset_dir / action_name
    if not folder.exists():
        return 0

    existing = list(folder.glob("sequence_*.npy"))
    if not existing:
        return 0

    indices = []
    for f in existing:
        try:
            idx = int(f.stem.replace("sequence_", ""))
            indices.append(idx)
        except ValueError:
            continue

    return max(indices) + 1 if indices else 0


# ═══════════════════════════════════════════════════════════════════════
#  SEQUENCE RECORDING & STORAGE
# ═══════════════════════════════════════════════════════════════════════

def landmarks_to_flat(landmarks) -> list[float]:
    """
    Convert 21 MediaPipe hand landmarks to a flat wrist-normalized list.

    Parameters
    ----------
    landmarks : list of NormalizedLandmark
        The 21 hand landmarks from MediaPipe.

    Returns
    -------
    list[float] — 63 values (21 × 3 xyz, wrist-normalized)
    """
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z

    flat = []
    for lm in landmarks:
        flat.extend([
            lm.x - base_x,
            lm.y - base_y,
            lm.z - base_z,
        ])
    return flat


def save_sequence(action_name: str, frames: list[list[float]],
                  sequence_idx: int | None = None) -> Path:
    """
    Save a single sequence (list of frame feature vectors) as a .npy file.

    Parameters
    ----------
    action_name : str
        The action/gesture label.
    frames : list[list[float]]
        List of SEQUENCE_LENGTH frame vectors, each of length NUM_FEATURES.
    sequence_idx : int or None
        If None, auto-assigns the next available index.

    Returns
    -------
    Path — the saved .npy file path.
    """
    folder = create_action_folder(action_name)

    if sequence_idx is None:
        sequence_idx = get_next_sequence_index(action_name)

    # Pad or truncate to SEQUENCE_LENGTH
    if len(frames) < SEQUENCE_LENGTH:
        # Pad with the last frame repeated
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1] if frames else [0.0] * NUM_FEATURES)
    elif len(frames) > SEQUENCE_LENGTH:
        frames = frames[:SEQUENCE_LENGTH]

    arr = np.array(frames, dtype=np.float32)
    assert arr.shape == (SEQUENCE_LENGTH, NUM_FEATURES), (
        f"Expected shape ({SEQUENCE_LENGTH}, {NUM_FEATURES}), got {arr.shape}"
    )

    path = folder / f"sequence_{sequence_idx}.npy"
    np.save(path, arr)
    return path


def overwrite_sequence(action_name: str, sequence_idx: int,
                       frames: list[list[float]]) -> Path:
    """Re-record / overwrite a specific sequence."""
    return save_sequence(action_name, frames, sequence_idx=sequence_idx)


def delete_action(action_name: str) -> bool:
    """Delete all data for the given action. Returns True if deleted."""
    folder = _current_dataset_dir / action_name
    if folder.exists():
        shutil.rmtree(folder)
        return True
    return False


def delete_all_actions() -> int:
    """Delete the entire dataset. Returns number of actions removed."""
    if not _current_dataset_dir.exists():
        return 0

    actions = list_actions()
    for a in actions:
        delete_action(a)
    return len(actions)


def load_all_sequences(dataset_dir: Path | None = None):
    """
    Load all sequences from the dataset directory.

    Returns
    -------
    X : np.ndarray — shape (N, SEQUENCE_LENGTH, NUM_FEATURES)
    y : np.ndarray — shape (N,) action labels as strings
    actions : list[str] — sorted unique action names
    """
    ds_dir = dataset_dir or _current_dataset_dir

    X_list = []
    y_list = []

    if not ds_dir.exists():
        return np.array([]), np.array([]), []

    actions = sorted([
        d.name for d in ds_dir.iterdir()
        if d.is_dir() and any(d.glob("*.npy"))
    ])

    for action in actions:
        action_dir = ds_dir / action
        for npy_file in sorted(action_dir.glob("sequence_*.npy")):
            seq = np.load(npy_file)
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X_list.append(seq)
                y_list.append(action)

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y, actions
