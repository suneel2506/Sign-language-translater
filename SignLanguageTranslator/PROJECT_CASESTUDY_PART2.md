# Case Study Part 2 — Learning Mode, Action Recorder & Motion Model

---

## 5. FILE: learning_mode.py — ML Training Pipeline (Static Gestures)

### Purpose
This is the brain of the **Static Mode**. It handles:
1. Converting hand landmarks into rich feature vectors (feature engineering)
2. Saving/loading training data from CSV
3. Data augmentation (adding noise for robustness)
4. Training a RandomForest classifier with cross-validation
5. Predicting gestures using the trained model
6. Temporal smoothing (reducing jittery predictions)
7. Model management (save, load, delete named models)

### Constants & Paths
```python
_BASE_DIR = Path(__file__).parent            # Directory where this script lives
DATA_PATH = _BASE_DIR / "gesture_data.csv"   # CSV file storing all training samples
MODEL_PATH = _BASE_DIR / "gesture_model.pkl" # Saved trained model (pickle format)
SCALER_PATH = _BASE_DIR / "gesture_scaler.pkl" # Saved feature scaler

_NUM_LANDMARKS = 21    # MediaPipe detects 21 points on each hand
_COORDS_PER_LM = 3     # Each landmark has x, y, z coordinates
_RAW_FEATURES = 63     # 21 × 3 = 63 raw coordinate features
```

### Landmark Indices
```python
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
# ... (similar for MIDDLE, RING, PINKY)

FINGER_TIPS = [4, 8, 12, 16, 20]   # All 5 fingertip indices
FINGER_PIPS = [3, 6, 10, 14, 18]   # All 5 PIP joint indices
FINGER_MCPS = [2, 5, 9, 13, 17]    # All 5 MCP joint indices
```
**Why:** Named constants make the code readable. Instead of writing `landmarks[8]`, we write `landmarks[INDEX_TIP]`.

### Feature Engineering (`landmarks_to_features`)

This is the most important function — it converts 21 raw landmarks into **98 meaningful features**:

```python
def landmarks_to_features(landmarks):
```

#### Feature Group 1: Normalized XYZ (63 features)
```python
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    norm_xyz = []
    for lm in landmarks:
        norm_xyz.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
```
**Why:** Raw coordinates depend on where the hand is on screen. By subtracting the wrist position (landmark 0), all coordinates become **relative to the wrist**. This means the same gesture at different screen positions produces the same features. This is called **translation invariance**.

#### Feature Group 2: Fingertip-to-Wrist Distances (5 features)
```python
    tip_wrist_dists = [_dist(landmarks[t], landmarks[WRIST]) for t in FINGER_TIPS]
```
**Why:** Measures how far each fingertip is from the wrist. Extended fingers have larger distances. This helps distinguish open hand vs closed fist.

#### Feature Group 3: Fingertip-to-Palm-Center Distances (5 features)
```python
    palm_indices = [WRIST] + list(FINGER_MCPS)
    palm_x = sum(landmarks[i].x for i in palm_indices) / len(palm_indices)
    # ... (average of wrist + 5 MCP joints = approximate palm center)
    tip_palm_dists = [_dist(landmarks[t], palm) for t in FINGER_TIPS]
```
**Why:** Palm center is approximated as the average position of wrist + all MCP joints. Distance from fingertips to palm center gives additional spatial information.

#### Feature Group 4: PIP Joint Curl Angles (5 features)
```python
    finger_chains = [(THUMB_MCP, THUMB_IP, THUMB_TIP), ...]
    for a, b, c in finger_chains:
        pip_angles.append(_angle(landmarks[a], landmarks[b], landmarks[c]))
```
**Why:** The angle at the PIP joint tells us how much each finger is curled. A straight finger ≈ 180°, a fully curled finger ≈ 30-60°.

#### Feature Group 5: MCP Joint Angles (5 features)
```python
    mcp_chains = [(WRIST, THUMB_CMC, THUMB_MCP), ...]
```
**Why:** Similar to PIP angles but at the base (MCP) joint. Together with PIP angles, these capture the full finger pose.

#### Feature Group 6: Inter-Fingertip Distances (10 features)
```python
    for i in range(len(FINGER_TIPS)):
        for j in range(i + 1, len(FINGER_TIPS)):
            inter_tip_dists.append(_dist(landmarks[FINGER_TIPS[i]], landmarks[FINGER_TIPS[j]]))
```
**Why:** All 10 pairwise distances between fingertips (thumb↔index, thumb↔middle, ..., ring↔pinky). This captures the **spread pattern** of the hand — crucial for distinguishing gestures like "Peace" (index+middle spread) from "Hello" (all fingers spread).

#### Feature Group 7: Extension Ratios (5 features)
```python
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        tip_dist = _dist(landmarks[tip_idx], landmarks[WRIST])
        mcp_dist = _dist(landmarks[mcp_idx], landmarks[WRIST]) + 1e-8  # avoid division by zero
        ext_ratios.append(tip_dist / mcp_dist)
```
**Why:** Ratio of (fingertip-to-wrist distance) / (MCP-to-wrist distance). When a finger is extended, this ratio > 1. When curled, ratio ≈ 1 or < 1. The `1e-8` prevents division by zero.

**Total: 63 + 5 + 5 + 5 + 5 + 10 + 5 = 98 features per hand pose**

### Helper Functions

```python
def _dist(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
```
**Why:** Standard 3D Euclidean distance formula between two landmarks.

```python
def _angle(a, b, c):
    # Vectors BA and BC
    ba = [a.x-b.x, a.y-b.y, a.z-b.z]
    bc = [c.x-b.x, c.y-b.y, c.z-b.z]
    # Dot product → angle
    cos_angle = dot / (mag_ba * mag_bc)
    return math.degrees(math.acos(cos_angle))
```
**Why:** Calculates the angle at point B formed by points A-B-C using the dot product formula. Used for measuring joint bend angles.

### Data Collection

```python
def save_landmarks(label, landmarks_batch):
    with open(DATA_PATH, "a", newline="") as f:  # "a" = append mode
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_get_csv_header())    # Write header only once
        for features in landmarks_batch:
            writer.writerow([label] + features)   # [gesture_name, feat1, feat2, ...]
```
**Why:** Appends feature vectors to a CSV file. Each row is: `label, x0, y0, z0, x1, y1, z1, ..., ext_r4`. The header row names each column.

### Data Augmentation
```python
def _augment_data(X, y, factor=2, noise_level=0.005):
    for _ in range(factor):
        noise = np.random.normal(0, noise_level, X.shape)  # Gaussian noise
        augmented_X.append(X + noise)                        # Add noise to features
        augmented_y.append(y)                                # Same labels
    return np.vstack(augmented_X), np.concatenate(augmented_y)
```
**Why:** Creates synthetic training data by adding small random noise. With `factor=2`, we triple our dataset (original + 2 noisy copies). This makes the model more **robust** — it won't overfit to exact positions. `noise_level=0.005` means very slight variations (0.5% of coordinate range).

### Model Training (`train_model`)

```python
def train_model():
    X, y = load_dataset()                    # Load CSV → numpy arrays

    # Check we have enough data
    unique, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return {"error": "Not enough samples per class"}
    if len(unique) < 2:
        return {"error": "Need at least 2 gesture classes"}

    # Augment data (triple the dataset with noise)
    X_aug, y_aug = _augment_data(X, y, factor=2, noise_level=0.005)

    # Scale features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)
```
**Why StandardScaler:** Features have different scales (coordinates 0-1, angles 0-180°, distances 0-0.5). StandardScaler normalizes each feature to mean=0, std=1. Without this, large-scale features would dominate the model.

```python
    # RandomForest classifier
    clf = RandomForestClassifier(
        n_estimators=200,        # 200 decision trees in the forest
        max_depth=25,            # Each tree can be up to 25 levels deep
        min_samples_split=3,     # Need at least 3 samples to split a node
        min_samples_leaf=2,      # Each leaf must have at least 2 samples
        max_features="sqrt",     # Each tree considers √n features at each split
        random_state=42,         # Reproducible results
        n_jobs=-1,               # Use all CPU cores for parallel training
        class_weight="balanced", # Handle class imbalance automatically
    )
```
**Why RandomForest:**
- It's an **ensemble** of 200 decision trees — each tree votes, majority wins
- `class_weight="balanced"` — if you have 100 "Hello" samples but only 30 "Stop", it upweights "Stop" samples automatically
- `max_features="sqrt"` — each tree sees only a random subset of features, adding diversity

```python
    # Cross-validation on original (non-augmented) data
    X_orig_scaled = scaler.transform(X)
    n_folds = min(5, min_samples)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_orig_scaled, y, cv=skf, scoring="accuracy")
    accuracy = float(scores.mean())
```
**Why StratifiedKFold:** Splits data into k folds while maintaining the same class ratio in each fold. We evaluate on **original data only** (not augmented) to get a realistic accuracy estimate. We use augmented data only for training.

```python
    clf.fit(X_scaled, y_aug)               # Train on full augmented data
    joblib.dump(clf, MODEL_PATH)           # Save model to .pkl file
    joblib.dump(scaler, SCALER_PATH)       # Save scaler too (needed at prediction time)
```

### Prediction
```python
def predict_gesture(model, scaler, landmarks):
    features = landmarks_to_features(landmarks)  # 21 landmarks → 98 features
    X = np.array(features).reshape(1, -1)         # Shape: (1, 98) — single sample
    if scaler is not None:
        X = scaler.transform(X)                    # Must scale with same scaler
    label = model.predict(X)[0]                    # Predicted class name
    proba = model.predict_proba(X)[0]              # Probability for each class
    confidence = float(proba.max())                # Highest probability = confidence
    if confidence < 0.45:
        return "Unknown", confidence               # Reject low-confidence predictions
    return label, confidence
```

### Temporal Smoothing (`PredictionSmoother`)
```python
class PredictionSmoother:
    def __init__(self, window_size=7):
        self._buffer = []                  # Last 7 predictions

    def add(self, label, confidence):
        self._buffer.append((label, confidence))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)            # Remove oldest

    def get_smoothed(self):
        # Count votes per label, return the majority
        best_label = max(votes, key=lambda k: len(votes[k]))
        # Require at least 40% of window to agree
        if vote_ratio < 0.4:
            return "Unknown", avg_conf
        return best_label, avg_conf
```
**Why:** Raw frame-by-frame predictions flicker rapidly (e.g., "Hello" → "Yes" → "Hello" → "Hello"). The smoother keeps the last 7 predictions and returns the **majority vote**. This eliminates jitter and gives stable output.

### Model Management
```python
def save_model_as(name):     # Copy model+scaler+data to saved_models/<name>/
def load_model_by_name(name): # Load a previously saved model
def delete_saved_model(name): # Delete a saved model
def reset_current_model():    # Delete active model + all training data
```
**Why:** Allows users to save multiple trained models (e.g., "numbers_v1", "alphabet_v2") and switch between them.

---

## 6. FILE: action_recorder.py — Motion Dataset Manager

### Purpose
Manages folder-based storage for motion sequences. Each action (e.g., "wave") gets a folder containing `.npy` files (NumPy arrays) of recorded frame sequences.

### Storage Structure
```
dataset/
├── wave/
│   ├── sequence_0.npy    ← shape (15, 63)
│   ├── sequence_1.npy
│   └── sequence_2.npy
├── thumbs_up/
│   ├── sequence_0.npy
│   └── sequence_1.npy
```

### Key Constants
```python
SEQUENCE_LENGTH = 15     # 15 frames per sequence (~0.5 seconds at 30fps)
NUM_LANDMARKS = 21       # MediaPipe hand landmarks
COORDS_PER_LM = 3        # x, y, z per landmark
NUM_FEATURES = 63         # 21 × 3 = 63 features per frame
```

### Landmark Normalization
```python
def landmarks_to_flat(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    flat = []
    for lm in landmarks:
        flat.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return flat   # Returns 63 values
```
**Why:** Same wrist-normalization as in learning_mode.py. Subtracting wrist position makes features position-independent.

### Saving Sequences
```python
def save_sequence(action_name, frames, sequence_idx=None):
    folder = create_action_folder(action_name)

    # Pad or truncate to exactly SEQUENCE_LENGTH frames
    if len(frames) < SEQUENCE_LENGTH:
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])      # Repeat last frame as padding
    elif len(frames) > SEQUENCE_LENGTH:
        frames = frames[:SEQUENCE_LENGTH]   # Truncate extra frames

    arr = np.array(frames, dtype=np.float32)  # shape: (15, 63)
    np.save(folder / f"sequence_{idx}.npy", arr)
```
**Why:** The LSTM model requires fixed-length input. If recording stops early (< 15 frames), we pad by repeating the last frame. If too many frames, we truncate. This ensures every sequence is exactly (15, 63).

### Loading All Sequences
```python
def load_all_sequences(dataset_dir=None):
    for action in actions:                     # For each action folder
        for npy_file in action_dir.glob("sequence_*.npy"):
            seq = np.load(npy_file)            # Load numpy array
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X_list.append(seq)
                y_list.append(action)

    X = np.array(X_list, dtype=np.float32)     # shape: (N, 15, 63)
    y = np.array(y_list)                       # shape: (N,) — action labels
    return X, y, actions
```
**Why:** Loads all sequences into memory for LSTM training. Returns X (3D array of sequences), y (labels), and unique action names.

---

## 7. FILE: motion_model.py — LSTM Deep Learning Model

### Purpose
Builds, trains, and runs an **LSTM (Long Short-Term Memory)** neural network for temporal action recognition. Unlike the static RandomForest which looks at one frame, the LSTM looks at **15 consecutive frames** to recognize motions/actions.

### Device Selection
```python
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
**Why:** Uses GPU (NVIDIA CUDA) if available for faster training. Otherwise falls back to CPU.

### LSTM Architecture
```python
class ActionLSTM(nn.Module):
    def __init__(self, n_features, n_classes):
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)   # First LSTM layer
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)           # Second LSTM layer
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),     # Dense layer: 64 → 32
            nn.ReLU(),              # Activation function (non-linearity)
            nn.Dropout(0.3),        # Randomly drops 30% of neurons (prevents overfitting)
            nn.Linear(32, n_classes), # Output layer: 32 → number of actions
        )

    def forward(self, x):          # x shape: (batch, 15, 63)
        x, _ = self.lstm1(x)       # → (batch, 15, 64)
        x, _ = self.lstm2(x)       # → (batch, 15, 64)
        x = x[:, -1, :]           # Take LAST time step → (batch, 64)
        x = self.classifier(x)    # → (batch, n_classes)
        return x                   # Raw logits (CrossEntropyLoss includes softmax)
```
**Why LSTM:**
- LSTM is a type of Recurrent Neural Network (RNN) designed for **sequential data**
- It has "memory cells" that can remember patterns across time steps
- **2 stacked LSTM layers** (64 units each) — deeper = better pattern learning
- We take only the **last time step's output** because it contains information from all 15 frames
- **Dropout(0.3)** — during training, randomly zeroes 30% of neurons to prevent overfitting

### Training Pipeline (`train_lstm`)
```python
def train_lstm(dataset_path=None):
    X, y, actions = load_all_sequences(ds_dir)  # Load all sequences

    # Label encoding: "wave"→0, "thumbs_up"→1, etc.
    label_to_idx = {label: idx for idx, label in enumerate(actions)}
    y_encoded = np.array([label_to_idx[label] for label in y])

    # Train/validation split (stratified — keeps class balance)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.15, stratify=y_encoded)

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(_DEVICE)
    y_train_t = torch.LongTensor(y_train).to(_DEVICE)
```
**Why:**
- **Label encoding** — neural networks need numeric labels (0,1,2...) not strings
- **Stratified split** — ensures validation set has the same class distribution as training
- **Tensors** — PyTorch's data format, moved to GPU if available

```python
    model = build_lstm_model(n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()
```
**Why:**
- **Adam optimizer** — adaptive learning rate optimizer, good default choice
- **lr=1e-3 (0.001)** — learning rate controls how big each weight update is
- **ReduceLROnPlateau** — halves learning rate if validation loss stops improving for 10 epochs
- **CrossEntropyLoss** — standard loss function for classification; combines softmax + negative log-likelihood

```python
    # Training loop with early stopping
    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()          # Clear old gradients
            outputs = model(X_batch)       # Forward pass
            loss = criterion(outputs, y_batch)  # Calculate loss
            loss.backward()                # Backpropagation (compute gradients)
            optimizer.step()               # Update weights

        # Validation
        model.eval()
        with torch.no_grad():             # No gradient needed for validation
            val_outputs = model(X_val_t)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == y_val_t).float().mean()

        # Early stopping — stop if no improvement for 15 epochs
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break
```
**Why Early Stopping:** Without it, the model would train for all 100 epochs even if it stopped improving at epoch 30. Early stopping saves time and prevents overfitting.

### Prediction
```python
def predict_action(model, label_map, sequence):
    X = torch.FloatTensor(sequence).unsqueeze(0).to(_DEVICE)  # (1, 15, 63)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        proba = torch.softmax(logits, dim=1)[0]     # Convert to probabilities
        idx = int(torch.argmax(proba))               # Index of highest probability
        confidence = float(proba[idx])
    label = label_map.get(idx, f"Class_{idx}")       # Convert index → action name
    return label, confidence
```

### Sliding Window Buffer (`SequenceBuffer`)
```python
class SequenceBuffer:
    def __init__(self, sequence_length=15, slide_step=3):
        self._buffer = []

    def add_frame(self, frame_features):
        self._buffer.append(frame_features)
        if len(self._buffer) >= self.sequence_length:
            self._ready_for_prediction = True

    def get_sequence(self):
        seq = self._buffer[-self.sequence_length:]   # Take last 15 frames
        self._buffer = self._buffer[self.slide_step:]  # Slide by 3 frames
        return np.array(seq)
```
**Why Sliding Window:** Instead of waiting for a completely new 15-frame sequence each time, we slide by 3 frames. This means predictions happen every 3 frames (~100ms) instead of every 15 frames (~500ms), giving faster real-time response.
