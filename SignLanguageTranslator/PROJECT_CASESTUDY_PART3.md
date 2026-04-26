# Case Study Part 3 — Main Application (app.py) & Configuration

---

## 8. FILE: app.py — Main Streamlit Application (1677 lines)

This is the largest file — the complete UI and camera processing loop. It ties everything together.

### 8.1 Imports (Lines 1-66)
```python
import time                     # For cooldown timers and frame rate control
import cv2                      # OpenCV — camera capture and image processing
import numpy as np              # NumPy — array operations
import streamlit as st          # Streamlit — web UI framework

from gesture_detector import GestureDetector       # Hand detection
from translator import GESTURE_EMOJIS, LANGUAGES, get_all_translations  # Translation
from voice_output import speak                      # Text-to-speech
from learning_mode import (                         # Static ML pipeline
    landmarks_to_features, save_landmarks, get_dataset_info,
    train_model, load_model, predict_gesture, PredictionSmoother,
    list_saved_models, save_model_as, load_model_by_name,
    delete_saved_model, reset_current_model, MODEL_PATH,
)
from action_recorder import (                       # Motion dataset management
    set_dataset_path, create_action_folder, list_actions,
    get_action_info, save_sequence, delete_action, delete_all_actions,
    landmarks_to_flat, load_all_sequences, SEQUENCE_LENGTH, NUM_FEATURES,
)
from motion_model import (                          # LSTM model
    build_lstm_model, train_lstm, load_action_model, predict_action,
    reset_action_model, clear_model_cache, SequenceBuffer, ACTION_MODEL_PATH,
)
```
**Why:** Each module is imported for its specific functionality. This modular design keeps the code organized.

### 8.2 Page Configuration (Lines 73-78)
```python
st.set_page_config(
    page_title="AI Sign Language Translator",  # Browser tab title
    page_icon="🤟",                            # Browser tab icon
    layout="wide",                              # Use full screen width
    initial_sidebar_state="expanded",           # Sidebar open by default
)
```
**Why:** Must be the FIRST Streamlit call. `layout="wide"` uses the full browser width instead of a centered narrow column.

### 8.3 Session State Initialization (Lines 85-178)

```python
def _init_session_state():
    _DEFAULTS = {
        "messages": [],           # Chat message history
        "running": False,         # Is camera active?
        "cap": None,              # OpenCV VideoCapture object
        "detector": None,         # GestureDetector instance
        "pending_gesture": None,  # Gesture being confirmed
        "gesture_counter": 0,     # Frames the same gesture has been seen
        "last_gesture": None,     # Last confirmed gesture
        "last_detection_time": 0.0,  # Timestamp of last detection (for cooldown)
        "total_detections": 0,    # Counter for stats display
        "theme": "dark",          # Current theme (dark/light)
        "learning_mode": False,   # Is learning mode active?
        "recording": False,       # Is recording gestures?
        "recorded_landmarks": [], # Buffer for recorded features
        "ml_model": None,         # Trained RandomForest model
        "ml_scaler": None,        # StandardScaler for the model
        "use_ml_model": False,    # Toggle: use ML model vs rules
        "prediction_smoother": None,  # PredictionSmoother instance
        "motion_mode": False,     # Is motion mode active?
        "motion_recording": False,  # Recording motion sequences?
        "motion_frames": [],      # Buffer for motion frames
        "action_model": None,     # Trained LSTM model
        "sequence_buffer": None,  # SequenceBuffer for sliding window
    }
    for key, val in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val
```
**Why Session State:** Streamlit re-runs the entire script on every interaction (button click, toggle, etc.). `st.session_state` persists data between reruns. Without it, the camera object, model, and chat history would be lost every time.

```python
    # Auto-load models if they exist on disk
    if st.session_state.ml_model is None and MODEL_PATH.exists():
        st.session_state.ml_model, st.session_state.ml_scaler = load_model()

    if st.session_state.action_model is None and ACTION_MODEL_PATH.exists():
        m, lm = load_action_model()
        st.session_state.action_model = m
        st.session_state.action_label_map = lm
```
**Why:** If a model was trained in a previous session, it automatically loads when the app starts.

### 8.4 CSS Styling (Lines 185-568)

The app uses **CSS injection** via `st.markdown()` with `unsafe_allow_html=True`:

```python
def _get_theme_css(theme):
    if theme == "dark":
        return """
        :root {
            --primary: #6366f1;       /* Indigo - main brand color */
            --accent: #a855f7;        /* Purple - secondary color */
            --bg-dark: #0f0c29;       /* Deep navy background */
            --text: #e2e8f0;          /* Light gray text */
            --glass-bg: rgba(255,255,255,0.04);  /* Glassmorphism background */
        }"""
```
**Why CSS Variables:** Using `--primary`, `--bg-dark` etc. as CSS variables means changing one value updates the entire app's color scheme. This enables the dark/light theme toggle.

Key CSS effects:
- **Glassmorphism** — `backdrop-filter: blur(14px)` creates frosted glass effect on chat bubbles
- **Gradient Title** — `background: linear-gradient(...)` with `animation: gradient-flow` creates animated gradient text
- **Micro-animations** — `@keyframes fadeInUp` makes chat bubbles slide in smoothly
- **Hover effects** — buttons lift up (`translateY(-2px)`) and glow on hover

### 8.5 Camera Callbacks (Lines 584-641)

```python
def _start_camera():
    cap = cv2.VideoCapture(0)        # Open default webcam (index 0)
    if not cap.isOpened():
        # Show error in chat
        return
    st.session_state.cap = cap
    st.session_state.detector = GestureDetector()
    st.session_state.running = True
```
**Why:** `cv2.VideoCapture(0)` opens the default camera. The `cap` object is stored in session state so it persists across reruns. `GestureDetector()` initializes MediaPipe.

```python
def _stop_camera():
    st.session_state.cap.release()    # Release camera hardware
    st.session_state.detector.release()  # Release MediaPipe resources
    st.session_state.running = False
```
**Why:** Always release the camera when done, otherwise it stays locked and other apps can't use it.

### 8.6 HUD Overlay (Lines 648-752)

```python
def _overlay_hud(frame, gesture, counter, threshold, recently_detected, ...):
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (10, 8, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture label text
    cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Progress bar (how close to confirmation)
    progress = min(counter / max(threshold, 1), 1.0)
    cv2.rectangle(frame, (bar_x, 52), (bar_x + int(bar_w * progress), 68), fill_color, -1)

    # Green flash border on detection
    if recently_detected:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 255, 130), 4)
```
**Why:** The HUD (Heads-Up Display) overlays information directly on the camera feed:
- Semi-transparent dark bar at the top
- Current gesture name and confidence
- Progress bar showing confirmation progress
- Green border flash when a gesture is confirmed

### 8.7 Sidebar UI (Lines 759-1278)

The sidebar contains all controls organized in sections:

1. **Settings** — Theme toggle, voice toggle, sensitivity slider, cooldown slider
2. **Learning Mode (Static)** — Record gestures, train model, manage saved models
3. **Motion Mode (LSTM)** — Record sequences, train LSTM, start/stop detection
4. **Gesture Guide** — Visual reference of supported gestures

```python
sensitivity = st.select_slider("Detection Sensitivity", ["Low", "Medium", "High"])
_SENSITIVITY_MAP = {"Low": 25, "Medium": 15, "High": 8}
CONFIRM_FRAMES = _SENSITIVITY_MAP[sensitivity]
```
**Why:** Higher sensitivity = fewer frames needed to confirm a gesture. "High" needs only 8 consecutive frames, "Low" needs 25. This controls the trade-off between speed and accuracy.

### 8.8 Main Layout (Lines 1294-1396)

```python
col_cam, col_chat = st.columns([1, 1], gap="large")  # 50/50 split
```
The screen is divided into two equal columns:
- **Left:** Camera feed + controls
- **Right:** Chat conversation

```python
with col_chat:
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"], unsafe_allow_html=True)
```
**Why:** Each detected gesture creates a new chat message showing the translation in all languages. `height=500` makes it scrollable.

### 8.9 Camera Processing Loop (Lines 1398-1677) ⭐ CORE LOOP

This is the heart of the application — a continuous `while` loop that processes camera frames:

```python
if st.session_state.running and st.session_state.cap is not None:
    while st.session_state.running:
        ret, raw_frame = cap.read()          # Read one frame from camera
        if not ret:
            break                             # Camera disconnected

        now = time.time()
        recently_detected = now - st.session_state.last_detection_time < 0.6
```
**Why a while loop instead of st.rerun():** Using `st.rerun()` after each frame would cause the entire page to reload (flash/blink). The `while` loop only updates the `st.empty()` placeholders, giving smooth video without flickering.

#### Static Mode Processing:
```python
        if ml_model is not None:
            processed, raw_gesture, raw_confidence = detector.process_frame_ml(
                raw_frame, ml_model, ml_scaler)
            smoother.add(raw_gesture, raw_confidence)
            gesture, ml_confidence = smoother.get_smoothed()
        else:
            processed, gesture = detector.process_frame(raw_frame)  # Rule-based
```
**Why:** If a trained ML model exists, use it. Otherwise, fall back to rule-based detection.

#### Confirmation Logic:
```python
        if gesture == st.session_state.pending_gesture:
            st.session_state.gesture_counter += 1     # Same gesture, increment
        else:
            st.session_state.pending_gesture = gesture
            st.session_state.gesture_counter = 1      # New gesture, reset

        confirmed = (
            gesture_counter >= CONFIRM_FRAMES         # Seen enough times
            and (now - last_detection_time) > cooldown_sec  # Cooldown passed
        )
```
**Why Confirmation:** Prevents false positives. A gesture must be held for CONFIRM_FRAMES consecutive frames (e.g., 15 frames = ~0.5 seconds) AND the cooldown period must have passed since the last detection.

#### Motion Mode Processing:
```python
        if motion_detecting and flat_features is not None:
            seq_buffer.add_frame(flat_features)        # Add frame to buffer

            if seq_buffer.is_ready():                   # 15 frames collected?
                sequence = seq_buffer.get_sequence()
                raw_label, raw_conf = predict_action(   # LSTM prediction
                    action_model, action_label_map, sequence)

                motion_smoother.add(raw_label, raw_conf)
                sm_label, sm_conf = motion_smoother.get_smoothed()

                if sm_conf >= 0.40:                     # Confidence threshold
                    # Add to chat, speak translation, reset buffer
```

#### Frame Display:
```python
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # BGR → RGB for display
        video_placeholder.image(rgb, use_container_width=True)

        time.sleep(0.033)                    # ~30 FPS cap (1/30 ≈ 0.033 seconds)
```
**Why `time.sleep(0.033)`:** Without this, the loop would consume 100% CPU. Sleeping for 33ms limits the frame rate to ~30 FPS, which is smooth enough for real-time video while being CPU-friendly.

---

## 9. Configuration Files

### .streamlit/config.toml
```toml
[theme]
primaryColor = "#6366f1"                  # Indigo — buttons, links
backgroundColor = "#0f0c29"              # Deep navy — main background
secondaryBackgroundColor = "#1e1b4b"     # Slightly lighter — sidebar
textColor = "#e2e8f0"                    # Light gray — text
font = "sans serif"                       # Clean modern font
```
**Why:** Streamlit's built-in theme system. These colors match our CSS variables for a consistent dark theme.

### Procfile
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
**Why:** Used by deployment platforms (Render, Heroku). `$PORT` is assigned by the platform. `0.0.0.0` makes the app accessible from outside (not just localhost).

### render.yaml
```yaml
services:
  - type: web
    name: sign-language-translator
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run SignLanguageTranslator/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.9"
```
**Why:** Render Blueprint for automatic deployment. `--server.headless true` disables the "Are you sure you want to open this?" prompt on servers.

### requirements.txt
```
streamlit              # Web UI framework
numpy                  # Numerical computing
opencv-python-headless # Image processing (no GUI needed on server)
mediapipe              # Google's hand detection ML framework
scikit-learn           # Machine learning (RandomForest, cross-validation)
joblib                 # Model serialization (save/load .pkl files)
pandas                 # Data manipulation (CSV loading)
pyttsx3                # Text-to-speech engine
torch                  # PyTorch deep learning framework (LSTM model)
```

---

## 10. Summary — How Everything Works Together

### Detection Flow (Static Mode):
1. **Camera** captures frame via OpenCV
2. **MediaPipe** detects 21 hand landmarks
3. **Feature Engineering** converts landmarks → 98 features (or rule-based finger states)
4. **RandomForest** classifies the gesture (or rule-based pattern matching)
5. **Temporal Smoother** stabilizes predictions (majority vote over 7 frames)
6. **Confirmation** requires holding gesture for N consecutive frames + cooldown
7. **Translation** looks up the gesture in the dictionary
8. **Voice Output** speaks the English translation in a background thread
9. **Chat UI** displays the detection with all translations

### Detection Flow (Motion Mode):
1. **Camera** captures frame via OpenCV
2. **MediaPipe** detects 21 hand landmarks
3. **Normalization** converts to 63 wrist-relative features
4. **Sliding Window Buffer** collects 15 consecutive frames
5. **LSTM Model** processes the (15, 63) sequence → action prediction
6. **Temporal Smoother** stabilizes predictions
7. **Cooldown** prevents duplicate detections
8. **Translation + Voice + Chat** same as static mode

### Key Design Decisions:
| Decision | Reason |
|----------|--------|
| RandomForest for static | Fast training, works well with small datasets |
| LSTM for motion | Captures temporal patterns across frames |
| 98 engineered features | Much more accurate than raw 63 coordinates |
| Data augmentation (noise) | Prevents overfitting with small datasets |
| Sliding window (step=3) | Fast predictions without waiting for full new sequence |
| Temporal smoothing | Eliminates flickering/jittery predictions |
| Confirmation frames | Prevents accidental false positive detections |
| Cooldown timer | Prevents the same gesture from being detected repeatedly |
| Background TTS thread | UI never freezes while speaking |
| CSS glassmorphism | Modern, premium visual design |
| Wrist normalization | Position-independent gesture recognition |

---

*Document prepared for project presentation and code review.*
*Project by: Suneel Kumar | GitHub: github.com/suneel2506/Sign-language-translater*
