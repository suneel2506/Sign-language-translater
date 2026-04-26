# AI Sign Language Translator — Full Case Study & Code Review

## 1. Project Overview

**Project Name:** AI Sign Language Translator  
**Technology Stack:** Python, Streamlit, MediaPipe, OpenCV, scikit-learn, PyTorch  
**Purpose:** Real-time hand gesture recognition that translates sign language gestures into text in multiple languages (English, Tamil, Hindi) with voice output.

### Key Features
- **Static Mode** — Single-frame gesture classification using RandomForest (ML)
- **Motion Mode** — 30-frame temporal action recognition using LSTM (Deep Learning)
- **Learning Mode** — Users can teach the system new gestures by recording samples
- **Multi-language Translation** — English, Tamil, Hindi
- **Voice Output** — Text-to-Speech using pyttsx3
- **Beautiful UI** — Glassmorphism design with dark/light themes

### Architecture Diagram
```
Camera (OpenCV) → MediaPipe Hand Detection → Landmark Extraction
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            Static Mode                      Motion Mode
                        (RandomForest ML)                   (LSTM Deep Learning)
                            ↓                                   ↓
                    Gesture Label + Confidence          Action Label + Confidence
                                    ↓                               ↓
                                    └───────────────┬───────────────┘
                                                    ↓
                                    Translation (English/Tamil/Hindi)
                                                    ↓
                                        Voice Output (pyttsx3)
                                                    ↓
                                    Chat UI Display (Streamlit)
```

### File Structure
```
SignLanguageTranslator/
├── app.py                  ← Main Streamlit application (UI + camera loop)
├── gesture_detector.py     ← MediaPipe hand detection + rule-based gestures
├── learning_mode.py        ← ML training pipeline (RandomForest + feature engineering)
├── action_recorder.py      ← Dataset management for motion sequences
├── motion_model.py         ← LSTM model for temporal action recognition
├── translator.py           ← Multi-language translation dictionary
├── voice_output.py         ← Text-to-Speech module
├── hand_landmarker.task    ← MediaPipe pre-trained hand model file
├── gesture_data.csv        ← Training data for static gestures
├── gesture_model.pkl       ← Trained RandomForest model
├── gesture_scaler.pkl      ← Feature scaler for the model
├── requirements.txt        ← Python dependencies
├── Procfile                ← Deployment start command
├── render.yaml             ← Render deployment config
└── .streamlit/config.toml  ← Streamlit theme configuration
```

---

## 2. FILE: gesture_detector.py — Hand Detection Module

### Purpose
This file uses **MediaPipe's Hand Landmarker** to detect 21 hand landmarks from camera frames and classify simple gestures using rule-based finger position analysis.

### Line-by-Line Explanation

```python
from pathlib import Path          # For file path handling
import cv2                        # OpenCV — image/video processing
import numpy as np                # NumPy — numerical array operations
import mediapipe as mp            # MediaPipe — Google's ML framework for hand detection
```
**Why:** These are the core libraries. `cv2` handles camera frames, `mediapipe` detects hands.

```python
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
```
**Why:** These are the MediaPipe Tasks API classes. We use the newer "Tasks API" (not the legacy Solutions API) because it's more accurate and actively maintained.

```python
_MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")
```
**Why:** `__file__` gives the current script's path. `parent` gets its directory. We join it with the model filename. This ensures the model file is found regardless of where the app is run from.

### The GestureDetector Class

```python
class GestureDetector:
    # Landmark indices — MediaPipe assigns numbers 0-20 to each hand point
    THUMB_TIP = 4      # Tip of thumb
    INDEX_TIP = 8      # Tip of index finger
    MIDDLE_TIP = 12    # Tip of middle finger
    RING_TIP = 16      # Tip of ring finger
    PINKY_TIP = 20     # Tip of pinky
    WRIST = 0          # Wrist point (base of hand)
```
**Why:** MediaPipe detects 21 landmarks on each hand. Each has a fixed index number. Tips are at indices 4,8,12,16,20.

```python
    GESTURES = {
        "Hello":     [1, 1, 1, 1, 1],  # All 5 fingers open
        "Yes":       [1, 0, 0, 0, 0],  # Only thumb up
        "No":        [0, 1, 0, 0, 0],  # Only index finger up
        "Thank You": [1, 0, 0, 0, 1],  # Thumb + pinky (shaka sign)
        "Stop":      [0, 0, 0, 0, 0],  # Closed fist
    }
```
**Why:** Each gesture is defined as a pattern of 5 values [thumb, index, middle, ring, pinky]. `1` = finger extended, `0` = finger curled. This is the rule-based approach — no ML needed.

### Constructor (`__init__`)
```python
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
```
**Why:**
- `model_asset_path` — points to the downloaded `.task` model file
- `RunningMode.IMAGE` — processes one frame at a time (not video stream mode)
- `detection_confidence=0.7` — 70% minimum confidence to consider a hand detected
- `create_from_options()` — creates the actual detector object

### Finger State Detection (`_get_finger_states`)
```python
    def _get_finger_states(self, landmarks, handedness_label):
        fingers = []
        # Thumb — special case: uses X-axis (horizontal comparison)
        if handedness_label == "Right":
            fingers.append(1 if landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x else 0)
        else:
            fingers.append(1 if landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x else 0)

        # Other 4 fingers — uses Y-axis (vertical: tip above PIP = extended)
        tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pips = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        for tip, pip_ in zip(tips, pips):
            fingers.append(1 if landmarks[tip].y < landmarks[pip_].y else 0)
        return fingers
```
**Why:**
- **Thumb** moves sideways (left-right), so we compare X coordinates
- **Other fingers** move up-down, so we compare Y coordinates
- If the tip is above (lower Y value because screen Y is inverted) the PIP joint, the finger is extended
- `handedness_label` matters because thumb direction flips for left vs right hand

### Frame Processing (`process_frame`)
```python
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)           # Mirror the image (selfie view)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, MediaPipe needs RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)  # Run hand detection

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                frame = self._draw_hand_landmarks(frame, hand_lms)  # Draw skeleton

        # Match gesture pattern
        gesture = None
        if result.hand_landmarks and result.handedness:
            finger_states = self._get_finger_states(hand_lms, handedness_label)
            gesture = self._match_gesture(finger_states)

        return frame, gesture
```
**Why:**
1. Flip the frame so it looks like a mirror (natural for the user)
2. Convert BGR→RGB because MediaPipe expects RGB format
3. Detect hand landmarks using the pre-trained model
4. Draw the hand skeleton on the frame for visual feedback
5. Get which fingers are up/down → match against known gesture patterns

### ML-Based Processing (`process_frame_ml`)
```python
    def process_frame_ml(self, frame, model, scaler=None):
        frame, result = self._detect_landmarks(frame)
        if result.hand_landmarks:
            gesture, confidence = predict_gesture(model, scaler, result.hand_landmarks[0])
            if confidence < 0.4:       # Reject low confidence predictions
                gesture = None
        return frame, gesture, confidence
```
**Why:** Instead of rule-based matching, this uses a trained ML model (RandomForest) to classify the gesture. It passes the raw landmarks to the `learning_mode.predict_gesture()` function. Predictions below 40% confidence are rejected to avoid false positives.

---

## 3. FILE: translator.py — Translation Module

### Purpose
Provides instant translations for detected gestures using a hardcoded dictionary (no API calls needed — works offline).

```python
LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "ta": {"name": "Tamil",   "flag": "🇮🇳"},
    "hi": {"name": "Hindi",   "flag": "🇮🇳"},
}
```
**Why:** Defines the supported languages with their codes, names, and flag emojis for the UI.

```python
GESTURE_EMOJIS = {
    "Hello": "✋", "Yes": "👍", "No": "☝️", "Thank You": "🤙", "Stop": "✊",
}
```
**Why:** Maps each gesture name to a visual emoji for display in the chat.

```python
TRANSLATIONS = {
    "Hello": {"en": "Hello", "ta": "வணக்கம்", "hi": "नमस्ते"},
    "Yes":   {"en": "Yes",   "ta": "ஆம்",     "hi": "हाँ"},
    # ... more gestures
}
```
**Why:** A nested dictionary where each gesture has translations in all 3 languages. This is hardcoded for reliability — no internet or API dependency.

```python
def get_all_translations(text):
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    return {code: text for code in LANGUAGES}  # Fallback: return the original text
```
**Why:** Returns all translations at once. If the gesture isn't in the dictionary (e.g., a custom trained gesture), it returns the raw text as a fallback.

---

## 4. FILE: voice_output.py — Text-to-Speech Module

```python
try:
    import pyttsx3              # Offline TTS engine
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False       # Graceful fallback on servers without audio
```
**Why:** `pyttsx3` requires an audio device. On servers like Render (no speakers), the import would crash the app. The try/except prevents that.

```python
def speak(text, rate=150, volume=0.9):
    if not _TTS_AVAILABLE:
        return                   # Skip silently on headless servers

    def _worker():
        engine = pyttsx3.init()         # Initialize TTS engine
        engine.setProperty("rate", rate)     # Words per minute
        engine.setProperty("volume", volume) # Volume 0.0-1.0
        engine.say(text)                # Queue the text
        engine.runAndWait()             # Block until speech finishes
        engine.stop()                   # Clean up

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
```
**Why:** Speech runs in a **background daemon thread** so it never blocks the main UI. Without threading, the app would freeze while speaking. `daemon=True` means the thread dies when the main program exits.
