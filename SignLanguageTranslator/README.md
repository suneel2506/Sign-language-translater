# 🤟 AI Sign Language Translator

A real-time Sign Language Detection system with a chat-style interface. Uses your **webcam** to detect hand gestures, **translates** them into multiple languages, and **speaks** the output — all inside a beautiful dark-themed Streamlit UI.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| ✋ **Gesture Detection** | MediaPipe Hands + rule-based finger counting (no training needed) |
| 🌍 **Multi-Language Translation** | English, Tamil (தமிழ்), Hindi (हिन्दी) |
| 🔊 **Voice Output** | Offline text-to-speech via pyttsx3 |
| 💬 **Chat-Style UI** | Beautiful dark theme with animated gradient title |
| ⚙️ **Configurable** | Sensitivity, cooldown, and voice toggle in the sidebar |

### Supported Gestures

| Gesture | Hand Pose | Emoji |
|---------|-----------|-------|
| **Hello** | All 5 fingers open (open palm) | ✋ |
| **Yes** | Only thumb extended (thumbs up) | 👍 |
| **No** | Only index finger extended | ☝️ |
| **Thank You** | Thumb + pinky extended (shaka) | 🤙 |
| **Stop** | Closed fist (no fingers up) | ✊ |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** – webcam capture
- **MediaPipe** – hand landmark detection
- **Streamlit** – chat-style web UI
- **pyttsx3** – offline text-to-speech

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10** or newer
- A working **webcam**
- **Windows / macOS / Linux**

### Installation

```bash
# 1. Clone or navigate to the project folder
cd SignLanguageTranslator

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**.

---

## 🎮 How to Use

1. Click **▶️ Start Camera** to begin webcam capture.
2. Hold your hand in front of the camera showing one of the **5 gestures**.
3. Keep the gesture **steady** until the progress bar fills up (confirmation).
4. The detected gesture appears in the **chat window** with translations.
5. If **voice output** is enabled, the word is spoken aloud.
6. Click **⏹️ Stop Camera** when finished.

### Sidebar Controls

| Control | Description |
|---------|-------------|
| 🔊 Voice Output | Toggle text-to-speech on/off |
| 🎚️ Sensitivity | Low / Medium / High – controls how many frames to confirm a gesture |
| ⏱️ Cooldown | Seconds to wait between detections (prevents repeats) |
| 🗑️ Clear Chat | Wipe the conversation history |

---

## 📁 Project Structure

```
SignLanguageTranslator/
├── .streamlit/
│   └── config.toml          # Streamlit dark theme config
├── app.py                   # Main Streamlit application
├── gesture_detector.py      # MediaPipe hand tracking & gesture rules
├── translator.py            # Translation dictionary
├── voice_output.py          # pyttsx3 text-to-speech wrapper
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Make sure no other app is using the webcam |
| MediaPipe import error | Run `pip install mediapipe --upgrade` |
| pyttsx3 crashes | On Linux, install `espeak`: `sudo apt install espeak` |
| Gesture not detected | Ensure good lighting and keep your full hand visible |
| Wrong hand detected | The app defaults to 1 hand; make sure only one hand is in frame |

---

## 📜 License

This project is open-source and free to use for educational purposes.
