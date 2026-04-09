"""
AI Sign Language Translator – Main Streamlit Application
========================================================
Real-time hand gesture recognition → text → translation → voice output,
all inside a beautiful chat-style interface.

Run with:  streamlit run app.py
"""

import time

import cv2
import numpy as np
import streamlit as st

from gesture_detector import GestureDetector
from translator import (
    GESTURE_EMOJIS,
    LANGUAGES,
    get_all_translations,
)
from voice_output import speak
from learning_mode import (
    landmarks_to_features,
    save_landmarks,
    get_dataset_info,
    train_model,
    load_model,
    predict_gesture,
    MODEL_PATH,
)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "messages": [],
    "running": False,
    "cap": None,
    "detector": None,
    "pending_gesture": None,
    "gesture_counter": 0,
    "last_gesture": None,
    "last_detection_time": 0.0,
    "total_detections": 0,
    "initialized": False,
    # ── Theme ──
    "theme": "dark",
    # ── Sidebar Toggle ──
    "sidebar_visible": True,
    # ── Learning Mode ──
    "learning_mode": False,
    "recording": False,
    "recorded_landmarks": [],
    "ml_model": None,
    "use_ml_model": False,
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# Welcome message (shown only once)
if not st.session_state.initialized:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to AI Sign Language Translator!**\n\n"
                "I can recognize **hand gestures** and translate them into "
                "English, Tamil, and Hindi.\n\n"
                "Click **▶️ Start Camera** to begin!"
            ),
        }
    )
    st.session_state.initialized = True

# Try to load ML model on first run if it exists
if st.session_state.ml_model is None and MODEL_PATH.exists():
    st.session_state.ml_model = load_model()


# ═══════════════════════════════════════════════════════════════════════
#  THEME-AWARE CSS
# ═══════════════════════════════════════════════════════════════════════

def _get_theme_css(theme: str) -> str:
    """Return the full CSS string for the given theme."""
    if theme == "light":
        return """
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --accent: #a855f7;
    --bg-dark: #f0f0f8;
    --bg-card: rgba(255, 255, 255, 0.65);
    --border: rgba(100, 100, 180, 0.15);
    --text: #1e1b4b;
    --text-dim: #64748b;
    --success: #16a34a;
    --chat-ai-bg: rgba(99, 102, 241, 0.08);
    --chat-ai-border: rgba(99, 102, 241, 0.18);
    --chat-user-bg: rgba(168, 85, 247, 0.08);
    --chat-user-border: rgba(168, 85, 247, 0.18);
    --sidebar-bg-top: #e8e6f5;
    --sidebar-bg-bottom: #f0f0f8;
    --glass-bg: rgba(255,255,255,0.55);
    --glass-border: rgba(100,100,180,0.12);
    --app-bg: linear-gradient(135deg, #f0f0f8 0%, #e8e6f5 40%, #ddd8f0 70%, #f0f0f8 100%);
}
"""
    else:  # dark (default)
        return """
:root {
    --primary: #6366f1;
    --primary-light: #818cf8;
    --accent: #a855f7;
    --bg-dark: #0f0c29;
    --bg-card: rgba(255, 255, 255, 0.04);
    --border: rgba(255, 255, 255, 0.08);
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --success: #22c55e;
    --chat-ai-bg: rgba(99, 102, 241, 0.06);
    --chat-ai-border: rgba(99, 102, 241, 0.12);
    --chat-user-bg: rgba(168, 85, 247, 0.06);
    --chat-user-border: rgba(168, 85, 247, 0.12);
    --sidebar-bg-top: #1e1b4b;
    --sidebar-bg-bottom: #0f0c29;
    --glass-bg: rgba(255,255,255,0.04);
    --glass-border: rgba(255,255,255,0.08);
    --app-bg: linear-gradient(135deg, #0f0c29 0%, #1e1b4b 40%, #312e81 70%, #0f0c29 100%);
}
"""


# Build the complete CSS
_THEME_VARS = _get_theme_css(st.session_state.theme)

st.markdown(
    f"""
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Theme Variables ────────────────────────────────────────────── */
{_THEME_VARS}

/* ── App Background ──────────────────────────────────────────────── */
.stApp {{
    background: var(--app-bg) !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── Animated Gradient Title ─────────────────────────────────────── */
.gradient-title {{
    font-family: 'Inter', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(120deg, #c084fc, #818cf8, #60a5fa, #c084fc);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-flow 4s ease-in-out infinite;
    text-align: center;
    margin-bottom: 0 !important;
}}
@keyframes gradient-flow {{
    0%, 100% {{ background-position: 0% 50%; }}
    50%      {{ background-position: 100% 50%; }}
}}

.subtitle {{
    text-align: center;
    color: var(--text-dim) !important;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
    font-weight: 300;
}}

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {{
    color: var(--text) !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}}

/* ── Chat Bubble Animations ─────────────────────────────────────── */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(12px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* ── Chat Messages — Glassmorphism Bubbles ───────────────────────── */
[data-testid="stChatMessage"] {{
    background: var(--glass-bg) !important;
    backdrop-filter: blur(14px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(14px) saturate(1.4) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    padding: 1rem 1.3rem !important;
    margin-bottom: 0.7rem !important;
    animation: fadeInUp 0.4s ease-out both;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}}
[data-testid="stChatMessage"]:hover {{
    background: rgba(99, 102, 241, 0.10) !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.12);
    transform: translateY(-1px);
}}

/* ── Stagger chat bubble animations ─────────────────────────────── */
[data-testid="stChatMessage"]:nth-child(1) {{ animation-delay: 0.00s; }}
[data-testid="stChatMessage"]:nth-child(2) {{ animation-delay: 0.05s; }}
[data-testid="stChatMessage"]:nth-child(3) {{ animation-delay: 0.10s; }}
[data-testid="stChatMessage"]:nth-child(4) {{ animation-delay: 0.15s; }}
[data-testid="stChatMessage"]:nth-child(5) {{ animation-delay: 0.20s; }}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}}
.stButton > button:disabled {{
    opacity: 0.35 !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Camera Feed ─────────────────────────────────────────────────── */
[data-testid="stImage"] {{
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--sidebar-bg-top), var(--sidebar-bg-bottom)) !important;
    border-right: 1px solid var(--border) !important;
    transition: margin-left 0.3s ease, opacity 0.3s ease !important;
}}

/* ── Alerts ──────────────────────────────────────────────────────── */
.stAlert {{ border-radius: 12px !important; }}

/* ── Custom Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.1); border-radius: 3px; }}
::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.3); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(99,102,241,0.5); }}

/* ── Hide Streamlit Chrome ───────────────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}

/* ── Gesture Guide Cards ─────────────────────────────────────────── */
.gesture-item {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem;
    text-align: center;
    margin-bottom: 0.4rem;
    transition: all 0.3s ease;
}}
.gesture-item:hover {{
    border-color: var(--primary);
    background: rgba(99,102,241,0.08);
}}

/* ── Pulse for live indicator ────────────────────────────────────── */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.4; }}
}}
.live-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s ease-in-out infinite;
}}

/* ── Recording indicator ─────────────────────────────────────────── */
@keyframes rec-pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 8px rgba(239,68,68,0.5); }}
    50%      {{ opacity: 0.6; box-shadow: 0 0 16px rgba(239,68,68,0.8); }}
}}
.rec-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #ef4444;
    animation: rec-pulse 1.2s ease-in-out infinite;
}}

/* ── Glass Card Utility ──────────────────────────────────────────── */
.glass-card {{
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.8rem;
}}

/* ── Sidebar Toggle Button ───────────────────────────────────────── */
.sidebar-toggle {{
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 999999;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
    border: none;
    border-radius: 12px;
    width: 40px;
    height: 40px;
    font-size: 1.2rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
    transition: all 0.3s ease;
}}
.sidebar-toggle:hover {{
    transform: scale(1.1);
    box-shadow: 0 6px 24px rgba(99,102,241,0.6);
}}

/* ── Stat Card ───────────────────────────────────────────────────── */
.stat-card {{
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    text-align: center;
}}
.stat-card .stat-value {{
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.stat-card .stat-label {{
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-top: 2px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar visibility CSS injection ────────────────────────────────
if not st.session_state.sidebar_visible:
    st.markdown(
        """<style>
        [data-testid="stSidebar"] { display: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════

def _start_camera():
    """Open webcam and initialise the gesture detector."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "❌ **Camera Error** – could not open webcam. "
                "Make sure no other app is using it.",
            }
        )
        return

    st.session_state.cap = cap
    st.session_state.detector = GestureDetector()
    st.session_state.running = True
    st.session_state.gesture_counter = 0
    st.session_state.pending_gesture = None
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                '<span class="live-dot"></span>'
                "**Camera started!** Show me a hand gesture…"
            ),
        }
    )


def _stop_camera():
    """Release webcam and detector resources."""
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    if st.session_state.detector is not None:
        st.session_state.detector.release()
        st.session_state.detector = None
    st.session_state.running = False
    st.session_state.gesture_counter = 0
    st.session_state.pending_gesture = None
    st.session_state.recording = False  # stop recording if active
    st.session_state.messages.append(
        {"role": "assistant", "content": "⏹️ Camera stopped."}
    )


def _clear_chat():
    """Wipe chat history."""
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "🗑️ Chat cleared. Click **▶️ Start Camera** to begin!",
        }
    ]


# ═══════════════════════════════════════════════════════════════════════
#  FRAME ANNOTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _overlay_hud(
    frame: np.ndarray,
    gesture: str | None,
    counter: int,
    threshold: int,
    recently_detected: bool,
    *,
    is_recording: bool = False,
    ml_confidence: float | None = None,
) -> np.ndarray:
    """Draw a translucent HUD overlay on the camera frame."""
    h, w = frame.shape[:2]

    # ── Top bar background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (10, 8, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if gesture:
        emoji = GESTURE_EMOJIS.get(gesture, "")
        label = f"{emoji}  {gesture}"
        if ml_confidence is not None:
            label += f"  ({ml_confidence:.0%})"
        cv2.putText(
            frame, label, (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (130, 255, 200), 2, cv2.LINE_AA,
        )
        # Progress bar
        progress = min(counter / max(threshold, 1), 1.0)
        bar_w = int(w * 0.55)
        bar_x = (w - bar_w) // 2
        cv2.rectangle(frame, (bar_x, 52), (bar_x + bar_w, 68), (40, 40, 50), -1)
        fill_color = (0, 220, 120) if progress < 1.0 else (0, 215, 255)
        cv2.rectangle(
            frame, (bar_x, 52), (bar_x + int(bar_w * progress), 68),
            fill_color, -1,
        )
        cv2.rectangle(frame, (bar_x, 52), (bar_x + bar_w, 68), (80, 80, 100), 2)
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(
            frame, pct_text, (bar_x + bar_w + 10, 66),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 220), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "Show a hand gesture...", (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 170), 2, cv2.LINE_AA,
        )

    # Green flash border on recent detection
    if recently_detected:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 130), 4)

    # ── Recording indicator (red dot + text in top-right) ──
    if is_recording:
        rec_text = "REC"
        cv2.circle(frame, (w - 70, 24), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, rec_text, (w - 55, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return frame


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Theme Toggle ──
    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button(
            "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark",
            use_container_width=True,
            key="theme_toggle",
        ):
            st.session_state.theme = (
                "light" if st.session_state.theme == "dark" else "dark"
            )
            st.rerun()
    with theme_col2:
        if st.button(
            "📌 Hide Sidebar",
            use_container_width=True,
            key="hide_sidebar_btn",
        ):
            st.session_state.sidebar_visible = False
            st.rerun()

    st.divider()

    voice_enabled = st.toggle("🔊 Voice Output", value=True)
    sensitivity = st.select_slider(
        "🎚️ Detection Sensitivity",
        options=["Low", "Medium", "High"],
        value="Medium",
        help="Higher = fewer confirmation frames needed.",
    )
    cooldown_sec = st.slider(
        "⏱️ Cooldown (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        help="Minimum gap between two detections.",
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        _clear_chat()

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    #  LEARNING MODE (inside sidebar)
    # ═══════════════════════════════════════════════════════════════════

    st.markdown("### 🧠 Learning Mode")

    learn_mode = st.toggle("Enable Learning Mode", key="learn_toggle")
    st.session_state.learning_mode = learn_mode

    if learn_mode:
        new_label = st.text_input(
            "Enter gesture name:",
            placeholder="e.g. Peace, Love, OK",
            key="gesture_label_input",
        )

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            if st.button(
                "🔴 Start Rec",
                disabled=st.session_state.recording or not new_label,
                use_container_width=True,
                key="start_rec_btn",
            ):
                st.session_state.recording = True
                st.session_state.recorded_landmarks = []
        with rec_col2:
            if st.button(
                "⏹ Stop & Save",
                disabled=not st.session_state.recording,
                use_container_width=True,
                key="stop_rec_btn",
            ):
                st.session_state.recording = False
                n_frames = len(st.session_state.recorded_landmarks)
                if n_frames > 0:
                    total = save_landmarks(
                        new_label, st.session_state.recorded_landmarks
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"💾 **Saved {n_frames} frames** for gesture "
                                f"**\"{new_label}\"**.\n\n"
                                f"Total dataset: {total} samples."
                            ),
                        }
                    )
                    st.session_state.recorded_landmarks = []
                else:
                    st.warning("No frames recorded. Show your hand to the camera.")

        # Recording status
        if st.session_state.recording:
            n = len(st.session_state.recorded_landmarks)
            st.markdown(
                f'<div class="rec-badge">● Recording — {n} frames</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Press **Start Rec** then show your gesture to the camera.")

        st.divider()

        # ── Train Model ──
        st.markdown("#### 🧪 Train Model")
        ds_info = get_dataset_info()
        if ds_info["total_rows"] > 0:
            st.caption(
                f"**{ds_info['total_rows']}** samples · "
                f"**{len(ds_info['labels'])}** classes: "
                + ", ".join(ds_info["labels"].keys())
            )
        else:
            st.caption("No data yet. Record some gestures first.")

        can_train = ds_info["total_rows"] >= 30 and len(ds_info["labels"]) >= 2
        if st.button(
            "🧠 Train Model",
            disabled=not can_train,
            use_container_width=True,
            key="train_btn",
        ):
            with st.spinner("Training model…"):
                result = train_model()
            st.session_state.ml_model = load_model()
            st.session_state.use_ml_model = True
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"🎓 **Model trained!**\n\n"
                        f"- Accuracy: **{result['accuracy']:.1%}**\n"
                        f"- Samples: **{result['n_samples']}**\n"
                        f"- Gestures: {', '.join(result['labels'])}\n\n"
                        f"ML-based detection is now **active**."
                    ),
                }
            )
            st.rerun()

        if not can_train and ds_info["total_rows"] > 0:
            st.info("Need ≥ 30 samples and ≥ 2 gesture classes to train.")

        # ── Use ML Model Toggle ──
        if st.session_state.ml_model is not None:
            st.session_state.use_ml_model = st.toggle(
                "Use ML Model for Detection",
                value=st.session_state.use_ml_model,
                key="use_ml_toggle",
            )
        st.divider()

    # ── Gesture Guide ──
    st.markdown("### 🤲 Gesture Guide")
    _guide = [
        ("✋", "Hello", "All fingers open"),
        ("👍", "Yes", "Only thumb up"),
        ("☝️", "No", "Only index finger"),
        ("🤙", "Thank You", "Thumb + pinky"),
        ("✊", "Stop", "Closed fist"),
    ]
    for emoji, name, desc in _guide:
        st.markdown(
            f'<div class="gesture-item"><b>{emoji} {name}</b><br>'
            f'<small style="color:var(--text-dim)">{desc}</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Built with ❤️ using MediaPipe + Streamlit")

# Map sensitivity label → frame threshold
_SENSITIVITY_MAP = {"Low": 25, "Medium": 15, "High": 8}
CONFIRM_FRAMES = _SENSITIVITY_MAP[sensitivity]


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR TOGGLE BUTTON (shown in main area when sidebar is hidden)
# ═══════════════════════════════════════════════════════════════════════

if not st.session_state.sidebar_visible:
    if st.button("☰ Show Sidebar", key="show_sidebar_btn"):
        st.session_state.sidebar_visible = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    '<h1 class="gradient-title">🤟 AI Sign Language Translator</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Real-time hand gesture recognition · '
    "Multi-language translation · Voice output</p>",
    unsafe_allow_html=True,
)

col_cam, col_chat = st.columns([1, 1], gap="large")

# ── Camera Column ────────────────────────────────────────────────────
with col_cam:
    st.markdown('<p class="section-header">📷 Camera Feed</p>', unsafe_allow_html=True)

    btn1, btn2 = st.columns(2)
    with btn1:
        st.button(
            "▶️ Start Camera",
            on_click=_start_camera,
            disabled=st.session_state.running,
            use_container_width=True,
        )
    with btn2:
        st.button(
            "⏹️ Stop Camera",
            on_click=_stop_camera,
            disabled=not st.session_state.running,
            use_container_width=True,
        )

    video_placeholder = st.empty()
    status_placeholder = st.empty()

    if not st.session_state.running:
        video_placeholder.info("📷 Camera is off. Click **Start Camera** to begin.")

# ── Chat Column ──────────────────────────────────────────────────────
with col_chat:
    st.markdown(
        '<p class="section-header">💬 Conversation</p>',
        unsafe_allow_html=True,
    )
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else "🧑"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"], unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────────────
if st.session_state.total_detections > 0:
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{st.session_state.total_detections}</div>'
            f'<div class="stat-label">🔍 Total Detections</div></div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{st.session_state.last_gesture or "—"}</div>'
            f'<div class="stat-label">🤚 Last Gesture</div></div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{"On" if voice_enabled else "Off"}</div>'
            f'<div class="stat-label">🔊 Voice Output</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════
#  CAMERA PROCESSING LOOP  (smooth streaming — no st.rerun flicker)
# ═══════════════════════════════════════════════════════════════════════
#
#  Instead of reading one frame and calling st.rerun(), we loop
#  continuously inside this single script run, updating only the
#  st.empty() placeholders.  This eliminates blinking entirely.
#  The loop breaks when:
#    • The user clicks "Stop Camera" (sets running = False)
#    • The camera fails to read a frame
#

if st.session_state.running and st.session_state.cap is not None:
    cap: cv2.VideoCapture = st.session_state.cap
    detector: GestureDetector = st.session_state.detector
    ml_model = st.session_state.ml_model if st.session_state.use_ml_model else None

    # Smooth streaming loop — updates placeholders only, no full rerun
    while st.session_state.running:
        ret, raw_frame = cap.read()

        if not ret:
            video_placeholder.warning("⚠️ Could not read from camera.")
            break

        # ── Choose detection mode (ML vs rule-based) ─────────────
        ml_confidence = None
        if ml_model is not None:
            processed, gesture, ml_confidence = detector.process_frame_ml(
                raw_frame, ml_model
            )
        else:
            processed, gesture = detector.process_frame(raw_frame)

        now = time.time()
        recently_detected = now - st.session_state.last_detection_time < 0.6

        # ── Learning Mode: record landmarks if recording ─────────
        if st.session_state.recording and st.session_state.learning_mode:
            _, landmarks = detector.get_raw_landmarks(raw_frame)
            if landmarks is not None:
                features = landmarks_to_features(landmarks)
                st.session_state.recorded_landmarks.append(features)

        # ── Confirmation logic ───────────────────────────────────
        if gesture:
            if gesture == st.session_state.pending_gesture:
                st.session_state.gesture_counter += 1
            else:
                st.session_state.pending_gesture = gesture
                st.session_state.gesture_counter = 1

            confirmed = (
                st.session_state.gesture_counter >= CONFIRM_FRAMES
                and (now - st.session_state.last_detection_time) > cooldown_sec
            )

            if confirmed:
                # Reset
                st.session_state.last_gesture = gesture
                st.session_state.last_detection_time = now
                st.session_state.gesture_counter = 0
                st.session_state.total_detections += 1

                # Translations
                translations = get_all_translations(gesture)
                emoji = GESTURE_EMOJIS.get(gesture, "✋")

                lines = [f"**{emoji} Detected Sign → {gesture}**\n"]
                for code, info in LANGUAGES.items():
                    lines.append(
                        f"{info['flag']} **{info['name']}:** {translations[code]}"
                    )
                msg_text = "\n".join(lines)

                st.session_state.messages.append(
                    {"role": "assistant", "content": msg_text}
                )

                # Voice output
                if voice_enabled:
                    speak(translations["en"])

                status_placeholder.success(f"✅ Detected: **{gesture}**")
        else:
            st.session_state.gesture_counter = 0
            st.session_state.pending_gesture = None

        # ── Render annotated frame ───────────────────────────────
        annotated = _overlay_hud(
            processed,
            gesture,
            st.session_state.gesture_counter,
            CONFIRM_FRAMES,
            recently_detected,
            is_recording=st.session_state.recording,
            ml_confidence=ml_confidence,
        )
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, use_container_width=True)

        if not recently_detected and gesture:
            status_placeholder.info(
                f"👀 Detecting **{gesture}** — hold still… "
                f"({st.session_state.gesture_counter}/{CONFIRM_FRAMES})"
            )
        elif not gesture:
            status_placeholder.info("👀 Waiting for a hand gesture…")

        # ── Frame rate cap (~30 FPS) — prevents CPU hogging ──────
        time.sleep(0.033)
