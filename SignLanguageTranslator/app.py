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
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
/* ── Google Font ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Design Tokens ───────────────────────────────────────────────── */
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
}

/* ── App Background ──────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #1e1b4b 40%, #312e81 70%, var(--bg-dark) 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Animated Gradient Title ─────────────────────────────────────── */
.gradient-title {
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
}
@keyframes gradient-flow {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

.subtitle {
    text-align: center;
    color: var(--text-dim) !important;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
    font-weight: 300;
}

/* ── Section Headers ─────────────────────────────────────────────── */
.section-header {
    color: var(--text) !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ── Chat Messages ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: rgba(99, 102, 241, 0.06) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-radius: 16px !important;
    padding: 0.8rem 1.2rem !important;
    margin-bottom: 0.6rem !important;
    transition: all 0.3s ease;
}
[data-testid="stChatMessage"]:hover {
    background: rgba(99, 102, 241, 0.10) !important;
    border-color: rgba(99, 102, 241, 0.22) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
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
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}
.stButton > button:disabled {
    opacity: 0.35 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Camera Feed ─────────────────────────────────────────────────── */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #0f0c29) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────── */
.stAlert { border-radius: 12px !important; }

/* ── Custom Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }

/* ── Hide Streamlit Chrome ───────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Gesture Guide Cards ─────────────────────────────────────────── */
.gesture-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem;
    text-align: center;
    margin-bottom: 0.4rem;
    transition: all 0.3s ease;
}
.gesture-item:hover {
    border-color: var(--primary);
    background: rgba(99,102,241,0.08);
}

/* ── Pulse for live indicator ────────────────────────────────────── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
.live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s ease-in-out infinite;
}
</style>
""",
    unsafe_allow_html=True,
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
                "I can recognize **5 hand gestures** and translate them into "
                "English, Tamil, and Hindi.\n\n"
                "Click **▶️ Start Camera** to begin!"
            ),
        }
    )
    st.session_state.initialized = True


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

    return frame


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Settings")
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
            f'<small style="color:#94a3b8">{desc}</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Built with ❤️ using MediaPipe + Streamlit")

# Map sensitivity label → frame threshold
_SENSITIVITY_MAP = {"Low": 25, "Medium": 15, "High": 8}
CONFIRM_FRAMES = _SENSITIVITY_MAP[sensitivity]


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
            with st.chat_message(msg["role"], avatar="🤖"):
                st.markdown(msg["content"], unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────────────
if st.session_state.total_detections > 0:
    m1, m2, m3 = st.columns(3)
    m1.metric("🔍 Total Detections", st.session_state.total_detections)
    m2.metric("🤚 Last Gesture", st.session_state.last_gesture or "—")
    m3.metric("🔊 Voice", "On" if voice_enabled else "Off")


# ═══════════════════════════════════════════════════════════════════════
#  CAMERA PROCESSING LOOP  (one frame per Streamlit rerun)
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.running and st.session_state.cap is not None:
    cap: cv2.VideoCapture = st.session_state.cap
    detector: GestureDetector = st.session_state.detector

    ret, raw_frame = cap.read()

    if ret:
        processed, gesture = detector.process_frame(raw_frame)
        now = time.time()
        recently_detected = now - st.session_state.last_detection_time < 0.6

        # ── Confirmation logic ───────────────────────────────────────
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

        # ── Render annotated frame ───────────────────────────────────
        annotated = _overlay_hud(
            processed,
            gesture,
            st.session_state.gesture_counter,
            CONFIRM_FRAMES,
            recently_detected,
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

    else:
        video_placeholder.warning("⚠️ Could not read from camera.")

    # Small delay then re-run for the next frame
    time.sleep(0.05)
    st.rerun()
