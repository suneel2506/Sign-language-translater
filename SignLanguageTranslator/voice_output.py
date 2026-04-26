"""
Voice Output Module
===================
Provides text-to-speech using pyttsx3 (fully offline).
Speech runs in a background thread so it never blocks the UI.
"""

import threading

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False


def speak(text: str, rate: int = 150, volume: float = 0.9) -> None:
    """
    Speak *text* in a background daemon thread.

    Parameters
    ----------
    text : str
        The text to speak aloud.
    rate : int
        Words per minute (default 150).
    volume : float
        Volume level 0.0 – 1.0 (default 0.9).
    """
    if not _TTS_AVAILABLE:
        return

    def _worker():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:          # noqa: BLE001
            # Silently ignore TTS errors so the main app keeps running
            print(f"[TTS] Could not speak: {exc}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
