"""
Translation Module
==================
Provides instant translations for detected gestures.
Uses a hardcoded dictionary for reliability (no API dependency).
"""

# ── Language Metadata ────────────────────────────────────────────────
LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧"},
    "ta": {"name": "Tamil",   "flag": "🇮🇳"},
    "hi": {"name": "Hindi",   "flag": "🇮🇳"},
}

# ── Gesture → Emoji Mapping ─────────────────────────────────────────
GESTURE_EMOJIS = {
    "Hello":     "✋",
    "Yes":       "👍",
    "No":        "☝️",
    "Thank You": "🤙",
    "Stop":      "✊",
}

# ── Translation Dictionary ──────────────────────────────────────────
TRANSLATIONS: dict[str, dict[str, str]] = {
    "Hello": {
        "en": "Hello",
        "ta": "வணக்கம்",
        "hi": "नमस्ते",
    },
    "Yes": {
        "en": "Yes",
        "ta": "ஆம்",
        "hi": "हाँ",
    },
    "No": {
        "en": "No",
        "ta": "இல்லை",
        "hi": "नहीं",
    },
    "Thank You": {
        "en": "Thank You",
        "ta": "நன்றி",
        "hi": "धन्यवाद",
    },
    "Stop": {
        "en": "Stop",
        "ta": "நிறுத்து",
        "hi": "रुको",
    },
}


def translate(text: str, target_lang: str = "en") -> str:
    """Translate a detected gesture into the target language."""
    if text in TRANSLATIONS:
        return TRANSLATIONS[text].get(target_lang, text)
    return text


def get_all_translations(text: str) -> dict[str, str]:
    """Return translations for every supported language."""
    if text in TRANSLATIONS:
        return TRANSLATIONS[text]
    return {code: text for code in LANGUAGES}
