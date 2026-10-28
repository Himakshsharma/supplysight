"""Optional Gemini client (REST). The copilot works fully offline when no key is set."""

from __future__ import annotations

import requests

from supplysight.config import get_settings
from supplysight.utils import get_logger

log = get_logger(__name__)
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def llm_available() -> bool:
    """True when a Gemini API key is configured."""
    return bool(get_settings().gemini_api_key)


def generate(prompt: str, temperature: float = 0.1, timeout: float = 20.0) -> str | None:
    """Call Gemini; returns text or None on any failure (callers fall back to offline mode)."""
    s = get_settings()
    if not s.gemini_api_key:
        return None
    try:  # pragma: no cover - network
        r = requests.post(
            _URL.format(model=s.gemini_model),
            params={"key": s.gemini_api_key},
            timeout=timeout,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 800},
            },
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:  # pragma: no cover
        log.warning("Gemini call failed, falling back to offline mode: %s", exc)
        return None
