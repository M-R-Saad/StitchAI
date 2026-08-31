"""
Decoupled explanation layer (whitepaper Section 4.4). Detection must keep working even
if this module / its API is unavailable (Section 3.8) - that's why it lives in its own
folder and is only ever called AFTER a verdict is already computed by the backbone.

Uses the current `google-genai` SDK (the older `google-generativeai` package is
deprecated / end-of-life) and `gemini-2.5-flash` (the current, actively-served flash
model - `gemini-1.5-flash` has been phased out).
"""
import os
from pathlib import Path
from typing import Optional

from explanation.prompt_templates import build_explanation_prompt

CACHE_DIR = Path(__file__).parent / "cached_explanations"

_client = None


def _get_client():
    """Lazily create the Gemini client so import of this module never requires an API
    key to be present (e.g. during tests, or before Phase 2 setup is done)."""
    global _client
    if _client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment/.env")
        _client = genai.Client(api_key=api_key)
    return _client


def explain_anomaly(
    category: str,
    score: float,
    region_description: str = "a region flagged by the anomaly heatmap",
    image_id: Optional[str] = None,
) -> str:
    """
    Return a plain-language explanation for a flagged anomaly.

    Tries the live VLM API first; falls back to a cached explanation (by image_id) if
    the API call fails, per the demo-safety mitigation in whitepaper Section 3.8.
    """
    prompt = build_explanation_prompt(category, score, region_description)

    try:
        return _call_gemini(prompt)
    except Exception:
        cached = _load_cached_fallback(image_id)
        if cached:
            return cached
        return (
            f"[Explanation unavailable] The image was flagged as anomalous in the "
            f"'{category}' category with confidence {score:.2f}. Automated explanation "
            f"service is currently unreachable - please review the heatmap directly."
        )


def _call_gemini(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        #model="gemini-2.5-flash",
        model="gemini-3.6-flash",
        contents=prompt,
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


def _load_cached_fallback(image_id: Optional[str]) -> Optional[str]:
    """Load a pre-generated fallback explanation for a known demo image, if one exists."""
    if not image_id:
        return None
    candidate = CACHE_DIR / f"{image_id}.txt"
    if candidate.exists():
        return candidate.read_text().strip()
    return None
