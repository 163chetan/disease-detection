"""
Gemini AI Health Advisor
========================
Generates personalised diet, lifestyle, precaution and follow-up guidance after a
disease prediction, using Google's Gemini API (google-genai SDK).

Get a free API key at https://aistudio.google.com/apikey
Set it as the GEMINI_API_KEY environment variable, in .streamlit/secrets.toml,
or paste it into the sidebar while the app is running.
"""
import os

# Tried in order. Override the first choice with the GEMINI_MODEL env var / secret.
_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


def _build_prompt(disease_name: str, is_positive: bool, patient_summary: str) -> str:
    if is_positive:
        result_context = f"**{disease_name} HAS BEEN DETECTED** by the screening model."
        advice_focus = f"Since {disease_name} was flagged, give clear, practical, non-alarmist guidance."
        heading = "⚠️ Understanding Your Result"
    else:
        result_context = f"**{disease_name} was NOT detected** — the screening result is negative."
        advice_focus = f"Since no {disease_name} was flagged, focus on preventive care."
        heading = "✅ Your Result Looks Good — Keep It Up!"

    return f"""You are a knowledgeable and empathetic AI health advisor supporting a clinician-facing screening tool.

**Prediction Result:** {result_context}
**Patient Key Measurements:** {patient_summary}
**Context:** {advice_focus}

Respond ONLY with the sections below, in markdown. Be specific, practical and concise.
Remember this is a statistical screening result, not a diagnosis.

## {heading}
_(2 sentences explaining what this result means and its limits)_

## 🥗 Diet & Nutrition
**✅ Foods to Include:**
- _(5 specific beneficial foods)_

**❌ Foods to Limit or Avoid:**
- _(5 foods to limit or avoid)_

## 🏃 Exercise & Lifestyle
- _(3–4 specific, actionable recommendations)_

## 💊 Precautions & Warning Signs
- _(4–5 key precautions, including warning signs that need urgent medical attention)_

## 🩺 Medical Follow-up
- _(Recommended specialist, confirmatory tests, and monitoring frequency)_

## 🧘 Mental Wellness & Sleep
- _(2–3 tips on stress, sleep and emotional well-being)_

---
> ⚠️ *AI-generated guidance for educational purposes only. It does not replace a certified medical professional.*

Keep the total response under 450 words. Use bullet points, not long paragraphs."""


def _candidates(client, preferred: str | None):
    """Preferred model first, then built-ins, then any live 'flash' model the key can use."""
    seen, order = set(), []
    for m in ([preferred] if preferred else []) + _MODEL_CANDIDATES:
        if m and m not in seen:
            seen.add(m)
            order.append(m)
    return order


def _live_flash_models(client):
    try:
        names = []
        for m in client.models.list():
            n = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in n and "generateContent" in actions and "preview" not in n:
                names.append(n)
        return names
    except Exception:
        return []


def _friendly_error(msg: str, model_id: str) -> str | None:
    low = msg.lower()
    if "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low \
            or "401" in low or "403" in low:
        return "❌ **Invalid or unauthorised Gemini API key.** Please check the key and try again."
    if "429" in low or "resource_exhausted" in low or "quota" in low:
        return "⏳ **Gemini rate limit / quota reached.** Please wait a minute and try again."
    if any(k in low for k in ("timeout", "timed out", "connect", "network", "name or service", "getaddrinfo", "errno")):
        return "🌐 **Could not reach the Gemini API.** Check your internet connection and try again."
    return None


def get_ai_advice(disease_name: str, is_positive: bool, patient_summary: str,
                  api_key: str, model: str | None = None) -> str:
    """
    Returns markdown advice from Gemini, or a short human-readable error message.
    Never raises, so the prediction page always keeps working.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ("⚠️ **The `google-genai` package is not installed.**\n\n"
                "Run `pip install google-genai` and restart the app.")

    if not api_key:
        return "⚠️ No Gemini API key configured."

    prompt = _build_prompt(disease_name, is_positive, patient_summary)
    preferred = model or os.environ.get("GEMINI_MODEL")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:  # pragma: no cover
        return f"❌ **Could not initialise Gemini client:** {e}"

    tried, last_err = [], ""
    queue = _candidates(client, preferred)
    refreshed = False
    while queue:
        model_id = queue.pop(0)
        tried.append(model_id)
        # 1st try: no 'thinking' tokens (faster, cheaper). 2nd: plain config for models that reject it.
        for use_thinking_cfg in (True, False):
            try:
                cfg = dict(temperature=0.6, max_output_tokens=2048)
                if use_thinking_cfg:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                resp = client.models.generate_content(
                    model=model_id, contents=prompt,
                    config=types.GenerateContentConfig(**cfg))
                text = (getattr(resp, "text", None) or "").strip()
                if text:
                    return text
                return ("⚠️ Gemini returned no text for this request (it may have been blocked by a "
                        "safety filter). Please try again.")
            except Exception as e:
                last_err = str(e)
                low = last_err.lower()
                if use_thinking_cfg and ("thinking" in low or "invalid_argument" in low):
                    continue  # retry same model without thinking config
                friendly = _friendly_error(last_err, model_id)
                if friendly:
                    return friendly
                break  # model not found / unsupported → try next candidate
        if not queue and not refreshed:
            refreshed = True
            queue = [m for m in _live_flash_models(client) if m not in tried][:3]

    return f"❌ **Gemini API error** (tried: {', '.join(tried)}): {last_err[:300]}"
