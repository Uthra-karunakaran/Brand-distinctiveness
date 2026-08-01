"""
Tone alignment — the one place an LLM judge beats vector math.

Embeddings encode topic far more strongly than tone: "we help teams ship faster"
and "we empower organisations to accelerate delivery" sit close in vector space
while being tonally opposite. So tone gets a judge, not a cosine.

Design notes that matter:
  - The judge NEVER sees the score it is meant to produce. It rates the input on
    named axes, and we compute the distance to the brand's own axis profile in
    Python. That keeps the number reproducible and auditable.
  - The brand's tone profile is itself judged once, at fingerprint build time,
    and cached — not re-derived per request.
  - If no API key is present the scorer degrades to a lexical heuristic so the
    demo never hard-fails on stage.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

AXES = ["formal_casual", "serious_playful", "corporate_human",
        "restrained_bold", "technical_accessible"]

JUDGE_SYSTEM = """You are a brand-voice rating instrument. You do not write copy \
and you do not give opinions.

Rate the SAMPLE on each axis from 0 to 10:
- formal_casual: 0 = highly formal, 10 = highly casual
- serious_playful: 0 = entirely serious, 10 = openly playful
- corporate_human: 0 = corporate/institutional, 10 = personal/human
- restrained_bold: 0 = understated, 10 = bold and declarative
- technical_accessible: 0 = dense and technical, 10 = plain and accessible

Return ONLY a JSON object mapping each axis name to an integer. No prose, no \
markdown fences."""


@dataclass
class ToneScores:
    alignment: float                 # 0-100 -> consistency signal
    input_profile: dict[str, float]
    brand_profile: dict[str, float]
    biggest_gap: str
    judge: str                       # "llm" | "heuristic"


# ---------- heuristic fallback ----------

_CONTRACTIONS = re.compile(r"\b\w+'(?:s|re|ll|ve|t|d|m)\b", re.I)
_CORPORATE = {"leverage", "solution", "solutions", "enterprise", "seamless",
              "robust", "scalable", "innovative", "empower", "optimize",
              "streamline", "synergy", "holistic", "cutting-edge", "best-in-class"}
_LONG_WORD = re.compile(r"\b\w{10,}\b")


def _heuristic_profile(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z'\-]+", text)
    n = max(len(words), 1)
    contractions = len(_CONTRACTIONS.findall(text)) / n
    corporate = sum(1 for w in words if w.lower() in _CORPORATE) / n
    excl = text.count("!") / n
    # "our" is what every corporation says; "you" is what human copy says.
    second_person = sum(1 for w in words if w.lower() in {"you", "your"}) / n
    first_plural = sum(1 for w in words if w.lower() in {"we", "our", "us"}) / n
    long_words = len(_LONG_WORD.findall(text)) / n
    sents = max(len(re.findall(r"[.!?]", text)), 1)
    avg_len = n / sents

    clamp = lambda x: float(max(0.0, min(10.0, x)))
    return {
        "formal_casual": clamp(contractions * 90 + excl * 60 + 3 - avg_len / 8),
        "serious_playful": clamp(excl * 120 + contractions * 50),
        "corporate_human": clamp(6 - corporate * 110 + second_person * 40
                                 - first_plural * 20 - long_words * 35
                                 + contractions * 40),
        "restrained_bold": clamp(excl * 90 + (10 - avg_len / 2.5)),
        "technical_accessible": clamp(10 - long_words * 80 - max(0.0, avg_len - 12) / 2),
    }


# ---------- LLM judge ----------

def _llm_profile(text: str, model: str = "claude-sonnet-4-6") -> dict[str, float] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"SAMPLE:\n{text}"}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {a: float(data[a]) for a in AXES}
    except Exception:
        return None


def profile(text: str) -> tuple[dict[str, float], str]:
    p = _llm_profile(text)
    if p is not None:
        return p, "llm"
    return _heuristic_profile(text), "heuristic"


class ToneScorer:
    def __init__(self, brand_voice_exemplars: list[str]):
        profiles, judges = [], set()
        for ex in brand_voice_exemplars:
            p, j = profile(ex)
            profiles.append(p)
            judges.add(j)
        self.brand_profile = {
            a: sum(p[a] for p in profiles) / len(profiles) for a in AXES
        }
        self.judge = "llm" if "llm" in judges else "heuristic"

    def score(self, text: str) -> ToneScores:
        p, judge = profile(text)
        gaps = {a: abs(p[a] - self.brand_profile[a]) for a in AXES}
        mean_gap = sum(gaps.values()) / len(gaps)
        return ToneScores(
            alignment=round(max(0.0, 1 - mean_gap / 4.5) * 100, 1),
            input_profile={k: round(v, 1) for k, v in p.items()},
            brand_profile={k: round(v, 1) for k, v in self.brand_profile.items()},
            biggest_gap=max(gaps, key=gaps.get),
            judge=judge,
        )
