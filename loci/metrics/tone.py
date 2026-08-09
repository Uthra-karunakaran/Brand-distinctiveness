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
  - Per-request judging is the hot path (every POST /score can trigger a live
    call), so results are cached by exact input text, and a process-wide daily
    call budget (ANTHROPIC_CALL_BUDGET, default 200) caps worst-case spend on
    an unauthenticated public demo — once hit, requests fall back to the
    heuristic until the budget resets at UTC midnight. `usage.snapshot()`
    reports current counts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv

import platform_admin

load_dotenv()

logger = logging.getLogger("loci.tone")

AXES = ["formal_casual", "serious_playful", "corporate_human",
        "restrained_bold", "technical_accessible"]

DAILY_CALL_BUDGET = int(os.environ.get("ANTHROPIC_CALL_BUDGET", "200"))

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


# ---------- usage tracking + call budget ----------

class ToneUsage:
    """Process-wide, in-memory only — resets on restart, same as everything
    else in this demo. Counters roll over at UTC midnight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = time.strftime("%Y-%m-%d", time.gmtime())
        self.llm_calls = 0
        self.llm_errors = 0
        self.cache_hits = 0
        self.heuristic_fallbacks = 0
        self.budget_exhausted = 0

    def _roll_if_new_day_locked(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self._day:
            self._day = today
            self.llm_calls = self.llm_errors = self.cache_hits = 0
            self.heuristic_fallbacks = self.budget_exhausted = 0

    def under_budget(self) -> bool:
        with self._lock:
            self._roll_if_new_day_locked()
            return self.llm_calls < DAILY_CALL_BUDGET

    def record(self, field: str) -> None:
        with self._lock:
            self._roll_if_new_day_locked()
            setattr(self, field, getattr(self, field) + 1)

    def snapshot(self) -> dict:
        with self._lock:
            self._roll_if_new_day_locked()
            return {
                "date_utc": self._day,
                "daily_budget": DAILY_CALL_BUDGET,
                "llm_calls": self.llm_calls,
                "llm_errors": self.llm_errors,
                "cache_hits": self.cache_hits,
                "heuristic_fallbacks": self.heuristic_fallbacks,
                "budget_exhausted": self.budget_exhausted,
            }


usage = ToneUsage()

# Exact-text cache: profile(text) is pure, and the likeliest repeat pattern
# (re-scoring the same draft, rate-limit testers replaying one payload) is an
# exact resubmission, not a near-duplicate — so a hash-keyed cache captures
# most of the win without fuzzy-matching complexity. Bounded and unordered
# eviction is fine; this is a cost guard, not a correctness-critical store.
_CACHE_MAX = 2000
_cache_lock = threading.Lock()
_cache: dict[str, tuple[dict[str, float], str]] = {}


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


# ---------- LLM judge ----------

def _llm_profile(text: str, model: str = "claude-sonnet-4-6") -> dict[str, float] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        usage.record("llm_calls")
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"SAMPLE:\n{text}"}],
        )
        logger.info(
            "tone judge call model=%s input_tokens=%s output_tokens=%s",
            model, resp.usage.input_tokens, resp.usage.output_tokens,
        )
        platform_admin.record_llm_usage(resp.usage.input_tokens, resp.usage.output_tokens)
        raw = "".join(b.text for b in resp.content if b.type == "text")
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {a: float(data[a]) for a in AXES}
    except Exception:
        usage.record("llm_errors")
        logger.warning("tone judge call failed, falling back to heuristic", exc_info=True)
        return None


def compute_profiles_for_caching(exemplars: list[str]) -> tuple[dict[str, float], str]:
    """
    Pre-compute tone profiles for a set of voice exemplars during offline
    embedding generation. Returns the averaged profile and judge type.
    Used by vector_generation to cache profiles in the manifest, avoiding
    API calls at server startup.
    """
    profiles, judges = [], set()
    for ex in exemplars:
        p, j = profile(ex)
        profiles.append(p)
        judges.add(j)
    brand_profile = {
        a: sum(p[a] for p in profiles) / len(profiles) for a in AXES
    }
    judge = "llm" if "llm" in judges else "heuristic"
    return brand_profile, judge


def profile(text: str) -> tuple[dict[str, float], str]:
    key = _cache_key(text)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        usage.record("cache_hits")
        return hit

    budget_ok = usage.under_budget()
    p = _llm_profile(text) if budget_ok else None

    if p is not None:
        result = (p, "llm")
    else:
        usage.record("budget_exhausted" if not budget_ok else "heuristic_fallbacks")
        result = (_heuristic_profile(text), "heuristic")

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = result
    return result


class ToneScorer:
    def __init__(self, brand_voice_exemplars: list[str], cached_brand_profile: dict[str, float] | None = None):
        if cached_brand_profile is not None:
            self.brand_profile = cached_brand_profile
            self.judge = "llm"  # if cached, it was computed with LLM
        else:
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
