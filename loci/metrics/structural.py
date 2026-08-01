"""
Structural style (stylometry).

A brand's shape is as recognisable as its vocabulary: how long its sentences run,
how much it varies them, how readable it is, whether it writes in bullets, whether
it hides behind passive voice.

We build a style vector for the brand corpus, a style vector for the input, and
score the input on how far it sits from the brand's style envelope (mean +- sd).
"""
from __future__ import annotations

import re
import statistics as st
from dataclasses import dataclass

SENT_SPLIT = re.compile(r"[.!?]+(?:\s|$)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
VOWEL_GROUPS = re.compile(r"[aeiouy]+")
PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\b\s+\w+(?:ed|en)\b", re.I)


def syllables(word: str) -> int:
    w = word.lower().rstrip("e")
    return max(len(VOWEL_GROUPS.findall(w)), 1)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


@dataclass
class StyleVector:
    mean_sentence_len: float
    sentence_len_sd: float
    flesch_kincaid: float
    bullet_ratio: float
    passive_ratio: float
    exclaim_per_100w: float

    def as_tuple(self) -> tuple[float, ...]:
        return (self.mean_sentence_len, self.sentence_len_sd, self.flesch_kincaid,
                self.bullet_ratio, self.passive_ratio, self.exclaim_per_100w)


def style_vector(text: str) -> StyleVector:
    sents = sentences(text) or [text]
    words = WORD_RE.findall(text)
    n_words = max(len(words), 1)
    lens = [len(WORD_RE.findall(s)) for s in sents] or [n_words]
    syl = sum(syllables(w) for w in words)

    # Flesch-Kincaid grade level
    fk = 0.39 * (n_words / len(sents)) + 11.8 * (syl / n_words) - 15.59

    lines = [l for l in text.splitlines() if l.strip()]
    bullets = sum(1 for l in lines if l.strip()[:2] in ("- ", "* ", "• ") or
                  re.match(r"^\d+[.)]\s", l.strip()))

    return StyleVector(
        mean_sentence_len=st.mean(lens),
        sentence_len_sd=st.pstdev(lens) if len(lens) > 1 else 0.0,
        flesch_kincaid=fk,
        bullet_ratio=bullets / max(len(lines), 1),
        passive_ratio=len(PASSIVE.findall(text)) / max(len(sents), 1),
        exclaim_per_100w=100 * text.count("!") / n_words,
    )


@dataclass
class StructuralScores:
    style_match: float          # 0-100 -> consistency signal
    per_feature: dict[str, dict[str, float]]


class StructuralScorer:
    """Learns the brand's style envelope, scores how far new copy deviates."""

    def __init__(self, brand_texts: list[str]):
        vecs = [style_vector(t).as_tuple() for t in brand_texts if t.strip()]
        cols = list(zip(*vecs))
        self.mean = [st.mean(c) for c in cols]
        # floor the sd so a uniform corpus doesn't make the envelope infinitely tight
        self.sd = [max(st.pstdev(c), abs(st.mean(c)) * 0.25, 0.5) for c in cols]
        self.names = ["mean_sentence_len", "sentence_len_sd", "flesch_kincaid",
                      "bullet_ratio", "passive_ratio", "exclaim_per_100w"]

    def score(self, text: str) -> StructuralScores:
        v = style_vector(text).as_tuple()
        per, zs = {}, []
        for name, val, mu, sd in zip(self.names, v, self.mean, self.sd):
            z = abs(val - mu) / sd
            zs.append(z)
            per[name] = {"input": round(val, 2), "brand_mean": round(mu, 2),
                         "z": round(z, 2)}
        mean_z = sum(zs) / len(zs)
        # z of 0 -> 100, z of 2.5+ -> 0
        return StructuralScores(
            style_match=round(max(0.0, 1 - mean_z / 2.5) * 100, 1),
            per_feature=per,
        )
