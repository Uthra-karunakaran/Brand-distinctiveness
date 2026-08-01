"""
Lexical metrics — classical NLP, no model required.

Three signals:
  1. Type-token ratio (MSTTR, length-normalised) -> vocabulary richness
  2. Signature-phrase hit rate -> does this copy use words THIS brand owns?
  3. Cliché density -> does this copy use words the whole INDUSTRY uses?

(2) feeds consistency. (3) feeds distinctiveness. They are computed against
different corpora, same as the two centroids.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")

# Function words carry voice signal for stylometry but are noise for
# signature/cliche extraction, where we want content terms only.
STOP = {
    "that", "this", "with", "from", "your", "their", "them", "they", "have",
    "will", "been", "were", "what", "when", "which", "there", "here", "than",
    "then", "into", "over", "more", "most", "some", "such", "only", "also",
    "just", "like", "very", "much", "each", "every", "about", "would", "could",
    "should", "them", "these", "those", "other", "because", "while", "where",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def msttr(tokens: list[str], window: int = 40) -> float:
    """Mean segmental TTR — TTR is length-biased, this isn't."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [
        len(set(tokens[i:i + window])) / window
        for i in range(0, len(tokens) - window + 1, window)
    ]
    return sum(ratios) / len(ratios)


@dataclass
class LexicalProfile:
    """Term-frequency profile of a corpus, with distinctive-term extraction."""
    counts: Counter = field(default_factory=Counter)
    total: int = 0

    @classmethod
    def build(cls, texts: list[str]) -> "LexicalProfile":
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        return cls(counts=c, total=max(sum(c.values()), 1))

    def rate(self, term: str) -> float:
        return self.counts[term] / self.total


def signature_terms(brand: LexicalProfile, generic: LexicalProfile,
                    top_k: int = 40, min_count: int = 2,
                    min_lift: float = 2.0) -> list[tuple[str, float]]:
    """Terms the brand over-uses relative to the industry baseline (log-odds-ish).

    min_lift is what stops shared domain vocabulary ("language", "learning")
    from being claimed as anyone's signature."""
    scored = []
    for term, n in brand.counts.items():
        if n < min_count or len(term) < 4 or term in STOP:
            continue
        lift = (brand.rate(term) + 1e-6) / (generic.rate(term) + 1e-6)
        if lift >= min_lift:
            scored.append((term, lift))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def cliche_terms(generic: LexicalProfile, brand: LexicalProfile,
                 top_k: int = 60, min_count: int = 2,
                 min_lift: float = 2.0) -> list[tuple[str, float]]:
    """Terms the industry over-uses relative to this brand — the boilerplate list.

    A term the brand also uses heavily is not a cliche for THIS brand: "free"
    is boilerplate for most SaaS and load-bearing for Duolingo. The lift floor
    is what keeps a word off both lists at once."""
    scored = []
    for term, n in generic.counts.items():
        if n < min_count or len(term) < 4 or term in STOP:
            continue
        lift = (generic.rate(term) + 1e-6) / (brand.rate(term) + 1e-6)
        if lift >= min_lift:
            scored.append((term, lift))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


@dataclass
class LexicalScores:
    diversity: float            # 0-100
    signature_hit: float        # 0-100  -> consistency signal
    cliche_free: float          # 0-100  -> distinctiveness signal
    matched_signature: list[str]
    matched_cliche: list[str]


class LexicalScorer:
    def __init__(self, brand_texts: list[str], generic_texts: list[str]):
        self.brand = LexicalProfile.build(brand_texts)
        self.generic = LexicalProfile.build(generic_texts)
        self.signatures = {t for t, _ in signature_terms(self.brand, self.generic)}
        self.cliches = {t for t, _ in cliche_terms(self.generic, self.brand)}

    def score(self, text: str) -> LexicalScores:
        toks = tokenize(text)
        if not toks:
            return LexicalScores(0, 0, 0, [], [])

        content = [t for t in toks if len(t) >= 4]
        hit = sorted({t for t in content if t in self.signatures})
        clich = sorted({t for t in content if t in self.cliches})

        denom = max(len(set(content)), 1)
        sig_rate = len(hit) / denom
        cli_rate = len(clich) / denom

        return LexicalScores(
            diversity=round(min(msttr(toks) / 0.75, 1.0) * 100, 1),
            # 25% signature-term density is already a strong brand-voice signal
            signature_hit=round(min(sig_rate / 0.25, 1.0) * 100, 1),
            # 30% cliché density = fully generic
            cliche_free=round(max(1.0 - cli_rate / 0.30, 0.0) * 100, 1),
            matched_signature=hit[:12],
            matched_cliche=clich[:12],
        )
