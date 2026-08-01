"""
The one embedding operation allowed at request time: projecting a single new
piece of copy into a vector space that was already fit offline.

InputEncoder has no .fit(). It loads a vectorizer + term weights that
vector_generation/generate_embeddings.py already produced and persisted, and
can only ever transform new text through them — it cannot create or refit the
space itself. That asymmetry is the point: embedding GENERATION lives in
vector_generation/, embedding USE (of one new input at a time) lives here.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .vectormath import l2_normalise


class InputEncoder:
    def __init__(self, vectorizer, term_weights: np.ndarray):
        self._vec = vectorizer
        self._term_weights = term_weights

    @classmethod
    def load(cls, folder: Path) -> "InputEncoder":
        vectorizer = joblib.load(folder / "vectorizer.joblib")
        term_weights = np.load(folder / "term_weights.npy")
        return cls(vectorizer, term_weights)

    def coverage(self, text: str) -> float:
        toks = self._vec.build_analyzer()(text)
        if not toks:
            return 0.0
        seen = sum(1 for t in toks if t in self._vec.vocabulary_)
        return seen / len(toks)

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.asarray(self._vec.transform(texts).todense(), dtype=np.float64)
        m = m * self._term_weights
        return l2_normalise(m)
