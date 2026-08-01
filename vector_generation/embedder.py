"""
Embedding substrate — vector GENERATION only.

This module is deliberately not importable from loci/ or api.py. Anything
that can FIT a vector space (TfidfEmbedder.fit / fit_discriminative,
SentenceTransformerEmbedder's model load) lives here, and only here, so it's
unambiguous where "an embedding gets created" happens: inside
generate_embeddings.py, offline, once per brand.

At runtime, loci/input_encoder.py re-implements just the encode()/coverage()
half of TfidfEmbedder against the artifacts this module writes to disk
(vectorizer.joblib + term_weights.npy) — it has no .fit(), so it cannot
regenerate the space, only project new text into the one that already exists.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loci.vectormath import l2_normalise


class Embedder(Protocol):
    dim: int
    def fit(self, corpus: list[str]) -> "Embedder": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """
    Character+word TF-IDF. Sparse, but cosine geometry behaves the same way,
    so every downstream calculation (centroids, calibration, quadrants) is
    identical to what it will be with dense embeddings.
    """
    def __init__(self, max_features: int = 4000):
        self._vec = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None,       # stopwords ARE voice signal — keep them
            lowercase=True,
        )
        self.dim = max_features
        self._fitted = False
        self._term_weights = None

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vec

    @property
    def term_weights(self) -> np.ndarray:
        if self._term_weights is None:
            return np.ones(len(self._vec.vocabulary_))
        return self._term_weights

    def fit(self, corpus: list[str]) -> "TfidfEmbedder":
        self._vec.fit(corpus)
        self.dim = len(self._vec.vocabulary_)
        self._fitted = True
        return self

    def fit_discriminative(self, brand: list[str], generic: list[str],
                           strength: float = 1.0) -> "TfidfEmbedder":
        """
        Fit on the joint corpus, then down-weight terms that appear at similar
        rates in BOTH corpora.

        Why this matters: raw similarity is dominated by TOPIC, not brand. Two
        language-learning companies share "language", "learn", "lesson" — those
        terms carry zero discriminative signal but eat most of the cosine. This
        reweights the space so distance reflects how a brand talks, not what
        industry it is in. (The same failure mode exists with dense embeddings,
        which is why tone gets an LLM judge rather than a cosine.)
        """
        self.fit(brand + generic)
        vocab = self._vec.vocabulary_
        w = np.ones(len(vocab))

        b = np.asarray(self._vec.transform(brand).sum(axis=0)).ravel()
        g = np.asarray(self._vec.transform(generic).sum(axis=0)).ravel()
        b = b / max(b.sum(), 1e-9)
        g = g / max(g.sum(), 1e-9)

        lift = np.abs(np.log((b + 1e-6) / (g + 1e-6)))
        w = (lift / max(lift.max(), 1e-9)) ** strength
        self._term_weights = np.clip(w, 0.05, 1.0)
        return self

    def coverage(self, text: str) -> float:
        """
        Fraction of the input's content tokens that exist in the fitted vocabulary.

        TF-IDF silently DROPS out-of-vocabulary words. So a piece of copy full of
        language neither corpus uses ("enrolment", "tuition", "supervised") gets
        reduced to whichever few words it happens to share with the brand — and
        then scores as highly consistent on the strength of that remnant. Coverage
        is the correction: low coverage means we saw little of the input, so the
        similarity we measured is not evidence of much.

        This correction is specific to sparse vectors; with dense sentence
        embeddings there is no OOV and coverage is always 1.0.
        """
        toks = self._vec.build_analyzer()(text)
        if not toks:
            return 0.0
        seen = sum(1 for t in toks if t in self._vec.vocabulary_)
        return seen / len(toks)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() on the joint corpus before encode().")
        m = np.asarray(self._vec.transform(texts).todense(), dtype=np.float64)
        if self._term_weights is not None:
            m = m * self._term_weights
        return l2_normalise(m)


class SentenceTransformerEmbedder:
    """Real build. Requires `pip install sentence-transformers` + model download."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def fit(self, corpus: list[str]) -> "SentenceTransformerEmbedder":
        return self  # pretrained; nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        return l2_normalise(np.asarray(self._model.encode(texts), dtype=np.float64))
