"""
Pure vector math shared by the offline embedder (vector_generation) and the
runtime scorer (loci). No fitting, no state — just the arithmetic both sides
need to agree on.
"""
from __future__ import annotations

import numpy as np


def l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def centroid(vectors: np.ndarray) -> np.ndarray:
    """Mean-pool then re-normalise, so centroid lives on the same unit sphere."""
    c = vectors.mean(axis=0, keepdims=True)
    return l2_normalise(c)[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already L2-normalised
