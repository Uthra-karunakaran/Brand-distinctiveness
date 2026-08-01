"""
Dual-centroid scoring — the core mechanism.

Raw cosine similarities are not interpretable. A cosine of 0.31 to a brand
centroid means nothing on its own; it depends on the embedder, the corpus size
and the industry. So every raw similarity is calibrated against two empirical
anchors drawn from the corpora themselves:

  Consistency anchors
    high = brand's own chunks vs brand centroid (leave-one-out) -> "this is what
           being on-brand actually looks like for THIS brand"
    low  = generic chunks vs brand centroid                     -> floor

  Distinctiveness anchors
    low  = generic chunks vs generic centroid  -> "this is what boilerplate scores"
    high = brand's own chunks vs generic centroid -> "this is how far this brand
           normally sits from the baseline"

Both scores come from independent comparisons. Neither is derived from the other.

This model takes brand/generic vectors PRECOMPUTED by vector_generation/ — it
never embeds a corpus itself. The only live embedding call it makes is
encoding the single new input text in .score(), via an InputEncoder loaded
from disk (see loci/input_encoder.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vectormath import centroid, cosine


def _rescale(value: float, low: float, high: float) -> float:
    """Map value from [low, high] onto [0, 100], clamped. Handles inverted ranges."""
    if abs(high - low) < 1e-9:
        return 50.0
    pct = (value - low) / (high - low)
    return float(max(0.0, min(1.0, pct)) * 100.0)


@dataclass
class CentroidScores:
    consistency: float          # 0-100, "are you being you?"
    distinctiveness: float      # 0-100, "are you not sounding like everyone else?"
    raw_sim_brand: float
    raw_sim_generic: float


class DualCentroidModel:
    MIN_CALIB_WORDS = 8  # stubs like a brand name have ~0 similarity to everything
                         # and would drag every anchor toward zero.
                         # (Enforced upstream, when vector_generation selects
                         # calibration rows — kept here only as documentation.)

    def __init__(self, encoder, brand_vecs: np.ndarray, generic_vecs: np.ndarray,
                 calib_brand_vecs: np.ndarray | None = None,
                 calib_generic_vecs: np.ndarray | None = None):
        """
        encoder: an InputEncoder (loci/input_encoder.py) — used ONLY to embed
                 the single new input text in .score(). brand_vecs/generic_vecs
                 and the calibration subsets are already-computed arrays loaded
                 from vector_generation's .npy output; nothing is embedded here.
        """
        if len(brand_vecs) < 2 or len(generic_vecs) < 2:
            raise ValueError("Need >=2 vectors in each corpus to form stable centroids.")

        self.encoder = encoder
        self.brand_vecs = brand_vecs
        self.generic_vecs = generic_vecs

        # Centroids use everything; anchors use only substantive chunks.
        self.calib_b = calib_brand_vecs if calib_brand_vecs is not None and len(calib_brand_vecs) >= 2 else self.brand_vecs
        self.calib_g = calib_generic_vecs if calib_generic_vecs is not None and len(calib_generic_vecs) >= 2 else self.generic_vecs

        self.brand_centroid = centroid(self.brand_vecs)
        self.generic_centroid = centroid(self.generic_vecs)

        self._calibrate()

    def _loo_sims(self, vecs: np.ndarray) -> list[float]:
        """Leave-one-out similarity of each chunk to its own centroid — otherwise
        the chunk inflates the centroid it's being measured against."""
        sims = []
        for i in range(len(vecs)):
            rest = np.delete(vecs, i, axis=0)
            sims.append(cosine(vecs[i], centroid(rest)))
        return sims

    def _calibrate(self) -> None:
        brand_self = self._loo_sims(self.calib_b)
        brand_vs_generic = [cosine(v, self.generic_centroid) for v in self.calib_b]
        generic_self = self._loo_sims(self.calib_g)
        generic_vs_brand = [cosine(v, self.brand_centroid) for v in self.calib_g]

        # Consistency: high anchor = typical on-brand chunk, low anchor = boilerplate
        self.cons_high = float(np.percentile(brand_self, 85))
        self.cons_low = float(np.percentile(generic_vs_brand, 60))

        # Distinctiveness: LOW similarity to generic == HIGH distinctiveness,
        # so anchors are inverted on purpose.
        self.dist_generic_anchor = float(np.percentile(generic_self, 40))    # -> score 0
        self.dist_brand_anchor = float(np.percentile(brand_vs_generic, 60))  # -> score 100

    def score(self, text: str) -> CentroidScores:
        # The one live embedding call in the whole scoring path: the new
        # input text, projected through an already-fitted encoder.
        v = self.encoder.encode([text])[0]
        sb = cosine(v, self.brand_centroid)
        sg = cosine(v, self.generic_centroid)

        # Discount similarity by how much of the input we could actually see.
        cov = getattr(self.encoder, "coverage", None)
        if cov is not None:
            sb *= cov(text) ** 0.5
        return CentroidScores(
            consistency=_rescale(sb, self.cons_low, self.cons_high),
            distinctiveness=_rescale(sg, self.dist_generic_anchor, self.dist_brand_anchor),
            raw_sim_brand=sb,
            raw_sim_generic=sg,
        )
