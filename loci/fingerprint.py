"""
Brand Fingerprint schema — the input contract for Loci.

Two ingest paths:
  1. MVBF (Minimum Viable Brand Fingerprint) — 6 fields, always available.
  2. Extended assets — anything else the company has (blog, docs, ads, job posts...).

Everything is normalised into `Chunk` objects tagged with a LAYER, because
scoring is reported per layer, not as one global number.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field


class Layer(str, Enum):
    IDENTITY = "identity"       # Who are we?
    MESSAGING = "messaging"     # What do we repeatedly communicate?
    VOICE = "voice"             # How do we sound?
    POSITIONING = "positioning"  # Why choose us?  (stretch layer)
    PROOF = "proof"             # Can we back it up?


# Which asset type feeds which layer. This is the mapping that turns a messy
# pile of scraped text into a structured fingerprint.
ASSET_LAYER_MAP: dict[str, Layer] = {
    "name": Layer.IDENTITY,
    "tagline": Layer.IDENTITY,
    "mission": Layer.IDENTITY,
    "vision": Layer.IDENTITY,
    "values": Layer.IDENTITY,
    "about": Layer.IDENTITY,
    "founder_story": Layer.IDENTITY,
    "homepage": Layer.MESSAGING,
    "product_page": Layer.MESSAGING,
    "landing_page": Layer.MESSAGING,
    "cta": Layer.MESSAGING,
    "email": Layer.MESSAGING,
    "blog": Layer.VOICE,
    "social": Layer.VOICE,
    "ad": Layer.VOICE,
    "support_doc": Layer.VOICE,
    "job_post": Layer.VOICE,
    "sales_deck": Layer.POSITIONING,
    "investor_deck": Layer.POSITIONING,
    "comparison_page": Layer.POSITIONING,
    "case_study": Layer.PROOF,
    "testimonial": Layer.PROOF,
    "review": Layer.PROOF,
}

MVBF_FIELDS = ("name", "tagline", "mission", "vision", "values", "about")


class ScorabilityError(Exception):
    """Raised when a caller opts into strict enforcement of is_scorable()."""

    def __init__(self, message: str, fields: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.fields = fields or []


class Chunk(BaseModel):
    """One embeddable unit of brand language."""
    text: str
    asset_type: str
    layer: Layer
    source_url: str | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


class BrandFingerprint(BaseModel):
    brand_id: str
    brand_name: str
    chunks: list[Chunk] = Field(default_factory=list)

    # ---------- construction ----------

    @classmethod
    def from_assets(cls, brand_id: str, brand_name: str,
                    assets: dict[str, list[str] | str]) -> "BrandFingerprint":
        """
        assets: {"mission": "...", "blog": ["post1", "post2"], ...}
        Unknown asset types are dropped loudly rather than silently mis-layered.
        """
        chunks: list[Chunk] = []
        for asset_type, value in assets.items():
            layer = ASSET_LAYER_MAP.get(asset_type)
            if layer is None:
                raise ValueError(
                    f"Unmapped asset_type '{asset_type}'. Add it to ASSET_LAYER_MAP."
                )
            texts = [value] if isinstance(value, str) else list(value)
            for t in texts:
                t = t.strip()
                if t:
                    chunks.append(Chunk(text=t, asset_type=asset_type, layer=layer))
        return cls(brand_id=brand_id, brand_name=brand_name, chunks=chunks)

    # ---------- validation ----------

    def mvbf_status(self) -> dict[str, bool]:
        present = {c.asset_type for c in self.chunks}
        return {f: (f in present) for f in MVBF_FIELDS}

    def is_scorable(self) -> tuple[bool, str]:
        """Floor check: without the MVBF, the brand centroid is meaningless."""
        status = self.mvbf_status()
        missing = [f for f, ok in status.items() if not ok]
        if missing:
            return False, f"Missing MVBF fields: {', '.join(missing)}"
        if len(self.chunks) < 6:
            return False, "Fewer than 6 chunks — centroid would be unstable."
        return True, "ok"

    # ---------- access ----------

    def layer_chunks(self, layer: Layer) -> list[Chunk]:
        return [c for c in self.chunks if c.layer == layer]

    def layers_present(self) -> list[Layer]:
        return sorted({c.layer for c in self.chunks}, key=lambda l: l.value)

    def texts(self, layer: Layer | None = None) -> list[str]:
        src: Iterable[Chunk] = self.chunks if layer is None else self.layer_chunks(layer)
        return [c.text for c in src]


class GenericCorpus(BaseModel):
    """
    The second reference point. Competitor pages + industry-report language +
    templated marketing copy. Without this, you are only measuring consistency.
    """
    industry: str
    chunks: list[Chunk] = Field(default_factory=list)

    @classmethod
    def from_texts(cls, industry: str, items: list[dict]) -> "GenericCorpus":
        chunks = [
            Chunk(
                text=i["text"],
                asset_type=i.get("asset_type", "homepage"),
                layer=ASSET_LAYER_MAP.get(i.get("asset_type", "homepage"), Layer.MESSAGING),
                source_url=i.get("source_url"),
            )
            for i in items
        ]
        return cls(industry=industry, chunks=chunks)

    def texts(self, layer: Layer | None = None) -> list[str]:
        if layer is None:
            return [c.text for c in self.chunks]
        hits = [c.text for c in self.chunks if c.layer == layer]
        # Generic corpora are thin per-layer; fall back to the whole corpus.
        return hits if len(hits) >= 3 else [c.text for c in self.chunks]


class InputCopy(BaseModel):
    """The thing being scored — a new piece of marketing language."""
    text: str
    intended_layer: Layer = Layer.MESSAGING
    channel: str = "landing_page"
    label: str | None = None
