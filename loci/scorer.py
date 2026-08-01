"""
Aggregation layer.

Consistency and Distinctiveness are carried as two separate numbers all the way
through, per layer, and only combined at the very last step — into a quadrant,
not a blended score. A blend is exactly what hides the two failure modes the
product exists to catch.

This scorer is built from a BrandVectorStore (loci/vector_store.py) — brand
and generic vectors, calibration subsets, and the FAISS indices are all loaded
from disk, already computed by vector_generation/generate_embeddings.py. The
only embedding call anywhere in .score() is encoding the single new input text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .fingerprint import InputCopy, Layer
from .metrics.centroid import DualCentroidModel
from .metrics.lexical import LexicalScorer
from .metrics.structural import StructuralScorer
from .metrics.tone import ToneScorer
from .vector_store import BrandVectorStore

# Which metric contributes to which axis, per layer.
# Identity/Messaging/Positioning are semantic -> centroid-led.
# Voice is stylistic -> lexical/structural/tone-led.
LAYER_WEIGHTS: dict[Layer, dict[str, dict[str, float]]] = {
    Layer.IDENTITY: {
        "consistency":     {"centroid": 0.60, "lexical_sig": 0.20, "tone": 0.20},
        "distinctiveness": {"centroid": 0.60, "lexical_cliche": 0.40},
    },
    Layer.MESSAGING: {
        "consistency":     {"centroid": 0.40, "lexical_sig": 0.20,
                            "structural": 0.15, "tone": 0.25},
        "distinctiveness": {"centroid": 0.45, "lexical_cliche": 0.40,
                            "lexical_div": 0.15},
    },
    Layer.VOICE: {
        "consistency":     {"centroid": 0.20, "lexical_sig": 0.20,
                            "structural": 0.25, "tone": 0.35},
        "distinctiveness": {"centroid": 0.30, "lexical_cliche": 0.45,
                            "lexical_div": 0.25},
    },
    Layer.POSITIONING: {   # stretch layer
        "consistency":     {"centroid": 0.70, "lexical_sig": 0.30},
        "distinctiveness": {"centroid": 0.55, "lexical_cliche": 0.45},
    },
}

QUADRANTS = {
    (True, True):   ("IDEAL", "On-brand and stands out from the market."),
    (True, False):  ("ON-BRAND BUT GENERIC",
                     "Consistent with itself, indistinguishable from competitors."),
    (False, True):  ("UNIQUE BUT OFF-BRAND",
                     "Stands out, but doesn't sound like this brand."),
    (False, False): ("LOST", "Neither on-brand nor distinctive."),
}

THRESHOLD = 55.0  # tunable; the axis split point for the 2x2


@dataclass
class LayerVerdict:
    layer: str
    consistency: float
    distinctiveness: float
    quadrant: str
    quadrant_note: str
    contributions: dict = field(default_factory=dict)


@dataclass
class Report:
    brand: str
    input_label: str | None
    overall_consistency: float
    overall_distinctiveness: float
    overall_quadrant: str
    overall_note: str
    layers: list[LayerVerdict]
    evidence: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _blend(parts: dict[str, float], weights: dict[str, float]) -> tuple[float, dict]:
    avail = {k: w for k, w in weights.items() if parts.get(k) is not None}
    total = sum(avail.values()) or 1.0
    score = sum(parts[k] * (w / total) for k, w in avail.items())
    detail = {k: {"value": round(parts[k], 1), "weight": round(w / total, 2)}
              for k, w in avail.items()}
    return round(score, 1), detail


class BrandDistinctivenessScorer:
    def __init__(self, store: BrandVectorStore):
        self.store = store
        self.fp = store.fingerprint
        self.generic = store.generic
        self.warnings: list[str] = list(store.warnings)

        # One dual-centroid model per layer present in the fingerprint, built
        # from vectors already sitting in the store — no encoding happens here.
        self.models: dict[Layer, DualCentroidModel] = {}
        for layer in self.fp.layers_present():
            bvecs = store.brand_layer_vectors(layer)
            if len(bvecs) < 2:
                self.warnings.append(
                    f"Layer '{layer.value}' has <2 assets — falling back to full corpus.")
                bvecs = store.brand_vectors
            gvecs = store.generic_layer_vectors(layer)
            self.models[layer] = DualCentroidModel(
                store.encoder, bvecs, gvecs,
                calib_brand_vecs=store.brand_calibration_vectors(layer),
                calib_generic_vecs=store.generic_calibration_vectors(layer),
            )

        brand_all = self.fp.texts()
        generic_all = self.generic.texts()
        self.lexical = LexicalScorer(brand_all, generic_all)
        self.structural = StructuralScorer(brand_all)

        voice_exemplars = self.fp.texts(Layer.VOICE) or brand_all[:5]
        self.tone = ToneScorer(voice_exemplars[:5])

    @classmethod
    def from_folder(cls, folder) -> "BrandDistinctivenessScorer":
        return cls(BrandVectorStore.load(folder))

    # ------------------------------------------------------------------

    def score(self, copy: InputCopy) -> Report:
        text = copy.text
        lex = self.lexical.score(text)
        struct = self.structural.score(text)
        tone = self.tone.score(text)

        verdicts: list[LayerVerdict] = []
        for layer, model in self.models.items():
            if layer not in LAYER_WEIGHTS:
                continue
            cen = model.score(text)
            parts = {
                "centroid": cen.consistency,
                "lexical_sig": lex.signature_hit,
                "structural": struct.style_match,
                "tone": tone.alignment,
            }
            cons, cons_detail = _blend(parts, LAYER_WEIGHTS[layer]["consistency"])

            dparts = {
                "centroid": cen.distinctiveness,
                "lexical_cliche": lex.cliche_free,
                "lexical_div": lex.diversity,
            }
            dist, dist_detail = _blend(dparts, LAYER_WEIGHTS[layer]["distinctiveness"])

            q, note = QUADRANTS[(cons >= THRESHOLD, dist >= THRESHOLD)]
            verdicts.append(LayerVerdict(
                layer=layer.value, consistency=cons, distinctiveness=dist,
                quadrant=q, quadrant_note=note,
                contributions={"consistency": cons_detail,
                               "distinctiveness": dist_detail,
                               "raw_cosine": {"vs_brand": round(cen.raw_sim_brand, 4),
                                              "vs_generic": round(cen.raw_sim_generic, 4)}},
            ))

        # Overall = the layer the copy was written for, weighted 2x, rest 1x.
        def agg(attr: str) -> float:
            num = den = 0.0
            for v in verdicts:
                w = 2.0 if v.layer == copy.intended_layer.value else 1.0
                num += getattr(v, attr) * w
                den += w
            return round(num / max(den, 1), 1)

        oc, od = agg("consistency"), agg("distinctiveness")
        oq, on = QUADRANTS[(oc >= THRESHOLD, od >= THRESHOLD)]

        # FAISS lookup: the nearest brand chunk and nearest generic chunk to
        # this input, straight from the vector base mounted at startup.
        vec = self.store.encoder.encode([text])[0]
        nearest_brand = self.store.nearest_brand_chunk(vec)
        nearest_generic = self.store.nearest_generic_chunk(vec)

        return Report(
            brand=self.fp.brand_name,
            input_label=copy.label,
            overall_consistency=oc,
            overall_distinctiveness=od,
            overall_quadrant=oq,
            overall_note=on,
            layers=verdicts,
            evidence={
                "signature_terms_used": lex.matched_signature,
                "cliches_detected": lex.matched_cliche,
                "lexical_diversity": lex.diversity,
                "style": struct.per_feature,
                "tone_input": tone.input_profile,
                "tone_brand": tone.brand_profile,
                "tone_biggest_gap": tone.biggest_gap,
                "tone_judge": tone.judge,
                "nearest_brand_chunk": nearest_brand,
                "nearest_generic_chunk": nearest_generic,
            },
            warnings=list(self.warnings),
        )
