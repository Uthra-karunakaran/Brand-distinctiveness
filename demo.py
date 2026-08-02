"""
Worked example: score four candidate pieces of copy for the same brand,
one aimed at each quadrant of the 2x2.

Embeddings are generated offline, once, by vector_generation/generate_embeddings.py
into vector_generation/embeddings/<brand_id>/. This script only LOADS them —
if they don't exist yet it generates them once first, exactly like the API
does at startup.

    python demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from loci.fingerprint import InputCopy, Layer
from loci.scorer import BrandDistinctivenessScorer
from loci.vector_store import BrandVectorStore
from vector_generation.generate_embeddings import generate

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EMBEDDINGS_ROOT = ROOT / "vector_generation" / "embeddings"
BRAND_ID = "duolingo"

CANDIDATES = [
    InputCopy(
        label="A — written in-voice",
        intended_layer=Layer.MESSAGING,
        channel="landing_page",
        text=(
            "Your streak is 47 days old. Do not let a Tuesday kill it.\n"
            "Five minutes. One lesson. The owl is watching and the owl is free.\n"
            "Start now. It costs nothing, because it never has."
        ),
    ),
    InputCopy(
        label="B — right claims, category language",
        intended_layer=Layer.MESSAGING,
        channel="landing_page",
        text=(
            "Learn a language for free in just five minutes a day. Our platform "
            "makes learning fun and engaging, with short lessons and daily streaks "
            "designed to keep you motivated. Join millions of learners worldwide "
            "and start your journey to fluency today."
        ),
    ),
    InputCopy(
        label="C — distinctive, but not this brand",
        intended_layer=Layer.MESSAGING,
        channel="landing_page",
        text=(
            "Language acquisition is a discipline, not a diversion. "
            "Our curriculum is derived from forty years of applied linguistics "
            "research and is administered in ninety-minute supervised sessions. "
            "Enrolment is capped at twelve. Tuition is 1,400 dollars per term "
            "and there is a waiting list."
        ),
    ),
    InputCopy(
        label="D — neither",
        intended_layer=Layer.MESSAGING,
        channel="landing_page",
        text=(
            "We provide comprehensive solutions for the modern learner. "
            "Contact our team to learn more about our enterprise offerings and "
            "discover how we can support your organization's objectives."
        ),
    ),
]


def build_scorer() -> BrandDistinctivenessScorer:
    brand_dir = EMBEDDINGS_ROOT / BRAND_ID
    if not brand_dir.exists():
        print("No precomputed embeddings found — running the offline generation "
              "step once (see vector_generation/generate_embeddings.py)...\n")
        generate(DATA / "brand_duolingo.json", DATA / "generic_edtech.json", EMBEDDINGS_ROOT)
        print()

    store = BrandVectorStore.load(brand_dir)
    print(f"Fingerprint: {len(store.fingerprint.chunks)} chunks across "
          f"{len(store.fingerprint.layers_present())} layers | "
          f"MVBF: {store.manifest['scorable_message']}")
    print(f"Generic baseline: {len(store.generic.chunks)} chunks "
          f"({store.generic.industry})\n")
    return BrandDistinctivenessScorer(store)


def bar(v: float, width: int = 22) -> str:
    n = int(round(v / 100 * width))
    return "█" * n + "·" * (width - n)


def main() -> None:
    scorer = build_scorer()
    print("Signature terms learned:",
          ", ".join(sorted(list(scorer.lexical.signatures))[:14]))
    print("Cliché terms learned:   ",
          ", ".join(sorted(list(scorer.lexical.cliches))[:14]))
    print("\n" + "=" * 74)

    for c in CANDIDATES:
        r = scorer.score(c)
        print(f"\n{r.input_label}")
        print("-" * 74)
        print(f"  Consistency     {bar(r.overall_consistency)} "
              f"{r.overall_consistency:>5.1f}")
        print(f"  Distinctiveness {bar(r.overall_distinctiveness)} "
              f"{r.overall_distinctiveness:>5.1f}")
        print(f"  => {r.overall_quadrant}  — {r.overall_note}")
        print("\n  Per layer:")
        for lv in r.layers:
            print(f"    {lv.layer:<12} cons {lv.consistency:>5.1f} | "
                  f"dist {lv.distinctiveness:>5.1f} | {lv.quadrant}")
        ev = r.evidence
        print(f"\n  Signature terms used: {ev['signature_terms_used'] or '—'}")
        print(f"  Clichés detected:     {ev['cliches_detected'] or '—'}")
        print(f"  Tone gap (largest):   {ev['tone_biggest_gap']} "
              f"[judge={ev['tone_judge']}]")
        print("  Nearest brand chunks:")
        for chunk in ev["nearest_brand_chunks"]:
            print(f"    - {chunk[:70]}")
        print("  Nearest generic chunks:")
        for chunk in ev["nearest_generic_chunks"]:
            print(f"    - {chunk[:70]}")

    print("\n" + "=" * 74)
    print("\nFull JSON for candidate A:\n")
    print(json.dumps(scorer.score(CANDIDATES[0]).to_dict(), indent=2)[:2200] + "\n...")


if __name__ == "__main__":
    main()
