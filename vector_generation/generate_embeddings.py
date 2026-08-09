"""
Embedding generation for Loci.

This is the ONLY place a vectorizer gets fit and a corpus gets bulk-encoded.
generate_from_dicts() is the shared core; it is called from two places:

  1. The CLI wrapper generate() below (offline, once per brand, reads JSON
     files from disk) — unchanged behaviour from before this module was
     split: it always writes embeddings and records scorable: true/false in
     the manifest, it never raises on a failed MVBF check.

  2. api.py's POST /brands/{id}/embeddings background worker (live, triggered
     by onboarding) — calls generate_from_dicts(..., strict=True) so an
     unscorable brand fails the job before anything is written or hot-loaded,
     and passes on_stage so the job record can report progress.

Either way, the API and demo never fit or bulk-encode anything themselves —
they only load what this module writes.

    python -m vector_generation.generate_embeddings \\
        --brand data/brand_duolingo.json \\
        --generic data/generic_edtech.json \\
        --out vector_generation/embeddings

Writes vector_generation/embeddings/<brand_id>/:
    manifest.json          brand_id, brand_name, industry, counts, dims
    brand_vectors.npy      (N_brand, D) float32, L2-normalised
    brand_meta.json        per-row {text, asset_type, layer, words}
    generic_vectors.npy    (N_generic, D) float32, L2-normalised
    generic_meta.json      per-row {text, asset_type, layer, words}
    vectorizer.joblib      fitted sklearn TfidfVectorizer
    term_weights.npy       discriminative term weights, aligned to the vectorizer's vocab
    brand.index            FAISS IndexFlatIP over brand_vectors
    generic.index          FAISS IndexFlatIP over generic_vectors
    source_input.json      the raw {brand, generic} dicts that produced this folder —
                            lets a later re-run omit generic_corpus and reuse this one

So it's explicit what got embedded: open brand_meta.json / generic_meta.json
and every row is one chunk that fed brand_vectors.npy / generic_vectors.npy,
in the same order.

generate() also best-effort registers --generic's industry into the shared,
immutable registry at vector_generation/industries/<industry_id>.json (see
vector_generation/industries.py) if it isn't there already — this is what
lets a later live brand reference that industry by id instead of resending
the corpus. Registration never overwrites; it never changes what this run
actually embeds either, which always comes straight from --generic.

The output folder is written to a temp directory first and swapped into place
with os.replace() only once every artifact is written, so a reader (the API's
hot-reload, or a restart's startup scan) never sees a half-written folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import faiss
import joblib
import numpy as np

from loci.fingerprint import BrandFingerprint, Chunk, GenericCorpus, Layer, ScorabilityError
from loci.metrics.tone import compute_profiles_for_caching

from vector_generation import industries
from vector_generation.embedder import TfidfEmbedder
from vector_generation.jobs import JobStage

ROOT = Path(__file__).parent
INDUSTRIES_ROOT = ROOT / "industries"

# Below this many generic-corpus chunks the centroid is thin enough to be
# noisy (README calls 15-30 chunks "enough") — not a hard floor, just a
# warning surfaced back to the caller.
GENERIC_CORPUS_WARNING_THRESHOLD = 15


def _meta_rows(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "asset_type": c.asset_type,
            "layer": c.layer.value,
            "words": c.words,
        }
        for c in chunks
    ]


def _faiss_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])  # vectors are L2-normalised -> inner product == cosine
    index.add(vecs)
    return index


def generate_from_dicts(
    brand: dict,
    generic: dict,
    out_root: Path,
    on_stage: Callable[[JobStage], None] | None = None,
    strict: bool = False,
) -> tuple[Path, dict]:
    """
    brand:   {"brand_id": ..., "brand_name": ..., "assets": {...}}
    generic: {"industry": ..., "items": [{"text": ..., "asset_type": ...}, ...]}

    If strict, raises ScorabilityError instead of proceeding when
    fp.is_scorable() is False — used by the live endpoint, which needs a
    clean pass/fail signal before it writes or hot-loads anything. The CLI
    wrapper below calls this with strict=False, its long-standing behaviour.

    Returns (out_dir, manifest).
    """
    def stage(s: JobStage) -> None:
        if on_stage is not None:
            on_stage(s)

    stage(JobStage.FINGERPRINTING)
    fp = BrandFingerprint.from_assets(brand["brand_id"], brand["brand_name"], brand["assets"])
    generic_corpus = GenericCorpus.from_texts(generic["industry"], generic["items"])
    ok, msg = fp.is_scorable()
    if strict and not ok:
        missing = [f for f, present in fp.mvbf_status().items() if not present]
        raise ScorabilityError(msg, fields=missing)

    warnings: list[dict] = []
    if len(generic_corpus.chunks) < GENERIC_CORPUS_WARNING_THRESHOLD:
        warnings.append({
            "code": "thin_generic_corpus",
            "message": (
                f"Generic corpus has {len(generic_corpus.chunks)} chunks; "
                f"{GENERIC_CORPUS_WARNING_THRESHOLD}+ recommended for a stable centroid."
            ),
        })

    brand_texts = fp.texts()
    generic_texts = generic_corpus.texts()

    stage(JobStage.FITTING_EMBEDDER)
    embedder = TfidfEmbedder().fit_discriminative(brand_texts, generic_texts)

    stage(JobStage.ENCODING_VECTORS)
    brand_vecs = embedder.encode(brand_texts).astype("float32")
    generic_vecs = embedder.encode(generic_texts).astype("float32")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".tmp-{fp.brand_id}-", dir=out_root))
    try:
        np.save(tmp_dir / "brand_vectors.npy", brand_vecs)
        np.save(tmp_dir / "generic_vectors.npy", generic_vecs)
        (tmp_dir / "brand_meta.json").write_text(json.dumps(_meta_rows(fp.chunks), indent=2))
        (tmp_dir / "generic_meta.json").write_text(json.dumps(_meta_rows(generic_corpus.chunks), indent=2))

        joblib.dump(embedder.vectorizer, tmp_dir / "vectorizer.joblib")
        np.save(tmp_dir / "term_weights.npy", embedder.term_weights)

        stage(JobStage.BUILDING_INDEX)
        faiss.write_index(_faiss_index(brand_vecs), str(tmp_dir / "brand.index"))
        faiss.write_index(_faiss_index(generic_vecs), str(tmp_dir / "generic.index"))

        stage(JobStage.WRITING_MANIFEST)

        # Pre-compute tone profiles for voice exemplars to avoid API calls at startup
        tone_profiles = None
        voice_exemplars = fp.texts(Layer.VOICE) or fp.texts()
        if voice_exemplars:
            try:
                tone_profile, tone_judge = compute_profiles_for_caching(voice_exemplars[:5])
                tone_profiles = {
                    "brand_profile": tone_profile,
                    "judge": tone_judge,
                }
            except Exception:
                # If tone profiling fails, continue without caching — startup will
                # fall back to computing profiles on demand.
                pass

        manifest = {
            "brand_id": fp.brand_id,
            "brand_name": fp.brand_name,
            "industry": generic_corpus.industry,
            "dims": int(brand_vecs.shape[1]),
            "brand_chunks": len(fp.chunks),
            "generic_chunks": len(generic_corpus.chunks),
            "layers_present": [l.value for l in fp.layers_present()],
            "scorable": ok,
            "scorable_message": msg,
            "warnings": warnings,
        }
        if tone_profiles is not None:
            manifest["tone_profiles"] = tone_profiles
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (tmp_dir / "source_input.json").write_text(json.dumps({"brand": brand, "generic": generic}, indent=2))

        # Validate every artifact a BrandVectorStore.load() will read is present
        # before swapping this folder into place.
        expected = (
            "brand_vectors.npy", "generic_vectors.npy", "brand_meta.json", "generic_meta.json",
            "vectorizer.joblib", "term_weights.npy", "brand.index", "generic.index", "manifest.json",
        )
        missing_files = [f for f in expected if not (tmp_dir / f).exists()]
        if missing_files:
            raise RuntimeError(f"Generation incomplete, missing: {', '.join(missing_files)}")

        out_dir = out_root / fp.brand_id
        old_dir = out_root / f".old-{fp.brand_id}"
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if out_dir.exists():
            out_dir.rename(old_dir)
        try:
            tmp_dir.replace(out_dir)  # os.replace equivalent, atomic on the same filesystem
        except Exception:
            if old_dir.exists():
                old_dir.rename(out_dir)
            raise
        if old_dir.exists():
            shutil.rmtree(old_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    print(f"Wrote embeddings for '{fp.brand_id}' -> {out_dir}")
    print(f"  brand chunks: {len(fp.chunks)}  generic chunks: {len(generic_corpus.chunks)}  "
          f"dims: {brand_vecs.shape[1]}  scorable: {ok} ({msg})")
    return out_dir, manifest


def generate(brand_json: Path, generic_json: Path, out_root: Path) -> Path:
    """
    CLI entry point — reads JSON files from disk. Generation behaviour is
    unchanged from before the industry registry existed: this always
    generates from the local --generic file exactly as given, never reads
    the registry back to decide what to embed.

    The one addition is a best-effort side effect: if this industry hasn't
    been registered yet, it gets registered from this file so a later live
    POST /brands/{id}/embeddings call can reference it by industry_id instead
    of resending the corpus. If it's already registered (by an earlier CLI
    run or a live brand), registration is a no-op — the registry keeps
    whatever it already has, even if this local file has since diverged from
    it. The registry is a discovery convenience for the live path, not a
    cache this command reads from.
    """
    b = json.loads(Path(brand_json).read_text())
    g = json.loads(Path(generic_json).read_text())

    industries.load_all(INDUSTRIES_ROOT)
    industries.get_or_create(INDUSTRIES_ROOT, g["industry"], g["items"])

    out_dir, _manifest = generate_from_dicts(b, g, Path(out_root))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", type=Path, default=ROOT.parent / "data" / "brand_duolingo.json")
    ap.add_argument("--generic", type=Path, default=ROOT.parent / "data" / "generic_edtech.json")
    ap.add_argument("--out", type=Path, default=ROOT / "embeddings")
    args = ap.parse_args()
    generate(args.brand, args.generic, args.out)


if __name__ == "__main__":
    main()
