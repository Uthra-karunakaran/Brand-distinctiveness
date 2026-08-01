"""
Offline embedding generation for Loci.

This is the ONLY place a vectorizer gets fit and a corpus gets bulk-encoded.
Run it once per brand (or whenever brand assets / the competitor corpus
change); the API and demo never fit or bulk-encode anything themselves —
they only load what this script writes.

    python -m vector_generation.generate_embeddings \\
        --brand data/brand_duolingo.json \\
        --generic data/generic_edtech.json \\
        --out vector_generation/embeddings

Writes vector_generation/embeddings/<brand_id>/:
    manifest.json         brand_id, brand_name, industry, counts, dims
    brand_vectors.npy      (N_brand, D) float32, L2-normalised
    brand_meta.json        per-row {text, asset_type, layer, source_url, words}
    generic_vectors.npy    (N_generic, D) float32, L2-normalised
    generic_meta.json      per-row {text, asset_type, layer, source_url, words}
    vectorizer.joblib       fitted sklearn TfidfVectorizer
    term_weights.npy        discriminative term weights, aligned to the vectorizer's vocab
    brand.index             FAISS IndexFlatIP over brand_vectors
    generic.index           FAISS IndexFlatIP over generic_vectors

So it's explicit what got embedded: open brand_meta.json / generic_meta.json
and every row is one chunk that fed brand_vectors.npy / generic_vectors.npy,
in the same order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import joblib
import numpy as np

from loci.fingerprint import BrandFingerprint, Chunk, GenericCorpus

from vector_generation.embedder import TfidfEmbedder

ROOT = Path(__file__).parent


def _meta_rows(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "text": c.text,
            "asset_type": c.asset_type,
            "layer": c.layer.value,
            "source_url": c.source_url,
            "words": c.words,
        }
        for c in chunks
    ]


def _faiss_index(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])  # vectors are L2-normalised -> inner product == cosine
    index.add(vecs)
    return index


def generate(brand_json: Path, generic_json: Path, out_root: Path) -> Path:
    b = json.loads(Path(brand_json).read_text())
    g = json.loads(Path(generic_json).read_text())

    fp = BrandFingerprint.from_assets(b["brand_id"], b["brand_name"], b["assets"])
    generic = GenericCorpus.from_texts(g["industry"], g["items"])
    ok, msg = fp.is_scorable()

    brand_texts = fp.texts()
    generic_texts = generic.texts()

    embedder = TfidfEmbedder().fit_discriminative(brand_texts, generic_texts)

    brand_vecs = embedder.encode(brand_texts).astype("float32")
    generic_vecs = embedder.encode(generic_texts).astype("float32")

    out_dir = Path(out_root) / fp.brand_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "brand_vectors.npy", brand_vecs)
    np.save(out_dir / "generic_vectors.npy", generic_vecs)
    (out_dir / "brand_meta.json").write_text(json.dumps(_meta_rows(fp.chunks), indent=2))
    (out_dir / "generic_meta.json").write_text(json.dumps(_meta_rows(generic.chunks), indent=2))

    joblib.dump(embedder.vectorizer, out_dir / "vectorizer.joblib")
    np.save(out_dir / "term_weights.npy", embedder.term_weights)

    faiss.write_index(_faiss_index(brand_vecs), str(out_dir / "brand.index"))
    faiss.write_index(_faiss_index(generic_vecs), str(out_dir / "generic.index"))

    manifest = {
        "brand_id": fp.brand_id,
        "brand_name": fp.brand_name,
        "industry": generic.industry,
        "dims": int(brand_vecs.shape[1]),
        "brand_chunks": len(fp.chunks),
        "generic_chunks": len(generic.chunks),
        "layers_present": [l.value for l in fp.layers_present()],
        "scorable": ok,
        "scorable_message": msg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote embeddings for '{fp.brand_id}' -> {out_dir}")
    print(f"  brand chunks: {len(fp.chunks)}  generic chunks: {len(generic.chunks)}  "
          f"dims: {brand_vecs.shape[1]}  scorable: {ok} ({msg})")
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
