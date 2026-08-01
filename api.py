"""
Loci service — thin FastAPI wrapper around the scorer.

No embedding is ever fit or bulk-encoded here. Every brand's vectors, FAISS
indices and fitted encoder are already sitting on disk under
vector_generation/embeddings/<brand_id>/ (written once, offline, by
vector_generation/generate_embeddings.py). At startup this process loads every
brand folder it finds into memory and holds it there until the process stops —
no request re-loads or re-generates anything.

  GET  /brands                  list brands mounted at startup
  POST /brands/{id}/score       fast, per request — the only live embedding
                                 call anywhere is encoding this one input text
  GET  /brands/{id}/vocabulary  what the brand owns vs. what the category owns

To add or refresh a brand: run vector_generation/generate_embeddings.py, then
restart the service.

    uvicorn api:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from loci.fingerprint import InputCopy, Layer
from loci.scorer import BrandDistinctivenessScorer
from loci.vector_store import BrandVectorStore

EMBEDDINGS_ROOT = Path(__file__).parent / "vector_generation" / "embeddings"

# In-memory cache: brand_id -> scorer, populated once at startup from disk.
_SCORERS: dict[str, BrandDistinctivenessScorer] = {}


def _load_all_brands() -> None:
    if not EMBEDDINGS_ROOT.exists():
        return
    for folder in sorted(p for p in EMBEDDINGS_ROOT.iterdir() if p.is_dir()):
        store = BrandVectorStore.load(folder)
        _SCORERS[store.manifest["brand_id"]] = BrandDistinctivenessScorer(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_all_brands()
    yield
    _SCORERS.clear()


app = FastAPI(title="Loci — Brand Distinctiveness", lifespan=lifespan)


class ScoreRequest(BaseModel):
    text: str
    intended_layer: Layer = Layer.MESSAGING
    channel: str = "landing_page"
    label: str | None = None


def _get_scorer(brand_id: str) -> BrandDistinctivenessScorer:
    scorer = _SCORERS.get(brand_id)
    if scorer is None:
        raise HTTPException(
            404,
            f"Unknown brand_id '{brand_id}'. Run "
            f"'python -m vector_generation.generate_embeddings' for it and restart the service.",
        )
    return scorer


@app.get("/brands")
def list_brands() -> dict:
    return {
        "brands": [
            {"brand_id": bid, "brand_name": s.fp.brand_name, "warnings": s.warnings}
            for bid, s in _SCORERS.items()
        ]
    }


@app.post("/brands/{brand_id}/score")
def score_copy(brand_id: str, req: ScoreRequest) -> dict:
    scorer = _get_scorer(brand_id)
    copy = InputCopy(
        text=req.text, intended_layer=req.intended_layer,
        channel=req.channel, label=req.label,
    )
    return scorer.score(copy).to_dict()


@app.get("/brands/{brand_id}/vocabulary")
def vocabulary(brand_id: str) -> dict:
    scorer = _get_scorer(brand_id)
    return {
        "brand_id": brand_id,
        "signature_terms": sorted(scorer.lexical.signatures),
        "cliche_terms": sorted(scorer.lexical.cliches),
    }
