"""
Loci service — thin FastAPI wrapper around the scorer.

No embedding is ever fit or bulk-encoded here. Every brand's vectors, FAISS
indices and fitted encoder are already sitting on disk under
vector_generation/embeddings/<brand_id>/ (written once, offline, by
vector_generation/generate_embeddings.py). At startup this process loads every
brand folder it finds into memory and holds it there until the process stops —
no request re-loads or re-generates anything.

  GET  /brands                     list brands mounted at startup
  POST /brands/{id}/score          fast, per request — the only live embedding
                                    call anywhere is encoding this one input text
  GET  /brands/{id}/vocabulary     what the brand owns vs. what the category owns
  POST /brands/{id}/embeddings     submit assets + a reference to an industry corpus,
                                    generate embeddings in the background, hot-load
                                    the result into this process on success
  GET  /jobs/{job_id}              poll a submitted job's status/stage/warnings/error
  GET  /industries                 list registered industry corpora + which brands use each
  GET  /industries/{industry_id}   one industry corpus's stats

To add or refresh a brand offline: run vector_generation/generate_embeddings.py,
then restart the service. To do it live, without a restart, use
POST /brands/{id}/embeddings and poll GET /jobs/{job_id} — see README for the
job lifecycle, error codes and warning codes.

    uvicorn api:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from loci.fingerprint import InputCopy, Layer, ScorabilityError
from loci.scorer import BrandDistinctivenessScorer
from loci.vector_store import BrandVectorStore
from vector_generation import industries, jobs
from vector_generation.generate_embeddings import generate_from_dicts
from vector_generation.jobs import JobRecord, JobStage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("loci.api")

EMBEDDINGS_ROOT = Path(__file__).parent / "vector_generation" / "embeddings"
INDUSTRIES_ROOT = Path(__file__).parent / "vector_generation" / "industries"

# In-memory cache: brand_id -> scorer. Populated at startup from disk, and
# updated in place by _run_embedding_job() whenever a live job succeeds —
# no restart required to pick up a freshly generated brand.
_SCORERS: dict[str, BrandDistinctivenessScorer] = {}


def _load_all_brands() -> None:
    if not EMBEDDINGS_ROOT.exists():
        return
    for folder in sorted(p for p in EMBEDDINGS_ROOT.iterdir() if p.is_dir()):
        store = BrandVectorStore.load(folder)
        _SCORERS[store.manifest["brand_id"]] = BrandDistinctivenessScorer(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    industries.load_all(INDUSTRIES_ROOT)
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
            f"Unknown brand_id '{brand_id}'. Submit it via POST /brands/{brand_id}/embeddings, "
            f"or run 'python -m vector_generation.generate_embeddings' and restart the service.",
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


class GenericCorpusRef(BaseModel):
    industry: str
    # Only meaningful when `industry` isn't registered yet — that's the one
    # case this call is allowed to create it. If `industry` already exists,
    # items (if sent anyway) are ignored; the registry is never overwritten.
    items: list[dict] | None = None


class EmbeddingsRequest(BaseModel):
    brand_name: str
    assets: dict[str, list[str] | str]
    generic_corpus: GenericCorpusRef


def _resolve_industry(job_id: str, ref: GenericCorpusRef) -> dict:
    """
    Raises ValueError (-> unknown_industry) if `ref.industry` isn't registered
    and no `items` were given to create it. Never overwrites an existing
    industry — if it exists and `items` was also sent, that's a warning, not
    a write.
    """
    record, created = industries.get_or_create(INDUSTRIES_ROOT, ref.industry, ref.items)
    if not created and ref.items:
        jobs.add_warning(
            job_id, "industry_corpus_ignored",
            f"Industry '{ref.industry}' already exists; submitted items were not used.",
        )
    return {"industry": record["industry_id"], "items": record["items"]}


def _run_embedding_job(job_id: str, brand_id: str, req: EmbeddingsRequest) -> None:
    jobs.start_job(job_id)
    try:
        generic = _resolve_industry(job_id, req.generic_corpus)
    except ValueError as e:
        jobs.fail_job(job_id, "unknown_industry", str(e))
        return
    except Exception as e:
        logger.exception("job %s failed unexpectedly while resolving industry", job_id)
        jobs.fail_job(job_id, "internal_error", str(e))
        return

    brand = {"brand_id": brand_id, "brand_name": req.brand_name, "assets": req.assets}

    try:
        out_dir, manifest = generate_from_dicts(
            brand, generic, EMBEDDINGS_ROOT,
            on_stage=lambda stage: jobs.update_stage(job_id, stage),
            strict=True,
        )
    except ScorabilityError as e:
        jobs.fail_job(job_id, "mvbf_not_met", e.message, fields=e.fields)
        return
    except ValueError as e:
        jobs.fail_job(job_id, "unmapped_asset_type", str(e))
        return
    except Exception as e:
        logger.exception("job %s failed unexpectedly", job_id)
        jobs.fail_job(job_id, "internal_error", str(e))
        return

    for w in manifest.get("warnings", []):
        jobs.add_warning(job_id, w["code"], w["message"])

    store = BrandVectorStore.load(out_dir)
    _SCORERS[brand_id] = BrandDistinctivenessScorer(store)

    jobs.complete_job(job_id)


@app.post("/brands/{brand_id}/embeddings", status_code=202)
def submit_embeddings(brand_id: str, req: EmbeddingsRequest, background_tasks: BackgroundTasks) -> dict:
    job_id = str(uuid4())
    if not jobs.try_acquire_brand_lock(brand_id, job_id):
        raise HTTPException(409, f"A job is already in flight for brand '{brand_id}'.")
    jobs.create_job(job_id, brand_id)
    background_tasks.add_task(_run_embedding_job, job_id, brand_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> JobRecord:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'.")
    return job


@app.get("/industries")
def list_industries() -> dict:
    brands_by_industry: dict[str, list[str]] = {}
    for bid, s in _SCORERS.items():
        brands_by_industry.setdefault(s.store.manifest["industry"], []).append(bid)
    return {
        "industries": [
            {
                "industry_id": rec["industry_id"],
                "chunk_count": rec["chunk_count"],
                "brands_using": sorted(brands_by_industry.get(rec["industry_id"], [])),
            }
            for rec in industries.list_industries()
        ]
    }


@app.get("/industries/{industry_id}")
def get_industry(industry_id: str) -> dict:
    record = industries.get_industry(industry_id)
    if record is None:
        raise HTTPException(404, f"Unknown industry_id '{industry_id}'.")
    return record
