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
import os
import re
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi import _rate_limit_exceeded_handler as _default_rate_limit_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

import platform_admin
from loci.fingerprint import ASSET_LAYER_MAP, MVBF_FIELDS, InputCopy, Layer, ScorabilityError
from loci.metrics import tone as tone_metrics
from loci.scorer import LAYER_WEIGHTS, BrandDistinctivenessScorer
from loci.vector_store import BrandVectorStore
from vector_generation import industries, jobs
from vector_generation.generate_embeddings import generate_from_dicts
from vector_generation.jobs import JobRecord, JobStage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("loci.api")

DEMO_DISCLAIMER = (
    "This is a temporary demo deployment, not a production service. Data is "
    "held in memory and on local disk only — a redeploy, restart, or "
    "inactivity spin-down wipes any brand created via POST /brands/{id}/embeddings. "
    "Only brands baked into the deployment at build time are guaranteed to be there. "
    "Text submitted to POST /brands/{id}/score may be sent to Anthropic's Claude API "
    "for tone-of-voice judging (see /brands/{id}/score docs) — it falls back to a "
    "local heuristic once a daily call budget is reached or no API key is configured. "
    "Do not submit confidential or sensitive text."
)

limiter = Limiter(key_func=get_remote_address)

# Caps total concurrent embedding-generation jobs regardless of caller — a
# temp demo's CPU shouldn't be pegged by several different IPs at once, which
# per-IP rate limiting alone wouldn't catch.
_JOB_SLOTS = threading.Semaphore(2)

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


app = FastAPI(title="Loci — Brand Distinctiveness", description=DEMO_DISCLAIMER, lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(platform_admin.MaintenanceModeMiddleware)


def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    platform_admin.record_rate_limit_rejection("scorer")
    return _default_rate_limit_handler(request, exc)


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

_score_daily_ip_cap = platform_admin.make_daily_cap("scorer", "ip", "DAILY_IP_CAP_SCORE", default=300)
_score_daily_visitor_cap = platform_admin.make_daily_cap("scorer", "visitor", "DAILY_VISITOR_CAP_SCORE", default=150)

# No CORS middleware unless explicitly configured — set ALLOWED_ORIGINS
# (comma-separated) in the deployment's env to the actual deployed frontend
# origin(s) once that domain is known. Fails closed: cross-origin browser
# calls get no Access-Control-Allow-Origin header until this is set.
_allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "Loci — Brand Distinctiveness",
        "capabilities": [
            "GET /brands", "POST /brands/{id}/score", "GET /brands/{id}/vocabulary",
            "POST /brands/{id}/embeddings", "GET /jobs/{job_id}",
            "GET /industries", "GET /industries/{industry_id}",
        ],
        "disclaimer": DEMO_DISCLAIMER,
        "docs": "/docs",
    }


MAX_SCORE_TEXT_LENGTH = 4000  # matches the frontend's client-side cap (ScorerPage.jsx MAX_TEXT_LENGTH)


class ScoreRequest(BaseModel):
    text: str = Field(..., max_length=MAX_SCORE_TEXT_LENGTH)
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


class CreateBrandRequest(BaseModel):
    brand_name: str


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "brand"


def _mint_brand_id(brand_name: str) -> str:
    base = _slugify(brand_name)
    for _ in range(20):
        candidate = f"{base}-{secrets.token_hex(3)}"
        if candidate not in _SCORERS and not (EMBEDDINGS_ROOT / candidate).exists():
            return candidate
    raise HTTPException(500, "Could not mint a unique brand_id, try again.")


@app.post("/brands", status_code=201)
@limiter.limit("20/hour")
def create_brand(
    request: Request, req: CreateBrandRequest,
    _key: None = Depends(platform_admin.require_client_key),
) -> dict:
    return {"brand_id": _mint_brand_id(req.brand_name)}


@app.get("/brands/{brand_id}")
def get_brand(brand_id: str) -> dict:
    scorer = _get_scorer(brand_id)
    mvbf_status = scorer.fp.mvbf_status()
    missing_fields = [f for f, present in mvbf_status.items() if not present]
    return {
        "brand_id": brand_id,
        "brand_name": scorer.fp.brand_name,
        "industry": scorer.store.manifest["industry"],
        "mvbf": {"met": not missing_fields, "missing_fields": missing_fields},
        "layers_present": [l.value for l in scorer.fp.layers_present()],
        "scorable": scorer.store.manifest["scorable"],
        "scorable_message": scorer.store.manifest["scorable_message"],
        "signature_terms": sorted(scorer.lexical.signatures),
        "cliche_terms": sorted(scorer.lexical.cliches),
        "warnings": scorer.warnings,
    }


@app.post("/brands/{brand_id}/score")
@limiter.limit("60/minute")
def score_copy(
    request: Request, brand_id: str, req: ScoreRequest,
    _key: None = Depends(platform_admin.require_client_key),
    _ip_cap: None = Depends(_score_daily_ip_cap),
    _visitor_cap: None = Depends(_score_daily_visitor_cap),
    _visitor_seen: None = Depends(platform_admin.record_visitor_seen),
) -> dict:
    scorer = _get_scorer(brand_id)
    copy = InputCopy(
        text=req.text, intended_layer=req.intended_layer,
        channel=req.channel, label=req.label,
    )
    result = scorer.score(copy).to_dict()
    platform_admin.record_score_call()
    return result


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
    try:
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
    finally:
        _JOB_SLOTS.release()


@app.post("/brands/{brand_id}/embeddings", status_code=202)
@limiter.limit("5/hour")
def submit_embeddings(
    request: Request, brand_id: str, req: EmbeddingsRequest, background_tasks: BackgroundTasks,
    _key: None = Depends(platform_admin.require_client_key),
) -> dict:
    if not _JOB_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Server is at capacity for embedding generation right now — try again shortly.")
    job_id = str(uuid4())
    if not jobs.try_acquire_brand_lock(brand_id, job_id):
        _JOB_SLOTS.release()
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


@app.get("/schema/assets")
def asset_schema() -> dict:
    layers: dict[str, list[str]] = {}
    for asset_type, layer in ASSET_LAYER_MAP.items():
        layers.setdefault(layer.value, []).append(asset_type)
    return {
        "mvbf_fields": list(MVBF_FIELDS),
        "layers": layers,
        "scored_layers": sorted(l.value for l in LAYER_WEIGHTS),
    }
