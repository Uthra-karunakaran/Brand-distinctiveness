"""
Job tracking for live (in-process) embedding generation, triggered via
api.py's POST /brands/{id}/embeddings.

In-memory only, mirroring api.py's _SCORERS pattern: everything that needs to
survive a restart already lives on disk under vector_generation/embeddings/;
job bookkeeping does not need to. If jobs ever need to survive a process
restart or run across multiple instances, replace this module with a real
queue/store — not needed for the prototype.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger("loci.jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


class JobStage(str, Enum):
    FINGERPRINTING = "fingerprinting"
    FITTING_EMBEDDER = "fitting_embedder"
    ENCODING_VECTORS = "encoding_vectors"
    BUILDING_INDEX = "building_index"
    WRITING_MANIFEST = "writing_manifest"


class JobWarning(BaseModel):
    code: str
    message: str


class JobError(BaseModel):
    code: str
    message: str
    fields: list[str] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    brand_id: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage | None = None
    warnings: list[JobWarning] = Field(default_factory=list)
    error: JobError | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


_JOBS: dict[str, JobRecord] = {}
_BRAND_LOCKS: dict[str, str] = {}  # brand_id -> job_id currently in flight
_LOCK = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def try_acquire_brand_lock(brand_id: str, job_id: str) -> bool:
    """Returns False if a job is already in flight for this brand_id."""
    with _LOCK:
        if brand_id in _BRAND_LOCKS:
            return False
        _BRAND_LOCKS[brand_id] = job_id
        return True


def release_brand_lock(brand_id: str) -> None:
    with _LOCK:
        _BRAND_LOCKS.pop(brand_id, None)


def create_job(job_id: str, brand_id: str) -> JobRecord:
    job = JobRecord(job_id=job_id, brand_id=brand_id, created_at=_now())
    _JOBS[job_id] = job
    logger.info("job %s created for brand '%s'", job_id, brand_id)
    return job


def start_job(job_id: str) -> None:
    job = _JOBS[job_id]
    job.status = JobStatus.RUNNING
    job.started_at = _now()


def update_stage(job_id: str, stage: JobStage) -> None:
    job = _JOBS[job_id]
    job.stage = stage
    logger.info("job %s -> stage '%s'", job_id, stage.value)


def add_warning(job_id: str, code: str, message: str) -> None:
    job = _JOBS[job_id]
    job.warnings.append(JobWarning(code=code, message=message))
    logger.warning("job %s warning [%s]: %s", job_id, code, message)


def complete_job(job_id: str) -> None:
    job = _JOBS[job_id]
    job.status = JobStatus.READY
    job.completed_at = _now()
    release_brand_lock(job.brand_id)
    logger.info("job %s ready", job_id)


def fail_job(job_id: str, code: str, message: str, fields: list[str] | None = None) -> None:
    job = _JOBS[job_id]
    job.status = JobStatus.FAILED
    job.error = JobError(code=code, message=message, fields=fields or [])
    job.completed_at = _now()
    release_brand_lock(job.brand_id)
    logger.error("job %s failed [%s]: %s", job_id, code, message)


def get_job(job_id: str) -> JobRecord | None:
    return _JOBS.get(job_id)
