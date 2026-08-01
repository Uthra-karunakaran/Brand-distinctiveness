"""
Industry corpus registry — an immutable, append-only store of generic /
competitor corpora, shared across every brand in that industry.

This is what makes a brand's generic_corpus.industry a *reference* instead of
a *copy*: the first brand (or the CLI) to name an industry registers its
competitor corpus once; every later brand in that same industry just points
at the industry_id and reuses it. Registering is one-way — an industry_id,
once created, is never edited or deleted. To change a corpus, create a new
industry_id (e.g. "outdoor_gear_apparel_v2") and re-point the brands that
should pick it up; brands still referencing the old id are left untouched.

Mirrors vector_generation/jobs.py's in-memory + on-disk pattern: state is
loaded from disk once at startup (api.py's lifespan) and updated in place as
new industries get created.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("loci.industries")

_INDUSTRIES: dict[str, dict] = {}
_LOCK = threading.Lock()


def load_all(root: Path) -> None:
    if not root.exists():
        return
    for f in sorted(root.glob("*.json")):
        record = json.loads(f.read_text())
        _INDUSTRIES[record["industry_id"]] = record


def list_industries() -> list[dict]:
    return sorted(_INDUSTRIES.values(), key=lambda r: r["industry_id"])


def get_industry(industry_id: str) -> dict | None:
    return _INDUSTRIES.get(industry_id)


def get_or_create(root: Path, industry_id: str, items: list[dict] | None) -> tuple[dict, bool]:
    """
    Returns (record, created).

    - industry_id already registered -> returns the EXISTING record unchanged,
      created=False, regardless of what `items` was passed (never overwrite).
    - industry_id not registered and `items` given -> creates it, created=True.
    - industry_id not registered and no `items` -> raises ValueError.

    The whole check-then-create sequence is held under one lock so two
    concurrent first-time requests for the same brand-new industry_id can't
    both pass the "doesn't exist yet" check and race each other's write.
    """
    with _LOCK:
        existing = _INDUSTRIES.get(industry_id)
        if existing is not None:
            return existing, False
        if not items:
            raise ValueError(f"Industry '{industry_id}' is not registered.")

        record = {"industry_id": industry_id, "items": items, "chunk_count": len(items)}
        root.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".tmp-{industry_id}-", suffix=".json", dir=root)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, indent=2)
            Path(tmp_path).replace(root / f"{industry_id}.json")
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        _INDUSTRIES[industry_id] = record
        logger.info("industry '%s' created (%d chunks)", industry_id, len(items))
        return record, True
