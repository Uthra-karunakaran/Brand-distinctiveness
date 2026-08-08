"""
Shared operational plumbing for the public demo deployment: a maintenance-mode
kill switch, a client-key check that filters bots/scanners hitting the API
directly (not real security — anything shipped in a public SPA bundle is
readable, just a coarse filter), a daily per-IP and per-visitor request cap
backed by Upstash Redis (layered on top of the existing in-memory slowapi
per-minute limiter — slowapi itself stays in-memory since its Redis backend
needs a raw TCP connection string, not Upstash's REST credentials), and the
usage counters an operator reads directly from the Upstash console.

Every Redis-touching function fails open: if UPSTASH_REDIS_REST_URL/TOKEN
aren't set (e.g. local dev) or Upstash is briefly unreachable, guardrails and
stats become no-ops rather than 500s — this is a public demo, not a system
where losing a rate-limit check for a few seconds is dangerous.
"""
from __future__ import annotations

import logging
import os
import secrets
import time

from fastapi import Header, HTTPException, Request
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("platform_admin")

# ---------- Upstash client ----------

_redis = None
try:
    if os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get("UPSTASH_REDIS_REST_TOKEN"):
        from upstash_redis import Redis as _UpstashRedis
        _redis = _UpstashRedis.from_env()
except Exception:
    logger.warning("Upstash Redis unavailable at startup; guardrails/stats degrade to no-ops", exc_info=True)


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


# ---------- maintenance-mode kill switch ----------

class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """503s every request except the health check when MAINTENANCE_MODE is
    truthy, so it can be flipped from the Render dashboard (which restarts
    the process on env-var save) without a git push/rebuild."""

    async def dispatch(self, request: Request, call_next):
        flag = os.environ.get("MAINTENANCE_MODE", "").strip().lower()
        if flag in ("1", "true", "yes", "on") and request.url.path != "/":
            return JSONResponse(
                {"detail": "This demo is temporarily in maintenance mode. Please try again shortly."},
                status_code=503,
            )
        return await call_next(request)


# ---------- client-key check (bot/scanner filter, not real auth) ----------

def require_client_key(x_client_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("CLIENT_KEY")
    if not expected:
        return  # not configured (e.g. local dev) — skip the check
    if not x_client_key or not secrets.compare_digest(x_client_key, expected):
        raise HTTPException(401, "Missing or invalid X-Client-Key header.")


# ---------- daily per-IP / per-visitor cap ----------

def _subject_value(request: Request, subject: str) -> str | None:
    if subject == "ip":
        return get_remote_address(request)
    if subject == "visitor":
        return request.headers.get("x-visitor-id")
    raise ValueError(f"unknown subject kind: {subject!r}")


def make_daily_cap(service: str, subject: str, env_var: str, default: int):
    """Returns a FastAPI dependency enforcing a per-`subject`-per-UTC-day
    request cap for `service`, on top of the existing per-minute slowapi
    limit. `subject` is "ip" or "visitor" — a missing visitor ID (e.g. a
    direct curl request with no frontend JS) skips enforcement rather than
    blocking the request, since the cap is a volume guard, not a gate."""
    cap = int(os.environ.get(env_var, str(default)))

    def _dependency(request: Request) -> None:
        if _redis is None:
            return
        value = _subject_value(request, subject)
        if value is None:
            return
        key = f"guard:{service}:daily:{_today()}:{subject}:{value}"
        try:
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 26 * 3600)  # buffer past UTC midnight rollover
        except Exception:
            logger.warning("Upstash unreachable; failing open on daily %s cap", subject, exc_info=True)
            return
        if count > cap:
            record_rate_limit_rejection(service)
            raise HTTPException(
                429,
                f"Daily {subject} limit ({cap} requests) reached for this endpoint. Try again after UTC midnight.",
            )

    return _dependency


# ---------- stats recording (read directly in the Upstash console) ----------

_DAILY_TTL_SECONDS = 400 * 86400  # keep trend buckets for >1 year, not forever


def record_visitor_seen(request: Request) -> None:
    if _redis is None:
        return
    visitor_id = request.headers.get("x-visitor-id")
    if not visitor_id:
        return
    try:
        _redis.sadd(f"stats:daily:{_today()}:visitors", visitor_id)
    except Exception:
        logger.warning("failed to record visitor-seen stat", exc_info=True)


def record_score_call() -> None:
    if _redis is None:
        return
    try:
        pipe = _redis.pipeline()
        pipe.incr("stats:scorer:score_total")
        pipe.hincrby(f"stats:daily:{_today()}", "score_calls", 1)
        pipe.expire(f"stats:daily:{_today()}", _DAILY_TTL_SECONDS)
        pipe.exec()
    except Exception:
        logger.warning("failed to record score-call stat", exc_info=True)


def record_llm_usage(input_tokens: int, output_tokens: int) -> None:
    if _redis is None:
        return
    # Claude Sonnet 4.6: $3/$15 per million input/output tokens.
    cost = (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0
    day = _today()
    try:
        pipe = _redis.pipeline()
        pipe.incr("stats:llm:calls_total")
        pipe.incrby("stats:llm:input_tokens_total", input_tokens)
        pipe.incrby("stats:llm:output_tokens_total", output_tokens)
        pipe.incrbyfloat("stats:llm:cost_usd_total", cost)
        pipe.hincrby(f"stats:daily:{day}", "llm_calls", 1)
        pipe.hincrby(f"stats:daily:{day}", "llm_input_tokens", input_tokens)
        pipe.hincrby(f"stats:daily:{day}", "llm_output_tokens", output_tokens)
        pipe.hincrbyfloat(f"stats:daily:{day}", "llm_cost_usd", cost)
        pipe.exec()
    except Exception:
        logger.warning("failed to record LLM usage stat", exc_info=True)


def record_rate_limit_rejection(service: str) -> None:
    if _redis is None:
        return
    try:
        pipe = _redis.pipeline()
        pipe.incr(f"stats:{service}:rejected_ratelimit_total")
        pipe.hincrby(f"stats:daily:{_today()}", "rate_limit_rejections", 1)
        pipe.exec()
    except Exception:
        logger.warning("failed to record rate-limit rejection stat", exc_info=True)
