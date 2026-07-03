"""FastAPI application for the AI Code Reviewer webhook server.

Receives GitHub ``pull_request`` webhook events, verifies the payload
signature (HMAC-SHA256), and dispatches accepted events to the review
pipeline as background tasks so the webhook responds within GitHub's
10-second window.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

import app.metrics  # noqa: F401 — registers all metrics with the default registry
from app.models import GitHubPREvent
from app.reviewer import handle_pr

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level readiness flag ──────────────────────────────────────────────

_chromadb_ready: bool = False


def _init_chromadb() -> None:
    """Initialise the ChromaDB vector store via :func:`app.rag.init_chromadb`.

    Sets the module-level ``_chromadb_ready`` flag to ``True`` once the
    collection is available.  On failure the flag stays ``False`` so the
    ``/ready`` probe returns 503.
    """
    global _chromadb_ready  # noqa: PLW0603
    try:
        from app.rag import init_chromadb

        init_chromadb()
        _chromadb_ready = True
    except Exception:
        logger.exception("Failed to initialise ChromaDB — /ready will return 503")
        _chromadb_ready = False


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler.

    * **Startup**: initialise ChromaDB so the ``/ready`` probe passes.
    * **Shutdown**: (reserved for future cleanup).
    """
    _init_chromadb()
    yield
    logger.info("Shutting down …")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Code Reviewer",
    description=(
        "Receives GitHub PR webhooks, analyses diffs with RAG + LLM, "
        "and posts inline review comments."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Prometheus /metrics endpoint ─────────────────────────────────────────────
# Mount the prometheus_client ASGI app at /metrics.  This is a sub-application
# so it bypasses FastAPI middleware but is served on the same port, which is
# what Prometheus expects when scraping a single target.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Helpers ──────────────────────────────────────────────────────────────────

ACTIONS_TO_PROCESS: set[str] = {"opened", "synchronize"}


async def _run_review(event: GitHubPREvent) -> None:
    """Safe background wrapper around :func:`app.reviewer.handle_pr`.

    Catches *all* unhandled exceptions so a bug in the review pipeline
    never propagates into Starlette's background-task runner and crashes
    the server process.  The PR number and repo name are always logged so
    failures are easy to correlate in production logs.

    Args:
        event: Validated GitHub webhook payload forwarded from the
            ``/webhook`` endpoint.
    """
    pr_number = event.pull_request.number
    repo = event.repository.full_name
    try:
        await handle_pr(event)
    except Exception:
        logger.exception(
            "Unhandled error in review pipeline for PR #%d on %s",
            pr_number,
            repo,
        )


def _verify_signature(payload: bytes, signature_header: str | None) -> None:
    """Validate the ``X-Hub-Signature-256`` header against the raw body.

    GitHub sends the header as ``sha256=<hex-digest>``.  We recompute the
    HMAC-SHA256 using ``WEBHOOK_SECRET`` and compare in constant time.

    Args:
        payload:          Raw request body bytes.
        signature_header: Value of the ``X-Hub-Signature-256`` header.

    Raises:
        HTTPException: 401 if the signature is missing, malformed, or invalid.
    """
    secret = os.environ.get("WEBHOOK_SECRET")
    if not secret:
        logger.error("WEBHOOK_SECRET environment variable is not set")
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: webhook secret not set",
        )

    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing signature header")

    # Header format: "sha256=<hex>"
    if not signature_header.startswith("sha256="):
        raise HTTPException(
            status_code=401,
            detail="Unsupported signature algorithm",
        )

    expected_sig = signature_header.removeprefix("sha256=")

    computed_sig = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(computed_sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns OK if the process is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe — returns 200 only after ChromaDB is initialised.

    Returns 503 (Service Unavailable) while the vector store is still
    starting up, so load-balancers won't route traffic prematurely.
    """
    if not _chromadb_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready"},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready"},
    )


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> dict[str, str]:
    """Receive and process a GitHub webhook event.

    1. Read the raw body and verify the HMAC-SHA256 signature.
    2. Parse the body as a ``GitHubPREvent``.
    3. Ignore actions we don't care about (e.g. ``closed``).
    4. Enqueue ``reviewer.handle_pr`` as a background task.
    5. Return ``{"status": "accepted"}`` immediately.

    Args:
        request:              The incoming HTTP request.
        background_tasks:     FastAPI background-task scheduler.
        x_hub_signature_256:  ``X-Hub-Signature-256`` header from GitHub.
        x_github_event:       ``X-GitHub-Event`` header (e.g. ``pull_request``).

    Returns:
        A JSON acknowledgement.
    """
    raw_body: bytes = await request.body()

    # 1. Signature verification
    _verify_signature(raw_body, x_hub_signature_256)

    # 2. Only handle pull_request events
    if x_github_event != "pull_request":
        logger.info("Ignoring event type: %s", x_github_event)
        return {"status": "ignored", "reason": f"event type '{x_github_event}' not handled"}

    # 3. Parse payload
    try:
        event = GitHubPREvent.model_validate_json(raw_body)
    except Exception as exc:
        logger.exception("Failed to parse webhook payload")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc

    # 4. Filter actions
    if event.action not in ACTIONS_TO_PROCESS:
        logger.info(
            "Ignoring action '%s' for PR #%d",
            event.action,
            event.pull_request.number,
        )
        return {"status": "ignored", "reason": f"action '{event.action}' not handled"}

    # 5. Dispatch to background
    logger.info(
        "Accepted PR #%d (%s) on %s — dispatching review",
        event.pull_request.number,
        event.action,
        event.repository.full_name,
    )
    background_tasks.add_task(_run_review, event)

    return {"status": "accepted"}


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
