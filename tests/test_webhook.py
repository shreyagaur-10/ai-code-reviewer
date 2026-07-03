"""Webhook endpoint tests for the AI Code Reviewer FastAPI app.

Tests cover:
- Valid HMAC-SHA256 signature → 200 accepted
- Invalid signature → 401
- Missing signature header → 401
- Non-PR GitHub event (push) → ignored gracefully
- PR action "closed" → ignored
- PR action "opened" → accepted and dispatched
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Provide env vars before importing the app so the import doesn't crash
os.environ.setdefault("WEBHOOK_SECRET", "test-secret-12345")
os.environ.setdefault("GITHUB_TOKEN", "fake-gh-token")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma-test")

# Patch ChromaDB init before importing the app
with patch("app.rag.init_chromadb", return_value=None):
    from app.main import app  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────

WEBHOOK_SECRET = "test-secret-12345"

_VALID_PR_PAYLOAD: dict = {
    "action": "opened",
    "pull_request": {
        "number": 42,
        "title": "Add new feature",
        "head": {"sha": "abc123", "ref": "feature-branch"},
        "base": {"sha": "def456", "ref": "main"},
        "diff_url": "https://github.com/octocat/Hello-World/pull/42.diff",
        "html_url": "https://github.com/octocat/Hello-World/pull/42",
    },
    "repository": {
        "full_name": "octocat/Hello-World",
        "clone_url": "https://github.com/octocat/Hello-World.git",
    },
}


def _make_signature(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    """Compute the X-Hub-Signature-256 header value for *body*."""
    mac = hmac.new(secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


@pytest_asyncio.fixture
async def client():
    """Async HTTPX client bound to the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Health / ready probes ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    """GET /health must always return 200 with status ok."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["version"] == "1.0.0"


# ── Webhook signature verification ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_signature_accepted(client: AsyncClient) -> None:
    """A correctly signed PR opened event must be accepted with HTTP 200."""
    body = json.dumps(_VALID_PR_PAYLOAD).encode()
    sig = _make_signature(body)

    with patch("app.main._run_review", new_callable=AsyncMock) as mock_review:
        resp = await client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "accepted"}


@pytest.mark.asyncio
async def test_invalid_signature_returns_401(client: AsyncClient) -> None:
    """A webhook with a wrong signature must be rejected with HTTP 401."""
    body = json.dumps(_VALID_PR_PAYLOAD).encode()

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=deadbeefdeadbeefdeadbeefdeadbeef",
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 401
    assert "Invalid signature" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_missing_signature_returns_401(client: AsyncClient) -> None:
    """A webhook with no signature header must be rejected with HTTP 401."""
    body = json.dumps(_VALID_PR_PAYLOAD).encode()

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_wrong_algorithm_returns_401(client: AsyncClient) -> None:
    """A signature using sha1= instead of sha256= must be rejected."""
    body = json.dumps(_VALID_PR_PAYLOAD).encode()
    mac = hmac.new(WEBHOOK_SECRET.encode(), msg=body, digestmod=hashlib.sha1)
    bad_sig = f"sha1={mac.hexdigest()}"

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": bad_sig,
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 401


# ── Event type filtering ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_pr_event_is_ignored(client: AsyncClient) -> None:
    """A 'push' event must be gracefully ignored (not crash, not dispatch)."""
    # For a push event, the payload can be anything valid-ish
    body = json.dumps({"ref": "refs/heads/main", "commits": []}).encode()
    sig = _make_signature(body)

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": "push",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ignored"
    assert "push" in data["reason"]


@pytest.mark.asyncio
async def test_pr_closed_action_is_ignored(client: AsyncClient) -> None:
    """A PR 'closed' event must be ignored — we only handle opened/synchronize."""
    payload = {**_VALID_PR_PAYLOAD, "action": "closed"}
    body = json.dumps(payload).encode()
    sig = _make_signature(body)

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ignored"
    assert "closed" in data["reason"]


@pytest.mark.asyncio
async def test_pr_synchronize_action_is_accepted(client: AsyncClient) -> None:
    """A PR 'synchronize' event (new push to existing PR) must be dispatched."""
    payload = {**_VALID_PR_PAYLOAD, "action": "synchronize"}
    body = json.dumps(payload).encode()
    sig = _make_signature(body)

    with patch("app.main._run_review", new_callable=AsyncMock):
        resp = await client.post(
            "/webhook",
            content=body,
            headers={
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "Content-Type": "application/json",
            },
        )

    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted"}


@pytest.mark.asyncio
async def test_malformed_payload_returns_400(client: AsyncClient) -> None:
    """A correctly signed but structurally invalid payload must return 400."""
    body = b'{"action": "opened", "missing_required_fields": true}'
    sig = _make_signature(body)

    resp = await client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": sig,
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        },
    )

    assert resp.status_code == 400
