"""Pydantic v2 models for the AI Code Reviewer.

Defines schemas for incoming GitHub webhook payloads and
outgoing review results posted back as PR comments.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── GitHub Webhook Payload Models ────────────────────────────────────────────


class PRHead(BaseModel):
    """Head branch metadata from a pull-request event."""

    sha: str = Field(..., description="SHA of the head commit")
    ref: str = Field(..., description="Branch name of the head")


class PRBase(BaseModel):
    """Base branch metadata from a pull-request event."""

    sha: str = Field(..., description="SHA of the base commit")
    ref: str = Field(..., description="Branch name of the base")


class PullRequest(BaseModel):
    """Subset of pull-request fields we actually need."""

    number: int = Field(..., description="PR number")
    title: str = Field(..., description="PR title")
    head: PRHead
    base: PRBase
    diff_url: str = Field(..., description="URL to fetch the unified diff")
    html_url: str = Field(..., description="Human-readable PR URL")


class Repository(BaseModel):
    """Repository metadata from the webhook payload."""

    full_name: str = Field(
        ...,
        description="Owner/repo slug, e.g. 'octocat/Hello-World'",
    )
    clone_url: str = Field(
        default="",
        description="HTTPS clone URL (optional, used for RAG indexing)",
    )


class GitHubPREvent(BaseModel):
    """Top-level schema for a GitHub `pull_request` webhook event.

    Reference:
        https://docs.github.com/en/webhooks/webhook-events-and-payloads#pull_request
    """

    action: str = Field(
        ...,
        description="Webhook action, e.g. 'opened', 'synchronize', 'closed'",
    )
    pull_request: PullRequest
    repository: Repository


# ── Review Result Models ─────────────────────────────────────────────────────


class ReviewIssue(BaseModel):
    """A single issue found during code review."""

    severity: Literal["critical", "warning", "suggestion"] = Field(
        ...,
        description="How serious the issue is",
    )
    issue: str = Field(..., description="What the problem is")
    suggestion: str = Field(..., description="How to fix it")
    line_number: int = Field(
        ...,
        ge=1,
        description="1-based line number in the diff where the issue occurs",
    )
    file_path: str = Field(
        ...,
        description="Path of the file relative to the repo root",
    )


class ReviewResult(BaseModel):
    """Aggregated review output for an entire pull request."""

    pr_number: int = Field(..., description="PR number that was reviewed")
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description="All issues found across every file in the diff",
    )
    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Quality score from 0 (terrible) to 100 (perfect)",
    )
    summary: str = Field(
        ...,
        description="Human-readable summary of the review",
    )
