"""Prometheus metrics definitions for the AI Code Reviewer.

All metrics are created as module-level singletons so they are registered
exactly once with the default ``CollectorRegistry``.  Import this module
early (it is imported by ``main.py`` at startup) so every metric exists
before the first scrape.

Metric catalogue
----------------
pr_reviews_total
    Counter.  Incremented once per completed review.
    Labels: ``repo`` (owner/name), ``status`` (success | error).

review_duration_seconds
    Histogram.  Wall-clock time from webhook receipt to GitHub comment posted.
    Labels: ``repo``.

issues_found_total
    Counter.  One increment per issue emitted by the LLM.
    Labels: ``repo``, ``severity`` (critical | warning | suggestion).

groq_api_calls_total
    Counter.  One increment per Groq API attempt (not per review).
    Labels: ``status`` (success | rate_limited | error).

groq_latency_seconds
    Histogram.  Round-trip time for a single Groq ``chat.completions.create``
    call (excluding the JSON-retry round-trip).

active_reviews
    Gauge.  Number of PR reviews currently in progress (incremented on entry,
    decremented on exit regardless of success/failure).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Review lifecycle ──────────────────────────────────────────────────────────

pr_reviews_total: Counter = Counter(
    name="pr_reviews_total",
    documentation="Total number of PR reviews completed, labelled by outcome.",
    labelnames=["repo", "status"],
)
"""Incremented once when ``handle_pr`` finishes (status=success) or raises
an unhandled exception (status=error)."""

review_duration_seconds: Histogram = Histogram(
    name="review_duration_seconds",
    documentation=(
        "End-to-end wall-clock time (seconds) for a full PR review, "
        "from webhook received to GitHub comment posted."
    ),
    labelnames=["repo"],
    # Buckets tuned for typical LLM review latencies (1 s – 120 s)
    buckets=(1, 2, 5, 10, 20, 30, 45, 60, 90, 120, float("inf")),
)

active_reviews: Gauge = Gauge(
    name="active_reviews",
    documentation="Number of PR reviews currently being processed.",
)
"""Incremented at the start of ``handle_pr``, decremented at the end
(via a ``try/finally`` block) regardless of outcome."""

# ── Issue counts ─────────────────────────────────────────────────────────────

issues_found_total: Counter = Counter(
    name="issues_found_total",
    documentation="Total number of code issues found across all reviews.",
    labelnames=["repo", "severity"],
)
"""Incremented once per :class:`~app.models.ReviewIssue` emitted.
The ``severity`` label carries ``critical``, ``warning``, or ``suggestion``."""

# ── Groq API ─────────────────────────────────────────────────────────────────

groq_api_calls_total: Counter = Counter(
    name="groq_api_calls_total",
    documentation="Total Groq API call attempts, labelled by outcome.",
    labelnames=["status"],
)
"""Labels:
* ``success``      — response received and JSON parsed successfully.
* ``rate_limited`` — API returned 429.
* ``error``        — any other exception (network, timeout, etc.).
"""

groq_latency_seconds: Histogram = Histogram(
    name="groq_latency_seconds",
    documentation="Round-trip latency (seconds) for a single Groq API call.",
    # Buckets tuned for LLM inference latencies (100 ms – 60 s)
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, float("inf")),
)
