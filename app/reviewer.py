"""Review orchestration — prompt construction, LLM calls, and end-to-end PR review.

Builds structured prompts from PR diffs and RAG context, sends them to Groq's
Llama 3.1 70B model, parses the JSON response into :class:`ReviewIssue` objects,
and aggregates everything into a :class:`ReviewResult`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from groq import Groq, RateLimitError

from app.github_client import post_review_comments
from app.metrics import (
    active_reviews,
    groq_api_calls_total,
    groq_latency_seconds,
    issues_found_total,
    pr_reviews_total,
    review_duration_seconds,
)
from app.models import GitHubPREvent, ReviewIssue, ReviewResult
from app.rag import get_pr_diff, index_repository, retrieve_context

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "llama-3.1-70b-versatile"
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 1500

SEVERITY_PENALTIES: dict[str, int] = {
    "critical": 20,
    "warning": 10,
    "suggestion": 3,
}

# ── Custom exceptions ───────────────────────────────────────────────────────


class GroqRateLimitError(Exception):
    """Raised when the Groq API returns HTTP 429 (rate limit exceeded).

    Attributes:
        retry_after: Suggested wait time in seconds, if provided by the API.
    """

    def __init__(self, message: str = "Groq rate limit exceeded", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ── Prompt construction ─────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """\
You are a senior software engineer performing a thorough code review.
Your job is to identify **real bugs, security vulnerabilities, performance \
problems, and logic errors** in the changed code.

Rules:
- Only flag genuine issues that could cause bugs, crashes, security holes, or \
significant performance degradation.
- Do NOT flag style preferences, naming conventions, or minor formatting issues.
- Every issue MUST reference a specific line number from the CHANGED CODE section.
- Respond with ONLY a JSON array of issue objects. No markdown, no explanation, \
no code fences — just the raw JSON array.
- If there are no issues, respond with an empty array: []

Each object in the array must have exactly these fields:
  "severity": one of "critical", "warning", "suggestion"
  "issue": a concise description of the problem
  "suggestion": a concrete fix or improvement
  "line_number": the 1-based line number in the changed code
  "file_path": the file path as shown in the CHANGED CODE header\
"""

_FEW_SHOT_EXAMPLES: str = """\

EXAMPLE 1 — Correct output format:
[
  {
    "severity": "critical",
    "issue": "SQL query built via string concatenation is vulnerable to SQL injection",
    "suggestion": "Use parameterised queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
    "line_number": 12,
    "file_path": "app/db.py"
  }
]

EXAMPLE 2 — Multiple issues:
[
  {
    "severity": "warning",
    "issue": "File handle opened but never closed; will leak file descriptors under load",
    "suggestion": "Wrap in a `with` statement: `with open(path) as f:`",
    "line_number": 5,
    "file_path": "utils/io.py"
  },
  {
    "severity": "suggestion",
    "issue": "Catching bare `Exception` masks unexpected errors and makes debugging harder",
    "suggestion": "Catch a specific exception, e.g. `except ValueError:` or `except (IOError, OSError):`",
    "line_number": 18,
    "file_path": "utils/io.py"
  }
]\
"""


def _number_lines(code: str, start: int = 1) -> str:
    """Add line numbers to a block of code for the prompt.

    Args:
        code:  Raw source-code string.
        start: Number for the first line.

    Returns:
        The same code with ``{n}: `` prepended to each line.
    """
    lines = code.splitlines()
    return "\n".join(f"{start + i}: {line}" for i, line in enumerate(lines))


def build_review_prompt(
    file_path: str,
    changed_lines: str,
    context_chunks: list[dict[str, Any]],
) -> str:
    """Construct the user-side prompt for a single file review.

    The prompt contains:
    1. The changed lines (with line numbers) under a CHANGED CODE header.
    2. Up to 3 RAG context chunks under RELATED CODE FOR CONTEXT headers.
    3. The two few-shot JSON examples showing the expected output format.

    Args:
        file_path:      Relative path of the file being reviewed.
        changed_lines:  Added lines extracted from the PR diff.
        context_chunks: RAG results — list of dicts with ``file_path``,
            ``content``, ``similarity_score``, ``start_line``.

    Returns:
        The full user prompt string (the system prompt is sent separately).
    """
    sections: list[str] = []

    # ── Changed code ────────────────────────────────────────────────
    numbered = _number_lines(changed_lines)
    sections.append(
        f"=== CHANGED CODE: {file_path} ===\n{numbered}"
    )

    # ── Context chunks (max 3) ──────────────────────────────────────
    for i, chunk in enumerate(context_chunks[:3]):
        ctx_path = chunk.get("file_path", "unknown")
        ctx_start = chunk.get("start_line", 1)
        ctx_score = chunk.get("similarity_score", 0.0)
        ctx_body = _number_lines(chunk.get("content", ""), start=ctx_start)
        sections.append(
            f"=== RELATED CODE FOR CONTEXT #{i + 1}: {ctx_path} "
            f"(similarity: {ctx_score:.2f}, starting line {ctx_start}) ===\n"
            f"{ctx_body}"
        )

    # ── Output instructions + few-shot ──────────────────────────────
    sections.append(
        "=== INSTRUCTIONS ===\n"
        "Respond with ONLY a JSON array of issue objects. "
        "No markdown fences, no commentary.\n"
        "Only flag real issues — not style preferences.\n"
        f"{_FEW_SHOT_EXAMPLES}"
    )

    return "\n\n".join(sections)


# ── LLM interaction ─────────────────────────────────────────────────────────


def _extract_json_array(text: str) -> str:
    """Extract the first JSON array from *text*, stripping markdown fences.

    Args:
        text: Raw LLM response that may contain code fences or preamble.

    Returns:
        The substring from the first ``[`` to the matching ``]``.

    Raises:
        ValueError: If no JSON array is found.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    # Find the outermost [ … ]
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM response")

    # Walk forward to find the matching ]
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]

    raise ValueError("Unbalanced brackets in LLM response")


def _parse_issues(raw_json: str, file_path: str) -> list[ReviewIssue]:
    """Parse a JSON string into a list of :class:`ReviewIssue`.

    Args:
        raw_json:  JSON array string from the LLM.
        file_path: Fallback file path if the LLM omits it.

    Returns:
        Validated list of ``ReviewIssue`` objects.

    Raises:
        ValueError: If the JSON is invalid or doesn't match the schema.
    """
    data = json.loads(raw_json)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    issues: list[ReviewIssue] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict item in LLM response: %r", item)
            continue
        # Ensure file_path is present
        item.setdefault("file_path", file_path)
        try:
            issues.append(ReviewIssue.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed issue %r: %s", item, exc)
    return issues


def call_groq(prompt: str, file_path: str = "") -> list[ReviewIssue]:
    """Send a review prompt to the Groq API and parse the response.

    Uses Llama 3.1 70B with low temperature for deterministic output.
    On JSON parse failure the function retries **once** with an appended
    correction message.  On second failure it returns an empty list.

    Args:
        prompt:    The user-side prompt (system prompt is added internally).
        file_path: File path for fallback when parsing issues.

    Returns:
        List of validated :class:`ReviewIssue` objects (may be empty).

    Raises:
        GroqRateLimitError: If the Groq API responds with HTTP 429.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set")
        return []

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        _t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except RateLimitError as exc:
            groq_latency_seconds.observe(time.perf_counter() - _t0)
            groq_api_calls_total.labels(status="rate_limited").inc()
            retry_after = getattr(exc, "retry_after", None)
            logger.error("Groq rate limit hit: %s", exc)
            raise GroqRateLimitError(
                message=str(exc),
                retry_after=retry_after,
            ) from exc
        except Exception:
            groq_latency_seconds.observe(time.perf_counter() - _t0)
            groq_api_calls_total.labels(status="error").inc()
            logger.exception("Groq API call failed (attempt %d)", attempt + 1)
            return []

        groq_latency_seconds.observe(time.perf_counter() - _t0)
        raw_text = response.choices[0].message.content or ""
        logger.debug("Groq response (attempt %d): %.500s", attempt + 1, raw_text)

        try:
            json_str = _extract_json_array(raw_text)
            parsed = _parse_issues(json_str, file_path)
            groq_api_calls_total.labels(status="success").inc()
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 0:
                logger.warning(
                    "JSON parse failed on attempt 1 (%s) — retrying with correction",
                    exc,
                )
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Fix your JSON syntax and respond with ONLY a "
                            "valid JSON array of issue objects. No markdown, "
                            "no commentary — just the raw JSON array."
                        ),
                    },
                )
            else:
                groq_api_calls_total.labels(status="error").inc()
                logger.error(
                    "JSON parse failed on attempt 2 (%s) — returning empty list. "
                    "Raw response: %.500s",
                    exc,
                    raw_text,
                )
                return []

    return []  # unreachable, but keeps mypy happy


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_review(issues: list[ReviewIssue]) -> int:
    """Calculate a 0–100 quality score based on issue count and severity.

    Scoring starts at 100 and deducts:
    * **Critical** — 20 points each
    * **Warning** — 10 points each
    * **Suggestion** — 3 points each

    The score is clamped to a minimum of 0.

    Args:
        issues: List of review issues found in the PR.

    Returns:
        Integer score between 0 and 100 (inclusive).
    """
    score = 100
    for issue in issues:
        penalty = SEVERITY_PENALTIES.get(issue.severity, 0)
        score -= penalty
    return max(0, score)


# ── End-to-end orchestration ─────────────────────────────────────────────────


async def handle_pr(event: GitHubPREvent) -> ReviewResult:
    """Orchestrate a full review of a pull request.

    Pipeline:
        1. Ensure the repository is indexed in ChromaDB.
        2. Fetch the PR diff (added lines per file).
        3. For each changed file, retrieve similar context via RAG.
        4. Build a prompt and call the LLM.
        5. Aggregate all issues, compute a quality score.
        6. Post inline comments back to GitHub.

    Metrics recorded:
        * :data:`~app.metrics.active_reviews` — gauge incremented on entry,
          decremented on exit via ``try/finally``.
        * :data:`~app.metrics.review_duration_seconds` — wall-clock time for
          the entire pipeline.
        * :data:`~app.metrics.pr_reviews_total` — labelled ``success`` or
          ``error`` depending on outcome.
        * :data:`~app.metrics.issues_found_total` — one increment per issue.

    Args:
        event: Validated GitHub webhook payload.

    Returns:
        A :class:`ReviewResult` with all issues, score, and summary.
    """
    repo = event.repository.full_name
    pr_number = event.pull_request.number
    token = os.environ.get("GITHUB_TOKEN", "")

    logger.info("Starting review of PR #%d on %s", pr_number, repo)
    active_reviews.inc()
    _wall_start = time.perf_counter()

    try:
        # 1. Index the repo (idempotent — upserts existing chunks)
        try:
            indexed = index_repository(repo, token)
            logger.info("Repository index: %d chunks", indexed)
        except Exception:
            logger.exception("Failed to index repository %s — proceeding without RAG context", repo)

        # 2. Fetch the diff
        try:
            changed_files = get_pr_diff(repo, pr_number, token)
        except Exception:
            logger.exception("Failed to fetch diff for PR #%d", pr_number)
            pr_reviews_total.labels(repo=repo, status="error").inc()
            review_duration_seconds.labels(repo=repo).observe(
                time.perf_counter() - _wall_start
            )
            return ReviewResult(
                pr_number=pr_number,
                issues=[],
                overall_score=100,
                summary="Failed to fetch PR diff — no review performed.",
            )

        if not changed_files:
            logger.info("PR #%d has no reviewable changed files", pr_number)
            pr_reviews_total.labels(repo=repo, status="success").inc()
            review_duration_seconds.labels(repo=repo).observe(
                time.perf_counter() - _wall_start
            )
            return ReviewResult(
                pr_number=pr_number,
                issues=[],
                overall_score=100,
                summary="No reviewable code changes found in this PR.",
            )

        # 3–4. For each file: retrieve context → build prompt → call LLM
        all_issues: list[ReviewIssue] = []

        for file_path, added_code in changed_files.items():
            logger.info("Reviewing file: %s", file_path)

            # 3. RAG context retrieval
            try:
                context_chunks = retrieve_context(
                    {file_path: added_code},
                    repo,
                    top_k=3,
                )
            except Exception:
                logger.exception("RAG retrieval failed for %s — reviewing without context", file_path)
                context_chunks = []

            # 4. Build prompt and call LLM
            prompt = build_review_prompt(file_path, added_code, context_chunks)

            try:
                file_issues = call_groq(prompt, file_path=file_path)
                all_issues.extend(file_issues)
                logger.info(
                    "  %s: %d issues found",
                    file_path,
                    len(file_issues),
                )
            except GroqRateLimitError:
                logger.error(
                    "Rate limited by Groq while reviewing %s — stopping review",
                    file_path,
                )
                break
            except Exception:
                logger.exception("LLM call failed for %s", file_path)

        # 5. Score and record per-issue metrics
        overall_score = score_review(all_issues)

        for issue in all_issues:
            issues_found_total.labels(repo=repo, severity=issue.severity).inc()

        # Build summary
        critical = sum(1 for i in all_issues if i.severity == "critical")
        warnings = sum(1 for i in all_issues if i.severity == "warning")
        suggestions = sum(1 for i in all_issues if i.severity == "suggestion")

        summary_parts: list[str] = [
            f"Reviewed {len(changed_files)} file(s) in PR #{pr_number}.",
            f"Found {len(all_issues)} issue(s): "
            f"{critical} critical, {warnings} warning(s), {suggestions} suggestion(s).",
            f"Overall quality score: {overall_score}/100.",
        ]
        summary = " ".join(summary_parts)

        result = ReviewResult(
            pr_number=pr_number,
            issues=all_issues,
            overall_score=overall_score,
            summary=summary,
        )

        # 6. Post comments to GitHub
        try:
            post_review_comments(
                repo_full_name=repo,
                pr_number=pr_number,
                issues=all_issues,
                overall_score=overall_score,
                summary=summary,
                github_token=token,
            )
            logger.info("Review comments posted to PR #%d", pr_number)
        except Exception:
            logger.exception("Failed to post review comments to PR #%d", pr_number)

        # Record success metrics
        pr_reviews_total.labels(repo=repo, status="success").inc()
        review_duration_seconds.labels(repo=repo).observe(
            time.perf_counter() - _wall_start
        )

        logger.info(
            "Review complete for PR #%d — score: %d/100, issues: %d",
            pr_number,
            overall_score,
            len(all_issues),
        )

        return result

    except Exception:
        # Any unhandled exception in the pipeline
        pr_reviews_total.labels(repo=repo, status="error").inc()
        review_duration_seconds.labels(repo=repo).observe(
            time.perf_counter() - _wall_start
        )
        raise

    finally:
        active_reviews.dec()
