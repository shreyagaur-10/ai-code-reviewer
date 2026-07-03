"""GitHub API client — post inline review comments and query repo metadata.

All public functions use PyGithub and include rate-limit handling:
if GitHub returns HTTP 403 with ``X-RateLimit-Reset``, we sleep until the
reset timestamp and retry the call once before giving up.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

from app.models import ReviewIssue

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Buffer added to reset wait time to absorb clock skew.
_RATE_LIMIT_BUFFER_SECS: int = 5

# GitHub review event strings.
_EVENT_APPROVE: str = "APPROVE"
_EVENT_COMMENT: str = "COMMENT"
_EVENT_REQUEST_CHANGES: str = "REQUEST_CHANGES"


# ── Rate-limit helpers ───────────────────────────────────────────────────────


def _seconds_until_reset(exc: GithubException) -> float | None:
    """Extract the number of seconds to wait from a GitHub rate-limit error.

    GitHub includes the reset timestamp in the response headers; PyGithub
    surfaces them inside the exception's ``data`` dict.

    Args:
        exc: The :class:`~github.GithubException.GithubException` raised by PyGithub.

    Returns:
        Seconds to sleep, or ``None`` if the header is absent / unreadable.
    """
    try:
        # PyGithub stores parsed headers in exc.headers
        reset_ts = exc.headers.get("x-ratelimit-reset")
        if reset_ts is None:
            return None
        reset_epoch = int(reset_ts)
        now_epoch = int(datetime.now(UTC).timestamp())
        wait = max(0, reset_epoch - now_epoch) + _RATE_LIMIT_BUFFER_SECS
        return float(wait)
    except Exception:  # noqa: BLE001
        return None


def _is_rate_limit_error(exc: GithubException) -> bool:
    """Return ``True`` when *exc* is a 403 GitHub rate-limit response.

    Args:
        exc: Exception raised by PyGithub.
    """
    if exc.status != 403:
        return False
    data = exc.data or {}
    message: str = data.get("message", "").lower()
    return "rate limit" in message or "api rate" in message


def _call_with_rate_limit_retry(fn, *args, **kwargs):
    """Call *fn* and, on a GitHub rate-limit 403, sleep then retry once.

    Args:
        fn:      A callable that may raise :class:`~github.GithubException.GithubException`.
        *args:   Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        GithubException: If the second attempt also fails, or if the error
            is not a rate-limit 403.
    """
    try:
        return fn(*args, **kwargs)
    except GithubException as exc:
        if not _is_rate_limit_error(exc):
            raise

        wait = _seconds_until_reset(exc)
        if wait is None:
            logger.warning("GitHub rate limit hit but could not parse reset time — raising")
            raise

        logger.warning(
            "GitHub rate limit hit — sleeping %.0f s until reset then retrying",
            wait,
        )
        time.sleep(wait)
        return fn(*args, **kwargs)  # second attempt — let any exception propagate


# ── Review body builder ──────────────────────────────────────────────────────


def _build_review_body(
    issues: list[ReviewIssue],
    overall_score: int,
    summary: str,
) -> str:
    """Compose the markdown body for the GitHub pull-request review.

    The body contains:
    * A score heading with an emoji indicator.
    * A severity breakdown table.
    * The summary paragraph.
    * An attribution footer.

    Args:
        issues:        All ``ReviewIssue`` objects found in this review.
        overall_score: Integer 0–100 quality score.
        summary:       Human-readable review summary.

    Returns:
        Markdown string to use as the review body.
    """
    critical = sum(1 for i in issues if i.severity == "critical")
    warnings = sum(1 for i in issues if i.severity == "warning")
    suggestions = sum(1 for i in issues if i.severity == "suggestion")

    # Score emoji
    if overall_score >= 80:
        score_emoji = "✅"
    elif overall_score >= 50:
        score_emoji = "⚠️"
    else:
        score_emoji = "❌"

    lines: list[str] = [
        f"## Code Review Score: {overall_score}/100 {score_emoji}",
        "",
        "| Severity | Count |",
        "|----------|-------|",
        f"| 🔴 Critical | {critical} |",
        f"| 🟡 Warning | {warnings} |",
        f"| 🔵 Suggestion | {suggestions} |",
        "",
        summary,
        "",
        "---",
        "*Reviewed by AI Code Bot — powered by Llama 3.1 70B via Groq*",
    ]
    return "\n".join(lines)


# ── Diff position resolver ───────────────────────────────────────────────────


def _build_diff_position_map(pr: PullRequest) -> dict[tuple[str, int], int]:
    """Build a mapping of (file_path, absolute_line_number) → diff position.

    GitHub's review comment API requires a *position* — the 1-based index of
    the line within the diff hunk, not the file's absolute line number.
    This function parses the raw patch of each PR file to produce that map.

    Args:
        pr: A PyGithub :class:`~github.PullRequest.PullRequest` object.

    Returns:
        Dict mapping ``(file_path, line_number)`` tuples to diff positions.
        Line numbers that are unchanged or removed are excluded.
    """
    position_map: dict[tuple[str, int], int] = {}

    for pr_file in pr.get_files():
        patch = pr_file.patch
        if not patch:
            continue

        filename = pr_file.filename
        diff_position = 0   # 1-based position within the full diff output
        current_line = 0    # current absolute line number in the new file

        for raw_line in patch.splitlines():
            diff_position += 1

            if raw_line.startswith("@@"):
                # Parse "@@  -old_start,old_count +new_start,new_count @@"
                import re
                match = re.search(r"\+(\d+)", raw_line)
                if match:
                    current_line = int(match.group(1)) - 1  # will be incremented below
                continue

            if raw_line.startswith("+"):
                current_line += 1
                position_map[(filename, current_line)] = diff_position
            elif raw_line.startswith("-"):
                pass  # removed line — no new-file line number
            else:
                current_line += 1  # context line

    return position_map


# ── Public API ───────────────────────────────────────────────────────────────


def post_review_comments(
    repo_full_name: str,
    pr_number: int,
    issues: list[ReviewIssue],
    overall_score: int,
    summary: str,
    github_token: str,
) -> None:
    """Create a single GitHub pull-request review with inline comments.

    Each :class:`~app.models.ReviewIssue` becomes an inline comment on the
    exact file + line in the diff.  Issues whose line number doesn't appear in
    the diff (e.g. because they reference a context line or the LLM hallucinated
    a line number) are skipped rather than crashing.

    The review event is determined by ``overall_score``:
    * ≥ 80  → ``APPROVE``
    * 50–79 → ``COMMENT``
    * < 50  → ``REQUEST_CHANGES``

    Rate-limit handling: if GitHub returns HTTP 403 with a rate-limit message,
    the function sleeps until the reset timestamp and retries once.

    Args:
        repo_full_name: Repository slug, e.g. ``"octocat/Hello-World"``.
        pr_number:      Pull-request number.
        issues:         Validated list of review issues.
        overall_score:  Quality score 0–100 used to decide the review event.
        summary:        Human-readable review summary for the review body.
        github_token:   GitHub Personal Access Token with ``repo`` scope.

    Raises:
        GithubException: On unrecoverable GitHub API errors (non-403, or
            second-attempt failure after rate-limit sleep).
    """
    gh = Github(github_token)

    def _do_post() -> None:
        repo: Repository = gh.get_repo(repo_full_name)
        pr: PullRequest = repo.get_pull(pr_number)

        # Determine review event
        if overall_score >= 80:
            event = _EVENT_APPROVE
        elif overall_score >= 50:
            event = _EVENT_COMMENT
        else:
            event = _EVENT_REQUEST_CHANGES

        # Build diff position map to validate line numbers
        position_map = _build_diff_position_map(pr)

        # Build inline comment dicts accepted by create_review()
        comments: list[dict] = []
        skipped = 0
        for issue in issues:
            key = (issue.file_path, issue.line_number)
            position = position_map.get(key)
            if position is None:
                logger.debug(
                    "Skipping comment on %s:%d — line not in diff",
                    issue.file_path,
                    issue.line_number,
                )
                skipped += 1
                continue

            severity_prefix = {
                "critical": "🔴 **CRITICAL**",
                "warning": "🟡 **WARNING**",
                "suggestion": "🔵 **SUGGESTION**",
            }.get(issue.severity, issue.severity.upper())

            body = (
                f"{severity_prefix}: {issue.issue}\n\n"
                f"**Suggestion:** {issue.suggestion}"
            )
            comments.append(
                {
                    "path": issue.file_path,
                    "position": position,
                    "body": body,
                }
            )

        if skipped:
            logger.info(
                "Skipped %d comment(s) whose line numbers were not in the diff",
                skipped,
            )

        review_body = _build_review_body(issues, overall_score, summary)

        pr.create_review(
            body=review_body,
            event=event,
            comments=comments,
        )

        logger.info(
            "Posted review to %s PR #%d — event: %s, inline comments: %d, skipped: %d",
            repo_full_name,
            pr_number,
            event,
            len(comments),
            skipped,
        )

    _call_with_rate_limit_retry(_do_post)


def get_repo_languages(
    repo_full_name: str,
    github_token: str,
) -> list[str]:
    """Return the top 3 programming languages used in a repository.

    Uses the GitHub Languages API (``GET /repos/{owner}/{repo}/languages``),
    which returns language names sorted by number of bytes written in each.

    The result is used to tune LLM review prompts toward language-specific
    patterns (e.g. GIL issues in Python, null-safety in TypeScript).

    Rate-limit handling: sleeps until reset and retries once on 403.

    Args:
        repo_full_name: Repository slug, e.g. ``"octocat/Hello-World"``.
        github_token:   GitHub Personal Access Token with ``repo`` scope.

    Returns:
        List of up to 3 language name strings, ordered by byte count
        descending.  Returns an empty list on any API error.
    """
    gh = Github(github_token)

    def _fetch() -> list[str]:
        repo: Repository = gh.get_repo(repo_full_name)
        # get_languages() returns {language: byte_count} already sorted desc
        lang_dict: dict[str, int] = repo.get_languages()
        return list(lang_dict.keys())[:3]

    try:
        return _call_with_rate_limit_retry(_fetch)
    except GithubException:
        logger.exception("Failed to fetch languages for %s", repo_full_name)
        return []
