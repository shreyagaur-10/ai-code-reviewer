"""Reviewer module tests — prompt builder, scorer, and LLM call logic.

All external calls (Groq SDK, GitHub API, ChromaDB) are patched so
these tests run fully offline with no API keys.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("WEBHOOK_SECRET", "test-secret-12345")
os.environ.setdefault("GITHUB_TOKEN", "fake-gh-token")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma-test")

from app.models import ReviewIssue
from app.reviewer import (
    GroqRateLimitError,
    _extract_json_array,
    build_review_prompt,
    call_groq,
    score_review,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_issue(severity: str = "warning", line: int = 5, path: str = "app/foo.py") -> ReviewIssue:
    """Construct a minimal ReviewIssue for testing."""
    return ReviewIssue(
        severity=severity,  # type: ignore[arg-type]
        issue="Something is wrong",
        suggestion="Fix it this way",
        line_number=line,
        file_path=path,
    )


def _make_context_chunk(path: str = "app/bar.py", start: int = 10) -> dict[str, Any]:
    return {
        "file_path": path,
        "content": "def helper():\n    pass",
        "similarity_score": 0.87,
        "start_line": start,
    }


def _groq_response(content: str) -> MagicMock:
    """Build a fake Groq API response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── build_review_prompt ───────────────────────────────────────────────────────


class TestBuildReviewPrompt:
    """Tests for build_review_prompt."""

    def test_contains_changed_lines(self) -> None:
        """Changed lines must appear verbatim (with line numbers) in prompt."""
        changed = "def foo():\n    return 42"
        prompt = build_review_prompt("app/foo.py", changed, [])
        assert "1: def foo():" in prompt
        assert "2:     return 42" in prompt

    def test_contains_file_path_header(self) -> None:
        """Prompt must include the file path in the CHANGED CODE header."""
        prompt = build_review_prompt("src/utils.py", "x = 1", [])
        assert "CHANGED CODE: src/utils.py" in prompt

    def test_context_chunks_are_included(self) -> None:
        """Up to 3 context chunks must appear under RELATED CODE FOR CONTEXT."""
        chunks = [_make_context_chunk(f"other{i}.py", i * 10) for i in range(3)]
        prompt = build_review_prompt("app/foo.py", "x = 1", chunks)
        assert "RELATED CODE FOR CONTEXT #1" in prompt
        assert "RELATED CODE FOR CONTEXT #2" in prompt
        assert "RELATED CODE FOR CONTEXT #3" in prompt

    def test_max_three_context_chunks(self) -> None:
        """Only the first 3 context chunks must be included, not more."""
        chunks = [_make_context_chunk(f"f{i}.py") for i in range(5)]
        prompt = build_review_prompt("app/foo.py", "x = 1", chunks)
        assert "CONTEXT #4" not in prompt
        assert "CONTEXT #5" not in prompt

    def test_contains_json_instruction(self) -> None:
        """Prompt must instruct the model to respond with a JSON array."""
        prompt = build_review_prompt("app/foo.py", "x = 1", [])
        assert "JSON" in prompt

    def test_empty_context_still_works(self) -> None:
        """Prompt must be valid even with zero context chunks."""
        prompt = build_review_prompt("main.py", "pass", [])
        assert "CHANGED CODE: main.py" in prompt
        assert "INSTRUCTIONS" in prompt


# ── score_review ──────────────────────────────────────────────────────────────


class TestScoreReview:
    """Tests for score_review."""

    def test_no_issues_returns_100(self) -> None:
        assert score_review([]) == 100

    def test_one_critical_deducts_20(self) -> None:
        assert score_review([_make_issue("critical")]) == 80

    def test_one_warning_deducts_10(self) -> None:
        assert score_review([_make_issue("warning")]) == 90

    def test_one_suggestion_deducts_3(self) -> None:
        assert score_review([_make_issue("suggestion")]) == 97

    def test_six_critical_clamped_to_zero(self) -> None:
        """Six critical issues (6 × 20 = 120 penalty) must not go below 0."""
        issues = [_make_issue("critical") for _ in range(6)]
        assert score_review(issues) == 0

    def test_mixed_severity_combined(self) -> None:
        """1 critical + 1 warning + 1 suggestion = 100 - 20 - 10 - 3 = 67."""
        issues = [
            _make_issue("critical"),
            _make_issue("warning"),
            _make_issue("suggestion"),
        ]
        assert score_review(issues) == 67

    def test_score_is_always_non_negative(self) -> None:
        """Score must never go below zero regardless of how many issues."""
        issues = [_make_issue("critical") for _ in range(20)]
        assert score_review(issues) == 0


# ── _extract_json_array ───────────────────────────────────────────────────────


class TestExtractJsonArray:
    """Tests for the internal JSON extractor used by call_groq."""

    def test_plain_array(self) -> None:
        result = _extract_json_array('[{"a": 1}]')
        assert json.loads(result) == [{"a": 1}]

    def test_array_with_preamble(self) -> None:
        raw = "Sure, here are the issues:\n[{\"b\": 2}]"
        result = _extract_json_array(raw)
        assert json.loads(result) == [{"b": 2}]

    def test_markdown_fenced_json(self) -> None:
        raw = "```json\n[{\"c\": 3}]\n```"
        result = _extract_json_array(raw)
        assert json.loads(result) == [{"c": 3}]

    def test_empty_array(self) -> None:
        assert _extract_json_array("[]") == "[]"

    def test_no_array_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON array found"):
            _extract_json_array("no json here at all")


# ── call_groq ─────────────────────────────────────────────────────────────────


class TestCallGroq:
    """Tests for call_groq — mocks the Groq SDK client."""

    _GOOD_RESPONSE = json.dumps([
        {
            "severity": "warning",
            "issue": "Missing error handling",
            "suggestion": "Wrap in try/except",
            "line_number": 3,
            "file_path": "app/foo.py",
        }
    ])

    def test_successful_response_returns_issues(self) -> None:
        """A valid JSON response must be parsed into ReviewIssue objects."""
        with patch("app.reviewer.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _groq_response(self._GOOD_RESPONSE)
            )
            issues = call_groq("some prompt", file_path="app/foo.py")

        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].line_number == 3

    def test_malformed_json_retries_once(self) -> None:
        """On malformed JSON, call_groq must retry exactly once."""
        bad = "Here is my answer: not valid json!!"

        call_count = 0

        def fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            # First attempt: bad JSON; second attempt: good JSON
            content = bad if call_count == 1 else self._GOOD_RESPONSE
            return _groq_response(content)

        with patch("app.reviewer.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = fake_create
            issues = call_groq("some prompt", file_path="app/foo.py")

        assert call_count == 2, f"Expected 2 API calls (retry), got {call_count}"
        assert len(issues) == 1

    def test_two_malformed_responses_returns_empty(self) -> None:
        """If both attempts return bad JSON, must return [] without crashing."""
        with patch("app.reviewer.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _groq_response("not json at all")
            )
            issues = call_groq("some prompt")

        assert issues == []

    def test_rate_limit_raises_custom_error(self) -> None:
        """A 429 from Groq must raise GroqRateLimitError."""
        from groq import RateLimitError

        with patch("app.reviewer.Groq") as MockGroq:
            # RateLimitError needs (message, response, body)
            exc = RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
            MockGroq.return_value.chat.completions.create.side_effect = exc

            with pytest.raises(GroqRateLimitError):
                call_groq("some prompt")

    def test_missing_api_key_returns_empty(self) -> None:
        """If GROQ_API_KEY is not set, call_groq must return [] gracefully."""
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            issues = call_groq("some prompt")
        assert issues == []

    def test_file_path_fallback_in_response(self) -> None:
        """If the LLM omits file_path, call_groq must fill it from the argument."""
        response_without_path = json.dumps([
            {
                "severity": "suggestion",
                "issue": "Use a list comprehension",
                "suggestion": "Replace the loop",
                "line_number": 7,
                # No file_path here — should be filled from call arg
            }
        ])
        with patch("app.reviewer.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _groq_response(response_without_path)
            )
            issues = call_groq("prompt", file_path="app/utils.py")

        assert issues[0].file_path == "app/utils.py"
