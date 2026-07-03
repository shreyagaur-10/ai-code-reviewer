"""RAG module tests — diff parsing and context retrieval.

Uses an in-memory ChromaDB client so no filesystem or network access is
needed.  GitHub API calls are patched with lightweight fakes.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("WEBHOOK_SECRET", "test-secret-12345")
os.environ.setdefault("GITHUB_TOKEN", "fake-gh-token")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma-test")

from app.rag import (
    _chunk_content,
    _detect_language,
    _is_binary_patch,
    _make_chunk_id,
    _parse_added_lines,
    _should_skip_path,
    get_collection,
    init_chromadb,
    retrieve_context,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def in_memory_chromadb():
    """Replace the persistent ChromaDB client with an in-memory one.

    This fixture patches ``app.rag._client`` and ``app.rag._collection``
    so every test in this module operates on a fresh, isolated in-memory
    collection without touching the filesystem.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.EphemeralClient()
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name="codebase-test",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    with patch("app.rag._client", client), patch("app.rag._collection", collection):
        yield collection


# ── _parse_added_lines ────────────────────────────────────────────────────────


class TestParseAddedLines:
    """Tests for _parse_added_lines — extracts + lines from unified diffs."""

    def test_extracts_added_lines(self) -> None:
        patch_text = (
            "@@ -1,3 +1,4 @@\n"
            " unchanged\n"
            "-removed line\n"
            "+added line 1\n"
            "+added line 2\n"
            " context\n"
        )
        result = _parse_added_lines(patch_text)
        assert "added line 1" in result
        assert "added line 2" in result

    def test_strips_leading_plus(self) -> None:
        """The leading '+' diff marker must be stripped."""
        result = _parse_added_lines("+def foo():\n+    pass")
        lines = result.splitlines()
        assert lines[0] == "def foo():"
        assert lines[1] == "    pass"

    def test_excludes_removed_lines(self) -> None:
        result = _parse_added_lines("-removed\n+added")
        assert "removed" not in result
        assert "added" in result

    def test_excludes_file_header(self) -> None:
        """Lines starting with '+++' (file header) must not be included."""
        result = _parse_added_lines("+++ b/app/foo.py\n+actual code")
        assert "b/app/foo.py" not in result
        assert "actual code" in result

    def test_empty_patch_returns_empty(self) -> None:
        assert _parse_added_lines("") == ""

    def test_only_context_lines_returns_empty(self) -> None:
        result = _parse_added_lines(" context\n context2")
        assert result == ""


# ── _should_skip_path ─────────────────────────────────────────────────────────


class TestShouldSkipPath:
    """Tests for _should_skip_path."""

    def test_node_modules_skipped(self) -> None:
        assert _should_skip_path("node_modules/lodash/index.js") is True

    def test_git_dir_skipped(self) -> None:
        assert _should_skip_path("src/.git/config") is True

    def test_dist_skipped(self) -> None:
        assert _should_skip_path("dist/bundle.js") is True

    def test_build_skipped(self) -> None:
        assert _should_skip_path("app/build/output.py") is True

    def test_normal_source_not_skipped(self) -> None:
        assert _should_skip_path("src/utils/helpers.py") is False

    def test_pycache_skipped(self) -> None:
        assert _should_skip_path("app/__pycache__/main.cpython-311.pyc") is True


# ── _detect_language ──────────────────────────────────────────────────────────


class TestDetectLanguage:
    """Tests for _detect_language."""

    def test_python(self) -> None:
        assert _detect_language("app/main.py") == "python"

    def test_javascript(self) -> None:
        assert _detect_language("src/index.js") == "javascript"

    def test_typescript(self) -> None:
        assert _detect_language("src/app.ts") == "typescript"

    def test_java(self) -> None:
        assert _detect_language("Main.java") == "java"

    def test_go(self) -> None:
        assert _detect_language("main.go") == "go"

    def test_markdown_unsupported(self) -> None:
        assert _detect_language("README.md") is None

    def test_yaml_unsupported(self) -> None:
        assert _detect_language("config.yml") is None


# ── _chunk_content ────────────────────────────────────────────────────────────


class TestChunkContent:
    """Tests for _chunk_content — line-based chunking with overlap."""

    def test_small_file_single_chunk(self) -> None:
        content = "\n".join(f"line {i}" for i in range(10))
        chunks = _chunk_content(content, max_lines=40, overlap=5)
        assert len(chunks) == 1
        assert chunks[0][0] == 1  # start_line is 1-based

    def test_large_file_multiple_chunks(self) -> None:
        content = "\n".join(f"line {i}" for i in range(50))
        chunks = _chunk_content(content, max_lines=40, overlap=5)
        # 50 lines: chunk 1 = lines 1-40, chunk 2 = lines 36-50
        assert len(chunks) == 2
        assert chunks[0][0] == 1
        assert chunks[1][0] == 36

    def test_overlap_respected(self) -> None:
        """Second chunk must start at max_lines - overlap + 1."""
        content = "\n".join(f"line {i}" for i in range(50))
        chunks = _chunk_content(content, max_lines=20, overlap=5)
        assert chunks[1][0] == 16  # 20 - 5 + 1

    def test_empty_content_returns_empty(self) -> None:
        assert _chunk_content("") == []


# ── _is_binary_patch ──────────────────────────────────────────────────────────


class TestIsBinaryPatch:
    """Tests for _is_binary_patch."""

    def test_none_is_binary(self) -> None:
        assert _is_binary_patch(None) is True

    def test_binary_marker_detected(self) -> None:
        assert _is_binary_patch("Binary files a/img.png and b/img.png differ") is True

    def test_normal_patch_not_binary(self) -> None:
        assert _is_binary_patch("+added line\n-removed line") is False


# ── _make_chunk_id ────────────────────────────────────────────────────────────


class TestMakeChunkId:
    """Tests for _make_chunk_id."""

    def test_format(self) -> None:
        cid = _make_chunk_id("octocat/Hello-World", "app/main.py", 42)
        assert cid == "octocat/Hello-World:app/main.py:L42"

    def test_unique_per_line(self) -> None:
        id1 = _make_chunk_id("repo/x", "f.py", 1)
        id2 = _make_chunk_id("repo/x", "f.py", 41)
        assert id1 != id2


# ── retrieve_context ──────────────────────────────────────────────────────────


class TestRetrieveContext:
    """Tests for retrieve_context — ensures self-retrieval is excluded."""

    def _seed_collection(self, collection, repo: str, paths_and_content: list[tuple[str, str]]) -> None:
        """Insert chunks into the test collection."""
        ids, docs, metas = [], [], []
        for path, content in paths_and_content:
            ids.append(f"{repo}:{path}:L1")
            docs.append(content)
            metas.append({"repo": repo, "file_path": path, "start_line": 1, "language": "python"})
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def test_excludes_changed_files_from_results(self, in_memory_chromadb) -> None:
        """Chunks from the changed file itself must not appear in results."""
        repo = "owner/repo"
        changed_path = "app/changed.py"
        other_path = "app/other.py"

        self._seed_collection(in_memory_chromadb, repo, [
            (changed_path, "def changed():\n    return 1"),
            (other_path,   "def helper():\n    return 2"),
        ])

        results = retrieve_context(
            {changed_path: "def changed():\n    return 1"},
            repo,
            top_k=5,
        )

        returned_paths = {r["file_path"] for r in results}
        assert changed_path not in returned_paths, (
            f"Self-retrieval: {changed_path} appeared in results"
        )

    def test_returns_similar_chunks(self, in_memory_chromadb) -> None:
        """retrieve_context must return chunks from non-changed files."""
        repo = "owner/repo2"
        self._seed_collection(in_memory_chromadb, repo, [
            ("app/a.py", "def authenticate(user, password):\n    pass"),
            ("app/b.py", "class Database:\n    def connect(self):\n        pass"),
        ])

        results = retrieve_context(
            {"app/new.py": "def login(user, pw):\n    pass"},
            repo,
            top_k=2,
        )

        assert isinstance(results, list)
        for r in results:
            assert "file_path" in r
            assert "content" in r
            assert "similarity_score" in r
            assert "start_line" in r
            assert r["file_path"] != "app/new.py"

    def test_empty_changed_files_returns_empty(self, in_memory_chromadb) -> None:
        results = retrieve_context({}, "owner/repo3", top_k=3)
        assert results == []

    def test_result_capped_at_top_k(self, in_memory_chromadb) -> None:
        """No more than top_k results should be returned per file."""
        repo = "owner/repo4"
        # Seed 5 context files
        self._seed_collection(in_memory_chromadb, repo, [
            (f"app/ctx{i}.py", f"def func_{i}():\n    return {i}") for i in range(5)
        ])

        results = retrieve_context(
            {"app/main.py": "def run():\n    pass"},
            repo,
            top_k=2,
        )
        assert len(results) <= 2
