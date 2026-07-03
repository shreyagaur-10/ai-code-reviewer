"""RAG module — codebase indexing, diff fetching, and context retrieval.

Manages a ChromaDB vector store backed by ``all-MiniLM-L6-v2`` embeddings
(runs locally, no API key).  Provides three public functions:

* :func:`index_repository` — clone-free indexing via the GitHub tree API.
* :func:`get_pr_diff`      — fetch a PR's unified diff and extract added lines.
* :func:`retrieve_context`  — semantic search for relevant codebase context.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import sys
from pathlib import PurePosixPath
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from github import Github, GithubException

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
}
"""Map file suffix → language name stored in ChromaDB metadata."""

SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }
)
"""Directory names to skip during repository traversal."""

MAX_INDEX_FILE_SIZE: int = 50 * 1024  # 50 KB
MAX_DIFF_FILE_SIZE: int = 100 * 1024  # 100 KB

CHUNK_MAX_LINES: int = 40
CHUNK_OVERLAP: int = 5

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
COLLECTION_NAME: str = "codebase"

# ── Module-level state ───────────────────────────────────────────────────────

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


# ── Initialisation ───────────────────────────────────────────────────────────


def init_chromadb(persist_dir: str | None = None) -> chromadb.Collection:
    """Initialise (or re-use) the ChromaDB persistent client and collection.

    The embedding function is ``all-MiniLM-L6-v2`` from *sentence-transformers*,
    which runs entirely on the CPU — no API key required.

    Args:
        persist_dir: Filesystem path for ChromaDB storage.  Falls back to
            the ``CHROMA_PERSIST_DIR`` env-var, then ``./chroma_data``.

    Returns:
        The active :class:`chromadb.Collection`.
    """
    global _client, _collection  # noqa: PLW0603

    if _collection is not None:
        return _collection

    if persist_dir is None:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")

    logger.info("Initialising ChromaDB at '%s' …", persist_dir)

    _client = chromadb.PersistentClient(path=persist_dir)

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        "ChromaDB ready — collection '%s' has %d existing documents",
        COLLECTION_NAME,
        _collection.count(),
    )
    return _collection


def get_collection() -> chromadb.Collection:
    """Return the active ChromaDB collection, initialising on first call.

    Returns:
        The :class:`chromadb.Collection` used for codebase embeddings.

    Raises:
        RuntimeError: If ChromaDB fails to initialise.
    """
    if _collection is None:
        return init_chromadb()
    return _collection


# ── Private helpers ──────────────────────────────────────────────────────────


def _should_skip_path(path: str) -> bool:
    """Return ``True`` if *path* contains a directory we want to ignore.

    Args:
        path: POSIX-style relative path (e.g. ``src/utils/helpers.py``).
    """
    parts = PurePosixPath(path).parts
    return any(part in SKIP_DIRS for part in parts)


def _detect_language(file_path: str) -> str | None:
    """Map a file path to its language name via extension.

    Args:
        file_path: POSIX-style relative path.

    Returns:
        Language string or ``None`` if the extension is unsupported.
    """
    ext = PurePosixPath(file_path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def _chunk_content(
    content: str,
    max_lines: int = CHUNK_MAX_LINES,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, str]]:
    """Split file content into overlapping line-based chunks.

    Args:
        content:   Full text content of the file.
        max_lines: Maximum lines per chunk.
        overlap:   Number of overlapping lines between consecutive chunks.

    Returns:
        List of ``(start_line, chunk_text)`` tuples.  ``start_line`` is 1-based.
    """
    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[tuple[int, str]] = []
    start = 0
    step = max_lines - overlap  # how far to advance each iteration

    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append((start + 1, chunk_text))  # 1-based line number

        if end >= len(lines):
            break
        start += step

    return chunks


def _make_chunk_id(repo: str, file_path: str, start_line: int) -> str:
    """Produce a deterministic, human-readable ChromaDB document ID.

    Args:
        repo:       Repository slug (``owner/name``).
        file_path:  Relative path inside the repo.
        start_line: 1-based first line of the chunk.

    Returns:
        String ID like ``owner/name:src/app.py:L1``.
    """
    return f"{repo}:{file_path}:L{start_line}"


def _parse_added_lines(patch: str) -> str:
    """Extract the *added* lines from a unified-diff patch.

    Lines starting with ``+`` (but not the ``+++`` file header) are kept.
    The leading ``+`` marker is stripped so the result is clean source code.

    Args:
        patch: The ``patch`` field from a GitHub PullRequestFile.

    Returns:
        Concatenated added lines separated by newlines.
    """
    added: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])  # strip the leading '+'
    return "\n".join(added)


def _is_binary_patch(patch: str | None) -> bool:
    """Heuristic: treat a file as binary if it has no patch or the patch
    mentions 'Binary files'.

    Args:
        patch: Raw patch text (may be ``None`` for binary files).
    """
    if patch is None:
        return True
    if re.search(r"^Binary files .+ differ$", patch, re.MULTILINE):
        return True
    return False


# ── Public API ───────────────────────────────────────────────────────────────


def index_repository(repo_full_name: str, github_token: str) -> int:
    """Index all supported source files from a GitHub repository.

    Uses the Git Tree API to enumerate blobs (one API call), then fetches
    file contents individually.  Files are chunked into 40-line segments
    with 5-line overlap and upserted into ChromaDB.

    Args:
        repo_full_name: Repository slug (e.g. ``"octocat/Hello-World"``).
        github_token:   GitHub Personal Access Token with ``repo`` scope.

    Returns:
        Total number of chunks indexed.
    """
    gh = Github(github_token)
    repo = gh.get_repo(repo_full_name)
    default_branch = repo.default_branch

    logger.info(
        "Indexing %s (branch: %s) …",
        repo_full_name,
        default_branch,
    )

    # One API call to list every blob recursively
    tree = repo.get_git_tree(sha=default_branch, recursive=True)

    collection = get_collection()
    total_chunks = 0
    files_indexed = 0

    for element in tree.tree:
        # Only process file blobs
        if element.type != "blob":
            continue

        # Must be a supported language
        language = _detect_language(element.path)
        if language is None:
            continue

        # Skip banned directories
        if _should_skip_path(element.path):
            continue

        # Skip oversized files
        if element.size is not None and element.size > MAX_INDEX_FILE_SIZE:
            logger.debug(
                "Skipping %s (%.1f KB exceeds %d KB limit)",
                element.path,
                element.size / 1024,
                MAX_INDEX_FILE_SIZE // 1024,
            )
            continue

        # Fetch file content via the Git Blob API (handles files up to 100 MB)
        try:
            blob = repo.get_git_blob(element.sha)
            if blob.encoding == "base64":
                raw_bytes = base64.b64decode(blob.content)
            else:
                raw_bytes = blob.content.encode("utf-8")
            content = raw_bytes.decode("utf-8", errors="replace")
        except (GithubException, UnicodeDecodeError) as exc:
            logger.warning("Failed to fetch %s: %s", element.path, exc)
            continue

        # Chunk the file
        chunks = _chunk_content(content)
        if not chunks:
            continue

        # Prepare batch for upsert
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for start_line, chunk_text in chunks:
            ids.append(_make_chunk_id(repo_full_name, element.path, start_line))
            documents.append(chunk_text)
            metadatas.append(
                {
                    "repo": repo_full_name,
                    "file_path": element.path,
                    "start_line": start_line,
                    "language": language,
                }
            )

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        files_indexed += 1
        logger.info("  ✓ %s → %d chunks", element.path, len(chunks))

    logger.info(
        "Indexing complete: %d files, %d total chunks for %s",
        files_indexed,
        total_chunks,
        repo_full_name,
    )
    return total_chunks


def get_pr_diff(
    repo_full_name: str,
    pr_number: int,
    github_token: str,
) -> dict[str, str]:
    """Fetch a pull request's diff and return added lines per file.

    Uses PyGithub's ``PullRequest.get_files()`` which calls the
    ``GET /repos/{owner}/{repo}/pulls/{pull_number}/files`` endpoint.

    Args:
        repo_full_name: Repository slug (e.g. ``"octocat/Hello-World"``).
        pr_number:      Pull-request number.
        github_token:   GitHub Personal Access Token with ``repo`` scope.

    Returns:
        Dict mapping ``file_path → added_lines``.  Only files that have
        at least one added line are included.  Binary files and patches
        exceeding 100 KB are skipped.
    """
    gh = Github(github_token)
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    changed_files: dict[str, str] = {}

    for pr_file in pr.get_files():
        patch: str | None = pr_file.patch

        # Skip binary files
        if _is_binary_patch(patch):
            logger.debug("Skipping binary file: %s", pr_file.filename)
            continue

        assert patch is not None  # guaranteed by _is_binary_patch check above

        # Skip patches larger than 100 KB
        if len(patch.encode("utf-8")) > MAX_DIFF_FILE_SIZE:
            logger.debug(
                "Skipping oversized diff for %s (%.1f KB)",
                pr_file.filename,
                len(patch.encode("utf-8")) / 1024,
            )
            continue

        added_lines = _parse_added_lines(patch)
        if added_lines.strip():
            changed_files[pr_file.filename] = added_lines

    logger.info(
        "PR #%d diff: %d files with added lines",
        pr_number,
        len(changed_files),
    )
    return changed_files


def retrieve_context(
    changed_files: dict[str, str],
    repo_full_name: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Query ChromaDB for codebase chunks similar to the changed code.

    For each changed file the added code is used as the query text.
    Results from the *changed files themselves* are excluded to avoid
    self-retrieval.  Duplicate chunks (retrieved by multiple queries) are
    deduplicated by their ChromaDB document ID.

    Args:
        changed_files:  Dict of ``file_path → added_lines`` (output of
            :func:`get_pr_diff`).
        repo_full_name: Repository slug used to scope the search.
        top_k:          Number of context chunks to retrieve *per file*.

    Returns:
        Deduplicated list of dicts, each with keys:
        ``file_path``, ``content``, ``similarity_score``, ``start_line``.
    """
    collection = get_collection()
    changed_paths: set[str] = set(changed_files.keys())

    # Track seen IDs to deduplicate across queries
    seen_ids: set[str] = set()
    results: list[dict[str, Any]] = []

    for file_path, added_code in changed_files.items():
        query_text = added_code.strip()
        if not query_text:
            continue

        # Over-fetch to have room after filtering out self-matches
        n_results = top_k + len(changed_paths)

        try:
            query_result = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"repo": repo_full_name},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChromaDB query failed for %s: %s",
                file_path,
                exc,
            )
            continue

        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]
        ids = query_result.get("ids", [[]])[0]

        count = 0
        for i, doc in enumerate(documents):
            if count >= top_k:
                break

            meta = metadatas[i]
            doc_id = ids[i]

            # Skip chunks from any changed file (no self-retrieval)
            if meta.get("file_path") in changed_paths:
                continue

            # Deduplicate
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            # ChromaDB cosine distance = 1 - cosine_similarity
            distance = distances[i] if distances else 0.0
            similarity = round(1.0 - distance, 4)

            results.append(
                {
                    "file_path": meta["file_path"],
                    "content": doc,
                    "similarity_score": similarity,
                    "start_line": meta.get("start_line", 1),
                }
            )
            count += 1

    logger.info(
        "Retrieved %d context chunks for %d changed files",
        len(results),
        len(changed_files),
    )
    return results


# ── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m app.rag <owner/repo>")
        print("  Indexes all supported source files into ChromaDB.")
        print()
        print("Required env vars:")
        print("  GITHUB_TOKEN       — GitHub PAT with 'repo' scope")
        print("  CHROMA_PERSIST_DIR — (optional) ChromaDB storage path")
        sys.exit(1)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is not set")
        sys.exit(1)

    repo_slug = sys.argv[1]
    print(f"Indexing repository: {repo_slug}")

    init_chromadb()
    total = index_repository(repo_slug, token)
    print(f"Done — indexed {total} chunks from {repo_slug}")
