<div align="center">

# 🤖 AI Code Reviewer

### An autonomous pull request review bot powered by RAG + LLaMA 3.1 70B

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1_70B-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)

*Automatically reviews every Pull Request with inline comments, severity-ranked issues, and a 0–100 quality score — all within 30 seconds of opening a PR.*

</div>

---

## 🚀 What This Project Does

This project is a **production-grade, AI-powered code review bot** that integrates directly into GitHub's pull request workflow. When a developer opens or updates a Pull Request, GitHub sends a webhook event to a FastAPI server, which fetches the diff, uses **Retrieval-Augmented Generation (RAG)** to pull in relevant context from the existing codebase via ChromaDB and `sentence-transformers`, and then passes everything to **Groq's free-tier LLaMA 3.1 70B** for analysis. The bot posts structured inline review comments directly onto the PR — including file-level annotations, severity-ranked issues (`critical`, `warning`, `suggestion`), actionable suggestions, and an overall code quality score from 0–100. The entire system is Dockerised, deployed to **Railway** via a 5-stage GitHub Actions CI/CD pipeline that includes automated canary testing, Prometheus error-rate monitoring, and **automatic rollback** if the Groq error rate spikes above 5% in the 10 minutes post-deploy.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AI CODE REVIEWER ARCHITECTURE                         │
└──────────────────────────────────────────────────────────────────────────────┘

  Developer opens / updates a Pull Request on GitHub
              │
              │  POST /webhook
              │  X-Hub-Signature-256: sha256=<hmac>
              ▼
  ┌───────────────────────┐
  │      GitHub.com       │──── HMAC-SHA256 signature on every event
  └──────────┬────────────┘
             │  Webhook event (JSON payload)
             ▼
  ┌───────────────────────┐       ┌──────────────────────────────┐
  │  FastAPI Server       │       │  Prometheus + Grafana        │
  │  (uvicorn, port 8000) │──────▶│  /metrics scrape every 15s   │
  │                       │       │  9 dashboard panels          │
  │  • Verify HMAC sig    │       │  Auto-rollback if error>5%   │
  │  • Parse PR payload   │       └──────────────────────────────┘
  │  • Background task    │
  └──────────┬────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  ┌──────────┐   ┌───────────────────────────────┐
  │  GitHub  │   │  RAG Pipeline                  │
  │  API     │   │                               │
  │          │   │  1. index_repository()        │
  │  Fetch   │   │     • Lists all .py/.js/.ts   │
  │  PR diff │   │       .java/.go files via     │
  │          │   │       GitHub Git Tree API     │
  │  (added  │   │     • Chunks into 40-line     │
  │  lines   │   │       segments (5-line        │
  │  only)   │   │       overlap)                │
  └────┬─────┘   │     • Embeds with             │
       │         │       all-MiniLM-L6-v2        │
       │         │     • Upserts → ChromaDB      │
       │         │                               │
       │         │  2. retrieve_context()        │
       │         │     • Embeds changed lines    │
       │         │     • Cosine similarity       │
       │         │       search in ChromaDB      │
       │         │     • Returns top-3 chunks    │
       │         │       (excludes changed file) │
       └────┬────┘
            │
            │  changed_lines + context_chunks
            ▼
  ┌───────────────────────┐
  │  Prompt Builder       │
  │                       │
  │  • System: senior     │
  │    code reviewer role │
  │  • Changed lines      │
  │    (with line numbers)│
  │  • Related code       │
  │    context (RAG)      │
  │  • 2 few-shot JSON    │
  │    examples           │
  │  • Strict JSON-only   │
  │    output instruction │
  └──────────┬────────────┘
             │
             │  POST api.groq.com/chat/completions
             │  model: llama-3.1-70b-versatile
             │  temperature: 0.1
             ▼
  ┌───────────────────────┐
  │  Groq API             │
  │  LLaMA 3.1 70B        │
  │                       │
  │  Returns JSON array   │
  │  of ReviewIssue       │
  │  objects              │
  │                       │
  │  On bad JSON →        │
  │  retry once with      │
  │  correction message   │
  └──────────┬────────────┘
             │
             │  list[ReviewIssue]
             ▼
  ┌───────────────────────┐
  │  Score Engine         │
  │                       │
  │  100 - (critical×20   │
  │       + warning×10    │
  │       + suggest×3)    │
  │  Clamped to [0, 100]  │
  └──────────┬────────────┘
             │
             │  POST /repos/{owner}/{repo}/pulls/{n}/reviews
             ▼
  ┌───────────────────────┐
  │  GitHub PR Review     │
  │                       │
  │  • Score badge        │
  │  • Severity table     │
  │  • Inline comments    │
  │    on exact diff line │
  │  • APPROVE / COMMENT  │
  │    / REQUEST_CHANGES  │
  └───────────────────────┘
```

---

## ⚡ Quick Start

**Requirements:** Python 3.11, Docker Desktop, Git

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/ai-code-reviewer.git
cd ai-code-reviewer

# 2. Copy the environment template
cp .env.example .env
```

Open `.env` and fill in these **4 required values**:

```env
GITHUB_TOKEN=ghp_xxxx        # github.com/settings/tokens → repo scope
GROQ_API_KEY=gsk_xxxx        # console.groq.com → API Keys (free)
WEBHOOK_SECRET=any-random-string-you-invent
GRAFANA_PASSWORD=your-grafana-password
```

```bash
# 3. Launch the full stack (FastAPI + ChromaDB + Prometheus + Grafana)
docker compose up --build -d

# 4. Verify it's running
curl http://localhost:8000/health
# → {"status":"ok","version":"1.0.0"}
```

**Open in your browser:**

| Service | URL | Credentials |
|---|---|---|
| API Docs (Swagger) | http://localhost:8000/docs | — |
| Prometheus Metrics | http://localhost:8000/metrics | — |
| Prometheus UI | http://localhost:9090 | — |
| Grafana Dashboard | http://localhost:3000 | admin / your password |

---

## 🔗 Register the Webhook on GitHub

To make the bot review your pull requests, register a webhook on the repository you want reviewed:

**Step 1 — Go to your repository's webhook settings**
> Navigate to: `github.com/{you}/{your-repo}` → **Settings** → **Webhooks** → **Add webhook**

**Step 2 — Fill in the webhook form**
```
Payload URL:   https://your-railway-url.up.railway.app/webhook
Content type:  application/json
Secret:        (paste the same value you set as WEBHOOK_SECRET in .env)
```

**Step 3 — Choose which events to send**
> Select **"Let me select individual events"**
> Tick only: ✅ **Pull requests**
> Leave everything else unchecked.

**Step 4 — Save and test**
> Click **Add webhook**. GitHub immediately sends a `ping` event.
> Check the **Recent Deliveries** tab — you should see a green ✓ for the ping.

**Step 5 — Open a Pull Request**
> Create a new branch in that repo, make a code change, and open a PR.
> Within ~30 seconds the bot posts an automated review with inline comments.

---

## 🧠 How It Works

### RAG Context Retrieval

Retrieval-Augmented Generation solves a fundamental LLM limitation: the model has no knowledge of your specific codebase. On first review of a repository, the bot:

1. **Indexes the codebase** — uses the GitHub Git Tree API (one call to list all files) to fetch every `.py`, `.js`, `.ts`, `.java`, `.go` file under 50KB, skipping `node_modules`, `dist`, `build`, and `.git` directories.
2. **Chunks each file** — splits content into 40-line segments with a 5-line overlap so functions that span chunk boundaries are still captured in at least one chunk.
3. **Embeds chunks locally** — uses `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU, no API key, ~90MB model baked into the Docker image to avoid cold-start delays).
4. **Stores in ChromaDB** — upserts every chunk with metadata: `{repo, file_path, start_line, language}`.

When a PR arrives, the bot embeds the changed lines from each file and performs a **cosine similarity search** in ChromaDB to retrieve the 3 most semantically related chunks from *other* files (the changed file itself is excluded to avoid circular context). These chunks give the LLM visibility into how the codebase patterns look — catching bugs like calling a method that was renamed, or deviating from the error-handling style established elsewhere.

### LLM Prompt Strategy

The prompt is engineered for **maximum determinism and parsability**:

- **Temperature 0.1** — minimises hallucinations and keeps output structured.
- **System role** — the model is told it is a senior code reviewer with 10+ years of experience, specialised in the detected languages of the repository.
- **Numbered line context** — changed lines are prefixed with their absolute line numbers so the model can reference exact positions.
- **Few-shot JSON examples** — two complete `ReviewIssue` JSON objects are included so the model learns the exact schema without relying on abstract instructions.
- **Self-correction retry** — if the first response is not valid JSON, the bot appends the bad response back to the conversation with a correction instruction and tries once more. On second failure it returns an empty list rather than crashing.
- **Strict output constraint** — the prompt ends with: *"Respond ONLY with a valid JSON array. No markdown. No commentary. Raw JSON only."*

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers a 5-stage pipeline:

```
  push to main
       │
       ▼
  ┌──────────┐     pytest        ┌──────────┐   Docker build    ┌──────────┐
  │  1. test │ ──────────────▶  │ 2. build │ ────────────────▶ │ 3. deploy│
  │          │   fails → stop   │          │   push to GHCR    │          │
  │ 42 tests │                  │ SHA tag  │                   │ Railway  │
  │ pytest   │                  │ + latest │                   │ CLI up   │
  └──────────┘                  └──────────┘                   └────┬─────┘
                                                                     │
                                                            ┌────────▼─────────┐
                                                            │    4. verify      │
                                                            │                  │
                                                            │  sleep 60s       │
                                                            │  GET /health     │
                                                            │  GET /ready      │
                                                            │                  │
                                                            │  fail → rollback │
                                                            │  + Slack alert   │
                                                            └────────┬─────────┘
                                                                     │
                                                            ┌────────▼─────────┐
                                                            │    5. canary      │
                                                            │                  │
                                                            │  sleep 10 min    │
                                                            │  Query Prometheus│
                                                            │  groq error rate │
                                                            │                  │
                                                            │  > 5%  → rollback│
                                                            │          + Slack │
                                                            │                  │
                                                            │  ≤ 5%  → stable  │
                                                            │  update          │
                                                            │  LAST_STABLE_TAG │
                                                            └──────────────────┘
```

**Key pipeline features:**
- Docker layer caching via `cache-from: type=gha` — rebuilds in seconds when only app code changes
- Build provenance attestation (SLSA Level 2) on every pushed image
- The canary job queries `rate(groq_api_calls_total{status=~"error|rate_limited"}[10m])` via the Prometheus HTTP API — if Prometheus is unreachable it skips gracefully instead of false-rolling-back
- `LAST_STABLE_TAG` is stored as a GitHub Actions repository variable and updated on every successful canary pass, giving each deploy a known-good rollback target

---

## 🎯 Skills Demonstrated

<table>
<tr>
<td width="50%" valign="top">

### 🤖 AI Engineering

- **Retrieval-Augmented Generation (RAG)** — vector similarity search with ChromaDB and sentence-transformers to give LLM context about the specific codebase
- **LLM prompt engineering** — few-shot examples, temperature tuning, structured JSON output constraints, and self-correction retry loops
- **Semantic chunking** — overlapping line-based chunking strategy to preserve function-boundary context across chunks
- **Local embedding model** — `all-MiniLM-L6-v2` baked into the Docker image at build time for zero cold-start delay
- **Pydantic v2 validation** — strict schema enforcement on every LLM response with graceful degradation on parse failure
- **Groq API integration** — LLaMA 3.1 70B with rate-limit detection, sleep-until-reset retry, and per-call latency tracking

</td>
<td width="50%" valign="top">

### ⚙️ DevOps Engineering

- **Multi-stage Docker build** — builder stage compiles wheels; runtime stage is lean with no compilers, non-root user (`appuser`), and HEALTHCHECK
- **Docker Compose orchestration** — 4-service stack (app, chromadb, prometheus, grafana) on a single bridge network with health-check dependencies
- **GitHub Actions CI/CD** — 5-job pipeline: test → build → deploy → verify → canary with automatic rollback
- **Prometheus instrumentation** — 6 custom metrics (counters, histograms, gauge) with tuned bucket boundaries for LLM latencies
- **Grafana dashboards** — 9 panels auto-provisioned on `docker compose up` with zero manual configuration
- **HMAC-SHA256 webhook security** — constant-time signature comparison to prevent timing attacks
- **Canary deployment** — Prometheus error-rate query post-deploy; rollback + Slack notification if error rate > 5%

</td>
</tr>
</table>

---

## 📊 Monitoring & Observability

The Grafana dashboard includes 9 panels, auto-provisioned on startup:

| Panel | Type | Metric |
|---|---|---|
| PRs Reviewed Today | Stat | `increase(pr_reviews_total[24h])` |
| Avg Review Score (24h) | Stat | Derived from severity penalty formula |
| Groq API Error Rate | Stat | Error + rate-limited / total |
| Active Reviews | Stat | `active_reviews` gauge |
| 🔴 Alert: Review > 30s | Alert Stat | `max_over_time` over 5m window |
| Review Duration p50/p95/p99 | Time series | `histogram_quantile(review_duration_seconds_bucket)` |
| Issues by Severity | Time series (stacked) | `rate(issues_found_total{severity=...}[5m])` |
| Groq Latency p50/p95 | Time series | `histogram_quantile(groq_latency_seconds_bucket)` |
| Groq Calls by Status | Time series | `rate(groq_api_calls_total{status=...}[5m])` |

---

## 🔐 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | ✅ Yes | PAT with `repo` scope — used to fetch diffs and post reviews |
| `GROQ_API_KEY` | ✅ Yes | Groq API key — free at console.groq.com |
| `WEBHOOK_SECRET` | ✅ Yes | Random string used to verify GitHub webhook HMAC signatures |
| `GRAFANA_PASSWORD` | ✅ Yes | Password for the Grafana `admin` user |
| `CHROMA_PERSIST_DIR` | No | Path where ChromaDB stores its data (default: `./chroma_data`) |
| `CHROMA_HOST` | No | ChromaDB host (default: `localhost`; `chromadb` inside Docker) |
| `CHROMA_PORT` | No | ChromaDB port (default: `8001`) |
| `GROQ_MODEL` | No | Override the Groq model (default: `llama-3.1-70b-versatile`) |
| `LOG_LEVEL` | No | Python log level (default: `INFO`) |
| `PORT` | No | Port the FastAPI server listens on (default: `8000`) |
| `RAILWAY_APP_URL` | CI only | Full Railway deployment URL — used by the canary verify job |

---

## 📁 Project Structure

```
ai-code-reviewer/
│
├── app/                          # Application source code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, webhook endpoint, /metrics mount
│   ├── models.py                 # Pydantic v2 schemas (GitHubPREvent, ReviewIssue, etc.)
│   ├── rag.py                    # ChromaDB indexing, diff parsing, context retrieval
│   ├── reviewer.py               # Prompt builder, Groq LLM calls, end-to-end orchestrator
│   ├── github_client.py          # Post inline PR review comments, rate-limit retry
│   └── metrics.py                # Prometheus metric definitions (6 metrics)
│
├── tests/                        # pytest test suite (42 tests, 0 real API calls)
│   ├── __init__.py
│   ├── test_webhook.py           # Signature verification, event filtering
│   ├── test_reviewer.py          # Prompt structure, scoring, LLM retry logic
│   └── test_rag.py               # Diff parsing, chunking, self-retrieval exclusion
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml    # Auto-provisions Prometheus datasource on startup
│       └── dashboards/
│           ├── dashboard.yml     # Tells Grafana to scan this directory for JSON
│           └── code-reviewer.json  # 9-panel dashboard definition
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # 5-job CI/CD pipeline with canary rollback
│
├── Dockerfile                    # Multi-stage build: builder + slim runtime
├── docker-compose.yml            # 4-service stack: app, chromadb, prometheus, grafana
├── prometheus.yml                # Scrape config: app /metrics + chromadb every 15s
├── .dockerignore                 # Excludes .git, .env, tests/, caches from build context
├── requirements.txt              # Pinned Python dependencies
├── pytest.ini                    # asyncio_mode = auto, test discovery settings
├── Makefile                      # make test / build / up / down / logs / lint
└── .env.example                  # Template with all variables documented
```

---

## 🧪 Running Tests

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run full suite
pytest tests/ -v

# Run a specific file
pytest tests/test_webhook.py -v
pytest tests/test_reviewer.py -v
pytest tests/test_rag.py -v       # Note: loads embedding model (~30s first run)
```

All 42 tests run fully offline — the Groq API, GitHub API, and ChromaDB are mocked. No API keys needed to run tests.

---

## 🛠️ Makefile Reference

```bash
make test     # Run pytest test suite
make build    # Build Docker image locally (tag: ai-code-reviewer:local)
make up       # docker compose up --build -d (copies .env.example if .env missing)
make down     # docker compose down (keeps volumes)
make logs     # docker compose logs --follow --tail=100
make clean    # docker compose down --volumes (WARNING: deletes all data)
make lint     # ruff check app/ tests/
```

---

## 📦 Tech Stack

| Category | Technology | Why |
|---|---|---|
| **Web framework** | FastAPI 0.115 | Async-first, auto-generates OpenAPI docs, BackgroundTasks |
| **LLM** | Groq / LLaMA 3.1 70B | Free tier, fastest inference (200+ tokens/s), no GPU needed |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Runs locally, no API key, 384-dim vectors, excellent quality |
| **Vector store** | ChromaDB | Local, persistent, built-in cosine similarity, no infra cost |
| **GitHub SDK** | PyGithub | Single-call `create_review()` for batched inline comments |
| **Monitoring** | Prometheus + Grafana | Industry standard, zero-cost OSS, already integrates with everything |
| **Containerisation** | Docker + Compose | Reproducible environment, multi-stage build keeps image lean |
| **CI/CD** | GitHub Actions | Native GitHub integration, free for public repos |
| **Deployment** | Railway | Auto-detects Dockerfile, free tier, CLI-based deploys |
| **Schema validation** | Pydantic v2 | Strict runtime type safety on all LLM and webhook payloads |

---

## 📄 License

MIT — free to use, modify, and deploy.

---

<div align="center">

Built with 🧠 AI + ⚙️ DevOps for portfolio demonstration purposes.

*If this project helped you, consider starring the repo!* ⭐

</div>
