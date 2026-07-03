# ════════════════════════════════════════════════════════════════════
# Stage 1 — dependency builder
# Install all Python packages into an isolated prefix so stage 2 can
# copy only the installed files without build tools or pip cache.
# ════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

# Keeps Python from generating .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install system packages needed to compile some wheels (e.g. chromadb, hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first so Docker can cache this layer
COPY requirements.txt .

# Install into a prefix we can cleanly COPY across to the final stage
RUN pip install --prefix=/install -r requirements.txt

# ── Download sentence-transformers model at build time ───────────────────────
# This bakes the model weights into the image, eliminating the cold-start
# delay caused by downloading ~90 MB on first request in production.
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
# Downloads model files to the default HuggingFace cache inside /install
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model downloaded successfully:", model)
EOF


# ════════════════════════════════════════════════════════════════════
# Stage 2 — lean runtime image
# Only the installed packages and the application code are included.
# Build tools, gcc, and pip itself are NOT present in this image.
# ════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python where the packages installed in stage 1 live
    PYTHONPATH=/app \
    # Point HuggingFace to a writable cache directory
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    HF_HOME=/home/appuser/.cache/huggingface \
    # ChromaDB persist path (overridable via env_file)
    CHROMA_PERSIST_DIR=/data/chroma \
    PORT=8000

# Install only the minimal runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user for security ───────────────────────────────────────────────
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy the downloaded model cache from builder stage
COPY --from=builder /root/.cache/huggingface /home/appuser/.cache/huggingface

# Create data directory and set ownership
RUN mkdir -p /data/chroma && chown -R appuser:appuser /data /home/appuser

WORKDIR /app

# Copy application source code
COPY --chown=appuser:appuser app/ ./app/

# Switch to non-root user for all subsequent RUN/CMD instructions
USER appuser

EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Starts after 30 s (allows ChromaDB init), checks every 30 s,
# tolerates 3 consecutive failures before marking unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
