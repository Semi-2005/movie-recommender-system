# ─────────────────────────────────────────────────────────────────────────────
# CineMatch — FastAPI Backend Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies into a clean venv
#   Stage 2 (runtime): Copy only the venv + app code → lean final image
#
# Memory-mapped similarity matrix (.npy) is stored in a Render Persistent Disk
# mounted at /data. The ML artifacts directory is symlinked to it at startup.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY backend/app/requirements.txt .

# Create an isolated venv and install deps into it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd -m -u 1001 appuser

# Install wget + curl for artifact download in startup.sh
# (python:3.11-slim does NOT include these by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from builder (no build tools in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY backend/app ./app
COPY data/processed/movie_features.csv ./data/processed/movie_features.csv
COPY data/processed/collaborative_artifacts/movie_index_mapping.joblib \
     ./data/processed/collaborative_artifacts/movie_index_mapping.joblib

# The similarity matrix (.npy, ~1.5-2 GB) is NOT baked into the image.
# It is mounted from a Render Persistent Disk or downloaded at startup.
# See startup.sh for the download logic.
COPY startup.sh ./startup.sh
RUN chmod +x ./startup.sh

# Ensure appuser owns everything
RUN chown -R appuser:appuser /app
USER appuser

# Render injects PORT env var; default to 8000 for local runs
ENV PORT=8000
ENV PYTHONPATH=/app

EXPOSE 8000

# Use startup.sh as entrypoint so we can handle artifact downloads
# before the uvicorn server starts.
CMD ["./startup.sh"]
