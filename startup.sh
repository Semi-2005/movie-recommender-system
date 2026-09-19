#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# startup.sh — CineMatch Backend Startup Script
#
# Responsibilities:
#   1. Check if the similarity matrix (.npy) is present on the persistent disk.
#   2. If not, download it from cloud storage (Cloudflare R2 / S3-compatible).
#   3. Start the Uvicorn server.
#
# Environment variables required (set in Render dashboard or .env):
#   SIM_MATRIX_URL   — Public URL to download item_similarity_matrix.npy
#                      Example: https://pub-xxx.r2.dev/item_similarity_matrix.npy
#   PORT             — Injected by Render automatically (default: 8000)
#
# Persistent Disk (Render): Mount at /data and set ARTIFACT_DIR=/data/collaborative_artifacts
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ARTIFACT_DIR="${ARTIFACT_DIR:-/app/data/processed/collaborative_artifacts}"
SIM_MATRIX_FILE="${ARTIFACT_DIR}/item_similarity_matrix.npy"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  CineMatch Backend — Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Ensure artifact directory exists ──────────────────────────────────
mkdir -p "${ARTIFACT_DIR}"

# ── Step 2: Download similarity matrix if not present ────────────────────────
if [ ! -f "${SIM_MATRIX_FILE}" ]; then
    if [ -z "${SIM_MATRIX_URL:-}" ]; then
        echo "❌ ERROR: item_similarity_matrix.npy not found and SIM_MATRIX_URL is not set."
        echo "   Please either:"
        echo "   a) Mount a Render Persistent Disk at /data and upload the .npy file, OR"
        echo "   b) Set the SIM_MATRIX_URL environment variable to a downloadable URL."
        exit 1
    fi

    echo "📥 Similarity matrix not found. Downloading from: ${SIM_MATRIX_URL}"
    echo "   This may take a few minutes depending on file size (~1.5-2 GB)..."

    # curl -L: follow redirects (required for Hugging Face → CDN redirects)
    # --retry 3: retry on failure
    # -o: output file
    curl -L \
         --retry 3 \
         --retry-delay 5 \
         --progress-bar \
         -o "${SIM_MATRIX_FILE}" \
         "${SIM_MATRIX_URL}"

    echo "✅ Download complete: ${SIM_MATRIX_FILE}"
else
    SIZE=$(du -sh "${SIM_MATRIX_FILE}" | cut -f1)
    echo "✅ Similarity matrix found (${SIZE}): ${SIM_MATRIX_FILE}"
fi

echo ""
echo "🚀 Starting Uvicorn server on port ${PORT:-8000}..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 3: Start Uvicorn ─────────────────────────────────────────────────────
# - Single worker: mmap is per-process; multiple workers each get their own
#   mmap handle to the same .npy file (OS page cache is shared → efficient).
# - For Render Starter plan (512MB), use 1 worker.
# - For Render Standard plan (2GB+), you can increase to --workers 2.
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1 \
    --log-level info
