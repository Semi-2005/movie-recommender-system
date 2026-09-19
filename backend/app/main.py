

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(
    title="Movie Recommendation System",
    description=(
        "ML-powered movie recommendation API using content-based filtering, "
        "item-based collaborative filtering, and an adaptive hybrid ensemble "
        "that fuses both models for superior recommendation quality."
    ),
    version="2.0.0"
)


# ── CORS Configuration ────────────────────────────────────────────────────────
# In production: set FRONTEND_URL env var to your Vercel deployment URL
#   e.g. https://cinematch.vercel.app
# In local development: falls back to localhost origins automatically.
_frontend_url = os.getenv("FRONTEND_URL", "")

_allowed_origins: list[str] = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # Alternative local port
]

if _frontend_url:
    _allowed_origins.append(_frontend_url)
    # Also allow www-prefixed variant if it's a plain domain
    if _frontend_url.startswith("https://") and not _frontend_url.startswith("https://www."):
        _allowed_origins.append(_frontend_url.replace("https://", "https://www.", 1))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API Routes
app.include_router(router)


@app.get("/ping")
def ping():
    """
    Simple ping endpoint.
    """
    return {
        "message": "Server is running"
    }