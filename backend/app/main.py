

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router


app = FastAPI(
    title="Movie Recommendation System",
    description="ML-powered movie recommendation API using content-based filtering",
    version="1.0.0"
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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