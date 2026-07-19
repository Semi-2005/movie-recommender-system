

from typing import Optional

from fastapi import APIRouter, Query

from backend.app.services.recommender import (
    get_recommendations,
    get_collaborative_recommendations,
    get_hybrid_recommendations,
    search_movies,
    get_model_stats,
    health_check
)

router = APIRouter()


@router.get("/")
def home():
    """
    Root endpoint.
    """
    return {
        "message": "Movie Recommendation API is running"
    }


@router.get("/recommend")
def recommend_movie(
    movie: str = Query(..., description="Movie title"),
    top_n: int = Query(10, ge=1, le=50)
):
    """
    Return movie recommendations based on content similarity.
    """
    recommendations = get_recommendations(movie, top_n)

    if not recommendations:
        return {
            "message": "Movie not found",
            "recommendations": []
        }

    return {
        "input_movie": movie,
        "total_recommendations": len(recommendations),
        "recommendations": recommendations
    }


@router.get("/recommend/collaborative")
def recommend_collaborative(
    movie: str = Query(..., description="Movie title"),
    top_n: int = Query(10, ge=1, le=50)
):
    """
    Return similar movies using item-based collaborative filtering.

    Uses the pre-computed cosine-similarity matrix built from user
    rating patterns.  Accepts a movie **title** (with fuzzy matching
    support for typo tolerance).
    """
    result = get_collaborative_recommendations(movie, top_n)

    if result is None:
        return {
            "message": "Movie not found in collaborative model. "
                       "Check the title and try again, or use /search to find available movies.",
            "input_movie": movie,
            "recommendations": [],
        }

    return {
        "input_movie": result["input_movie"],
        "input_movie_id": result["input_movie_id"],
        "total_recommendations": len(result["recommendations"]),
        "recommendations": result["recommendations"],
    }


@router.get("/recommend/hybrid")
async def recommend_hybrid(
    movie: str = Query(..., description="Movie title"),
    top_n: int = Query(10, ge=1, le=50),
    alpha_override: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Override the adaptive alpha weight for A/B testing. "
            "0.0 = pure collaborative, 1.0 = pure content-based."
        ),
    ),
):
    """
    Return hybrid recommendations by fusing content-based and
    collaborative filtering models.

    Uses an adaptive weighted ensemble strategy where the blending
    weight (alpha) is determined per-request based on collaborative
    model confidence.  Optionally override alpha via query param
    for experimentation.

    Response includes ``strategy`` metadata indicating which fusion
    mode was used: ``hybrid_full``, ``hybrid_partial``, or
    ``content_based_only``.
    """
    result = await get_hybrid_recommendations(movie, top_n, alpha_override)

    if not result["recommendations"]:
        # Provide search suggestions so the user can self-correct
        suggestions = search_movies(movie, limit=5)
        return {
            "message": "Movie not found. Did you mean one of these?",
            "input_movie": movie,
            "suggestions": suggestions,
            "recommendations": [],
        }

    return result


@router.get("/search")
def search_movie(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search movie titles.
    """
    results = search_movies(q, limit)

    return {
        "query": q,
        "results_count": len(results),
        "results": results
    }


@router.get("/stats")
def model_stats():
    """
    Return model statistics.
    """
    return get_model_stats()


@router.get("/health")
def health():
    """
    Health check endpoint.
    """
    return health_check()