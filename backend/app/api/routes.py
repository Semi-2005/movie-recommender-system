


from fastapi import APIRouter, Query

from backend.app.services.recommender import (
    get_recommendations,
    get_collaborative_recommendations,
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