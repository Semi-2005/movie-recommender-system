from typing import Optional

from backend.app.models.content_based import recommender
from backend.app.models.collaborative import collaborative_recommender
from backend.app.models.hybrid import hybrid_recommender


def get_recommendations(movie_name: str, top_n: int = 10):
    """
    Return recommended movies for a given title.
    """
    return recommender.recommend(movie_name, top_n)


def get_collaborative_recommendations(movie_title: str, top_n: int = 10):
    """
    Return similar movies using item-based collaborative filtering.
    Accepts a movie title (with fuzzy matching support) and delegates
    to the pre-computed similarity matrix.
    """
    return collaborative_recommender.get_similar_movies(movie_title, top_n)


async def get_hybrid_recommendations(
    movie_title: str,
    top_n: int = 10,
    alpha_override: Optional[float] = None,
) -> dict:
    """
    Return hybrid recommendations by fusing content-based and
    collaborative filtering outputs.

    Uses adaptive weighted ensemble: the blending weight (alpha) is
    determined per-request based on collaborative model confidence,
    unless explicitly overridden via ``alpha_override``.

    Args:
        movie_title:    Movie title (fuzzy matching supported).
        top_n:          Number of final recommendations.
        alpha_override: Optional manual alpha for A/B testing [0, 1].

    Returns:
        A dict with fused recommendations, strategy metadata, and
        performance metrics.
    """
    return await hybrid_recommender.recommend_async(
        movie_title, top_n, alpha_override
    )


def search_movies(query: str, limit: int = 10):
    """
    Search movies by title.
    """
    return recommender.search(query, limit)


def get_model_stats():
    """
    Return recommender model statistics.
    """
    return {
        "content_based": recommender.stats(),
        "collaborative": collaborative_recommender.stats(),
    }


def health_check():
    """
    Basic service health response.
    """
    return {
        "status": "ok",
        "service": "movie-recommender",
        "model_loaded": True
    }