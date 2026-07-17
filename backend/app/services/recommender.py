from backend.app.models.content_based import recommender
from backend.app.models.collaborative import collaborative_recommender


def get_recommendations(movie_name: str, top_n: int = 10):
    """
    Return recommended movies for a given title.
    """
    return recommender.recommend(movie_name, top_n)


def get_collaborative_recommendations(movie_id: int, top_n: int = 10):
    """
    Return similar movies using item-based collaborative filtering.
    Delegates to the pre-computed similarity matrix.
    """
    return collaborative_recommender.get_similar_movies(movie_id, top_n)


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