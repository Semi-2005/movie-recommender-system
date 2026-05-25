from backend.app.models.content_based import recommender


def get_recommendations(movie_name: str, top_n: int = 10):
    """
    Return recommended movies for a given title.
    """
    return recommender.recommend(movie_name, top_n)


def search_movies(query: str, limit: int = 10):
    """
    Search movies by title.
    """
    return recommender.search(query, limit)


def get_model_stats():
    """
    Return recommender model statistics.
    """
    return recommender.stats()


def health_check():
    """
    Basic service health response.
    """
    return {
        "status": "ok",
        "service": "movie-recommender",
        "model_loaded": True
    }