from backend.app.models.content_based import recommender

results = recommender.recommend("Interstellar")

for movie in results:
    print(movie)