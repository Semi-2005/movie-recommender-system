from backend.app.models.content_based import recommender

print(recommender.recommend("Interstellar"))
print(recommender.stats())