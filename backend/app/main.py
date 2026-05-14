from backend.app.models.content_based import recommender

print(recommender.stats())
print(recommender.search("Batman"))
print(recommender.recommend("Interstellar"))