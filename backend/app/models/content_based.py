from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """
    Content-based movie recommender system using movie genres.
    Uses TF-IDF vectorization + cosine similarity.
    """

    def __init__(self):
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self._initialize_model()

    def _initialize_model(self):
        """Load data and build model artifacts."""
        self.df = self._load_data()
        self.df = self._preprocess_data(self.df)
        self.df = self._build_ranking_features(self.df)
        self._build_tfidf_matrix()

    def _load_data(self):
        """Load processed CSV dataset."""
        project_root = Path(__file__).resolve().parents[3]
        file_path = project_root / "data" / "processed" / "movie_features.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        return pd.read_csv(file_path)

    def _preprocess_data(self, df):
        """Clean genres column."""
        df = df.copy()

        df["genres"] = df["genres"].fillna("")
        df["genres"] = df["genres"].str.replace("Sci-Fi", "SciFi", regex=False)
        df["genres"] = df["genres"].str.replace("Film-Noir", "FilmNoir", regex=False)
        df["genres"] = df["genres"].str.replace("|", " ", regex=False)
        df["genres"] = df["genres"].replace("(no genres listed)", "")
        df["genres"] = df["genres"].str.lower().str.strip()

        return df

    def _build_ranking_features(self, df):
        """
        Create normalized ranking features used for
        popularity-aware recommendation scoring.
        """
        df = df.copy()

        # Normalize movie ratings
        rating_min = df["rating"].min()
        rating_max = df["rating"].max()

        df["normalized_rating"] = (
            (df["rating"] - rating_min)
            / (rating_max - rating_min)
        )

        # Log-scale rating counts to reduce popularity dominance
        df["log_rating_count"] = np.log1p(df["rating_count"])

        popularity_min = df["log_rating_count"].min()
        popularity_max = df["log_rating_count"].max()

        df["normalized_popularity"] = (
            (df["log_rating_count"] - popularity_min)
            / (popularity_max - popularity_min)
        )

        return df

    def _build_tfidf_matrix(self):
        """Create TF-IDF matrix."""
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            min_df=1
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.df["genres"])

    def recommend(self, movie_name, top_n=10):
        """
        Recommend similar movies.

        Args:
            movie_name (str): Input movie title
            top_n (int): Number of recommendations

        Returns:
            list[dict]
        """
        matches = self.df[
            self.df["title"].str.contains(movie_name, case=False, na=False)
        ]

        if matches.empty:
            return []

        movie_index = matches.index[0]

        movie_vector = self.tfidf_matrix[movie_index]

        similarity_scores = linear_kernel(
            movie_vector,
            self.tfidf_matrix
        ).flatten()

        similarity_scores = list(enumerate(similarity_scores))

        ranking_data = []

        for movie_idx, similarity_score in similarity_scores:
            if movie_idx == movie_index:
                continue

            movie = self.df.iloc[movie_idx]

            final_score = (
                (0.70 * similarity_score)
                + (0.20 * movie["normalized_rating"])
                + (0.10 * movie["normalized_popularity"])
            )

            ranking_data.append({
                "movie_index": movie_idx,
                "title": movie["title"],
                "genres": movie["genres"],
                "rating": round(movie["rating"], 2),
                "rating_count": int(movie["rating_count"]),
                "similarity_score": round(float(similarity_score), 3),
                "final_score": round(float(final_score), 3)
            })

        ranking_data = sorted(
            ranking_data,
            key=lambda item: item["final_score"],
            reverse=True
        )

        return ranking_data[:top_n]

    def search(self, query, limit=10):
        """Search movie titles."""
        matches = self.df[
            self.df["title"].str.contains(query, case=False, na=False)
        ]["title"].head(limit)

        return matches.tolist()

    def stats(self):
        """Return model statistics."""
        return {
            "movie_count": int(len(self.df)),
            "feature_count": int(len(self.tfidf.get_feature_names_out())),
            "tfidf_matrix_shape": tuple(self.tfidf_matrix.shape),
            "ranking_system": "popularity-aware",
            "similarity_strategy": "on-demand linear kernel",
        }


recommender = ContentBasedRecommender()