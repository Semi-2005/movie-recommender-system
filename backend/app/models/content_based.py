from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based movie recommender system using movie genres.
    Uses TF-IDF vectorization + cosine similarity.
    """

    def __init__(self):
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._initialize_model()

    def _initialize_model(self):
        """Load data and build model artifacts."""
        self.df = self._load_data()
        self.df = self._preprocess_data(self.df)
        self._build_tfidf_matrix()
        self._build_similarity_matrix()

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

    def _build_tfidf_matrix(self):
        """Create TF-IDF matrix."""
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            min_df=1
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.df["genres"])

    def _build_similarity_matrix(self):
        """Create cosine similarity matrix."""
        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix,
            self.tfidf_matrix
        )

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

        similarity_scores = list(enumerate(self.cosine_sim[movie_index]))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda item: item[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1: top_n + 1]
        movie_indices = [item[0] for item in similarity_scores]

        recommendations = self.df.iloc[movie_indices][["title", "genres"]].copy()

        recommendations["similarity_score"] = [
            round(item[1], 3) for item in similarity_scores
        ]

        return recommendations.to_dict(orient="records")

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
            "matrix_shape": tuple(self.tfidf_matrix.shape),
        }


recommender = ContentBasedRecommender()