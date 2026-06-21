from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
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

    # ------------------------------------------------------------------
    # Fuzzy Matching Helper
    # ------------------------------------------------------------------

    def _find_movie_index(
        self,
        movie_name: str,
        fuzzy_threshold: int = 75,
    ) -> Optional[int]:
        """
        Locate the DataFrame index of the best-matching movie title.

        Search Strategy (two-phase):
          1. **Exact Match** (case-insensitive): O(n) string comparison.
             Preferred when the user's input perfectly matches a title,
             e.g. "Toy Story" → "Toy Story (1995)" would NOT match here;
             the comparison is against the full stored title.
          2. **Fuzzy Match** via `rapidfuzz.process.extractOne`:
             Uses `fuzz.WRatio` (Weighted Ratio) which combines multiple
             fuzzy algorithms internally and is robust against partial
             matches, transpositions, and token reordering.
             Only accepted when `score >= fuzzy_threshold` to prevent
             wildly unrelated results from slipping through.

        Args:
            movie_name:      Raw title string from the user.
            fuzzy_threshold: Minimum similarity score (0–100) required
                             for a fuzzy match to be accepted. Default 75.

        Returns:
            The integer DataFrame index of the matched movie, or ``None``
            if no suitable match is found.
        """
        # ── Phase 1: Exact match (case-insensitive) ─────────────────────
        exact_mask = self.df["title"].str.lower() == movie_name.lower()
        exact_matches = self.df[exact_mask]

        if not exact_matches.empty:
            # Deterministic: always pick the first exact hit
            return int(exact_matches.index[0])

        # ── Phase 2: Fuzzy match ────────────────────────────────────────
        # Build a plain list of titles for rapidfuzz to score against.
        # `process.extractOne` returns (match_string, score, index) or None.
        result = process.extractOne(
            query=movie_name,
            choices=self.df["title"].tolist(),
            scorer=fuzz.WRatio,   # Handles partial matches & token reordering
            score_cutoff=fuzzy_threshold,
        )

        if result is None:
            # Score fell below threshold → treat as "not found"
            return None

        _matched_title, _score, matched_list_position = result

        # `matched_list_position` is the positional index in the *list*,
        # which equals the DataFrame's iloc position (same order).
        return int(self.df.index[matched_list_position])

    def recommend(self, movie_name: str, top_n: int = 10) -> list[dict]:
        """
        Recommend movies similar to ``movie_name``.

        Internally delegates title resolution to :meth:`_find_movie_index`,
        which performs an exact-match check first and falls back to fuzzy
        matching when necessary.  An empty list is returned when no
        sufficiently similar title can be found in the dataset.

        Args:
            movie_name: The movie title provided by the user. Typos and
                        minor casing differences are tolerated.
            top_n:      Maximum number of recommendations to return.

        Returns:
            A list of dicts ordered by ``final_score`` (descending), each
            containing: ``movie_index``, ``title``, ``genres``, ``rating``,
            ``rating_count``, ``similarity_score``, and ``final_score``.
            Returns an empty list when no match is found.
        """
        # Resolve the user's input to a concrete DataFrame index.
        movie_index = self._find_movie_index(movie_name)

        if movie_index is None:
            # No exact or fuzzy match above the threshold → bail out early.
            return []

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

            # Weighted blend: content similarity (70%), rating (20%), popularity (10%)
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

    def search(self, query: str, limit: int = 10) -> list[str]:
        """
        Search movie titles using a two-phase strategy.

        Phase 1 — Substring filter: quickly narrows the candidate pool to
        titles that contain the query string (case-insensitive).  This
        mirrors the original behaviour for normal, well-typed queries and
        is very fast.

        Phase 2 — Fuzzy ranking: when the substring filter yields no
        results (e.g. due to a typo), ``rapidfuzz.process.extract`` scores
        *all* titles and returns the top-``limit`` results that meet the
        similarity threshold.  This ensures the search box stays useful
        even when the user makes small mistakes.

        Args:
            query: The partial or full title string to search for.
            limit: Maximum number of title suggestions to return.

        Returns:
            A list of matching movie title strings (up to ``limit`` items).
            Returns an empty list when nothing passes the threshold.
        """
        # ── Phase 1: Fast substring filter ──────────────────────────────
        substring_matches = self.df[
            self.df["title"].str.contains(query, case=False, na=False, regex=False)
        ]["title"].head(limit)

        if not substring_matches.empty:
            return substring_matches.tolist()

        # ── Phase 2: Fuzzy fallback for typo-tolerant search ─────────────
        # `process.extract` returns a list of (title, score, index) tuples,
        # already sorted by score descending.
        fuzzy_results = process.extract(
            query=query,
            choices=self.df["title"].tolist(),
            scorer=fuzz.WRatio,
            limit=limit,
            score_cutoff=75,   # Same threshold as _find_movie_index for consistency
        )

        # Unpack only the matched title strings; discard score & position.
        return [title for title, _score, _pos in fuzzy_results]

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