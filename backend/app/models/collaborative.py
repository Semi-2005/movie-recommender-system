"""
Item-Based Collaborative Filtering — Runtime Model
====================================================

Loads pre-computed artifacts (similarity matrix + index mapping) produced
by :mod:`train_collaborative` and exposes a ``get_similar_movies`` function
for the API layer.

The module-level singleton ``collaborative_recommender`` is instantiated at
import time so the FastAPI application pays the cost once during startup,
identical to the pattern used by :mod:`content_based`.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from rapidfuzz import process, fuzz

# ── Logging ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "collaborative_artifacts"
MOVIE_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "movie_features.csv"


class CollaborativeRecommender:
    """
    Item-based collaborative filtering recommender.

    Relies on artifacts pre-computed by ``train_collaborative.py``:

    - ``item_similarity_matrix.joblib`` — dense float32 cosine-similarity
      matrix of shape (num_movies, num_movies).
    - ``movie_index_mapping.joblib`` — dict containing ``movie_id_to_idx``
      and ``idx_to_movie_id`` bidirectional mappings.

    Additionally loads ``movie_features.csv`` to enrich recommendations
    with human-readable metadata (title, genres, rating, rating_count).
    """

    def __init__(self) -> None:
        self.similarity_matrix: Optional[np.ndarray] = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.movie_metadata: Optional[pd.DataFrame] = None
        self._load_artifacts()

    # ------------------------------------------------------------------
    # Artifact Loading
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """
        Load the pre-computed similarity matrix, index mapping, and
        movie metadata into memory.

        Raises:
            FileNotFoundError: If any required artifact is missing.
                               Run ``train_collaborative.py`` first.
        """
        sim_path = ARTIFACT_DIR / "item_similarity_matrix.joblib"
        map_path = ARTIFACT_DIR / "movie_index_mapping.joblib"

        if not sim_path.exists() or not map_path.exists():
            raise FileNotFoundError(
                "Collaborative filtering artifacts not found. "
                f"Expected files in {ARTIFACT_DIR}/. "
                "Run `python -m backend.app.models.train_collaborative` first."
            )

        logger.info("Loading collaborative filtering artifacts …")

        self.similarity_matrix = joblib.load(sim_path)
        mapping = joblib.load(map_path)

        self.movie_id_to_idx = mapping["movie_id_to_idx"]
        self.idx_to_movie_id = mapping["idx_to_movie_id"]

        logger.info(
            "Loaded similarity matrix %s (%.1f MB) — %d movies indexed",
            self.similarity_matrix.shape,
            self.similarity_matrix.nbytes / (1024 ** 2),
            len(self.movie_id_to_idx),
        )

        # ── Movie metadata for response enrichment ──────────────────────
        self._load_movie_metadata()

    def _load_movie_metadata(self) -> None:
        """
        Load ``movie_features.csv`` and index by ``movieId`` for O(1)
        lookups when enriching recommendation responses.
        """
        if not MOVIE_FEATURES_PATH.exists():
            logger.warning(
                "movie_features.csv not found at %s — "
                "recommendations will lack title/genre metadata.",
                MOVIE_FEATURES_PATH,
            )
            return

        self.movie_metadata = pd.read_csv(MOVIE_FEATURES_PATH)
        self.movie_metadata.set_index("movieId", inplace=True)
        logger.info(
            "Loaded movie metadata for %d movies", len(self.movie_metadata)
        )

    # ------------------------------------------------------------------
    # Title Resolution
    # ------------------------------------------------------------------

    def resolve_movie_title(self, movie_id: int) -> str:
        """
        Resolve a ``movieId`` to its human-readable title.

        Falls back to ``"Unknown (movieId=<id>)"`` when metadata is
        unavailable so callers always receive a usable string.
        """
        if (
            self.movie_metadata is not None
            and movie_id in self.movie_metadata.index
        ):
            return str(self.movie_metadata.loc[movie_id, "title"])
        return f"Unknown (movieId={movie_id})"

    def find_movie_by_title(
        self,
        title: str,
        fuzzy_threshold: int = 65,
    ) -> Optional[dict]:
        """
        Resolve a user-provided title string to a ``movieId``.

        Search Strategy (three-phase):
          1. **Exact Match** (case-insensitive): full string comparison
             against all titles in ``movie_metadata``.
          2. **Substring Match** (case-insensitive): catches partial
             inputs like ``"Toy Story"`` → ``"Toy Story (1995)"``.
             When multiple matches exist, prefers the movie with the
             highest ``rating_count`` (most popular).
          3. **Fuzzy Match** via ``rapidfuzz.process.extractOne``:
             Uses ``fuzz.WRatio`` (Weighted Ratio) — robust against
             partial matches, transpositions, and token reordering.
             Only accepted when ``score >= fuzzy_threshold``.

        Args:
            title:           Raw title string from the user.
            fuzzy_threshold: Minimum similarity score (0–100) required
                             for a fuzzy match to be accepted. Default 75.

        Returns:
            A dict with ``movie_id`` (int) and ``matched_title`` (str),
            or ``None`` when no suitable match is found.
        """
        if self.movie_metadata is None:
            logger.warning("Movie metadata not loaded — title lookup unavailable.")
            return None

        titles_series = self.movie_metadata["title"]

        # ── Phase 1: Exact match (case-insensitive) ─────────────────────
        exact_mask = titles_series.str.lower() == title.lower()
        exact_matches = titles_series[exact_mask]

        if not exact_matches.empty:
            matched_movie_id = int(exact_matches.index[0])
            matched_title = str(exact_matches.iloc[0])
            logger.info(
                "Exact title match: '%s' → movieId=%d", matched_title, matched_movie_id
            )
            return {"movie_id": matched_movie_id, "matched_title": matched_title}

        # ── Phase 2: Substring match (case-insensitive) ─────────────────
        # Handles inputs like "Toy Story" → "Toy Story (1995)" where the
        # user omits the year suffix that is part of the stored title.
        substring_mask = titles_series.str.contains(
            title, case=False, na=False, regex=False
        )
        substring_matches = self.movie_metadata[substring_mask]

        if not substring_matches.empty:
            # When multiple substring matches exist (e.g. "Toy Story" matches
            # both "Toy Story (1995)" and "Toy Story 2 (1999)"), prefer the
            # most popular one by rating_count for deterministic results.
            if "rating_count" in substring_matches.columns:
                best = substring_matches.sort_values("rating_count", ascending=False)
            else:
                best = substring_matches
            matched_movie_id = int(best.index[0])
            matched_title = str(best.iloc[0]["title"])
            logger.info(
                "Substring title match: '%s' → '%s' (movieId=%d, %d candidates)",
                title, matched_title, matched_movie_id, len(substring_matches),
            )
            return {"movie_id": matched_movie_id, "matched_title": matched_title}

        # ── Phase 3: Fuzzy match ────────────────────────────────────────
        titles_list = titles_series.tolist()
        movie_ids_list = titles_series.index.tolist()

        result = process.extractOne(
            query=title,
            choices=titles_list,
            scorer=fuzz.WRatio,
            score_cutoff=fuzzy_threshold,
        )

        if result is None:
            logger.info("No fuzzy match found for '%s' (threshold=%d)", title, fuzzy_threshold)
            return None

        matched_title_str, score, matched_position = result
        matched_movie_id = int(movie_ids_list[matched_position])

        logger.info(
            "Fuzzy title match: '%s' → '%s' (score=%.1f, movieId=%d)",
            title, matched_title_str, score, matched_movie_id,
        )
        return {"movie_id": matched_movie_id, "matched_title": matched_title_str}

    # ------------------------------------------------------------------
    # Recommendation Logic
    # ------------------------------------------------------------------

    def get_similar_movies(
        self,
        title: str,
        top_n: int = 10,
    ) -> Optional[dict]:
        """
        Return the top-N most similar movies for a given movie **title**.

        Algorithm:
            1. Resolve ``title`` → ``movieId`` via :meth:`find_movie_by_title`
               (exact match first, then fuzzy fallback).
            2. Map ``movieId`` → matrix row index.
            3. Retrieve the similarity vector for that row.
            4. ``np.argsort`` descending, skip self, take top N.
            5. Enrich each neighbour with title, genres, rating metadata.

        Args:
            title: The movie title provided by the user. Typos and
                   minor casing differences are tolerated via fuzzy matching.
            top_n: Maximum number of similar movies to return.

        Returns:
            A dict with:
            - ``input_movie``  — matched title of the queried movie.
            - ``input_movie_id`` — the resolved movieId.
            - ``recommendations`` — list of dicts ordered by
              ``similarity_score`` (descending).

            Returns ``None`` when the title cannot be resolved or the
            resolved ``movieId`` is not in the collaborative model.
        """
        # ── Step 1: Title → movieId resolution ──────────────────────────
        match = self.find_movie_by_title(title)

        if match is None:
            logger.info(
                "Title '%s' could not be resolved — no recommendation possible.",
                title,
            )
            return None

        movie_id = match["movie_id"]
        matched_title = match["matched_title"]

        # ── Step 2: Check collaborative model coverage ──────────────────
        if movie_id not in self.movie_id_to_idx:
            logger.info(
                "movieId=%d ('%s') exists in metadata but not in the "
                "collaborative model — likely insufficient rating data.",
                movie_id,
                matched_title,
            )
            return None

        idx = self.movie_id_to_idx[movie_id]
        similarity_vector = self.similarity_matrix[idx]

        # Indices sorted by similarity (descending).  The movie itself
        # will be at position 0 (similarity = 1.0), so we skip it.
        sorted_indices = np.argsort(similarity_vector)[::-1]

        recommendations: list[dict] = []
        for neighbor_idx in sorted_indices:
            if len(recommendations) >= top_n:
                break

            neighbor_idx_int = int(neighbor_idx)

            # Skip self-similarity
            if neighbor_idx_int == idx:
                continue

            neighbor_movie_id = self.idx_to_movie_id[neighbor_idx_int]
            score = float(similarity_vector[neighbor_idx_int])

            # Title-first entry: human-readable title is the primary key
            entry: dict = {
                "title": self.resolve_movie_title(neighbor_movie_id),
                "movieId": neighbor_movie_id,
                "similarity_score": round(score, 4),
            }

            # Enrich with additional metadata when available
            if (
                self.movie_metadata is not None
                and neighbor_movie_id in self.movie_metadata.index
            ):
                meta = self.movie_metadata.loc[neighbor_movie_id]
                entry["genres"] = str(meta["genres"])
                entry["rating"] = round(float(meta["rating"]), 2)
                entry["rating_count"] = int(meta["rating_count"])

            recommendations.append(entry)

        return {
            "input_movie": matched_title,
            "input_movie_id": movie_id,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Model Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return model statistics."""
        return {
            "model_type": "item-based collaborative filtering",
            "similarity_metric": "cosine",
            "movie_count": len(self.movie_id_to_idx),
            "similarity_matrix_shape": (
                tuple(self.similarity_matrix.shape)
                if self.similarity_matrix is not None
                else None
            ),
            "similarity_matrix_dtype": (
                str(self.similarity_matrix.dtype)
                if self.similarity_matrix is not None
                else None
            ),
            "metadata_loaded": self.movie_metadata is not None,
        }


# ── Module-level singleton ───────────────────────────────────────────────
# Instantiated once when the module is first imported (i.e., at FastAPI
# startup).  Mirrors the pattern in content_based.py.
collaborative_recommender = CollaborativeRecommender()
