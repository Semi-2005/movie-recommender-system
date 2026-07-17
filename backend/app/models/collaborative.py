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
    # Recommendation Logic
    # ------------------------------------------------------------------

    def get_similar_movies(
        self,
        movie_id: int,
        top_n: int = 10,
    ) -> list[dict]:
        """
        Return the top-N most similar movies for a given ``movieId``.

        Algorithm:
            1. Map ``movie_id`` → matrix row index.
            2. Retrieve the similarity vector for that row.
            3. ``np.argsort`` descending, skip self, take top N.
            4. Enrich each result with metadata from ``movie_features.csv``
               (title, genres, rating, rating_count) when available.

        Args:
            movie_id: The integer movie ID to find neighbors for.
            top_n:    Maximum number of similar movies to return.

        Returns:
            A list of dicts ordered by ``similarity_score`` (descending),
            each containing: ``movieId``, ``similarity_score``, and
            optionally ``title``, ``genres``, ``rating``, ``rating_count``.
            Returns an empty list when ``movie_id`` is unknown.
        """
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        similarity_vector = self.similarity_matrix[idx]

        # Indices sorted by similarity (descending).  The movie itself
        # will be at position 0 (similarity = 1.0), so we skip it.
        sorted_indices = np.argsort(similarity_vector)[::-1]

        results: list[dict] = []
        for neighbor_idx in sorted_indices:
            if len(results) >= top_n:
                break

            neighbor_idx_int = int(neighbor_idx)

            # Skip self-similarity
            if neighbor_idx_int == idx:
                continue

            neighbor_movie_id = self.idx_to_movie_id[neighbor_idx_int]
            score = float(similarity_vector[neighbor_idx_int])

            entry: dict = {
                "movieId": neighbor_movie_id,
                "similarity_score": round(score, 4),
            }

            # Enrich with metadata when available
            if (
                self.movie_metadata is not None
                and neighbor_movie_id in self.movie_metadata.index
            ):
                meta = self.movie_metadata.loc[neighbor_movie_id]
                entry["title"] = str(meta["title"])
                entry["genres"] = str(meta["genres"])
                entry["rating"] = round(float(meta["rating"]), 2)
                entry["rating_count"] = int(meta["rating_count"])

            results.append(entry)

        return results

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
