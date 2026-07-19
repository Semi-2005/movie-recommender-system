"""
Movie Identity Bridge
=====================

Provides bidirectional mapping between the content-based model's
DataFrame positional index (``movie_index``) and the canonical
``movieId`` used by the collaborative filtering model and the
underlying ``movie_features.csv`` dataset.

This bridge is the prerequisite for hybrid score fusion — without it,
the two models' outputs cannot be joined on a common key.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models.content_based import recommender as cb_recommender

logger = logging.getLogger(__name__)


class MovieIdentityBridge:
    """
    Lightweight service that translates between the two identifier
    spaces used by the recommendation models.

    Backed by the content-based recommender's in-memory DataFrame
    (reference, not copy) to avoid duplicate memory usage.
    """

    def __init__(self) -> None:
        self._df = cb_recommender.df

        if "movieId" not in self._df.columns:
            raise ValueError(
                "movie_features.csv must contain a 'movieId' column. "
                f"Available columns: {list(self._df.columns)}"
            )

        # Pre-build a reverse lookup: movieId → DataFrame positional index
        # for O(1) reverse mapping.
        self._movie_id_to_index: dict[int, int] = {}
        for iloc_pos in range(len(self._df)):
            movie_id = int(self._df.iloc[iloc_pos]["movieId"])
            # First occurrence wins (there should be no duplicates)
            if movie_id not in self._movie_id_to_index:
                self._movie_id_to_index[movie_id] = iloc_pos

        logger.info(
            "MovieIdentityBridge initialized — %d movies indexed",
            len(self._movie_id_to_index),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_to_movie_id(self, movie_index: int) -> int:
        """
        Convert a content-based DataFrame positional index to ``movieId``.

        Args:
            movie_index: The iloc position in the CB model's DataFrame.

        Returns:
            The canonical ``movieId`` integer.

        Raises:
            IndexError: If ``movie_index`` is out of bounds.
        """
        if movie_index < 0 or movie_index >= len(self._df):
            raise IndexError(
                f"movie_index {movie_index} is out of range "
                f"[0, {len(self._df) - 1}]"
            )
        return int(self._df.iloc[movie_index]["movieId"])

    def movie_id_to_index(self, movie_id: int) -> Optional[int]:
        """
        Convert a ``movieId`` to the content-based DataFrame positional index.

        Args:
            movie_id: The canonical movie identifier.

        Returns:
            The iloc position, or ``None`` if the ``movieId`` is not
            present in the content-based dataset.
        """
        return self._movie_id_to_index.get(movie_id)

    def get_movie_metadata(self, movie_id: int) -> Optional[dict]:
        """
        Return metadata for a given ``movieId``.

        Returns:
            A dict with ``title``, ``genres``, ``rating``, ``rating_count``,
            or ``None`` if the ``movieId`` is not in the dataset.
        """
        idx = self.movie_id_to_index(movie_id)
        if idx is None:
            return None

        row = self._df.iloc[idx]
        return {
            "movie_id": movie_id,
            "title": str(row["title"]),
            "genres": str(row.get("genres", "")),
            "rating": round(float(row.get("rating", 0)), 2),
            "rating_count": int(row.get("rating_count", 0)),
        }


# ── Module-level singleton ───────────────────────────────────────────────
identity_bridge = MovieIdentityBridge()
