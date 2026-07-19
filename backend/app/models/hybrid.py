"""
Hybrid Recommendation Engine
=============================

Fuses content-based and collaborative filtering recommendations into
a unified ranked list using an **adaptive weighted ensemble** strategy.

Architecture
------------
1.  Call both model singletons concurrently (over-fetching ``top_n * 2``).
2.  Normalize CF scores per-request via min-max scaling.
3.  Map CB results from ``movie_index`` → ``movieId`` using the identity bridge.
4.  Build a **union** set of all candidate movies.
5.  For movies in *both* models: ``hybrid_score = α × CB + (1-α) × CF``.
6.  For movies in *only one* model: use that model's score × its weight.
7.  Sort by ``hybrid_score`` descending, truncate to ``top_n``.
8.  Tag response with ``strategy`` metadata for frontend adaptation.

The alpha (α) weight is determined dynamically per request based on
collaborative model confidence (see :mod:`score_utils.determine_alpha`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.models.content_based import recommender as cb_recommender
from app.models.collaborative import collaborative_recommender as cf_recommender
from app.services.movie_identity import identity_bridge
from app.utils.score_utils import min_max_normalize, determine_alpha

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Adaptive weighted ensemble that fuses content-based and
    collaborative filtering outputs.
    """

    def __init__(self) -> None:
        self._cb = cb_recommender
        self._cf = cf_recommender
        self._bridge = identity_bridge

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recommend_async(
        self,
        title: str,
        top_n: int = 10,
        alpha_override: Optional[float] = None,
    ) -> dict:
        """
        Produce hybrid recommendations by running both models
        concurrently and fusing their outputs.

        Args:
            title:          Movie title (fuzzy matching supported).
            top_n:          Number of final recommendations to return.
            alpha_override: Optional manual alpha for A/B testing.
                            Must be in [0, 1]. Overrides adaptive selection.

        Returns:
            A dict containing ``recommendations``, ``strategy``,
            ``alpha``, and performance metadata.
        """
        start_time = time.perf_counter()
        over_fetch = top_n * 2

        # ── Step 1: Run both models concurrently ────────────────────────
        cb_task = asyncio.to_thread(self._cb.recommend, title, over_fetch)
        cf_task = asyncio.to_thread(self._cf.get_similar_movies, title, over_fetch)

        cb_results, cf_results = await asyncio.gather(cb_task, cf_task)

        # ── Step 2: Determine alpha and strategy ────────────────────────
        cf_available = cf_results is not None
        cf_recs = cf_results.get("recommendations", []) if cf_available else []

        alpha, strategy = determine_alpha(
            cf_available=cf_available,
            cf_result_count=len(cf_recs),
        )

        if alpha_override is not None:
            alpha = max(0.0, min(1.0, alpha_override))
            strategy = f"{strategy}(alpha_override={alpha})"

        # ── Step 3: Normalize and index CB results by movieId ───────────
        cb_by_movie_id: dict[int, dict] = {}

        for item in (cb_results or []):
            movie_index = item["movie_index"]
            movie_id = self._bridge.index_to_movie_id(movie_index)
            cb_by_movie_id[movie_id] = {
                "movie_id": movie_id,
                "title": item["title"],
                "genres": item["genres"],
                "rating": item["rating"],
                "rating_count": item["rating_count"],
                "cb_score": item["final_score"],
            }

        # CB scores are already in [0, 1] from the content-based model.
        # No additional normalization needed.

        # ── Step 4: Normalize CF scores per-request ─────────────────────
        cf_by_movie_id: dict[int, dict] = {}

        if cf_recs:
            raw_cf_scores = [r["similarity_score"] for r in cf_recs]
            normalized_cf_scores = min_max_normalize(raw_cf_scores)

            for rec, norm_score in zip(cf_recs, normalized_cf_scores):
                movie_id = rec["movieId"]
                cf_by_movie_id[movie_id] = {
                    "movie_id": movie_id,
                    "title": rec["title"],
                    "genres": rec.get("genres", ""),
                    "rating": rec.get("rating", 0.0),
                    "rating_count": rec.get("rating_count", 0),
                    "cf_score": norm_score,
                }

        # ── Step 5: Union fusion ────────────────────────────────────────
        all_movie_ids = set(cb_by_movie_id.keys()) | set(cf_by_movie_id.keys())

        fused: list[dict] = []
        for movie_id in all_movie_ids:
            cb_entry = cb_by_movie_id.get(movie_id)
            cf_entry = cf_by_movie_id.get(movie_id)

            cb_score = cb_entry["cb_score"] if cb_entry else 0.0
            cf_score = cf_entry["cf_score"] if cf_entry else 0.0

            hybrid_score = (alpha * cb_score) + ((1 - alpha) * cf_score)

            # Use metadata from whichever model has it; prefer CB
            # (it always has enriched metadata from movie_features.csv)
            source = cb_entry or cf_entry

            fused.append({
                "movie_id": movie_id,
                "title": source["title"],
                "genres": source.get("genres", ""),
                "rating": source.get("rating", 0.0),
                "rating_count": source.get("rating_count", 0),
                "hybrid_score": round(hybrid_score, 4),
                "content_score": round(cb_score, 4) if cb_entry else None,
                "collaborative_score": round(cf_score, 4) if cf_entry else None,
            })

        # ── Step 6: Sort and truncate ───────────────────────────────────
        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
        recommendations = fused[:top_n]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "input_movie": title,
            "strategy": strategy,
            "alpha": round(alpha, 2),
            "cb_candidates": len(cb_by_movie_id),
            "cf_candidates": len(cf_by_movie_id),
            "total_recommendations": len(recommendations),
            "fusion_time_ms": round(elapsed_ms, 2),
            "recommendations": recommendations,
        }

    def recommend_sync(
        self,
        title: str,
        top_n: int = 10,
        alpha_override: Optional[float] = None,
    ) -> dict:
        """
        Synchronous wrapper around :meth:`recommend_async`.

        Useful for testing or non-async contexts. Runs both models
        sequentially (no concurrency benefit).
        """
        start_time = time.perf_counter()
        over_fetch = top_n * 2

        cb_results = self._cb.recommend(title, over_fetch)
        cf_results = self._cf.get_similar_movies(title, over_fetch)

        cf_available = cf_results is not None
        cf_recs = cf_results.get("recommendations", []) if cf_available else []

        alpha, strategy = determine_alpha(
            cf_available=cf_available,
            cf_result_count=len(cf_recs),
        )

        if alpha_override is not None:
            alpha = max(0.0, min(1.0, alpha_override))
            strategy = f"{strategy}(alpha_override={alpha})"

        # Normalize and index CB results
        cb_by_movie_id: dict[int, dict] = {}
        for item in (cb_results or []):
            movie_index = item["movie_index"]
            movie_id = self._bridge.index_to_movie_id(movie_index)
            cb_by_movie_id[movie_id] = {
                "movie_id": movie_id,
                "title": item["title"],
                "genres": item["genres"],
                "rating": item["rating"],
                "rating_count": item["rating_count"],
                "cb_score": item["final_score"],
            }

        # Normalize CF scores
        cf_by_movie_id: dict[int, dict] = {}
        if cf_recs:
            raw_cf_scores = [r["similarity_score"] for r in cf_recs]
            normalized_cf_scores = min_max_normalize(raw_cf_scores)
            for rec, norm_score in zip(cf_recs, normalized_cf_scores):
                movie_id = rec["movieId"]
                cf_by_movie_id[movie_id] = {
                    "movie_id": movie_id,
                    "title": rec["title"],
                    "genres": rec.get("genres", ""),
                    "rating": rec.get("rating", 0.0),
                    "rating_count": rec.get("rating_count", 0),
                    "cf_score": norm_score,
                }

        # Union fusion
        all_movie_ids = set(cb_by_movie_id.keys()) | set(cf_by_movie_id.keys())
        fused: list[dict] = []

        for movie_id in all_movie_ids:
            cb_entry = cb_by_movie_id.get(movie_id)
            cf_entry = cf_by_movie_id.get(movie_id)

            cb_score = cb_entry["cb_score"] if cb_entry else 0.0
            cf_score = cf_entry["cf_score"] if cf_entry else 0.0
            hybrid_score = (alpha * cb_score) + ((1 - alpha) * cf_score)

            source = cb_entry or cf_entry
            fused.append({
                "movie_id": movie_id,
                "title": source["title"],
                "genres": source.get("genres", ""),
                "rating": source.get("rating", 0.0),
                "rating_count": source.get("rating_count", 0),
                "hybrid_score": round(hybrid_score, 4),
                "content_score": round(cb_score, 4) if cb_entry else None,
                "collaborative_score": round(cf_score, 4) if cf_entry else None,
            })

        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
        recommendations = fused[:top_n]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "input_movie": title,
            "strategy": strategy,
            "alpha": round(alpha, 2),
            "cb_candidates": len(cb_by_movie_id),
            "cf_candidates": len(cf_by_movie_id),
            "total_recommendations": len(recommendations),
            "fusion_time_ms": round(elapsed_ms, 2),
            "recommendations": recommendations,
        }


# ── Module-level singleton ───────────────────────────────────────────────
hybrid_recommender = HybridRecommender()
