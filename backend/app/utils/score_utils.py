"""
Score Utilities
===============

Pure functions for score normalization and adaptive weight (alpha)
selection, used by the hybrid recommendation engine.

All functions are stateless and side-effect-free for easy unit testing.
"""

from __future__ import annotations

_EPSILON = 1e-8

# ── Default alpha thresholds ─────────────────────────────────────────────
# These values are starting points; they can be tuned via A/B testing.
ALPHA_FULL_HYBRID = 0.4       # CF has strong signal → trust CF more
ALPHA_PARTIAL_HYBRID = 0.6    # CF signal is weak → lean on CB
ALPHA_CB_ONLY = 1.0           # No CF data → pure content-based

# Minimum CF results to qualify for "full" hybrid mode
DEFAULT_MIN_CF_RESULTS = 3


def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Normalize a list of scores to [0, 1] using min-max scaling.

    Edge cases handled:
    - Empty list → returns empty list.
    - Single element → returns [1.0].
    - All identical values → returns list of [1.0] (no variance to scale).

    Args:
        scores: Raw score values from a recommendation model.

    Returns:
        A new list of normalized scores in the same order.
    """
    if not scores:
        return []

    min_val = min(scores)
    max_val = max(scores)
    score_range = max_val - min_val

    if score_range < _EPSILON:
        # All scores are effectively identical — assign maximum confidence
        return [1.0] * len(scores)

    return [(s - min_val) / (score_range + _EPSILON) for s in scores]


def determine_alpha(
    cf_available: bool,
    cf_result_count: int = 0,
    min_cf_results: int = DEFAULT_MIN_CF_RESULTS,
) -> tuple[float, str]:
    """
    Compute the adaptive content-based weight (alpha) and strategy label.

    The alpha value controls the blend ratio:
        ``hybrid_score = alpha × CB_score + (1 - alpha) × CF_score``

    Higher alpha → more content-based influence.

    Args:
        cf_available:    Whether the movie exists in the collaborative model.
        cf_result_count: Number of CF recommendations returned.
        min_cf_results:  Minimum CF results for "full" hybrid mode.

    Returns:
        A tuple of ``(alpha, strategy_label)`` where strategy_label is one
        of ``"hybrid_full"``, ``"hybrid_partial"``, or ``"content_based_only"``.
    """
    if not cf_available:
        return ALPHA_CB_ONLY, "content_based_only"

    if cf_result_count >= min_cf_results:
        return ALPHA_FULL_HYBRID, "hybrid_full"

    return ALPHA_PARTIAL_HYBRID, "hybrid_partial"
