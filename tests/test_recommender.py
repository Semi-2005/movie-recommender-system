"""
test_recommender.py — Unit tests for the ContentBasedRecommender ML model.

Test Strategy
-------------
These tests focus exclusively on the *model layer* (content_based.py) in
complete isolation from the HTTP layer.  By testing the model directly we
verify that the core algorithm is correct independently of any API concerns,
making failures easier to diagnose during code review.

The well-known title "Interstellar" is used as the primary test fixture
because it is a widely-rated film that is virtually guaranteed to exist in
any standard movie dataset, giving us a deterministic and stable test anchor.

All tests receive the shared `recommender_model` fixture from conftest.py.
"""

# ── Standard library ─────────────────────────────────────────────────────────
from typing import Any

# ── Third-party ──────────────────────────────────────────────────────────────
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_MOVIE_TITLE: str = "Interstellar"
NONEXISTENT_MOVIE_TITLE: str = "xQzAbsolutelyNotAMoviexQz"
EXPECTED_RECOMMENDATION_KEYS: frozenset[str] = frozenset({
    "title",
    "genres",
    "similarity_score",
    "final_score",
})
DEFAULT_TOP_N: int = 10
CUSTOM_TOP_N: int = 5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_recommendation_shape(recommendation: dict[str, Any]) -> None:
    """
    Assert that a single recommendation dict contains all required keys.

    Extracted as a helper so that the same structural contract is enforced
    consistently across multiple test functions without copy-pasting assertions.

    Args:
        recommendation: A single item from the list returned by recommend().
    """
    for expected_key in EXPECTED_RECOMMENDATION_KEYS:
        assert expected_key in recommendation, (
            f"Missing key '{expected_key}' in recommendation: {recommendation}"
        )


# ---------------------------------------------------------------------------
# Tests — recommend()
# ---------------------------------------------------------------------------

class TestRecommendMethod:
    """
    Test suite for ContentBasedRecommender.recommend().

    Groups all tests related to the recommend() public method under one
    class, making the test report easier to read in the PR diff.
    """

    def test_recommend_returns_a_list(self, recommender_model):
        """
        Verify that recommend() always returns a list, never None or another type.

        Why: The API layer and downstream consumers iterate over the result.
        A wrong return type would cause a silent runtime crash that the type
        checker might not catch if the annotation is not enforced at runtime.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        assert isinstance(recommendation_results, list), (
            f"Expected list, got {type(recommendation_results).__name__}"
        )

    def test_recommend_returns_non_empty_results_for_known_movie(self, recommender_model):
        """
        Verify that recommend() returns at least one result for a well-known title.

        Why: An empty list for a title that definitely exists in the dataset
        indicates a broken similarity pipeline or a data-loading regression.
        Catching this early protects Berat and Berkay from shipping a model
        that silently returns nothing.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        assert len(recommendation_results) > 0, (
            f"Expected at least 1 recommendation for '{KNOWN_MOVIE_TITLE}', got 0."
        )

    def test_recommend_every_result_contains_required_keys(self, recommender_model):
        """
        Verify that every returned recommendation dict contains the full contract set of keys.

        Why: The API serialises these dicts directly into the JSON response.
        A missing key produces an incomplete JSON payload that breaks the
        frontend without raising a server-side error — a subtle, hard-to-debug bug.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        for recommendation in recommendation_results:
            _assert_recommendation_shape(recommendation)

    def test_recommend_respects_top_n_limit(self, recommender_model):
        """
        Verify that recommend() returns no more than `top_n` results.

        Why: The API exposes `top_n` as a user-facing query parameter.
        If the model ignores this limit, users could receive unexpectedly
        large payloads and the API contract (max 50 per route definition) would be violated.
        """
        recommendation_results = recommender_model.recommend(
            KNOWN_MOVIE_TITLE,
            top_n=CUSTOM_TOP_N
        )

        assert len(recommendation_results) <= CUSTOM_TOP_N, (
            f"Expected at most {CUSTOM_TOP_N} results, got {len(recommendation_results)}."
        )

    def test_recommend_similarity_scores_are_valid_floats(self, recommender_model):
        """
        Verify that similarity_score in each result is a float in the range [0.0, 1.0].

        Why: Cosine similarity is mathematically bounded to [0, 1] for
        non-negative TF-IDF vectors.  A value outside this range signals a
        broken vectorisation step or a data normalisation error.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        for recommendation in recommendation_results:
            similarity_score = recommendation["similarity_score"]
            assert isinstance(similarity_score, float), (
                f"similarity_score must be float, got {type(similarity_score).__name__}"
            )
            assert 0.0 <= similarity_score <= 1.0, (
                f"similarity_score {similarity_score} is outside the valid [0.0, 1.0] range."
            )

    def test_recommend_results_are_sorted_by_final_score_descending(self, recommender_model):
        """
        Verify that recommend() returns results ordered by final_score (highest first).

        Why: The sorting guarantees that users always see the best matches at
        the top of the list.  If the sort is broken, the API returns correct
        data in the wrong order — a product-quality bug that can be subtle to spot.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        final_scores = [item["final_score"] for item in recommendation_results]
        assert final_scores == sorted(final_scores, reverse=True), (
            "Recommendations are not sorted by final_score in descending order."
        )

    def test_recommend_returns_empty_list_for_nonexistent_movie(self, recommender_model):
        """
        Verify that recommend() returns an empty list when no title matches.

        Why: The route handler checks `if not recommendations` to decide
        whether to return a 'Movie not found' message.  If the model raises
        an exception instead of returning [], the API would return a 500 error
        instead of a clean 'not found' response, degrading the user experience.
        """
        recommendation_results = recommender_model.recommend(NONEXISTENT_MOVIE_TITLE)

        assert isinstance(recommendation_results, list), (
            "recommend() must return a list even when no match is found."
        )
        assert len(recommendation_results) == 0, (
            f"Expected empty list for '{NONEXISTENT_MOVIE_TITLE}', "
            f"got {len(recommendation_results)} results."
        )

    def test_recommend_default_top_n_does_not_exceed_ten(self, recommender_model):
        """
        Verify that the default top_n=10 cap is honoured when no top_n is specified.

        Why: Documents the public API contract of the model method itself —
        callers that omit top_n must not receive more than 10 results.
        """
        recommendation_results = recommender_model.recommend(KNOWN_MOVIE_TITLE)

        assert len(recommendation_results) <= DEFAULT_TOP_N, (
            f"Default top_n={DEFAULT_TOP_N} was not respected; got {len(recommendation_results)} results."
        )


# ---------------------------------------------------------------------------
# Tests — search()
# ---------------------------------------------------------------------------

class TestSearchMethod:
    """
    Test suite for ContentBasedRecommender.search().

    Keeps search-related assertions separate from recommendation assertions
    so failures in one area do not obscure failures in the other during review.
    """

    def test_search_returns_a_list(self, recommender_model):
        """
        Verify that search() always returns a list type.

        Why: The /search route serialises this directly into JSON.
        A non-list return type would cause a Pydantic serialisation error.
        """
        search_results = recommender_model.search("Batman")

        assert isinstance(search_results, list), (
            f"Expected list from search(), got {type(search_results).__name__}"
        )

    def test_search_returns_string_titles(self, recommender_model):
        """
        Verify that every item in the search result list is a string.

        Why: The frontend autocomplete widget expects a flat list of title
        strings. Returning dicts or other types here would break the UI contract.
        """
        search_results = recommender_model.search("Batman")

        for title in search_results:
            assert isinstance(title, str), (
                f"Expected string title in search results, got {type(title).__name__}: {title}"
            )