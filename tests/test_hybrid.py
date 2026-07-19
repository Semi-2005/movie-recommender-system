"""
Tests for the Hybrid Recommendation System
===========================================

Covers:
- Score normalization utilities (score_utils)
- Movie identity bridge (movie_identity)
- Hybrid recommender integration (hybrid endpoint)
- Regression: existing endpoints remain functional
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.utils.score_utils import (
    min_max_normalize,
    determine_alpha,
    ALPHA_FULL_HYBRID,
    ALPHA_PARTIAL_HYBRID,
    ALPHA_CB_ONLY,
)
from backend.app.services.movie_identity import identity_bridge


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient for HTTP endpoint tests."""
    with TestClient(app) as client:
        yield client


# ══════════════════════════════════════════════════════════════════════════
# Unit Tests: score_utils
# ══════════════════════════════════════════════════════════════════════════


class TestMinMaxNormalize:
    """Tests for min_max_normalize()."""

    def test_empty_list(self):
        assert min_max_normalize([]) == []

    def test_single_element(self):
        result = min_max_normalize([0.5])
        assert result == [1.0]

    def test_all_identical(self):
        result = min_max_normalize([0.3, 0.3, 0.3])
        assert result == [1.0, 1.0, 1.0]

    def test_normal_range(self):
        result = min_max_normalize([0.1, 0.5, 0.9])
        assert len(result) == 3
        # Min should map to ~0.0, max to ~1.0
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[2] == pytest.approx(1.0, abs=0.01)
        # Middle should be ~0.5
        assert result[1] == pytest.approx(0.5, abs=0.01)

    def test_preserves_order(self):
        scores = [0.2, 0.8, 0.4, 0.6]
        result = min_max_normalize(scores)
        # Relative ordering must be preserved
        assert result[1] > result[3] > result[2] > result[0]

    def test_two_elements(self):
        result = min_max_normalize([0.3, 0.7])
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(1.0, abs=0.01)


class TestDetermineAlpha:
    """Tests for determine_alpha()."""

    def test_cf_not_available(self):
        alpha, strategy = determine_alpha(cf_available=False)
        assert alpha == ALPHA_CB_ONLY
        assert strategy == "content_based_only"

    def test_full_hybrid(self):
        alpha, strategy = determine_alpha(
            cf_available=True, cf_result_count=10
        )
        assert alpha == ALPHA_FULL_HYBRID
        assert strategy == "hybrid_full"

    def test_partial_hybrid(self):
        alpha, strategy = determine_alpha(
            cf_available=True, cf_result_count=2
        )
        assert alpha == ALPHA_PARTIAL_HYBRID
        assert strategy == "hybrid_partial"

    def test_boundary_at_min_cf_results(self):
        # Exactly at the threshold → full hybrid
        alpha, strategy = determine_alpha(
            cf_available=True, cf_result_count=3, min_cf_results=3
        )
        assert alpha == ALPHA_FULL_HYBRID
        assert strategy == "hybrid_full"

    def test_boundary_below_min_cf_results(self):
        alpha, strategy = determine_alpha(
            cf_available=True, cf_result_count=2, min_cf_results=3
        )
        assert alpha == ALPHA_PARTIAL_HYBRID
        assert strategy == "hybrid_partial"


# ══════════════════════════════════════════════════════════════════════════
# Unit Tests: movie_identity
# ══════════════════════════════════════════════════════════════════════════


class TestMovieIdentityBridge:
    """Tests for the MovieIdentityBridge singleton."""

    def test_bridge_initialized(self):
        """Bridge should be initialized at import time."""
        assert identity_bridge is not None

    def test_index_to_movie_id_valid(self):
        """First movie in dataset should have a valid movieId."""
        movie_id = identity_bridge.index_to_movie_id(0)
        assert isinstance(movie_id, int)
        assert movie_id > 0

    def test_index_to_movie_id_out_of_bounds(self):
        """Out-of-range index should raise IndexError."""
        with pytest.raises(IndexError):
            identity_bridge.index_to_movie_id(999999)

    def test_index_to_movie_id_negative(self):
        """Negative index should raise IndexError."""
        with pytest.raises(IndexError):
            identity_bridge.index_to_movie_id(-1)

    def test_round_trip(self):
        """movie_index → movieId → movie_index should round-trip."""
        original_index = 0
        movie_id = identity_bridge.index_to_movie_id(original_index)
        recovered_index = identity_bridge.movie_id_to_index(movie_id)
        assert recovered_index == original_index

    def test_round_trip_multiple(self):
        """Round-trip for several known indices."""
        for idx in [0, 1, 5, 10, 50]:
            movie_id = identity_bridge.index_to_movie_id(idx)
            recovered = identity_bridge.movie_id_to_index(movie_id)
            assert recovered == idx, (
                f"Round-trip failed for index {idx}: "
                f"movieId={movie_id}, recovered={recovered}"
            )

    def test_movie_id_to_index_unknown(self):
        """Unknown movieId should return None."""
        result = identity_bridge.movie_id_to_index(9999999)
        assert result is None

    def test_get_movie_metadata(self):
        """Metadata lookup should return title and genres."""
        movie_id = identity_bridge.index_to_movie_id(0)
        metadata = identity_bridge.get_movie_metadata(movie_id)

        assert metadata is not None
        assert "title" in metadata
        assert "genres" in metadata
        assert "rating" in metadata
        assert "rating_count" in metadata
        assert metadata["movie_id"] == movie_id

    def test_get_movie_metadata_unknown(self):
        """Unknown movieId should return None."""
        assert identity_bridge.get_movie_metadata(9999999) is None


# ══════════════════════════════════════════════════════════════════════════
# Integration Tests: Hybrid Endpoint
# ══════════════════════════════════════════════════════════════════════════


class TestHybridEndpoint:
    """Tests for GET /recommend/hybrid."""

    def test_hybrid_known_movie(self, api_client):
        """A popular movie should return hybrid recommendations."""
        response = api_client.get(
            "/recommend/hybrid", params={"movie": "Toy Story", "top_n": 5}
        )
        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert "strategy" in data
        assert "alpha" in data
        assert "fusion_time_ms" in data

        # Each recommendation should have the unified schema
        rec = data["recommendations"][0]
        assert "movie_id" in rec
        assert "title" in rec
        assert "hybrid_score" in rec

    def test_hybrid_response_strategy_field(self, api_client):
        """Response should contain a valid strategy label."""
        response = api_client.get(
            "/recommend/hybrid", params={"movie": "Toy Story"}
        )
        data = response.json()
        strategy = data.get("strategy", "")
        assert any(
            s in strategy
            for s in ["hybrid_full", "hybrid_partial", "content_based_only"]
        )

    def test_hybrid_alpha_override(self, api_client):
        """alpha_override should be reflected in the response."""
        response = api_client.get(
            "/recommend/hybrid",
            params={"movie": "Toy Story", "alpha_override": 0.8},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["alpha"] == 0.8

    def test_hybrid_alpha_override_pure_cb(self, api_client):
        """alpha=1.0 should produce content-based-only results."""
        response = api_client.get(
            "/recommend/hybrid",
            params={"movie": "Toy Story", "alpha_override": 1.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["alpha"] == 1.0
        # All collaborative scores should be present but weighted to 0
        for rec in data["recommendations"]:
            assert "hybrid_score" in rec

    def test_hybrid_unknown_movie(self, api_client):
        """Unknown movie should return suggestions, not crash."""
        response = api_client.get(
            "/recommend/hybrid",
            params={"movie": "xyznonexistentmovie123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recommendations"] == []
        assert "suggestions" in data or "message" in data

    def test_hybrid_invalid_alpha(self, api_client):
        """alpha_override outside [0, 1] should be rejected (422)."""
        response = api_client.get(
            "/recommend/hybrid",
            params={"movie": "Toy Story", "alpha_override": 1.5},
        )
        assert response.status_code == 422

    def test_hybrid_top_n_respected(self, api_client):
        """Should return at most top_n recommendations."""
        response = api_client.get(
            "/recommend/hybrid", params={"movie": "Toy Story", "top_n": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) <= 3

    def test_hybrid_scores_descending(self, api_client):
        """Recommendations should be sorted by hybrid_score descending."""
        response = api_client.get(
            "/recommend/hybrid", params={"movie": "Toy Story", "top_n": 10}
        )
        data = response.json()
        scores = [r["hybrid_score"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════
# Regression Tests: Existing Endpoints
# ══════════════════════════════════════════════════════════════════════════


class TestExistingEndpointsRegression:
    """Ensure existing endpoints still work after hybrid integration."""

    def test_content_based_endpoint(self, api_client):
        response = api_client.get(
            "/recommend", params={"movie": "Toy Story"}
        )
        assert response.status_code == 200
        assert "recommendations" in response.json()

    def test_collaborative_endpoint(self, api_client):
        response = api_client.get(
            "/recommend/collaborative", params={"movie": "Toy Story"}
        )
        assert response.status_code == 200

    def test_search_endpoint(self, api_client):
        response = api_client.get("/search", params={"q": "Toy"})
        assert response.status_code == 200
        assert "results" in response.json()

    def test_stats_endpoint(self, api_client):
        response = api_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "content_based" in data
        assert "collaborative" in data

    def test_health_endpoint(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200

    def test_ping_endpoint(self, api_client):
        response = api_client.get("/ping")
        assert response.status_code == 200
