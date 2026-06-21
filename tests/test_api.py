"""
test_api.py — Integration tests for the Movie Recommendation System REST API.

Test Strategy
-------------
These tests exercise the full HTTP request/response cycle using FastAPI's
`TestClient` (backed by `httpx` under the hood), which runs the ASGI app
in-process without a real network socket.  This gives us true end-to-end
coverage of routing, query-parameter validation, response serialisation, and
service-layer wiring — all without the overhead or flakiness of a live server.

All tests receive the shared `api_client` fixture from conftest.py.

Conventions used in this file:
- HTTP status codes are compared against the `http.HTTPStatus` enum values
  for readability (e.g. HTTPStatus.OK instead of the magic number 200).
- Response body keys are stored in named constants to prevent silent breakage
  if the API contract changes — a failing constant reference is easier to
  diagnose in a PR than a failing string literal.
"""

# ── Standard library ─────────────────────────────────────────────────────────
from http import HTTPStatus

# ── Third-party ──────────────────────────────────────────────────────────────
import pytest


# ---------------------------------------------------------------------------
# Constants — API query parameters & response body keys
# ---------------------------------------------------------------------------

KNOWN_MOVIE_TITLE: str = "Interstellar"
SEARCH_QUERY_BATMAN: str = "Batman"
RECOMMENDATIONS_TOP_N: int = 5

# Response body top-level keys
RESPONSE_KEY_MESSAGE: str = "message"
RESPONSE_KEY_RECOMMENDATIONS: str = "recommendations"
RESPONSE_KEY_INPUT_MOVIE: str = "input_movie"
RESPONSE_KEY_TOTAL_RECOMMENDATIONS: str = "total_recommendations"
RESPONSE_KEY_QUERY: str = "query"
RESPONSE_KEY_RESULTS: str = "results"
RESPONSE_KEY_RESULTS_COUNT: str = "results_count"
RESPONSE_KEY_STATUS: str = "status"


# ---------------------------------------------------------------------------
# Tests — Root & Health endpoints
# ---------------------------------------------------------------------------

class TestInfrastructureEndpoints:
    """
    Smoke tests for endpoints that verify the service is alive.

    These tests should always be the first to run (alphabetically they are)
    and should be the fastest to pass.  A failure here means the app cannot
    even start, which blocks all other tests from being meaningful.
    """

    def test_root_endpoint_returns_200_ok(self, api_client):
        """
        Verify that GET / returns HTTP 200 OK.

        Why: The root endpoint is the most basic liveness signal. If this
        fails, every downstream test will also fail due to the app not
        being routable at all. Catching it here isolates the root cause.
        """
        response = api_client.get("/")

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 from GET /, got {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_root_endpoint_body_contains_message_key(self, api_client):
        """
        Verify that GET / returns a JSON body with a 'message' key.

        Why: The frontend status banner reads `response.message` to display
        the 'API is running' notice. A missing key silently breaks the banner.
        """
        response = api_client.get("/")
        response_body = response.json()

        assert RESPONSE_KEY_MESSAGE in response_body, (
            f"Expected '{RESPONSE_KEY_MESSAGE}' key in response body: {response_body}"
        )

    def test_health_endpoint_returns_200_ok(self, api_client):
        """
        Verify that GET /health returns HTTP 200 OK.

        Why: /health is polled by container orchestration tools (e.g. Kubernetes
        liveness probes). A non-200 response causes the container to be
        restarted, creating a crash-loop — a production-critical contract.
        """
        response = api_client.get("/health")

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 from GET /health, got {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_health_endpoint_body_contains_status_key(self, api_client):
        """
        Verify that GET /health returns a JSON body with a 'status' key set to 'ok'.

        Why: External monitoring dashboards parse `response.status == 'ok'`
        to determine service health. Any deviation from this contract causes
        false-positive alerts for Berat and Berkay on-call.
        """
        response = api_client.get("/health")
        response_body = response.json()

        assert RESPONSE_KEY_STATUS in response_body, (
            f"Expected '{RESPONSE_KEY_STATUS}' key in /health response: {response_body}"
        )
        assert response_body[RESPONSE_KEY_STATUS] == "ok", (
            f"Expected status='ok', got '{response_body[RESPONSE_KEY_STATUS]}'"
        )

    def test_ping_endpoint_returns_200_ok(self, api_client):
        """
        Verify that GET /ping returns HTTP 200 OK.

        Why: /ping is a developer-facing lightweight check registered directly
        on the FastAPI `app` instance (not the router). Testing it confirms
        the ASGI app itself is functional, separate from router wiring.
        """
        response = api_client.get("/ping")

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 from GET /ping, got {response.status_code}."
        )


# ---------------------------------------------------------------------------
# Tests — GET /recommend
# ---------------------------------------------------------------------------

class TestRecommendEndpoint:
    """
    Test suite for the GET /recommend endpoint.

    These are the most business-critical tests in the suite because /recommend
    is the core value proposition of the application.
    """

    def test_recommend_returns_200_for_known_movie(self, api_client):
        """
        Verify that GET /recommend?movie=Interstellar&top_n=5 returns HTTP 200 OK.

        Why: A non-200 for a known movie indicates either a routing error, an
        unhandled exception in the service layer, or a data-loading failure.
        Confirming 200 is the minimum bar before inspecting the response body.
        """
        response = api_client.get(
            "/recommend",
            params={"movie": KNOWN_MOVIE_TITLE, "top_n": RECOMMENDATIONS_TOP_N}
        )

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 from GET /recommend?movie={KNOWN_MOVIE_TITLE}&top_n={RECOMMENDATIONS_TOP_N}, "
            f"got {response.status_code}. Body: {response.text}"
        )

    def test_recommend_returns_exactly_top_n_movies(self, api_client):
        """
        Verify that GET /recommend with top_n=5 returns exactly 5 movies in the response.

        Why: The `top_n` query parameter is part of the public API contract.
        Returning fewer or more than requested violates user expectations and
        breaks any frontend pagination logic that relies on knowing the exact count.
        """
        response = api_client.get(
            "/recommend",
            params={"movie": KNOWN_MOVIE_TITLE, "top_n": RECOMMENDATIONS_TOP_N}
        )
        response_body = response.json()

        assert RESPONSE_KEY_RECOMMENDATIONS in response_body, (
            f"Missing '{RESPONSE_KEY_RECOMMENDATIONS}' key in response body: {response_body}"
        )

        actual_recommendation_count = len(response_body[RESPONSE_KEY_RECOMMENDATIONS])
        assert actual_recommendation_count == RECOMMENDATIONS_TOP_N, (
            f"Expected exactly {RECOMMENDATIONS_TOP_N} recommendations, "
            f"got {actual_recommendation_count}."
        )

    def test_recommend_response_body_contains_required_keys(self, api_client):
        """
        Verify that the /recommend response body contains all required top-level keys.

        Why: The frontend destructures `input_movie`, `total_recommendations`,
        and `recommendations` from the response. Any missing key produces a
        JavaScript `undefined` read that is hard to debug without this test.
        """
        response = api_client.get(
            "/recommend",
            params={"movie": KNOWN_MOVIE_TITLE, "top_n": RECOMMENDATIONS_TOP_N}
        )
        response_body = response.json()

        expected_top_level_keys = {
            RESPONSE_KEY_INPUT_MOVIE,
            RESPONSE_KEY_TOTAL_RECOMMENDATIONS,
            RESPONSE_KEY_RECOMMENDATIONS,
        }

        for required_key in expected_top_level_keys:
            assert required_key in response_body, (
                f"Missing required key '{required_key}' in /recommend response: {response_body}"
            )

    def test_recommend_total_count_matches_recommendations_list_length(self, api_client):
        """
        Verify that `total_recommendations` in the response body equals the actual list length.

        Why: A mismatch between the declared count and the actual array length
        breaks client-side logic that uses the count for display (e.g. "Showing
        5 of 5 results") without iterating the full array.
        """
        response = api_client.get(
            "/recommend",
            params={"movie": KNOWN_MOVIE_TITLE, "top_n": RECOMMENDATIONS_TOP_N}
        )
        response_body = response.json()

        declared_total = response_body[RESPONSE_KEY_TOTAL_RECOMMENDATIONS]
        actual_list_length = len(response_body[RESPONSE_KEY_RECOMMENDATIONS])

        assert declared_total == actual_list_length, (
            f"total_recommendations ({declared_total}) does not match "
            f"len(recommendations) ({actual_list_length})."
        )

    def test_recommend_missing_movie_param_returns_422(self, api_client):
        """
        Verify that GET /recommend without the required `movie` param returns HTTP 422.

        Why: FastAPI automatically validates required Query parameters and
        returns 422 Unprocessable Entity on missing/invalid input.  This test
        documents that the validation guard is active — removing the `...`
        default from the route would accidentally make the param optional.
        """
        response = api_client.get("/recommend")

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
            f"Expected 422 for missing 'movie' param, got {response.status_code}."
        )

    def test_recommend_unknown_movie_returns_200_with_empty_recommendations(self, api_client):
        """
        Verify that an unrecognised movie title returns 200 with an empty recommendations list.

        Why: The route is designed to return a graceful 'not found' JSON message
        rather than a 404 error, because the client app renders this case with a
        custom UI state (not an error screen). This test guards against a future
        refactor accidentally turning a graceful 200 into a hard 404.
        """
        response = api_client.get(
            "/recommend",
            params={"movie": "xQzAbsolutelyNotAMoviexQz", "top_n": RECOMMENDATIONS_TOP_N}
        )
        response_body = response.json()

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 for unknown movie, got {response.status_code}."
        )
        assert RESPONSE_KEY_RECOMMENDATIONS in response_body, (
            f"Missing '{RESPONSE_KEY_RECOMMENDATIONS}' key in not-found response: {response_body}"
        )
        assert response_body[RESPONSE_KEY_RECOMMENDATIONS] == [], (
            f"Expected empty list for unknown movie, got: {response_body[RESPONSE_KEY_RECOMMENDATIONS]}"
        )


# ---------------------------------------------------------------------------
# Tests — GET /search
# ---------------------------------------------------------------------------

class TestSearchEndpoint:
    """
    Test suite for the GET /search endpoint.

    Search is a secondary feature supporting frontend autocomplete.
    These tests confirm the endpoint is reachable and returns the expected shape.
    """

    def test_search_returns_200_for_valid_query(self, api_client):
        """
        Verify that GET /search?q=Batman returns HTTP 200 OK.

        Why: /search is used by the frontend autocomplete on every keystroke.
        A non-200 response means the autocomplete widget is completely broken
        for all users, not just those searching for 'Batman'.
        """
        response = api_client.get(
            "/search",
            params={"q": SEARCH_QUERY_BATMAN}
        )

        assert response.status_code == HTTPStatus.OK, (
            f"Expected 200 from GET /search?q={SEARCH_QUERY_BATMAN}, "
            f"got {response.status_code}. Body: {response.text}"
        )

    def test_search_response_body_contains_required_keys(self, api_client):
        """
        Verify that GET /search response contains 'query', 'results_count', and 'results' keys.

        Why: The frontend reads all three keys to render the autocomplete dropdown.
        A missing key causes a silent undefined access in JavaScript.
        """
        response = api_client.get(
            "/search",
            params={"q": SEARCH_QUERY_BATMAN}
        )
        response_body = response.json()

        expected_search_response_keys = {
            RESPONSE_KEY_QUERY,
            RESPONSE_KEY_RESULTS_COUNT,
            RESPONSE_KEY_RESULTS,
        }

        for required_key in expected_search_response_keys:
            assert required_key in response_body, (
                f"Missing required key '{required_key}' in /search response: {response_body}"
            )

    def test_search_results_is_a_list(self, api_client):
        """
        Verify that the 'results' field in the /search response is a list.

        Why: The frontend maps over `response.results` to render suggestion items.
        A non-list type (e.g. None or a string) causes an uncaught TypeError.
        """
        response = api_client.get(
            "/search",
            params={"q": SEARCH_QUERY_BATMAN}
        )
        response_body = response.json()

        assert isinstance(response_body[RESPONSE_KEY_RESULTS], list), (
            f"Expected list for 'results', got {type(response_body[RESPONSE_KEY_RESULTS]).__name__}"
        )

    def test_search_results_count_matches_actual_results_length(self, api_client):
        """
        Verify that `results_count` matches the actual length of the `results` list.

        Why: Same principle as the recommend endpoint — a declared count that
        differs from the actual array size breaks client-side display logic.
        """
        response = api_client.get(
            "/search",
            params={"q": SEARCH_QUERY_BATMAN}
        )
        response_body = response.json()

        declared_count = response_body[RESPONSE_KEY_RESULTS_COUNT]
        actual_results_length = len(response_body[RESPONSE_KEY_RESULTS])

        assert declared_count == actual_results_length, (
            f"results_count ({declared_count}) does not match "
            f"len(results) ({actual_results_length})."
        )

    def test_search_missing_q_param_returns_422(self, api_client):
        """
        Verify that GET /search without the required `q` param returns HTTP 422.

        Why: Same rationale as the /recommend validation test — confirms that
        FastAPI's query parameter guard is properly configured and active.
        """
        response = api_client.get("/search")

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, (
            f"Expected 422 for missing 'q' param, got {response.status_code}."
        )
