"""
conftest.py — Shared pytest fixtures for the Movie Recommendation System test suite.

This module is automatically discovered by pytest and makes its fixtures
available to every test file in the `tests/` package without explicit imports.

Design decisions:
- `scope="module"`: The ML model and the FastAPI TestClient are both
  expensive to initialise (CSV loading, TF-IDF matrix construction).
  Module scope means they are created once per test file, not once per
  test function, keeping the full suite fast.
- Fixtures are kept here rather than duplicated across test files so that
  Berat and Berkay only need to change this single file when the
  application bootstrap logic changes.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.content_based import recommender as content_based_recommender


@pytest.fixture(scope="module")
def recommender_model():
    """
    Provide the singleton ContentBasedRecommender instance.

    The model is already initialised at import time (module-level singleton
    pattern in content_based.py), so this fixture simply exposes it to tests
    without triggering a second expensive initialisation cycle.
    """
    return content_based_recommender


@pytest.fixture(scope="module")
def api_client():
    """
    Provide a FastAPI TestClient configured against the production `app` instance.

    TestClient wraps the ASGI app with a `requests`-compatible interface so
    tests can make real HTTP calls without spinning up a live server.
    The `with` block ensures proper lifespan event handling (startup / shutdown).
    """
    with TestClient(app) as client:
        yield client
