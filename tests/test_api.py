"""
API endpoint tests for the research generator.

Uses FastAPI TestClient for integration testing.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_returns_status(self, client):
        """Health endpoint should return status field."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_returns_version(self, client):
        """Health endpoint should return version field."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "2.0.0"


class TestModelsEndpoint:
    """Tests for the models listing endpoint."""

    def test_models_returns_200(self, client):
        """Models endpoint should return 200 OK."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

    def test_models_returns_recommended_models(self, client):
        """Models endpoint should return recommended models."""
        response = client.get("/api/v1/models")
        data = response.json()
        assert "recommended_models" in data
        assert "best_quality" in data["recommended_models"]
        assert "balanced" in data["recommended_models"]
        assert "lightweight" in data["recommended_models"]


class TestResearchTypesEndpoint:
    """Tests for the research types listing endpoint."""

    def test_research_types_returns_200(self, client):
        """Research types endpoint should return 200 OK."""
        response = client.get("/api/v1/research-types")
        assert response.status_code == 200

    def test_research_types_returns_list(self, client):
        """Research types endpoint should return list of types."""
        response = client.get("/api/v1/research-types")
        data = response.json()
        assert "research_types" in data
        assert len(data["research_types"]) > 0

    def test_research_types_includes_systematic_review(self, client):
        """Research types should include systematic_review."""
        response = client.get("/api/v1/research-types")
        data = response.json()
        values = [t["value"] for t in data["research_types"]]
        assert "systematic_review" in values


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_app_info(self, client):
        """Root endpoint should return app information."""
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestResearchEndpoint:
    """Tests for the research generation endpoint."""

    def test_research_requires_topic(self, client):
        """Research endpoint should require topic field."""
        response = client.post(
            "/api/v1/research",
            json={
                "research_question": "How does AI affect learning?",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_research_requires_research_question(self, client):
        """Research endpoint should require research_question field."""
        response = client.post(
            "/api/v1/research",
            json={
                "topic": "Impact of AI on education",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_research_validates_topic_length(self, client):
        """Research endpoint should validate topic minimum length."""
        response = client.post(
            "/api/v1/research",
            json={
                "topic": "AI",  # Too short (min 10 chars)
                "research_question": "How does AI affect learning outcomes?",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_research_validates_paper_limit(self, client):
        """Research endpoint should validate paper_limit range."""
        response = client.post(
            "/api/v1/research",
            json={
                "topic": "Impact of AI on education",
                "research_question": "How does AI affect learning outcomes?",
                "paper_limit": 200,  # Too high (max 100)
            },
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.skip(reason="Requires LLM to be running")
    def test_research_generates_project(self, client):
        """Research endpoint should generate a project."""
        response = client.post(
            "/api/v1/research",
            json={
                "topic": "Impact of artificial intelligence on education",
                "research_question": "How does AI affect learning outcomes?",
                "research_type": "systematic_review",
                "discipline": "education",
                "use_llm": False,  # Disable LLM for faster test
                "search_papers": False,  # Disable search for faster test
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "request_id" in data
        assert "topic" in data
