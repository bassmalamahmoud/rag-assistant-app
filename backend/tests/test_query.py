import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "service" in data

def test_query_validation_error():
    # Empty query or missing required field should return HTTP 422 Unprocessable Entity
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422

    response_missing = client.post("/query", json={})
    assert response_missing.status_code == 422

    response_whitespace = client.post("/query", json={"question": "   "})
    assert response_whitespace.status_code == 422

def test_query_happy_path(monkeypatch):
    # Mock retrieval and generation to test API contract deterministically
    from app.services.retrieval import retrieval_service
    from app.services.generation import generation_service

    monkeypatch.setattr(
        retrieval_service,
        "retrieve",
        lambda q, top_k=None: (
            ["Mutable default arguments trap in Python functions."],
            [{"source": "02_python_functions.md", "header": "Common Mistake"}]
        )
    )
    monkeypatch.setattr(
        generation_service,
        "generate_answer",
        lambda q, docs, metas: (
            "Default parameter expressions are evaluated once at definition time. "
            "Use None as sentinel. [Source: 02_python_functions.md]"
        )
    )

    response = client.post("/query", json={"question": "What is the mutable default argument mistake?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "02_python_functions.md" in data["sources"]
    assert len(data["answer"]) > 10
