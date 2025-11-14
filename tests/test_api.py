# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """Teste que le endpoint racine répond correctement."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Hybrid Search API is online" in response.json()["message"]
