from fastapi import status
from fastapi.testclient import TestClient

from multimodal_embedding.adapter.fake_adapter import FakeModelAdapter
from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings


def test_health_live(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}


def test_health_ready_when_ready(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert data["model"] == "google/embeddinggemma-2"
    assert data["actual_device"] == "cpu"
    assert data["device_fallback"] is False
    assert data["queue_depth"] == 0
    assert data["active_workers"] == 0


def test_health_ready_when_not_ready(test_settings: Settings) -> None:
    adapter = FakeModelAdapter(test_settings)
    # Deliberately not calling adapter.load()
    app = create_app(settings=test_settings, adapter=adapter)
    # override ready state manually to test 503
    adapter._is_ready = False
    with TestClient(app) as test_client:
        adapter._is_ready = False
        response = test_client.get("/health/ready")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json()["status"] == "not_ready"
