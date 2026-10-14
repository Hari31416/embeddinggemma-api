from fastapi import status
from fastapi.testclient import TestClient


def test_list_models(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "data" in data
    assert len(data["data"]) == 1

    model_info = data["data"][0]
    assert model_info["test_mode"] is True
    assert model_info["id"] == "google/embeddinggemma-2"
    assert model_info["actual_device"] == "cpu"
    assert model_info["supported_inputs"] == ["text", "image", "audio", "video"]
    assert model_info["output_dimensions"] == [768, 512, 256, 128]

    limits = model_info["configured_limits"]
    assert limits["max_batch_size"] == 4
    assert limits["max_upload_bytes"] == 1048576
    assert limits["max_context_tokens"] == 8192
    assert limits["max_concurrency"] == 1
    assert limits["max_queue_size"] == 2
