import time
from pathlib import Path
from PIL import Image
from fastapi import status
from fastapi.testclient import TestClient

from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings


def test_upload_image(client: TestClient, fixtures_dir: Path) -> None:
    img_path = fixtures_dir / "sample.png"
    with open(img_path, "rb") as f:
        response = client.post(
            "/v1/uploads",
            files={"file": ("sample.png", f, "image/png")},
        )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "upload_id" in data
    assert data["filename"] == "sample.png"
    assert data["modality"] == "image"
    assert data["size_bytes"] > 0
    assert data["expires_at"] > data["created_at"]


def test_upload_audio(client: TestClient, fixtures_dir: Path) -> None:
    audio_path = fixtures_dir / "sample.wav"
    with open(audio_path, "rb") as f:
        response = client.post(
            "/v1/uploads",
            files={"file": ("sample.wav", f, "audio/wav")},
        )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["modality"] == "audio"
    assert data["filename"] == "sample.wav"


def test_upload_video(client: TestClient, fixtures_dir: Path) -> None:
    video_path = fixtures_dir / "sample.mp4"
    with open(video_path, "rb") as f:
        response = client.post(
            "/v1/uploads",
            files={"file": ("sample.mp4", f, "video/mp4")},
        )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["modality"] == "video"
    assert data["filename"] == "sample.mp4"


def test_upload_unsupported_format(client: TestClient) -> None:
    response = client.post(
        "/v1/uploads",
        files={"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["error"]["code"] == "invalid_input"


def test_upload_file_too_large(client: TestClient) -> None:
    # Test setting max_upload_bytes is 1MB (1048576 bytes)
    oversized = b"0" * (1048576 + 100)
    response = client.post(
        "/v1/uploads",
        files={"file": ("big.png", oversized, "image/png")},
    )
    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["error"]["code"] == "payload_too_large"


def test_upload_image_exceeds_max_pixels(
    temp_upload_dir: Path, fixtures_dir: Path
) -> None:
    # Set max_image_pixels very low (100 pixels)
    strict_settings = Settings(
        upload_dir=temp_upload_dir,
        max_image_pixels=100,
        use_fake_adapter=True,
    )
    app = create_app(settings=strict_settings)
    img_path = fixtures_dir / "sample.png"  # 64x64 = 4096 pixels > 100

    with TestClient(app) as test_client:
        with open(img_path, "rb") as f:
            response = test_client.post(
                "/v1/uploads",
                files={"file": ("sample.png", f, "image/png")},
            )
        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        assert response.json()["error"]["code"] == "payload_too_large"
        assert "exceeds maximum allowed limit" in response.json()["error"]["message"]


def test_upload_audio_exceeds_max_duration(
    temp_upload_dir: Path, fixtures_dir: Path
) -> None:
    # Set max_audio_duration_seconds very low (0.1 seconds)
    strict_settings = Settings(
        upload_dir=temp_upload_dir,
        max_audio_duration_seconds=0.1,
        use_fake_adapter=True,
    )
    app = create_app(settings=strict_settings)
    audio_path = fixtures_dir / "sample.wav"  # 1.0s > 0.1s

    with TestClient(app) as test_client:
        with open(audio_path, "rb") as f:
            response = test_client.post(
                "/v1/uploads",
                files={"file": ("sample.wav", f, "audio/wav")},
            )
        assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        assert response.json()["error"]["code"] == "payload_too_large"


def test_upload_expiry_cleanup(temp_upload_dir: Path, fixtures_dir: Path) -> None:
    # Set TTL to 1 second
    short_ttl_settings = Settings(
        upload_dir=temp_upload_dir,
        upload_ttl_seconds=1,
        use_fake_adapter=True,
    )
    app = create_app(settings=short_ttl_settings)
    img_path = fixtures_dir / "sample.png"

    with TestClient(app) as test_client:
        with open(img_path, "rb") as f:
            resp = test_client.post(
                "/v1/uploads",
                files={"file": ("sample.png", f, "image/png")},
            )
        assert resp.status_code == status.HTTP_200_OK
        upload_id = resp.json()["upload_id"]

        # Sleep past TTL
        time.sleep(1.2)

        # Trying to embed with expired upload ID must return 404
        embed_resp = test_client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {
                        "id": "item-1",
                        "parts": [{"type": "media", "upload_id": upload_id}],
                    }
                ]
            },
        )
        assert embed_resp.status_code == status.HTTP_404_NOT_FOUND
        assert embed_resp.json()["error"]["code"] == "not_found"
