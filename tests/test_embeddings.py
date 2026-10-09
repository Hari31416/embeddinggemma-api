import asyncio
import concurrent.futures
from pathlib import Path
import numpy as np
from fastapi import status
from fastapi.testclient import TestClient

from multimodal_embedding.adapter.fake_adapter import FakeModelAdapter
from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings


def test_text_embeddings_query_and_document(
    client: TestClient, fake_adapter: FakeModelAdapter
) -> None:
    # Query task
    q_resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {"id": "q1", "parts": [{"type": "text", "text": "Search query"}]}
            ],
            "task": "query",
        },
    )
    assert q_resp.status_code == status.HTTP_200_OK
    q_data = q_resp.json()
    assert q_data["task"] == "query"
    assert q_data["dimension"] == 768
    assert len(q_data["embeddings"]) == 1
    assert q_data["embeddings"][0]["id"] == "q1"
    assert len(q_data["embeddings"][0]["embedding"]) == 768

    # Document task
    d_resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {"id": "d1", "parts": [{"type": "text", "text": "Document text"}]}
            ],
            "task": "document",
        },
    )
    assert d_resp.status_code == status.HTTP_200_OK
    d_data = d_resp.json()
    assert d_data["task"] == "document"

    # Verify adapter call history recorded tasks
    assert fake_adapter.call_history[-2]["task"] == "query"
    assert fake_adapter.call_history[-1]["task"] == "document"


def test_output_ordering_in_batch(client: TestClient) -> None:
    items = [
        {"id": "c_item", "parts": [{"type": "text", "text": "Charlie"}]},
        {"id": "a_item", "parts": [{"type": "text", "text": "Alpha"}]},
        {"id": "b_item", "parts": [{"type": "text", "text": "Bravo"}]},
    ]
    resp = client.post("/v1/embeddings", json={"items": items})
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert len(data["embeddings"]) == 3
    # IDs must strictly preserve input ordering
    assert [e["id"] for e in data["embeddings"]] == ["c_item", "a_item", "b_item"]


def test_media_and_combined_embeddings(client: TestClient, fixtures_dir: Path) -> None:
    # Upload an image and an audio
    with open(fixtures_dir / "sample.png", "rb") as f:
        img_resp = client.post(
            "/v1/uploads", files={"file": ("sample.png", f, "image/png")}
        )
    img_id = img_resp.json()["upload_id"]

    with open(fixtures_dir / "sample.wav", "rb") as f:
        aud_resp = client.post(
            "/v1/uploads", files={"file": ("sample.wav", f, "audio/wav")}
        )
    aud_id = aud_resp.json()["upload_id"]

    # Combined item: text + image + audio
    resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {
                    "id": "combined-1",
                    "parts": [
                        {"type": "text", "text": "A scene with sound and picture"},
                        {"type": "media", "upload_id": img_id},
                        {"type": "media", "upload_id": aud_id},
                    ],
                }
            ]
        },
    )
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert len(data["embeddings"]) == 1
    assert data["embeddings"][0]["id"] == "combined-1"
    assert len(data["embeddings"][0]["embedding"]) == 768


def test_heterogeneous_batch(client: TestClient, fixtures_dir: Path) -> None:
    # Upload image, audio, video
    with open(fixtures_dir / "sample.png", "rb") as f:
        img_id = client.post(
            "/v1/uploads", files={"file": ("sample.png", f, "image/png")}
        ).json()["upload_id"]

    with open(fixtures_dir / "sample.wav", "rb") as f:
        aud_id = client.post(
            "/v1/uploads", files={"file": ("sample.wav", f, "audio/wav")}
        ).json()["upload_id"]

    with open(fixtures_dir / "sample.mp4", "rb") as f:
        vid_id = client.post(
            "/v1/uploads", files={"file": ("sample.mp4", f, "video/mp4")}
        ).json()["upload_id"]

    items = [
        {"id": "item_text", "parts": [{"type": "text", "text": "Just text"}]},
        {"id": "item_image", "parts": [{"type": "media", "upload_id": img_id}]},
        {"id": "item_audio", "parts": [{"type": "media", "upload_id": aud_id}]},
        {
            "id": "item_video_text",
            "parts": [
                {"type": "text", "text": "A video"},
                {"type": "media", "upload_id": vid_id},
            ],
        },
    ]

    resp = client.post("/v1/embeddings", json={"items": items})
    assert resp.status_code == status.HTTP_200_OK
    data = resp.json()
    assert len(data["embeddings"]) == 4
    for e in data["embeddings"]:
        assert len(e["embedding"]) == 768


def test_manual_placeholder_validation(client: TestClient, fixtures_dir: Path) -> None:
    with open(fixtures_dir / "sample.png", "rb") as f:
        img_id = client.post(
            "/v1/uploads", files={"file": ("sample.png", f, "image/png")}
        ).json()["upload_id"]

    # 1. Matching placeholder count: 1 <|image|> and 1 image part
    valid_resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {
                    "id": "item-1",
                    "parts": [
                        {"type": "text", "text": "Look at this: <|image|>"},
                        {"type": "media", "upload_id": img_id},
                    ],
                }
            ]
        },
    )
    assert valid_resp.status_code == status.HTTP_200_OK

    # 2. Mismatch placeholder count: 2 <|image|> in text, but only 1 image part
    mismatch_resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {
                    "id": "item-2",
                    "parts": [
                        {"type": "text", "text": "Compare <|image|> with <|image|>"},
                        {"type": "media", "upload_id": img_id},
                    ],
                }
            ]
        },
    )
    assert mismatch_resp.status_code == status.HTTP_400_BAD_REQUEST
    assert mismatch_resp.json()["error"]["code"] == "invalid_input"
    assert "placeholder count mismatch" in mismatch_resp.json()["error"]["message"]


def test_matryoshka_dimension_truncation(client: TestClient) -> None:
    for dim in [512, 256, 128]:
        resp = client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {"id": f"item-{dim}", "parts": [{"type": "text", "text": "Hello"}]}
                ],
                "truncate_dim": dim,
            },
        )
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["dimension"] == dim
        vec = data["embeddings"][0]["embedding"]
        assert len(vec) == dim
        # Must be normalized
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-3

    # Invalid dimension
    inv_resp = client.post(
        "/v1/embeddings",
        json={
            "items": [{"id": "inv", "parts": [{"type": "text", "text": "Hello"}]}],
            "truncate_dim": 100,
        },
    )
    assert inv_resp.status_code == status.HTTP_400_BAD_REQUEST
    assert inv_resp.json()["error"]["code"] == "invalid_input"


def test_batch_size_limit_exceeded(client: TestClient) -> None:
    # Test setting max_batch_size is 4
    items = [
        {"id": f"item-{i}", "parts": [{"type": "text", "text": f"Text {i}"}]}
        for i in range(5)
    ]
    resp = client.post("/v1/embeddings", json={"items": items})
    assert resp.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_nonexistent_upload_id(client: TestClient) -> None:
    resp = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {
                    "id": "item-1",
                    "parts": [{"type": "media", "upload_id": "nonexistent-id"}],
                }
            ]
        },
    )
    assert resp.status_code == status.HTTP_404_NOT_FOUND
    assert resp.json()["error"]["code"] == "not_found"


def test_empty_items_validation(client: TestClient) -> None:
    resp = client.post("/v1/embeddings", json={"items": []})
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert resp.json()["error"]["code"] == "invalid_input"


def test_queue_saturation(temp_upload_dir: Path) -> None:
    settings = Settings(
        upload_dir=temp_upload_dir,
        max_concurrency=1,
        max_queue_size=1,
        use_fake_adapter=True,
    )
    adapter = FakeModelAdapter(settings)
    adapter.load()
    adapter.sleep_duration = 0.3  # Slow down inference to trigger queue fill

    app = create_app(settings=settings, adapter=adapter)

    with TestClient(app) as test_client:

        def make_req(item_id: str):
            return test_client.post(
                "/v1/embeddings",
                json={
                    "items": [
                        {
                            "id": item_id,
                            "parts": [{"type": "text", "text": "hello"}],
                        }
                    ]
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 1 running, 1 queued, remaining should hit queue saturation (503)
            futures = [executor.submit(make_req, f"req-{i}") for i in range(5)]
            results = [f.result() for f in futures]

        status_codes = [r.status_code for r in results]
        assert status.HTTP_200_OK in status_codes
        # At least one request must have been rejected due to queue saturation
        assert status.HTTP_503_SERVICE_UNAVAILABLE in status_codes
        saturated_responses = [
            r for r in results if r.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        ]
        assert any(
            r.json()["error"]["code"] == "queue_saturation" for r in saturated_responses
        )
