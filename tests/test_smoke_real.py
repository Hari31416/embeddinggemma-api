import os
import time
from pathlib import Path
import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient

from multimodal_embedding.adapter.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings


def is_real_model_testing_enabled() -> bool:
    return os.environ.get("RUN_REAL_MODEL_TESTS") == "1"


@pytest.mark.skipif(
    not is_real_model_testing_enabled(),
    reason="Real model tests are opt-in. Set RUN_REAL_MODEL_TESTS=1 to run them.",
)
def test_real_model_cpu_smoke(temp_upload_dir: Path, fixtures_dir: Path) -> None:
    settings = Settings(
        device="cpu",
        upload_dir=temp_upload_dir,
        use_fake_adapter=False,
    )
    adapter = SentenceTransformersAdapter(settings)
    adapter.load()

    app = create_app(settings=settings, adapter=adapter)

    with TestClient(app) as client:
        # 1. Upload fixtures
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

        # 2. Text Query & Document
        q_resp = client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {
                        "id": "q1",
                        "parts": [{"type": "text", "text": "What is AI?"}],
                    }
                ],
                "task": "query",
            },
        )
        assert q_resp.status_code == 200
        q_vec = q_resp.json()["embeddings"][0]["embedding"]
        assert len(q_vec) == 768
        assert not any(np.isnan(q_vec))
        assert abs(np.linalg.norm(q_vec) - 1.0) < 1e-3

        # 3. Heterogeneous batch
        batch_items = [
            {"id": "text_item", "parts": [{"type": "text", "text": "A photo of city"}]},
            {"id": "image_item", "parts": [{"type": "media", "upload_id": img_id}]},
            {"id": "audio_item", "parts": [{"type": "media", "upload_id": aud_id}]},
            {"id": "video_item", "parts": [{"type": "media", "upload_id": vid_id}]},
            {
                "id": "combined_item",
                "parts": [
                    {"type": "text", "text": "City scene"},
                    {"type": "media", "upload_id": img_id},
                    {"type": "media", "upload_id": aud_id},
                ],
            },
        ]

        t0 = time.time()
        resp = client.post(
            "/v1/embeddings",
            json={"items": batch_items, "task": "document"},
        )
        duration = time.time() - t0
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == 5

        for item_emb in data["embeddings"]:
            vec = item_emb["embedding"]
            assert len(vec) == 768
            assert not any(np.isnan(vec))
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-3

        print(f"\nReal CPU heterogeneous batch inference latency: {duration:.3f}s")


@pytest.mark.skipif(
    not is_real_model_testing_enabled() or not torch.backends.mps.is_available(),
    reason="Real MPS tests require RUN_REAL_MODEL_TESTS=1 and Apple Silicon MPS availability.",
)
def test_real_model_mps_smoke(temp_upload_dir: Path, fixtures_dir: Path) -> None:
    settings = Settings(
        device="mps",
        upload_dir=temp_upload_dir,
        use_fake_adapter=False,
    )
    adapter = SentenceTransformersAdapter(settings)
    adapter.load()
    assert adapter.actual_device == "mps"
    assert adapter.device_fallback is False

    app = create_app(settings=settings, adapter=adapter)

    with TestClient(app) as client:
        with open(fixtures_dir / "sample.png", "rb") as f:
            img_id = client.post(
                "/v1/uploads", files={"file": ("sample.png", f, "image/png")}
            ).json()["upload_id"]

        resp = client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {
                        "id": "mps_img",
                        "parts": [{"type": "media", "upload_id": img_id}],
                    }
                ]
            },
        )
        assert resp.status_code == 200
        vec = resp.json()["embeddings"][0]["embedding"]
        assert len(vec) == 768
        assert not any(np.isnan(vec))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-3
