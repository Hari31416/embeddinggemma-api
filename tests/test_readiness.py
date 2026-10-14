import asyncio
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import PayloadTooLargeError
from multimodal_embedding.queue_manager import InferenceQueueManager
from multimodal_embedding.storage import StorageManager


def test_cancelled_waiter_releases_capacity():
    async def scenario():
        queue = InferenceQueueManager(1, 1)
        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            assert release.wait(5)

        running = asyncio.create_task(queue.run(work))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            waiting = asyncio.create_task(queue.run(lambda: None))
            await asyncio.sleep(0.01)
            assert queue.queue_depth == 1
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert queue.queue_depth == 0
        finally:
            release.set()
            await running
        assert queue.active_workers == 0
        assert await queue.run(lambda: 42) == 42

    asyncio.run(scenario())


def test_cancelled_running_job_retains_slot_until_thread_finishes():
    async def scenario():
        queue = InferenceQueueManager(1, 1)
        started, release = threading.Event(), threading.Event()
        second_started = threading.Event()

        def work():
            started.set()
            assert release.wait(5)

        running = asyncio.create_task(queue.run(work))
        while not started.is_set():
            await asyncio.sleep(0.001)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        following = asyncio.create_task(queue.run(second_started.set))
        try:
            await asyncio.sleep(0.02)
            assert queue.active_workers == 1
            assert not second_started.is_set()
        finally:
            release.set()
            await following
        assert queue.active_workers == queue.queue_depth == 0

    asyncio.run(scenario())


def test_storage_defaults_import_correctly(tmp_path):
    storage = StorageManager(base_dir=tmp_path)
    assert storage.settings.model_revision == "914f7f89142e33e77833254d9c9b90c3cef7303b"


def test_audio_cannot_exceed_model_window():
    with pytest.raises(ValidationError):
        Settings(max_audio_duration_seconds=31)


def test_prompt_mode_forwarded(client, fake_adapter):
    response = client.post(
        "/v1/embeddings",
        json={
            "items": [
                {
                    "id": "doc",
                    "parts": [
                        {"type": "text", "text": "title: Report | text: Revenue"}
                    ],
                }
            ],
            "prompt_mode": "none",
        },
    )
    assert response.status_code == 200
    assert fake_adapter.call_history[-1]["prompt_mode"] == "none"


def test_video_frame_group_validation(client, fixtures_dir):
    def upload(name, mime):
        with (fixtures_dir / name).open("rb") as file:
            response = client.post("/v1/uploads", files={"file": (name, file, mime)})
        assert response.status_code == 200
        return response.json()["upload_id"]

    image_id = upload("sample.png", "image/png")
    audio_id = upload("sample.wav", "audio/wav")
    for ids, expected in [
        ([image_id, image_id], 200),
        ([audio_id], 400),
        (["missing"], 404),
    ]:
        response = client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {
                        "id": "video",
                        "parts": [{"type": "video_frames", "upload_ids": ids}],
                    }
                ]
            },
        )
        assert response.status_code == expected


@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1", reason="Opt-in model test"
)
def test_real_limits_and_worker_parity(tmp_path, fixtures_dir):
    import sys
    from PIL import Image
    from multimodal_embedding.adapter.base import PreparedItem
    from multimodal_embedding.adapter.sentence_transformers_adapter import (
        SentenceTransformersAdapter,
    )

    cfg = Settings(device="cpu", upload_dir=tmp_path, max_context_tokens=8192)
    adapter = SentenceTransformersAdapter(cfg)
    adapter.load()
    # Exercise the actual worker adapter when this service is checked out in search-anything.
    worker_src = Path(__file__).resolve().parents[2] / "src"
    if not (worker_src / "search_anything").is_dir():
        pytest.skip("Worker parity requires the search-anything checkout")
    sys.path.insert(0, str(worker_src))
    try:
        from search_anything.embeddings import LocalEmbeddingModel, MODEL_REVISION

        worker = LocalEmbeddingModel(
            Path.home()
            / ".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots"
            / MODEL_REVISION,
            device="cpu",
        )
        with TestClient(create_app(settings=cfg, adapter=adapter)) as client:

            def upload(name, mime):
                with (fixtures_dir / name).open("rb") as file:
                    response = client.post(
                        "/v1/uploads", files={"file": (name, file, mime)}
                    )
                assert response.status_code == 200
                return response.json()["upload_id"]

            image_id = upload("sample.png", "image/png")
            audio_id = upload("sample.wav", "audio/wav")
            image = Image.open(fixtures_dir / "sample.png").convert("RGB")
            audio_path = fixtures_dir / "sample.wav"
            cases = [
                (
                    [
                        {
                            "type": "text",
                            "text": "title: Report | text: Quarterly revenue",
                        }
                    ],
                    worker.embed_texts(["title: Report | text: Quarterly revenue"])[0],
                    "none",
                    "document",
                ),
                (
                    [{"type": "text", "text": "city scene"}],
                    worker.embed_query("city scene"),
                    "auto",
                    "query",
                ),
                (
                    [{"type": "media", "upload_id": image_id}],
                    worker.embed_images([image])[0],
                    "none",
                    "document",
                ),
                (
                    [{"type": "media", "upload_id": audio_id}],
                    worker.embed_audio(audio_path),
                    "none",
                    "document",
                ),
                (
                    [{"type": "video_frames", "upload_ids": [image_id, image_id]}],
                    worker.embed_video([image, image]),
                    "none",
                    "document",
                ),
                (
                    [
                        {"type": "text", "text": "City scene"},
                        {"type": "media", "upload_id": image_id},
                        {"type": "media", "upload_id": audio_id},
                    ],
                    worker.embed_combined(
                        "City scene", image=image, audio_path=audio_path
                    ),
                    "none",
                    "document",
                ),
            ]
            for parts, expected, mode, task in cases:
                response = client.post(
                    "/v1/embeddings",
                    json={
                        "items": [{"id": "parity", "parts": parts}],
                        "prompt_mode": mode,
                        "task": task,
                    },
                )
                assert response.status_code == 200, response.text
                actual = response.json()["embeddings"][0]["embedding"]
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-4, atol=1e-5, err_msg=str(parts)
                )
            response = client.post(
                "/v1/embeddings",
                json={
                    "items": [
                        {
                            "id": "long",
                            "parts": [{"type": "text", "text": "word " * 9000}],
                        }
                    ]
                },
            )
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "payload_too_large"
            import io
            import soundfile as sf

            audio = io.BytesIO()
            sf.write(audio, np.zeros(31 * 16000, dtype=np.float32), 16000, format="WAV")
            response = client.post(
                "/v1/uploads",
                files={"file": ("long.wav", audio.getvalue(), "audio/wav")},
            )
            assert response.status_code == 413
        strict = Settings(device="cpu", max_context_tokens=10)
        adapter.settings = strict
        with pytest.raises(PayloadTooLargeError):
            adapter.encode([PreparedItem(id="strict", text="word " * 20)])
    finally:
        sys.path.remove(str(worker_src))


def test_video_sampling_limits(fixtures_dir):
    from multimodal_embedding.adapter.sentence_transformers_adapter import (
        SentenceTransformersAdapter,
    )

    video_path = str(fixtures_dir / "sample.mp4")
    adapter = SentenceTransformersAdapter(
        Settings(device="cpu", video_fps=2, max_video_frames=1)
    )
    assert len(adapter._video_frames(video_path)) == 1
    adapter.settings = Settings(device="cpu", video_fps=2, max_video_frames=2)
    assert len(adapter._video_frames(video_path)) == 2
    adapter.settings = Settings(device="cpu", max_image_pixels=100)
    with pytest.raises(PayloadTooLargeError):
        adapter._video_frames(video_path)
    adapter.settings = Settings(device="cpu", max_video_duration_seconds=0.1)
    with pytest.raises(PayloadTooLargeError):
        adapter._video_frames(video_path)


def test_video_frame_group_cap(temp_upload_dir):
    with TestClient(
        create_app(
            settings=Settings(
                upload_dir=temp_upload_dir, use_fake_adapter=True, max_video_frames=1
            )
        )
    ) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "items": [
                    {
                        "id": "video",
                        "parts": [
                            {"type": "video_frames", "upload_ids": ["one", "two"]}
                        ],
                    }
                ]
            },
        )
    assert response.status_code == 413
