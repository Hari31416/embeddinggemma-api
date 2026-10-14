import asyncio
import base64
import io
import json
import socket
import time
from unittest.mock import Mock

import pytest
from PIL import Image

from multimodal_embedding import media_inputs
from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import InvalidInputError, PayloadTooLargeError


@pytest.fixture
def png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
    return stream.getvalue()


def payload(*parts):
    return {"items": [{"id": "image", "parts": list(parts)}]}


def test_multipart_shared_file_and_cleanup(client, png, fake_adapter):
    body = {
        "items": [
            {"id": "a", "parts": [{"type": "media", "file": "photo"}]},
            {
                "id": "b",
                "parts": [
                    {"type": "text", "text": "red"},
                    {"type": "media", "file": "photo"},
                ],
            },
        ]
    }
    response = client.post(
        "/v1/embeddings",
        data={"payload": json.dumps(body)},
        files={"photo": ("red.png", png, "image/png")},
    )
    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()["embeddings"]] == ["a", "b"]
    assert not client.app.state.storage.uploads
    assert not list(client.app.state.settings.upload_dir.iterdir())


@pytest.mark.parametrize("data_url", [True, False])
def test_base64(client, png, data_url):
    data = base64.b64encode(png).decode()
    part = {
        "type": "media",
        "file_data": f"data:image/png;base64,{data}" if data_url else data,
    }
    if not data_url:
        part["content_type"] = "image/png"
    response = client.post("/v1/embeddings", json=payload(part))
    assert response.status_code == 200, response.text
    assert not client.app.state.storage.uploads


@pytest.mark.parametrize(
    "part",
    [
        {"type": "media"},
        {"type": "media", "upload_id": "id", "file_url": "https://example.com/a"},
        {"type": "media", "file_data": "aGVsbG8="},
        {"type": "media", "file_data": "data:image/png;base64,!!!!"},
        {"type": "media", "file_data": "data:image/png,AAAA"},
        {"type": "media", "file": "absent"},
    ],
)
def test_invalid_media_source(client, part):
    response = client.post("/v1/embeddings", json=payload(part))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


def test_cleanup_after_later_part_fails(client, png):
    response = client.post(
        "/v1/embeddings",
        json=payload(
            {
                "type": "media",
                "file_data": "data:image/png;base64," + base64.b64encode(png).decode(),
            },
            {"type": "media", "upload_id": "missing"},
        ),
    )
    assert response.status_code == 404
    assert not client.app.state.storage.uploads
    assert not list(client.app.state.settings.upload_dir.iterdir())


def test_media_and_request_size_limits(client):
    response = client.post(
        "/v1/embeddings",
        json=payload(
            {"type": "media", "file_data": "data:image/png;base64," + "A" * 1400000}
        ),
    )
    assert response.status_code == 413
    client.app.state.settings.max_embedding_request_bytes = 1024
    response = client.post(
        "/v1/embeddings",
        content=b" " * 1025,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_multipart_missing_extra_duplicate_and_invalid_json(client, png):
    cases = [
        [("photo", ("a.png", png, "image/png"))],
        [("payload", (None, "{}")), ("payload", (None, "{}"))],
        [("payload", (None, "invalid"))],
        [
            (
                "payload",
                (None, json.dumps(payload({"type": "media", "file": "photo"}))),
            ),
            ("other", ("a.png", png, "image/png")),
        ],
        [
            (
                "payload",
                (None, json.dumps(payload({"type": "media", "file": "photo"}))),
            ),
            ("photo", ("a.png", png, "image/png")),
            ("photo", ("b.png", png, "image/png")),
        ],
    ]
    for files in cases:
        response = client.post("/v1/embeddings", files=files)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_input"


def test_openapi(client):
    content = client.get("/openapi.json").json()["paths"]["/v1/embeddings"]["post"][
        "requestBody"
    ]["content"]
    assert set(content) == {"application/json", "multipart/form-data"}
    assert "items" in content["application/json"]["schema"]["properties"]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "100.64.0.1",
    ],
)
def test_url_blocks_nonpublic_addresses(monkeypatch, address):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))],
    )
    with pytest.raises(InvalidInputError, match="public"):
        asyncio.run(media_inputs.download_url("https://example.com/a.png", Settings()))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/a",
        "https://user:pass@example.com/a",
        "https://example.com:444/a",
        "file:///etc/passwd",
        "https://example.com/a#x",
    ],
)
def test_url_restrictions(url):
    with pytest.raises(InvalidInputError):
        asyncio.run(media_inputs.download_url(url, Settings()))


def test_url_allowlist():
    with pytest.raises(InvalidInputError, match="hostname"):
        asyncio.run(
            media_inputs.download_url(
                "https://other.example/a", Settings(url_allowed_hosts=["cdn.example"])
            )
        )


@pytest.fixture
def fake_https(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    response = Mock(status=200)
    headers = {"Content-Type": "image/png"}
    response.getheader.side_effect = lambda name, default=None: headers.get(
        name, default
    )
    response.read1.side_effect = [b"", b""]
    conn = Mock()
    conn.getresponse.return_value = response
    factory = Mock(return_value=conn)
    monkeypatch.setattr(media_inputs.http.client, "HTTPSConnection", factory)
    return conn, response, headers, factory


def test_url_success_preserves_signed_path_and_pins_ip(
    client, png, fake_https, monkeypatch
):
    conn, response, headers, factory = fake_https
    response.read1.side_effect = [png, b""]
    result = client.post(
        "/v1/embeddings",
        json=payload(
            {"type": "media", "file_url": "https://example.com/a.png?signature=secret"}
        ),
    )
    assert result.status_code == 200, result.text
    assert not client.app.state.storage.uploads
    assert factory.call_args.args == ("example.com",)
    assert conn.request.call_args.args == ("GET", "/a.png?signature=secret")
    create_connection = Mock()
    monkeypatch.setattr(socket, "create_connection", create_connection)
    conn._create_connection(("example.com", 443), 1, None)
    assert create_connection.call_args.args == (("93.184.216.34", 443),)
    conn.close.assert_called_once()
    response.close.assert_called_once()


@pytest.mark.parametrize(
    "kind", ["redirect", "length", "stream", "timeout", "compressed"]
)
def test_url_download_failures(fake_https, kind):
    conn, response, headers, factory = fake_https
    if kind == "redirect":
        response.status = 302
    elif kind == "length":
        headers["Content-Length"] = "100"
    elif kind == "stream":
        response.read1.side_effect = [b"x" * 11]
    elif kind == "timeout":
        response.read1.side_effect = TimeoutError
    else:
        headers["Content-Encoding"] = "gzip"
    expected = (
        PayloadTooLargeError if kind in ("length", "stream") else InvalidInputError
    )
    with pytest.raises(expected) as exc:
        asyncio.run(
            media_inputs.download_url(
                "https://example.com/a?secret=yes", Settings(url_download_max_bytes=10)
            )
        )
    assert "secret" not in str(exc.value)
    conn.close.assert_called_once()


def test_total_download_timeout(fake_https):
    conn, response, headers, factory = fake_https

    def slow_read(size):
        time.sleep(0.02)
        return b"x"

    response.read1.side_effect = slow_read
    with pytest.raises(InvalidInputError):
        asyncio.run(
            media_inputs.download_url(
                "https://example.com/a", Settings(url_download_timeout_seconds=0.05)
            )
        )


def test_cancelled_inference_keeps_request_media_until_thread_done(
    test_settings, fake_adapter, png, monkeypatch
):
    import threading
    from types import SimpleNamespace

    from multimodal_embedding.media_inputs import MediaResolver
    from multimodal_embedding.queue_manager import InferenceQueueManager
    from multimodal_embedding.routes.embeddings import create_embeddings
    from multimodal_embedding.schemas import EmbeddingRequest
    from multimodal_embedding.storage import StorageManager

    async def check():
        storage = StorageManager(settings=test_settings)
        resolver = MediaResolver(storage, test_settings, {})
        started = threading.Event()
        release = threading.Event()
        read_after_cancel = []

        def encode(items, **kwargs):
            started.set()
            assert release.wait(timeout=2)
            read_after_cancel.append(items[0].media[0].filepath)
            assert __import__("pathlib").Path(read_after_cancel[0]).read_bytes() == png
            return [[0.0] * 768]

        monkeypatch.setattr(fake_adapter, "encode", encode)
        state = SimpleNamespace(
            storage=storage,
            settings=test_settings,
            adapter=fake_adapter,
            queue_manager=InferenceQueueManager(),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        body = EmbeddingRequest.model_validate(
            payload(
                {
                    "type": "media",
                    "file_data": "data:image/png;base64,"
                    + base64.b64encode(png).decode(),
                }
            )
        )

        async def run_request():
            try:
                await create_embeddings(request, (body, resolver))
            finally:
                await resolver.close()

        task = asyncio.create_task(run_request())
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert storage.uploads
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert read_after_cancel
        assert not storage.uploads
        assert not list(test_settings.upload_dir.iterdir())

    asyncio.run(check())


def test_multipart_video_frame_order_reuse_and_cleanup(client, png, monkeypatch):
    seen = []

    def encode(items, **kwargs):
        frames = items[0].media[0].frame_paths
        assert frames[0] == frames[2]
        assert frames[0] != frames[1]
        names = {
            str(record.filepath): record.filename
            for record in client.app.state.storage.uploads.values()
        }
        assert [names[frame] for frame in frames] == [
            "second.png",
            "first.png",
            "second.png",
        ]
        seen.extend(frames)
        return [[0.0] * 768]

    monkeypatch.setattr(client.app.state.adapter, "encode", encode)
    response = client.post(
        "/v1/embeddings",
        data={
            "payload": json.dumps(
                payload(
                    {"type": "video_frames", "files": ["second", "first", "second"]}
                )
            )
        },
        files=[
            ("first", ("first.png", png, "image/png")),
            ("second", ("second.png", png, "image/png")),
        ],
    )
    assert response.status_code == 200, response.text
    assert len(seen) == 3
    assert not client.app.state.storage.uploads
    assert not list(client.app.state.settings.upload_dir.iterdir())


@pytest.mark.parametrize(
    "part",
    [
        {"type": "video_frames"},
        {"type": "video_frames", "files": []},
        {"type": "video_frames", "files": [""]},
        {"type": "video_frames", "files": ["missing"]},
        {"type": "video_frames", "files": ["a"], "upload_ids": ["b"]},
    ],
)
def test_invalid_video_frame_references(client, part):
    response = client.post("/v1/embeddings", json=payload(part))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


def test_multipart_video_frame_limit(client, png):
    count = client.app.state.settings.max_video_frames + 1
    response = client.post(
        "/v1/embeddings",
        data={
            "payload": json.dumps(
                payload({"type": "video_frames", "files": ["frame"] * count})
            )
        },
        files={"frame": ("a.png", png, "image/png")},
    )
    assert response.status_code == 413
    assert not client.app.state.storage.uploads


def test_multipart_video_rejects_audio_frames(client, fixtures_dir):
    response = client.post(
        "/v1/embeddings",
        data={
            "payload": json.dumps(payload({"type": "video_frames", "files": ["audio"]}))
        },
        files={
            "audio": ("a.wav", (fixtures_dir / "sample.wav").read_bytes(), "audio/wav")
        },
    )
    assert response.status_code == 400
    assert not client.app.state.storage.uploads
