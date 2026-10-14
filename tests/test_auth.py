import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings


def embedding_request(client, headers):
    return client.post(
        "/v1/embeddings",
        headers=headers,
        json={"items": [{"id": "1", "parts": [{"type": "text", "text": "hello"}]}]},
    )


@pytest.mark.parametrize("key", ["first-key", "second-key"])
def test_each_configured_key_is_accepted(test_settings, fake_adapter, key):
    test_settings.api_keys = Settings(
        _env_file=None, api_keys=["first-key", "second-key"]
    ).api_keys
    with TestClient(create_app(test_settings, fake_adapter)) as client:
        headers = {"Authorization": f"Bearer {key}"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert embedding_request(client, headers).status_code == 200
        image = BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        assert (
            client.post(
                "/v1/uploads",
                headers=headers,
                files={"file": ("image.png", image.getvalue(), "image/png")},
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    "authorization", [None, "Bearer wrong", "Basic first-key", "Bearer"]
)
@pytest.mark.parametrize("path", ["/v1/models", "/v1/uploads", "/v1/embeddings"])
def test_missing_or_invalid_key_is_rejected(
    test_settings, fake_adapter, authorization, path
):
    test_settings.api_keys = Settings(_env_file=None, api_keys=["first-key"]).api_keys
    with TestClient(create_app(test_settings, fake_adapter)) as client:
        headers = {"Authorization": authorization} if authorization else {}
        if path == "/v1/models":
            response = client.get(path, headers=headers)
        else:
            response = client.post(path, headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert response.json()["error"]["code"] == "unauthorized"
        assert fake_adapter.call_history == []


def test_health_remains_public_when_auth_enabled(test_settings, fake_adapter):
    test_settings.api_keys = Settings(_env_file=None, api_keys=["first-key"]).api_keys
    with TestClient(create_app(test_settings, fake_adapter)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        operation = client.get("/openapi.json").json()["paths"]["/v1/embeddings"][
            "post"
        ]
        assert operation["security"] == [{"HTTPBearer": []}]


@pytest.mark.parametrize("authorization", [None, "Bearer anything", "Basic anything"])
def test_empty_list_disables_auth(test_settings, fake_adapter, authorization):
    test_settings.api_keys = []
    with TestClient(create_app(test_settings, fake_adapter)) as client:
        headers = {"Authorization": authorization} if authorization else {}
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert embedding_request(client, headers).status_code == 200


@pytest.mark.parametrize("value", ["", "[]", "  ", '["first-key","second-key"]'])
def test_api_keys_from_environment(monkeypatch, value):
    monkeypatch.setenv("SEARCH_EMBEDDING_API_KEYS", value)
    cfg = Settings(_env_file=None)
    expected = ["first-key", "second-key"] if "first-key" in value else []
    assert [key.get_secret_value() for key in cfg.api_keys] == expected
    assert "first-key" not in repr(cfg)


@pytest.mark.parametrize("value", ['[""]', '["   "]', '"key"', "not-json"])
def test_invalid_key_configuration_fails(monkeypatch, value):
    monkeypatch.setenv("SEARCH_EMBEDDING_API_KEYS", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
