import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from multimodal_embedding.adapter.fake_adapter import FakeModelAdapter
from multimodal_embedding.app import create_app
from multimodal_embedding.config import Settings
from scripts.create_fixtures import create_fixtures


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    fdir = Path("tests/fixtures")
    create_fixtures(fdir)
    return fdir


@pytest.fixture
def temp_upload_dir() -> Generator[Path, None, None]:
    tmp_path = Path(tempfile.mkdtemp(prefix="test_uploads_"))
    try:
        yield tmp_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def fake_adapter() -> FakeModelAdapter:
    adapter = FakeModelAdapter()
    adapter.load()
    return adapter


@pytest.fixture
def test_settings(temp_upload_dir: Path) -> Settings:
    return Settings(
        upload_dir=temp_upload_dir,
        max_upload_bytes=1048576,  # 1 MB for testing
        max_batch_size=4,
        max_queue_size=2,
        max_concurrency=1,
        upload_ttl_seconds=300,
        max_image_pixels=1000000,
        max_audio_duration_seconds=10.0,
        max_video_duration_seconds=10.0,
        use_fake_adapter=True,
    )


@pytest.fixture
def client(
    test_settings: Settings, fake_adapter: FakeModelAdapter
) -> Generator[TestClient, None, None]:
    app = create_app(settings=test_settings, adapter=fake_adapter)
    with TestClient(app) as test_client:
        yield test_client
