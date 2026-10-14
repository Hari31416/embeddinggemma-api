import pytest
import torch

from multimodal_embedding.adapter.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from multimodal_embedding.config import Settings
from scripts.verify_phase1 import probe_device


@pytest.mark.parametrize("cuda_available", [True, False])
def test_explicit_cuda_from_environment(monkeypatch, cuda_available):
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    settings = Settings(_env_file=None)
    adapter = SentenceTransformersAdapter(settings)

    expected = "cuda" if cuda_available else "cpu"
    assert settings.device == adapter.requested_device == "cuda"
    assert adapter.actual_device == expected
    assert adapter.device_fallback is (not cuda_available)
    device, _ = probe_device("cuda")
    assert str(device) == expected


@pytest.mark.parametrize(
    ("mps_available", "cuda_available", "expected"),
    [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")],
)
def test_auto_device_selection(monkeypatch, mps_available, cuda_available, expected):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    adapter = SentenceTransformersAdapter(Settings(device="auto", _env_file=None))
    assert adapter.actual_device == expected
    assert adapter.device_fallback is False
