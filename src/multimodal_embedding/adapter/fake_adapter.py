import hashlib
import math
import time
from typing import Any
import numpy as np

from multimodal_embedding.adapter.base import ModelAdapter, PreparedItem
from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import ModelUnavailableError


class FakeModelAdapter(ModelAdapter):
    def __init__(self, settings: Settings | None = None) -> None:
        self._model_id = settings.model_id if settings else "google/embeddinggemma-2"
        self._revision = settings.model_revision if settings else "main"
        self._device = settings.device if settings else "cpu"
        self._is_ready = False
        self.call_history: list[dict[str, Any]] = []
        self.sleep_duration: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def actual_device(self) -> str:
        return "cpu"

    @property
    def requested_device(self) -> str:
        return self._device

    @property
    def device_fallback(self) -> bool:
        return False

    @property
    def supported_inputs(self) -> list[str]:
        return ["text", "image", "audio", "video"]

    @property
    def output_dimensions(self) -> list[int]:
        return [768, 512, 256, 128]

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load(self) -> None:
        self._is_ready = True

    def unload(self) -> None:
        self._is_ready = False

    def encode(
        self,
        items: list[PreparedItem],
        task: str = "document",
        truncate_dim: int | None = None,
    ) -> list[list[float]]:
        if not self._is_ready:
            raise ModelUnavailableError("Fake model adapter is not ready.")

        if self.sleep_duration > 0:
            time.sleep(self.sleep_duration)

        dim = truncate_dim if truncate_dim is not None else 768
        results: list[list[float]] = []

        for item in items:
            inp_repr = (
                f"{task}:{item.id}:{item.text}:{[m.upload_id for m in item.media]}"
            )
            # Generate deterministic pseudo-random vector from hash
            h = hashlib.sha256(inp_repr.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "big")
            rng = np.random.default_rng(seed)

            vec = rng.standard_normal(768).astype(np.float32)
            if truncate_dim is not None and truncate_dim < 768:
                vec = vec[:truncate_dim]

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            results.append(vec.tolist())

        self.call_history.append(
            {
                "item_count": len(items),
                "task": task,
                "truncate_dim": truncate_dim,
                "items": [item.id for item in items],
            }
        )

        return results
