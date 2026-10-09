import logging
import os
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from multimodal_embedding.adapter.base import ModelAdapter, PreparedItem
from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import InferenceError, ModelUnavailableError

logger = logging.getLogger(__name__)


class SentenceTransformersAdapter(ModelAdapter):
    def __init__(self, settings: Settings) -> None:
        self._model_id = settings.model_id
        self._revision = settings.model_revision
        self._requested_device = settings.device
        self._torch_dtype_str = settings.torch_dtype
        self._model: Any = None
        self._actual_device = "cpu"
        self._device_fallback = False
        self._is_ready = False

        self._probe_device()

    def _probe_device(self) -> None:
        mps_available = torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()

        if self._requested_device == "mps":
            if mps_available:
                self._actual_device = "mps"
                self._device_fallback = False
            else:
                self._actual_device = "cpu"
                self._device_fallback = True
                logger.warning(
                    "MPS device requested but not available. Falling back to CPU."
                )
        elif self._requested_device == "cpu":
            self._actual_device = "cpu"
            self._device_fallback = False
        elif self._requested_device == "auto":
            if mps_available:
                self._actual_device = "mps"
                self._device_fallback = False
            elif cuda_available:
                self._actual_device = "cuda"
                self._device_fallback = False
            else:
                self._actual_device = "cpu"
                self._device_fallback = False
        else:
            logger.warning(
                "Unrecognized device '%s'. Defaulting to CPU.",
                self._requested_device,
            )
            self._actual_device = "cpu"
            self._device_fallback = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def actual_device(self) -> str:
        return self._actual_device

    @property
    def requested_device(self) -> str:
        return self._requested_device

    @property
    def device_fallback(self) -> bool:
        return self._device_fallback

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
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading model %s (revision: %s) on %s ...",
            self._model_id,
            self._revision,
            self._actual_device,
        )

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if hf_token:
            logger.info("HF authentication token detected in environment.")
        else:
            logger.info("No HF authentication token found in environment.")

        dtype = torch.float32
        if self._torch_dtype_str == "bfloat16":
            dtype = torch.bfloat16

        t0 = time.time()
        try:
            self._model = SentenceTransformer(
                self._model_id,
                revision=self._revision,
                device=self._actual_device,
                model_kwargs={"torch_dtype": dtype},
            )
            self._is_ready = True
            logger.info(
                "Model loaded successfully in %.2f seconds on %s",
                time.time() - t0,
                self._actual_device,
            )
        except Exception as e:
            self._is_ready = False
            logger.error("Failed to load model %s: %s", self._model_id, e)
            raise ModelUnavailableError(
                f"Failed to load model {self._model_id}: {e}"
            ) from e

    def encode(
        self,
        items: list[PreparedItem],
        task: str = "document",
        truncate_dim: int | None = None,
    ) -> list[list[float]]:
        if not self._is_ready or self._model is None:
            raise ModelUnavailableError("Model is not initialized or ready.")

        if not items:
            return []

        prompt_name = "SearchQuery" if task == "query" else "Document"
        inputs = [item.to_input() for item in items]

        try:
            t0 = time.time()
            embeddings = self._model.encode(
                inputs,
                prompt_name=prompt_name,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            # Ensure 2D tensor
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)

            # MRL truncation and re-normalization
            if truncate_dim is not None and truncate_dim < 768:
                embeddings = embeddings[:, :truncate_dim]
                embeddings = F.normalize(embeddings.float(), p=2, dim=-1)

            # Ensure unit length L2 normalization in float32
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)

            result: list[list[float]] = embeddings.cpu().numpy().tolist()
            logger.info(
                "Encoded batch of %d item(s) in %.3f seconds (task=%s, dim=%d)",
                len(items),
                time.time() - t0,
                task,
                len(result[0]) if result else 0,
            )
            return result
        except Exception as e:
            logger.error("Error during model inference: %s", e)
            raise InferenceError(f"Model inference failed: {e}") from e
