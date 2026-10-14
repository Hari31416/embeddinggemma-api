import logging
import os
import time
import math
import av
import soundfile as sf
from PIL import Image
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from multimodal_embedding.adapter.base import ModelAdapter, PreparedItem
from multimodal_embedding.config import Settings
from multimodal_embedding.exceptions import (
    APIException,
    InferenceError,
    ModelUnavailableError,
    InvalidInputError,
    PayloadTooLargeError,
)

logger = logging.getLogger(__name__)


class SentenceTransformersAdapter(ModelAdapter):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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
        elif self._requested_device == "cuda":
            if cuda_available:
                self._actual_device = "cuda"
                self._device_fallback = False
            else:
                self._actual_device = "cpu"
                self._device_fallback = True
                logger.warning(
                    "CUDA device requested but not available. Falling back to CPU."
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

    def _video_frames(self, filepath: str) -> list[Any]:
        with av.open(filepath) as container:
            streams = container.streams.video
            if not streams:
                raise InvalidInputError("Video file has no video stream.")
            stream = streams[0]
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                raise InvalidInputError("Cannot determine video duration safely.")
            if duration <= 0:
                raise InvalidInputError("Video duration must be positive.")
            if duration > self.settings.max_video_duration_seconds:
                raise PayloadTooLargeError("Video duration exceeds configured limit.")
            count = min(
                self.settings.max_video_frames,
                max(1, math.ceil(duration * self.settings.video_fps)),
            )
            targets = np.linspace(0, duration, count, endpoint=False)
            frames = []
            first_timestamp = None
            for frame in container.decode(stream):
                if frame.width * frame.height > self.settings.max_image_pixels:
                    raise PayloadTooLargeError(
                        "Video frame resolution exceeds max_image_pixels."
                    )
                if frame.time is None:
                    raise InvalidInputError("Video frame has no timestamp.")
                if first_timestamp is None:
                    first_timestamp = frame.time
                if frame.time - first_timestamp >= targets[len(frames)]:
                    frames.append(frame.to_image().convert("RGB"))
                    if len(frames) == count:
                        break
            if not frames:
                raise InvalidInputError("Video contains no decodable frames.")
            return frames

    def _input(self, item: PreparedItem) -> Any:
        value = item.to_input()
        if not item.media:
            return value
        images, audios, videos = [], [], []
        for media in item.media:
            if media.modality == "image":
                with Image.open(media.filepath) as image:
                    images.append(image.convert("RGB"))
            elif media.modality == "audio":
                if (
                    sf.info(media.filepath).duration
                    > self.settings.max_audio_duration_seconds
                ):
                    raise PayloadTooLargeError(
                        "Audio exceeds model window; split it before embedding."
                    )
                audios.append(media.filepath)
            elif media.modality == "video":
                if media.frame_paths is not None:
                    if len(media.frame_paths) > self.settings.max_video_frames:
                        raise PayloadTooLargeError(
                            "Video frame group exceeds max_video_frames."
                        )
                    frames = []
                    for path in media.frame_paths:
                        with Image.open(path) as image:
                            frames.append(image.convert("RGB"))
                else:
                    frames = self._video_frames(media.filepath)
                videos.append(frames)
        # Match the worker's explicit placeholder schema. Keep caller placeholders.
        text = item.text or ""
        if not any(
            marker in text for marker in ("<|image|>", "<|audio|>", "<|video|>")
        ):
            markers = (
                ["<|image|>"] * len(images)
                + ["<|audio|>"] * len(audios)
                + ["<|video|>"] * len(videos)
            )
            text = " ".join([text, *markers]).strip()
        value = {"text": text}
        if images:
            value["image"] = images[0] if len(images) == 1 else images
        if audios:
            value["audio"] = audios[0] if len(audios) == 1 else audios
        if videos:
            value["video"] = videos
        return value

    def encode(
        self,
        items: list[PreparedItem],
        task: str = "document",
        truncate_dim: int | None = None,
        prompt_mode: str = "auto",
    ) -> list[list[float]]:
        if not self._is_ready or self._model is None:
            raise ModelUnavailableError("Model is not initialized or ready.")
        if not items:
            return []
        prompt_name = "SearchQuery" if task == "query" else "Document"
        prompt = self._model.prompts[prompt_name] if prompt_mode == "auto" else ""
        processing_kwargs = {
            "text": {"truncation": False},
            "image": {"max_soft_tokens": self.settings.vision_max_soft_tokens},
            "video": {
                "do_sample_frames": False,
                "max_soft_tokens": self.settings.video_max_soft_tokens,
            },
        }
        try:
            inputs = [self._input(item) for item in items]
            for item, value in zip(items, inputs):
                processed = self._model.preprocess(
                    [value], prompt=prompt, processing_kwargs=processing_kwargs
                )
                input_ids = processed.get("input_ids")
                if input_ids is None:
                    raise InferenceError("Model processor did not return token IDs.")
                tokens = int(input_ids.shape[-1])
                if tokens > self.settings.max_context_tokens:
                    raise PayloadTooLargeError(
                        f"Item '{item.id}' has {tokens} processed tokens; limit is {self.settings.max_context_tokens}."
                    )
            embeddings = self._model.encode(
                inputs,
                prompt=prompt,
                batch_size=min(len(inputs), self.settings.max_batch_size),
                processing_kwargs=processing_kwargs,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            if (
                tuple(embeddings.shape) != (len(items), 768)
                or not torch.isfinite(embeddings).all()
            ):
                raise InferenceError(
                    "Model returned invalid embedding values or shape."
                )
            embeddings = embeddings[:, :truncate_dim] if truncate_dim else embeddings
            if (torch.linalg.vector_norm(embeddings.float(), dim=-1) == 0).any():
                raise InferenceError("Model returned a zero embedding.")
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
            return embeddings.cpu().numpy().tolist()
        except APIException:
            raise
        except (ValueError, OSError) as exc:
            raise InvalidInputError(f"Cannot process embedding input: {exc}") from exc
        except Exception as exc:
            logger.exception("Model inference failed")
            raise InferenceError("Model inference failed.") from exc
