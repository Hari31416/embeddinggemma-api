from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, model_validator


# Error Responses
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


# Health Responses
class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: str
    actual_device: str
    device_fallback: bool
    queue_depth: int
    active_workers: int


# Model Metadata
class ConfiguredLimits(BaseModel):
    max_batch_size: int
    max_upload_bytes: int
    max_embedding_request_bytes: int
    url_download_max_bytes: int
    url_download_timeout_seconds: float
    max_image_pixels: int
    max_audio_duration_seconds: float
    max_video_duration_seconds: float
    max_context_tokens: int
    max_concurrency: int
    max_queue_size: int
    video_fps: int
    max_video_frames: int
    vision_max_soft_tokens: int
    video_max_soft_tokens: int


class ModelInfo(BaseModel):
    test_mode: bool
    id: str
    revision: str
    actual_device: str
    requested_device: str
    device_fallback: bool
    supported_inputs: list[str]
    output_dimensions: list[int]
    configured_limits: ConfiguredLimits


class ModelsResponse(BaseModel):
    data: list[ModelInfo]


# Upload Responses
class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    modality: Literal["image", "audio", "video"]
    content_type: str
    size_bytes: int
    created_at: float
    expires_at: float


# Embeddings Request & Response
class TextPart(BaseModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class MediaPart(BaseModel):
    type: Literal["media"]
    model_config = {"extra": "forbid"}

    upload_id: str | None = Field(default=None, min_length=1)
    file: str | None = Field(
        default=None, min_length=1, description="Multipart field name"
    )
    file_url: str | None = Field(default=None, min_length=1)
    file_data: str | None = Field(
        default=None, min_length=1, description="Base64 data URL or raw base64"
    )
    filename: str | None = Field(default=None, min_length=1)
    content_type: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_source(self):
        if (
            sum(
                value is not None
                for value in (self.upload_id, self.file, self.file_url, self.file_data)
            )
            != 1
        ):
            raise ValueError(
                "Media requires exactly one of upload_id, file, file_url, file_data."
            )
        if (
            self.file_data
            and not self.file_data.startswith("data:")
            and not self.content_type
        ):
            raise ValueError("Raw base64 requires content_type.")
        return self


class VideoFramesPart(BaseModel):
    type: Literal["video_frames"]
    upload_ids: list[str] = Field(min_length=1)


ContentPart = Annotated[
    Union[TextPart, MediaPart, VideoFramesPart], Field(discriminator="type")
]


class ItemRequest(BaseModel):
    id: str = Field(min_length=1)
    parts: list[ContentPart] = Field(min_length=1)


class EmbeddingRequest(BaseModel):
    items: list[ItemRequest] = Field(min_length=1)
    task: Literal["query", "document"] = "document"
    truncate_dim: Literal[128, 256, 512, 768] | None = None
    prompt_mode: Literal["auto", "none"] = "auto"


class ItemEmbedding(BaseModel):
    id: str
    embedding: list[float]
    dimension: int


class UsageInfo(BaseModel):
    total_items: int
    latency_ms: float


class EmbeddingResponse(BaseModel):
    model: str
    revision: str
    task: str
    dimension: int
    normalized: bool = True
    embeddings: list[ItemEmbedding]
    usage: UsageInfo
