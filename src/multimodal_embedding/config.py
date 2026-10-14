import os
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server settings
    host: str = Field(default="127.0.0.1", description="Host to bind server")
    port: int = Field(default=8000, description="Port to bind server")

    # Model settings
    model_id: str = Field(
        default="google/embeddinggemma-2",
        description="Hugging Face model identifier",
    )
    model_revision: str = Field(
        default="914f7f89142e33e77833254d9c9b90c3cef7303b",
        description="Model revision or commit hash",
    )
    device: Literal["cpu", "mps", "cuda", "auto"] = Field(
        default="auto",
        description="Target compute device: cpu, mps, cuda, or auto",
    )
    torch_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description="Torch dtype: float32 or bfloat16",
    )
    use_fake_adapter: bool = Field(
        default=False,
        description="Whether to use a mock adapter for tests/development",
    )

    # Concurrency and queue settings
    max_concurrency: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent inference jobs",
    )
    max_queue_size: int = Field(
        default=10,
        ge=1,
        description="Maximum queued inference requests before 503 queue saturation",
    )

    # Input validation limits
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum items allowed in a single embeddings request",
    )
    max_upload_bytes: int = Field(
        default=52428800,  # 50 MB
        ge=1024,
        description="Maximum upload file size in bytes",
    )
    max_embedding_request_bytes: int = Field(default=104857600, ge=1024)
    url_download_timeout_seconds: float = Field(default=20.0, gt=0)
    url_download_max_bytes: int = Field(default=52428800, ge=1)
    url_allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Optional exact HTTPS hostname allowlist; empty allows public hosts",
    )

    upload_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        ge=1,
        description="Time to live for uploaded files before automatic cleanup",
    )
    upload_dir: Path = Field(
        default=Path("/tmp/multimodal_embedding_api/uploads"),
        description="Directory for temporary media uploads",
    )

    # Media dimensions and duration limits
    max_image_pixels: int = Field(
        default=16777216,  # 4096 * 4096
        ge=1,
        description="Maximum total pixel count for decoded images",
    )
    max_audio_duration_seconds: float = Field(
        default=30.0,
        gt=0,
        le=30.0,
        description="Maximum allowed audio duration in seconds",
    )
    max_video_duration_seconds: float = Field(
        default=60.0,  # 1 minute
        gt=0,
        description="Maximum allowed video duration in seconds",
    )

    # Visual tokens and frame sampling
    video_fps: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Frames per second sampling rate for video",
    )
    max_video_frames: int = Field(
        default=32,
        ge=1,
        le=120,
        description="Maximum frames sampled from video",
    )
    vision_max_soft_tokens: Literal[70, 140, 280, 560, 1120] = Field(
        default=280,
        description="Soft token budget per image or video frame: 70, 140, 280, 560, 1120",
    )
    video_max_soft_tokens: Literal[70, 140, 280, 560, 1120] = 140
    max_context_tokens: int = Field(
        default=8192,
        ge=1,
        le=8192,
        description="Context window budget shared across all modalities",
    )

    @field_validator("vision_max_soft_tokens", "video_max_soft_tokens", mode="before")
    @classmethod
    def parse_soft_token_budget(cls, value):
        # Environment variables are strings; Literal[int] does not coerce them.
        return int(value) if isinstance(value, str) else value


settings = Settings()
