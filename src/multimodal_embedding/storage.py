import asyncio
import logging
import mimetypes
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from PIL import Image
import soundfile as sf
import av

from multimodal_embedding.config import Settings, settings as global_settings
from multimodal_embedding.exceptions import (
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    filepath: Path
    modality: Literal["image", "audio", "video"]
    content_type: str
    size_bytes: int
    created_at: float
    expires_at: float


class StorageManager:
    def __init__(
        self,
        base_dir: Path | None = None,
        ttl_seconds: int | None = None,
        max_upload_bytes: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or global_settings
        self.base_dir = base_dir or self.settings.upload_dir
        self.ttl_seconds = ttl_seconds or self.settings.upload_ttl_seconds
        self.max_upload_bytes = max_upload_bytes or self.settings.max_upload_bytes
        self.uploads: dict[str, UploadRecord] = {}
        self._lock = asyncio.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _determine_modality(
        self, filename: str, content_type: str
    ) -> Literal["image", "audio", "video"]:
        suffix = Path(filename).suffix.lower()
        if content_type.startswith("image/") or suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        if content_type.startswith("audio/") or suffix in SUPPORTED_AUDIO_EXTENSIONS:
            return "audio"
        if content_type.startswith("video/") or suffix in SUPPORTED_VIDEO_EXTENSIONS:
            return "video"
        raise InvalidInputError(
            f"Unsupported file format '{suffix or filename}' with content type '{content_type}'. Supported modalities are image, audio, video."
        )

    def _validate_image(self, filepath: Path) -> None:
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                total_pixels = width * height
                if total_pixels > self.settings.max_image_pixels:
                    raise PayloadTooLargeError(
                        f"Image resolution {width}x{height} ({total_pixels} pixels) exceeds maximum allowed limit of {self.settings.max_image_pixels} pixels.",
                        details={
                            "width": width,
                            "height": height,
                            "total_pixels": total_pixels,
                            "max_pixels": self.settings.max_image_pixels,
                        },
                    )
        except PayloadTooLargeError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Failed to decode image file: {e}") from e

    def _validate_audio(self, filepath: Path) -> None:
        try:
            info = sf.info(str(filepath))
            duration = info.duration
            if duration > self.settings.max_audio_duration_seconds:
                raise PayloadTooLargeError(
                    f"Audio duration {duration:.2f}s exceeds maximum allowed limit of {self.settings.max_audio_duration_seconds:.2f}s.",
                    details={
                        "duration_seconds": duration,
                        "max_duration_seconds": self.settings.max_audio_duration_seconds,
                    },
                )
        except PayloadTooLargeError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Failed to inspect audio file: {e}") from e

    def _validate_video(self, filepath: Path) -> None:
        try:
            container = av.open(str(filepath))
            duration = (
                float(container.duration) / 1000000.0
                if container.duration is not None
                else 0.0
            )
            container.close()
            if duration > self.settings.max_video_duration_seconds:
                raise PayloadTooLargeError(
                    f"Video duration {duration:.2f}s exceeds maximum allowed limit of {self.settings.max_video_duration_seconds:.2f}s.",
                    details={
                        "duration_seconds": duration,
                        "max_duration_seconds": self.settings.max_video_duration_seconds,
                    },
                )
        except PayloadTooLargeError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Failed to inspect video file: {e}") from e

    async def save_upload(
        self,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> UploadRecord:
        size_bytes = len(data)
        if size_bytes > self.max_upload_bytes:
            raise PayloadTooLargeError(
                f"File size {size_bytes} bytes exceeds maximum upload limit of {self.max_upload_bytes} bytes.",
                details={"size_bytes": size_bytes, "max_bytes": self.max_upload_bytes},
            )

        if not content_type:
            guessed, _ = mimetypes.guess_type(filename)
            content_type = guessed or "application/octet-stream"

        modality = self._determine_modality(filename, content_type)
        upload_id = str(uuid.uuid4())
        safe_suffix = Path(filename).suffix.lower()
        stored_filename = f"{upload_id}{safe_suffix}"
        target_path = self.base_dir / stored_filename

        try:
            target_path.write_bytes(data)

            if modality == "image":
                self._validate_image(target_path)
            elif modality == "audio":
                self._validate_audio(target_path)
            elif modality == "video":
                self._validate_video(target_path)

            now = time.time()
            record = UploadRecord(
                upload_id=upload_id,
                filename=filename,
                filepath=target_path,
                modality=modality,
                content_type=content_type,
                size_bytes=size_bytes,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )

            async with self._lock:
                self.uploads[upload_id] = record

            logger.info(
                "Upload saved successfully: id=%s modality=%s size=%d bytes",
                upload_id,
                modality,
                size_bytes,
            )
            return record

        except Exception:
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            raise

    async def get_upload(self, upload_id: str) -> UploadRecord:
        async with self._lock:
            record = self.uploads.get(upload_id)
            if not record:
                raise NotFoundError(f"Upload '{upload_id}' not found or expired.")

            if time.time() > record.expires_at:
                self._remove_record(record)
                raise NotFoundError(f"Upload '{upload_id}' has expired.")

            if not record.filepath.exists():
                self.uploads.pop(upload_id, None)
                raise NotFoundError(
                    f"Upload '{upload_id}' file is missing from storage."
                )

            return record

    def _remove_record(self, record: UploadRecord) -> None:
        self.uploads.pop(record.upload_id, None)
        try:
            if record.filepath.exists():
                record.filepath.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to remove file %s: %s", record.filepath, e)

    async def delete_upload(self, upload_id: str) -> None:
        async with self._lock:
            record = self.uploads.get(upload_id)
            if record is not None:
                self._remove_record(record)

    async def cleanup_expired(self) -> int:
        now = time.time()
        to_delete: list[UploadRecord] = []
        async with self._lock:
            for record in self.uploads.values():
                if now > record.expires_at:
                    to_delete.append(record)

            for record in to_delete:
                self._remove_record(record)

        if to_delete:
            logger.info("Cleaned up %d expired upload(s).", len(to_delete))
        return len(to_delete)

    async def clear_all(self) -> None:
        async with self._lock:
            for record in list(self.uploads.values()):
                self._remove_record(record)
            self.uploads.clear()
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir, ignore_errors=True)
                self.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all upload records and storage directory.")
