import logging
from fastapi import APIRouter, File, Request, UploadFile

from multimodal_embedding.config import settings
from multimodal_embedding.exceptions import InvalidInputError, PayloadTooLargeError
from multimodal_embedding.schemas import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["uploads"])


@router.post("/uploads", response_model=UploadResponse)
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    storage = request.app.state.storage

    if not file.filename:
        raise InvalidInputError("File must have a filename.")

    # Stream read with size enforcement to avoid reading arbitrarily huge payloads
    content_chunks = []
    total_bytes = 0

    while True:
        chunk = await file.read(65536)  # 64 KB
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"Uploaded file exceeds maximum allowed limit of {settings.max_upload_bytes} bytes."
            )
        content_chunks.append(chunk)

    data = b"".join(content_chunks)

    record = await storage.save_upload(
        filename=file.filename,
        content_type=file.content_type,
        data=data,
    )

    return UploadResponse(
        upload_id=record.upload_id,
        filename=record.filename,
        modality=record.modality,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )
