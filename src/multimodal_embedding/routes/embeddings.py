import asyncio
import anyio
import logging
import time
from fastapi import APIRouter, Depends, Request

from multimodal_embedding.embedding_request import (
    parse_embedding_request,
    request_body_schema,
)

from multimodal_embedding.adapter.base import PreparedItem, PreparedMedia
from multimodal_embedding.config import settings
from multimodal_embedding.exceptions import (
    InvalidInputError,
    PayloadTooLargeError,
)
from multimodal_embedding.schemas import (
    EmbeddingResponse,
    ItemEmbedding,
    MediaPart,
    TextPart,
    UsageInfo,
    VideoFramesPart,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post(
    "/embeddings", response_model=EmbeddingResponse, openapi_extra=request_body_schema()
)
async def create_embeddings(
    request: Request,
    inputs: tuple = Depends(parse_embedding_request),
) -> EmbeddingResponse:
    payload, resolver = inputs
    adapter = request.app.state.adapter
    queue_manager = request.app.state.queue_manager
    cfg = getattr(request.app.state, "settings", settings)

    # 1. Validate batch size
    if len(payload.items) > cfg.max_batch_size:
        raise PayloadTooLargeError(
            f"Batch size {len(payload.items)} exceeds maximum batch limit of {cfg.max_batch_size}."
        )

    t0 = time.time()
    prepared_items: list[PreparedItem] = []

    # 2. Prepare each item
    for item in payload.items:
        text_segments: list[str] = []
        media_list: list[PreparedMedia] = []

        for part in item.parts:
            if isinstance(part, TextPart):
                text_segments.append(part.text)
            elif isinstance(part, VideoFramesPart):
                references = part.files if part.files is not None else part.upload_ids
                if len(references) > cfg.max_video_frames:
                    raise PayloadTooLargeError(
                        "Video frame group exceeds max_video_frames."
                    )
                paths = []
                for reference in references:
                    record = await resolver.resolve(
                        MediaPart(type="media", file=reference)
                        if part.files is not None
                        else MediaPart(type="media", upload_id=reference)
                    )
                    if record.modality != "image":
                        raise InvalidInputError("Video frames must reference images.")
                    paths.append(str(record.filepath))
                media_list.append(
                    PreparedMedia(
                        upload_id="", filepath="", modality="video", frame_paths=paths
                    )
                )
            elif isinstance(part, MediaPart):
                record = await resolver.resolve(part)
                media_list.append(
                    PreparedMedia(
                        upload_id=record.upload_id,
                        filepath=str(record.filepath),
                        modality=record.modality,
                    )
                )

        full_text = " ".join(text_segments) if text_segments else None

        # Check manual placeholders if text is present
        if full_text:
            num_img_placeholders = full_text.count("<|image|>")
            num_audio_placeholders = full_text.count("<|audio|>")
            num_video_placeholders = full_text.count("<|video|>")

            total_placeholders = (
                num_img_placeholders + num_audio_placeholders + num_video_placeholders
            )
            if total_placeholders > 0:
                actual_images = sum(1 for m in media_list if m.modality == "image")
                actual_audios = sum(1 for m in media_list if m.modality == "audio")
                actual_videos = sum(1 for m in media_list if m.modality == "video")

                if num_img_placeholders != actual_images:
                    raise InvalidInputError(
                        f"Item '{item.id}': placeholder count mismatch for <|image|>. Found {num_img_placeholders} in text, but {actual_images} image media part(s) provided."
                    )
                if num_audio_placeholders != actual_audios:
                    raise InvalidInputError(
                        f"Item '{item.id}': placeholder count mismatch for <|audio|>. Found {num_audio_placeholders} in text, but {actual_audios} audio media part(s) provided."
                    )
                if num_video_placeholders != actual_videos:
                    raise InvalidInputError(
                        f"Item '{item.id}': placeholder count mismatch for <|video|>. Found {num_video_placeholders} in text, but {actual_videos} video media part(s) provided."
                    )

        prepared_items.append(
            PreparedItem(id=item.id, text=full_text, media=media_list)
        )

    # 3. Execute inference via queue manager
    inference = asyncio.create_task(
        queue_manager.run(
            adapter.encode,
            prepared_items,
            task=payload.task,
            truncate_dim=payload.truncate_dim,
            prompt_mode=payload.prompt_mode,
        )
    )
    try:
        embeddings = await asyncio.shield(inference)
    except asyncio.CancelledError:
        # The model thread may still be reading media. Keep request-local files
        # until it finishes, before the dependency removes them.
        with anyio.CancelScope(shield=True):
            try:
                await inference
            except Exception:
                logger.warning("Inference failed after request cancellation.")
        raise

    duration_ms = (time.time() - t0) * 1000.0

    # 4. Format response
    dim = payload.truncate_dim if payload.truncate_dim is not None else 768
    item_embeddings = [
        ItemEmbedding(id=item.id, embedding=emb, dimension=len(emb))
        for item, emb in zip(payload.items, embeddings)
    ]

    logger.info(
        "Successfully computed embeddings for %d item(s) in %.2f ms",
        len(item_embeddings),
        duration_ms,
    )

    return EmbeddingResponse(
        model=adapter.model_id,
        revision=adapter.revision,
        task=payload.task,
        dimension=dim,
        normalized=True,
        embeddings=item_embeddings,
        usage=UsageInfo(
            total_items=len(item_embeddings),
            latency_ms=duration_ms,
        ),
    )
