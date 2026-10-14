from fastapi import APIRouter, Request

from multimodal_embedding.config import settings
from multimodal_embedding.schemas import ConfiguredLimits, ModelInfo, ModelsResponse

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    adapter = request.app.state.adapter
    cfg = getattr(request.app.state, "settings", settings)

    limits = ConfiguredLimits(
        max_batch_size=cfg.max_batch_size,
        max_upload_bytes=cfg.max_upload_bytes,
        max_image_pixels=cfg.max_image_pixels,
        max_audio_duration_seconds=cfg.max_audio_duration_seconds,
        max_video_duration_seconds=cfg.max_video_duration_seconds,
        max_context_tokens=cfg.max_context_tokens,
        max_concurrency=cfg.max_concurrency,
        max_queue_size=cfg.max_queue_size,
        video_fps=cfg.video_fps,
        max_video_frames=cfg.max_video_frames,
        vision_max_soft_tokens=cfg.vision_max_soft_tokens,
        video_max_soft_tokens=cfg.video_max_soft_tokens,
    )

    info = ModelInfo(
        id=adapter.model_id,
        revision=adapter.revision,
        actual_device=adapter.actual_device,
        requested_device=adapter.requested_device,
        device_fallback=adapter.device_fallback,
        supported_inputs=adapter.supported_inputs,
        output_dimensions=adapter.output_dimensions,
        configured_limits=limits,
    )

    return ModelsResponse(data=[info])
