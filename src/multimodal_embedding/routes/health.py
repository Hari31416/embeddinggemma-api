from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from multimodal_embedding.schemas import LiveResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=LiveResponse)
async def health_live() -> LiveResponse:
    return LiveResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
async def health_ready(request: Request) -> JSONResponse:
    adapter = getattr(request.app.state, "adapter", None)
    queue_manager = getattr(request.app.state, "queue_manager", None)

    is_ready = adapter is not None and adapter.is_ready

    data = {
        "status": "ready" if is_ready else "not_ready",
        "model": adapter.model_id if adapter else "none",
        "actual_device": adapter.actual_device if adapter else "none",
        "device_fallback": adapter.device_fallback if adapter else False,
        "queue_depth": queue_manager.queue_depth if queue_manager else 0,
        "active_workers": queue_manager.active_workers if queue_manager else 0,
    }

    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=status_code, content=data)
