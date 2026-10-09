import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from multimodal_embedding.adapter.base import ModelAdapter
from multimodal_embedding.adapter.fake_adapter import FakeModelAdapter
from multimodal_embedding.adapter.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from multimodal_embedding.config import Settings, settings as global_settings
from multimodal_embedding.exceptions import APIException
from multimodal_embedding.queue_manager import InferenceQueueManager
from multimodal_embedding.routes import (
    embeddings_router,
    health_router,
    models_router,
    uploads_router,
)
from multimodal_embedding.storage import StorageManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multimodal_embedding_api")


def create_app(
    settings: Settings | None = None,
    adapter: ModelAdapter | None = None,
) -> FastAPI:
    cfg = settings or global_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Initializing service...")

        # 1. Initialize storage and settings
        app.state.settings = cfg
        storage = StorageManager(
            base_dir=cfg.upload_dir,
            ttl_seconds=cfg.upload_ttl_seconds,
            max_upload_bytes=cfg.max_upload_bytes,
            settings=cfg,
        )
        app.state.storage = storage

        # 2. Initialize queue manager
        queue_mgr = InferenceQueueManager(
            max_concurrency=cfg.max_concurrency,
            max_queue_size=cfg.max_queue_size,
        )
        app.state.queue_manager = queue_mgr

        # 3. Initialize and load adapter
        if adapter is not None:
            active_adapter = adapter
        elif cfg.use_fake_adapter:
            logger.info("Using FakeModelAdapter as configured.")
            active_adapter = FakeModelAdapter(cfg)
        else:
            active_adapter = SentenceTransformersAdapter(cfg)

        if not active_adapter.is_ready:
            active_adapter.load()
        app.state.adapter = active_adapter

        # 4. Background cleanup task for expired uploads
        stop_event = asyncio.Event()

        async def cleanup_loop() -> None:
            while not stop_event.is_set():
                try:
                    await asyncio.sleep(30)
                    await storage.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning("Error during upload cleanup: %s", e)

        cleanup_task = asyncio.create_task(cleanup_loop())

        logger.info("Service initialized and ready to receive requests.")
        yield

        # Shutdown
        logger.info("Shutting down service...")
        stop_event.set()
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        await storage.clear_all()
        logger.info("Shutdown completed.")

    app = FastAPI(
        title="Multimodal Embedding API",
        description="Standalone API service for google/embeddinggemma-2 embeddings across text, image, audio, and video.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Error Handlers
    @app.exception_handler(APIException)
    async def handle_api_exception(request: Request, exc: APIException) -> JSONResponse:
        logger.warning(
            "APIException [%s]: %s (path=%s)",
            exc.code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Sanitize details so raw user data is not leaked into logs or responses
        sanitized_errors = []
        for err in exc.errors():
            sanitized_errors.append(
                {
                    "loc": [str(x) for x in err.get("loc", [])],
                    "msg": err.get("msg", ""),
                    "type": err.get("type", ""),
                }
            )
        logger.warning("Validation error on %s: %s", request.url.path, sanitized_errors)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "invalid_input",
                    "message": "Invalid request parameters.",
                    "details": {"errors": sanitized_errors},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_generic_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s: %s",
            request.url.path,
            type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred during request processing.",
                    "details": {},
                }
            },
        )

    # Register Routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(uploads_router)
    app.include_router(embeddings_router)

    return app


# Default app instance
app = create_app()
