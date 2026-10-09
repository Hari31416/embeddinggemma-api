from multimodal_embedding.routes.embeddings import router as embeddings_router
from multimodal_embedding.routes.health import router as health_router
from multimodal_embedding.routes.models import router as models_router
from multimodal_embedding.routes.uploads import router as uploads_router

__all__ = [
    "embeddings_router",
    "health_router",
    "models_router",
    "uploads_router",
]
