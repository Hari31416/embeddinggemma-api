import uvicorn
from multimodal_embedding.config import settings


def main() -> None:
    uvicorn.run(
        "multimodal_embedding.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
