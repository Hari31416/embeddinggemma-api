import logging
import sys
from pathlib import Path
import httpx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("client_example")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def run_client(base_url: str = "http://127.0.0.1:8000") -> None:
    client = httpx.Client(base_url=base_url, timeout=60.0)

    # 1. Health check
    logger.info("Checking service readiness at %s/health/ready ...", base_url)
    try:
        ready_resp = client.get("/health/ready")
        ready_resp.raise_for_status()
        logger.info("Service readiness: %s", ready_resp.json())
    except httpx.HTTPError as e:
        logger.error("Service is not ready or unreachable: %s", e)
        sys.exit(1)

    # 2. Model info
    models_resp = client.get("/v1/models")
    models_resp.raise_for_status()
    logger.info("Available models: %s", models_resp.json())

    # 3. Upload test fixtures
    fixtures_dir = Path("tests/fixtures")
    img_path = fixtures_dir / "sample.png"
    if not img_path.exists():
        logger.error(
            "Fixture %s not found. Please run scripts/create_fixtures.py first.",
            img_path,
        )
        sys.exit(1)

    logger.info("Uploading image %s ...", img_path)
    with open(img_path, "rb") as f:
        upload_resp = client.post(
            "/v1/uploads",
            files={"file": ("sample.png", f, "image/png")},
        )
        upload_resp.raise_for_status()
        upload_data = upload_resp.json()
        image_upload_id = upload_data["upload_id"]
        logger.info("Uploaded image ID: %s", image_upload_id)

    # 4. Generate embeddings for a search query and candidates
    query_payload = {
        "items": [
            {
                "id": "query-1",
                "parts": [{"type": "text", "text": "blue abstract graphic"}],
            }
        ],
        "task": "query",
    }
    logger.info("Generating query embedding...")
    q_resp = client.post("/v1/embeddings", json=query_payload)
    q_resp.raise_for_status()
    query_emb = q_resp.json()["embeddings"][0]["embedding"]

    candidate_payload = {
        "items": [
            {
                "id": "doc-text-match",
                "parts": [
                    {"type": "text", "text": "A modern graphic in blue and teal."}
                ],
            },
            {
                "id": "doc-image-match",
                "parts": [{"type": "media", "upload_id": image_upload_id}],
            },
            {
                "id": "doc-text-unrelated",
                "parts": [{"type": "text", "text": "A recipe for chocolate cake."}],
            },
        ],
        "task": "document",
    }
    logger.info("Generating candidate embeddings...")
    c_resp = client.post("/v1/embeddings", json=candidate_payload)
    c_resp.raise_for_status()
    candidates = c_resp.json()["embeddings"]

    # 5. Compute similarities
    logger.info("Similarity results against query 'blue abstract graphic':")
    for cand in candidates:
        sim = cosine_similarity(query_emb, cand["embedding"])
        logger.info("Candidate ID: %s | Cosine Similarity: %.4f", cand["id"], sim)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    run_client(url)
