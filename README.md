# EmbeddingGemma API

Standalone FastAPI embedding service for `google/embeddinggemma-2` using Sentence Transformers. The service maps text, image, audio, video, and combined inputs into a single shared 768-dimensional embedding space with support for Matryoshka Representation Learning (MRL) truncation.

## Goal

Provide a reusable local inference service running natively on macOS with Apple Silicon MPS acceleration and explicit CPU fallback.

Search, chunking, OCR, and storage infrastructure are intentionally outside the scope of this service.

## Architecture

- Web framework: FastAPI with Uvicorn worker.
- Inference engine: Sentence Transformers (`>=6.1.0`) with `google/embeddinggemma-2`.
- Media decoders: Pillow for images, SoundFile and Librosa for audio, PyAV and TorchCodec for video.
- Execution model: Bounded queue with semaphore-gated concurrency (`max_concurrency=1` by default). Blocking model inference runs in worker threads outside the asyncio event loop.
- Storage: Ephemeral upload storage for media files with time-to-live (TTL) expiry and background cleanup.

## Hardware Requirements and Verification Results

The model has 740M total parameters (270M text backbone, 170M vision encoder, 300M audio encoder).

### Model Access and Download

- Repository: `google/embeddinggemma-2` on Hugging Face.
- Gated model: No (publicly available under Apache 2.0).
- Checkpoint size: Approximately 1.45 GB (`model.safetensors` is 1.42 GB).
- Hugging Face credentials: Read optionally from `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` without logging token values.

### Observed Verification on macOS (Apple Silicon M-Series)

- Model load time: ~7.2 seconds on CPU, ~6.8 seconds on MPS.
- Memory footprint: ~1.5 GB in float32.
- Text query inference latency: ~88 ms (CPU), ~120 ms (MPS).
- Text document inference latency: ~30 ms (CPU), ~50 ms (MPS).
- Image inference latency (64x64): ~712 ms (CPU), ~1450 ms (MPS).
- Audio inference latency (1.0s 16kHz): ~524 ms (CPU), ~1800 ms (MPS).
- Video inference latency (2 frames): ~336 ms (CPU), ~640 ms (MPS).
- Combined input latency (text + image + audio): ~1219 ms (CPU), ~658 ms (MPS).
- Mixed batch latency (5 items): ~2888 ms (CPU), ~2636 ms (MPS).
- Normalization: L2 unit-length verified across all modalities and MRL truncations (768, 512, 256, 128).

## Installation and Setup

### Prerequisites

- macOS (Apple Silicon recommended)
- `uv` package manager (`brew install uv`)
- Python 3.11 to 3.14

### Setup Virtual Environment and Dependencies

Install all dependencies pinned in `uv.lock`:

```bash
uv sync
```

Or install with pip inside your virtual environment:

```bash
uv venv
source .venv/bin/activate
uv add "fastapi" "uvicorn[standard]" "pydantic" "pydantic-settings" "python-multipart" "torch" "torchvision" "torchaudio" "pillow" "sentence-transformers" "soundfile" "librosa" "av" "torchcodec" "pytest" "httpx" "black"
```

### Test Fixtures

Generate local test fixtures for all four modalities:

```bash
uv run python scripts/create_fixtures.py
```

This creates:
- `tests/fixtures/sample.txt` (text snippet)
- `tests/fixtures/sample.png` (64x64 PNG image)
- `tests/fixtures/sample.wav` (1-second 16kHz mono WAV audio)
- `tests/fixtures/sample.mp4` (2-frame 64x64 MP4 video)

## Configuration

Configuration values are managed by Pydantic Settings and loaded from environment variables or `.env`. Copy the example file to get started:

```bash
cp .env.example .env
```

Key configuration options:

- `HOST`: Host interface to bind (default: `127.0.0.1`).
- `PORT`: Port to listen on (default: `8000`).
- `DEVICE`: Target compute device: `cpu`, `mps`, or `auto` (default: `auto`).
- `TORCH_DTYPE`: Numerical precision: `float32` or `bfloat16` (default: `float32`).
- `USE_FAKE_ADAPTER`: Set `true` to use mock adapter for fast testing without loading weights (default: `false`).
- `MAX_CONCURRENCY`: Concurrency limit for active inference (default: `1`).
- `MAX_QUEUE_SIZE`: Maximum pending inference jobs before rejecting requests with 503 (default: `10`).
- `MAX_BATCH_SIZE`: Maximum items per embedding request (default: `32`).
- `MAX_UPLOAD_BYTES`: Maximum size for uploaded media in bytes (default: `52428800` = 50 MB).
- `UPLOAD_TTL_SECONDS`: Upload expiration in seconds (default: `300`).
- `MAX_IMAGE_PIXELS`: Maximum decoded image pixels (default: `16777216` = 4096x4096).
- `MAX_AUDIO_DURATION_SECONDS`: Maximum allowed audio duration (default: `30.0`, cannot exceed the model window).
- `MAX_VIDEO_DURATION_SECONDS`: Maximum allowed video duration (default: `60.0`).
- `MAX_CONTEXT_TOKENS`: Shared token budget across all modalities (default: `8192`). The service checks actual processor token IDs, including prompts and media, before inference and returns 413 when exceeded.
- `VIDEO_FPS`: Target sampling rate for raw video uploads (default: `1`).
- `MAX_VIDEO_FRAMES`: Maximum sampled frames or supplied frames per video group (default: `32`). Raw video sampling spreads these frames across the clip when the target rate would exceed the cap.
- `VISION_MAX_SOFT_TOKENS`: Token budget per image (default: `280`).
- `VIDEO_MAX_SOFT_TOKENS`: Token budget per video frame (default: `140`, matching the pinned checkpoint).

## Running the Service

Start the service on localhost:

```bash
uv run python -m multimodal_embedding.server
```

Or run via Uvicorn directly:

```bash
uv run uvicorn multimodal_embedding.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## API Contract

### Health Endpoints

#### `GET /health/live`

Liveness probe confirming the HTTP server process is running.

Response `200 OK`:

```json
{
  "status": "ok"
}
```

#### `GET /health/ready`

Readiness probe confirming the model adapter is loaded and ready to serve inference.

Response `200 OK` (when ready):

```json
{
  "status": "ready",
  "model": "google/embeddinggemma-2",
  "actual_device": "mps",
  "device_fallback": false,
  "queue_depth": 0,
  "active_workers": 0
}
```

Response `503 Service Unavailable` (when loading or failed):

```json
{
  "status": "not_ready",
  "model": "google/embeddinggemma-2",
  "actual_device": "none",
  "device_fallback": false,
  "queue_depth": 0,
  "active_workers": 0
}
```

### Models Metadata

#### `GET /v1/models`

Lists active model configuration, supported modalities, output dimensions, and enforced limits.

Response `200 OK`:

```json
{
  "data": [
    {
      "id": "google/embeddinggemma-2",
      "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
      "actual_device": "mps",
      "requested_device": "auto",
      "device_fallback": false,
      "supported_inputs": ["text", "image", "audio", "video"],
      "output_dimensions": [768, 512, 256, 128],
      "configured_limits": {
        "max_batch_size": 32,
        "max_upload_bytes": 52428800,
        "max_image_pixels": 16777216,
        "max_audio_duration_seconds": 30.0,
        "max_video_duration_seconds": 60.0,
        "max_context_tokens": 8192,
        "max_concurrency": 1,
        "max_queue_size": 10,
        "video_fps": 1,
        "max_video_frames": 32,
        "vision_max_soft_tokens": 280,
        "video_max_soft_tokens": 140
      }
    }
  ]
}
```

### Media Uploads

#### `POST /v1/uploads`

Uploads an image, audio, or video file for subsequent embedding requests. Each file is size-limited and stored temporarily with automatic expiration.

Request: `multipart/form-data` with form field `file`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/uploads \
  -F "file=@tests/fixtures/sample.png"
```

Response `200 OK`:

```json
{
  "upload_id": "09f5ebd8-a58d-4621-b0be-207f6711582c",
  "filename": "sample.png",
  "modality": "image",
  "content_type": "image/png",
  "size_bytes": 185,
  "created_at": 1791546823.65,
  "expires_at": 1791547123.65
}
```

### Embeddings Generation

#### `POST /v1/embeddings`

Computes dense vector representations for a batch of items.

Request Body:

- `items` (required): List of items to embed (max `max_batch_size`).
  - `id` (required): Unique item identifier.
  - `parts` (required): Ordered list of content parts.
    - Text part: `{"type": "text", "text": "..."}`
    - Media part: `{"type": "media", "upload_id": "<upload_id>"}`
- `task` (optional): `query` or `document` (default: `document`).
- `prompt_mode` (optional): `auto` applies the model's query/document prompt; `none` accepts already formatted text without adding another prompt.
- Video frame part: `{"type": "video_frames", "upload_ids": ["<image_upload_id>", "<image_upload_id>"]}`. Image uploads are treated as one ordered video sequence, without resampling.
- `truncate_dim` (optional): `768`, `512`, `256`, or `128` for MRL truncation with re-normalization.

#### Example 1: Text Query

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "query-1",
        "parts": [{"type": "text", "text": "Search query here"}]
      }
    ],
    "task": "query"
  }'
```

#### Example 2: Uploaded Media

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "img-doc-1",
        "parts": [{"type": "media", "upload_id": "09f5ebd8-a58d-4621-b0be-207f6711582c"}]
      }
    ],
    "task": "document"
  }'
```

#### Example 3: Combined Multimodal Item

Multiple parts in a single item are combined into one embedding:

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id": "combined-1",
        "parts": [
          {"type": "text", "text": "Photo of sunset with bird songs"},
          {"type": "media", "upload_id": "<image_upload_id>"},
          {"type": "media", "upload_id": "<audio_upload_id>"}
        ]
      }
    ],
    "task": "document"
  }'
```

#### Example 4: Mixed Heterogeneous Batch with 256d Truncation

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "item-text", "parts": [{"type": "text", "text": "City skyline"}]},
      {"id": "item-image", "parts": [{"type": "media", "upload_id": "<image_upload_id>"}]},
      {"id": "item-audio", "parts": [{"type": "media", "upload_id": "<audio_upload_id>"}]},
      {"id": "item-video", "parts": [{"type": "media", "upload_id": "<video_upload_id>"}]}
    ],
    "truncate_dim": 256
  }'
```

Response `200 OK`:

```json
{
  "model": "google/embeddinggemma-2",
  "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
  "task": "document",
  "dimension": 256,
  "normalized": true,
  "embeddings": [
    {
      "id": "item-text",
      "embedding": [0.031, -0.012, 0.084],
      "dimension": 256
    }
  ],
  "usage": {
    "total_items": 1,
    "latency_ms": 42.1
  }
}
```

### Error Responses

All errors follow a standard typed format:

```json
{
  "error": {
    "code": "invalid_input",
    "message": "Item 'item-1': placeholder count mismatch for <|image|>.",
    "details": {}
  }
}
```

Standard error codes:

- `400 Bad Request`: `invalid_input` (malformed input, placeholder mismatch, invalid dimension).
- `404 Not Found`: `not_found` (unknown or expired `upload_id`).
- `413 Payload Too Large`: `payload_too_large` (file exceeds upload bytes, image resolution, audio/video duration, or batch size limit).
- `503 Service Unavailable`: `model_unavailable` (model still loading) or `queue_saturation` (queue capacity reached).
- `500 Internal Server Error`: `inference_failed` (internal model failure).

## Testing

The test suite separates fast mock-based unit tests from opt-in real model runs.

### Mock-Based Unit and Contract Tests

Runs in ~2 seconds without loading weights:

```bash
uv run pytest tests/test_health.py tests/test_models.py tests/test_uploads.py tests/test_embeddings.py -v
```

Covers:
- Liveness and readiness states (including 503 when not ready).
- Model metadata and limit reporting.
- Media upload validation, dimension/duration limits, and expiry purging.
- Query vs. document prompt selection.
- Input batch ordering preservation.
- Manual placeholder validation and mismatch rejections.
- Matryoshka truncation across 512d, 256d, 128d with re-normalization.
- Concurrency and queue saturation under heavy traffic.

### Real Model Smoke Tests

Opt-in verification on actual `google/embeddinggemma-2` weights:

```bash
RUN_REAL_MODEL_TESTS=1 uv run pytest tests/test_smoke_real.py -v -s
```

Verifies real CPU and Apple Silicon MPS execution across all four input modalities and heterogeneous batches.

## Client Usage Example

A complete client script demonstrates checking health, querying model metadata, uploading media, generating embeddings, and ranking results by cosine similarity:

```bash
uv run python scripts/client_example.py http://127.0.0.1:8000
```

## Search-anything integration

This repository is checked out at `search-anything/embeddinggemma-api` as a Git submodule. For a fresh parent checkout, run `git submodule update --init --recursive`.

The default model revision is pinned to `914f7f89142e33e77833254d9c9b90c3cef7303b`, matching the worker's existing index. Clients must check `/v1/models` for the expected model, revision, and 768-dimensional output before using the service with that index.

Worker-generated document text already contains `title: ... | text: ...`. Send it with `prompt_mode: "none"`. Plain query text uses `task: "query"` with `prompt_mode: "auto"`. For worker-prepared media and combined inputs, use `prompt_mode: "none"` to match existing embeddings. Upload audio windows of at most 30 seconds. Upload prepared images individually and reference them in a `video_frames` part to preserve the worker's selected frame order.

Keep the pinned tokenizer in the worker for token counting and offsets used by chunk planning. The service performs the final context-budget check with the model processor. Tokenization does not require loading model weights.

The service contract is ready for an HTTP client; the parent worker still uses its local adapter until that client is configured. Query prioritization and cross-request dynamic batching are not implemented.

Run the regression tests, including real CPU inference and parity with the parent worker adapter:

```bash
HF_HUB_OFFLINE=1 RUN_REAL_MODEL_TESTS=1 uv run pytest -q
```

The worker parity test runs when the service is inside the parent checkout. MPS verification requires a process with MPS access. CPU tests verify document, query, image, audio, video frame groups, and combined embeddings against the actual worker adapter. Tests also cover cancelled requests, oversized context, and audio-window rejection.
