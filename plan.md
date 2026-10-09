# Local multimodal model API

## Goal

Build a standalone FastAPI service for `google/embeddinggemma-2`, using Sentence Transformers. Support text, images, audio, video, and combined text/media inputs. CPU on macOS is required; Apple Silicon MPS acceleration is optional. Keep the service reusable by other applications.

Use Python, uv, Pydantic, logging, Black, and PyTorch. Search, storage infrastructure, transcription, OCR, chunking, and frontend are outside this project's scope.

## Phase 1: Verify model support

- Read the current model card and official integration documentation:
  - https://huggingface.co/google/embeddinggemma-2
  - https://huggingface.co/docs/transformers/main/model_doc/embedding_gemma2
- Start with the documented Sentence Transformers integration, currently recommending >=6.1.0. Pin a tested dependency set in `uv.lock`.
- Run real CPU inference on small fixtures for all four input types and combined inputs. Confirm query/document prompts, pooling, normalization, and output dimensions against the documentation.
- Support `device=cpu|mps|auto`. CPU must work independently of MPS. Probe MPS support and make any CPU fallback explicit in logs and health metadata.
- Document model access requirements, download size, observed memory usage, and startup commands. Read Hugging Face credentials from the environment without logging them.

Done when actual CPU inference works for every input type, or a specific upstream blocker is documented. Do not silently substitute another model.

## Phase 2: Implement the service

- Load one configured model during application lifespan and reuse it. Default to one server worker, bounded inference concurrency, and a bounded queue. Run blocking inference outside the async event loop.
- Provide `GET /health/live`, `GET /health/ready`, and `GET /v1/models`. Report model identifier/revision, actual device, supported inputs, output dimensions, and configured limits.
- Provide multipart `POST /v1/uploads` for media. Use a bounded temporary directory with expiry and cleanup. Accept service-issued upload IDs in embedding requests; remote URLs and arbitrary filesystem paths are outside the first version.
- Provide JSON `POST /v1/embeddings` accepting a batch of items. Each item contains an ID and ordered typed content parts: text or media referenced by upload ID. Multiple parts in one item produce one combined embedding. Items in a batch may use different input types.
- Support `task=query|document` using documented prompts. Return results in input order with IDs, embeddings, model/revision, dimensions, task, and normalization metadata. Keep preprocessing inside the model adapter.
- Validate batch size, upload bytes, decoded image size, audio/video duration, and context limits. Allow bounded video frame sampling and visual token budget settings. Reject oversized inputs clearly rather than silently truncating them.
- Return typed errors for invalid input, unavailable model, queue saturation, and inference failure. Log timing and failures without user content. Bind to localhost by default.

Done when curl examples work for each input type, a combined item, and a mixed batch. Document the API contract and limits in README and OpenAPI.

## Phase 3: Test and document

- Use a fake model adapter to test validation, output ordering, prompt selection, limits, upload cleanup, and concurrency behavior.
- Add opt-in real-model CPU and MPS smoke checks. Verify finite, normalized embeddings and matching dimensions across input types. Record latency and memory for small representative inputs.
- Provide `.env.example`, `uv` setup/run commands, fixture instructions, and a short client example. Prioritize native macOS execution.

Done when contract tests pass and real CPU smoke results are documented. Clearly distinguish mocked tests from real model runs.

## Later: Reranking

Reserve a separate adapter and `POST /v1/rerank` contract with query, candidates, optional top-N, and returned original indices/scores. Select and benchmark a CPU-compatible text reranker before implementation. Multimodal reranking requires a separately verified model and contract. Cosine similarity is not a reranking model.

Stop after the embedding service works and is documented. Reranker selection and implementation require a separate decision.
