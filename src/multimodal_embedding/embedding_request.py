"""Bounded JSON and multipart parsing for the embeddings endpoint."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request as StarletteRequest

from multimodal_embedding.exceptions import InvalidInputError, PayloadTooLargeError
from multimodal_embedding.media_inputs import MediaResolver
from multimodal_embedding.schemas import EmbeddingRequest, MediaPart


def request_body_schema():
    schema = EmbeddingRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(value):
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].split("/")[-1]])
            return {
                key: (
                    {"propertyName": child["propertyName"]}
                    if key == "discriminator"
                    else inline(child)
                )
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [inline(child) for child in value]
        return value

    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": inline(schema)},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload"],
                        "properties": {
                            "payload": {
                                "type": "string",
                                "description": "JSON EmbeddingRequest; media.file references a named binary field",
                            }
                        },
                        "additionalProperties": {"type": "string", "format": "binary"},
                    }
                },
            },
        }
    }


async def parse_embedding_request(request: Request):
    cfg = request.app.state.settings
    content_type = (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    )
    if content_type not in ("application/json", "multipart/form-data"):
        raise InvalidInputError("Use application/json or multipart/form-data.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > cfg.max_embedding_request_bytes:
            raise PayloadTooLargeError(
                "Embedding request exceeds max_embedding_request_bytes."
            )
        body.extend(chunk)

    async def receive():
        return {"type": "http.request", "body": bytes(body), "more_body": False}

    parsed_request = StarletteRequest(request.scope, receive)
    form = None
    resolver = None
    try:
        files = {}
        if content_type == "multipart/form-data":
            form = await parsed_request.form(
                max_files=128,
                max_fields=1,
                max_part_size=cfg.max_embedding_request_bytes,
            )
            payloads = form.getlist("payload")
            if len(payloads) != 1 or not isinstance(payloads[0], str):
                raise InvalidInputError(
                    "Multipart requires exactly one JSON payload field."
                )
            raw = payloads[0]
            for name, value in form.multi_items():
                if name == "payload":
                    continue
                if not isinstance(value, UploadFile) or name in files:
                    raise InvalidInputError(
                        "Multipart file fields must have unique names."
                    )
                files[name] = value
        else:
            raw = bytes(body)
        try:
            payload = EmbeddingRequest.model_validate_json(raw)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from None
        if len(payload.items) > cfg.max_batch_size:
            raise PayloadTooLargeError("Batch exceeds max_batch_size.")
        referenced = {
            part.file
            for item in payload.items
            for part in item.parts
            if isinstance(part, MediaPart) and part.file is not None
        }
        if referenced != set(files):
            raise InvalidInputError(
                "Multipart file fields must match the media.file references exactly."
            )
        resolver = MediaResolver(request.app.state.storage, cfg, files)
        yield payload, resolver
    except HTTPException:
        raise InvalidInputError(
            "Malformed multipart body or too many fields/files."
        ) from None
    finally:
        if resolver is not None:
            await resolver.close()
        if form is not None:
            await form.close()
