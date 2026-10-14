"""Optional bearer-key authentication for the service API."""

from secrets import compare_digest

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from multimodal_embedding.exceptions import APIException

bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    if credentials is not None:
        supplied = credentials.credentials.encode("utf-8")
        if any(
            compare_digest(supplied, key.get_secret_value().encode("utf-8"))
            for key in keys
        ):
            return
    raise APIException(
        code="unauthorized",
        message="A valid bearer API key is required.",
        status_code=401,
    )
