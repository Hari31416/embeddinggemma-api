from typing import Any


class APIException(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InvalidInputError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="invalid_input",
            message=message,
            status_code=400,
            details=details,
        )


class PayloadTooLargeError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="payload_too_large",
            message=message,
            status_code=413,
            details=details,
        )


class NotFoundError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="not_found",
            message=message,
            status_code=404,
            details=details,
        )


class ModelUnavailableError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="model_unavailable",
            message=message,
            status_code=503,
            details=details,
        )


class QueueSaturationError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="queue_saturation",
            message=message,
            status_code=503,
            details=details,
        )


class InferenceError(APIException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            code="inference_failed",
            message=message,
            status_code=500,
            details=details,
        )
