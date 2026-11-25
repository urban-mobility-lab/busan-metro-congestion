"""API 오류 응답을 {"code", "message"} 형식으로 통일한다."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from busan_metro_congestion.api.predictions import PredictionStoreUnavailableError

logger = logging.getLogger(__name__)

_HTTP_MESSAGES = {
    404: "요청한 경로를 찾을 수 없습니다.",
    405: "허용되지 않는 요청 방식입니다.",
}
_NOT_READY_MESSAGE = "예측 모델이 준비되지 않았습니다. 잠시 후 다시 시도해 주세요."


class ApiError(Exception):
    """API가 의도적으로 돌려주는 오류."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"code": code, "message": message}
    )


def _describe_validation_error(exc: RequestValidationError) -> str:
    details = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err.get("loc") else "요청"
        if err.get("type") == "missing":
            details.append(f"{field} 값이 필요합니다")
        else:
            details.append(f"{field} 값의 형식이 올바르지 않습니다")
    return "요청 값이 올바르지 않습니다. " + ", ".join(details) + "."


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(422, "INVALID_REQUEST", _describe_validation_error(exc))

    @app.exception_handler(PredictionStoreUnavailableError)
    async def _store_unavailable(
        _: Request, exc: PredictionStoreUnavailableError
    ) -> JSONResponse:
        return error_response(503, "PREDICTION_NOT_READY", _NOT_READY_MESSAGE)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = _HTTP_MESSAGES.get(exc.status_code, "요청을 처리할 수 없습니다.")
        return error_response(exc.status_code, "INVALID_REQUEST", message)

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("예측 처리 중 오류: %s %s", request.method, request.url)
        return error_response(503, "PREDICTION_NOT_READY", _NOT_READY_MESSAGE)
