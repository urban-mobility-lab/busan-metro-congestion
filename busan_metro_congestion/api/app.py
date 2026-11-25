"""사전 추론 혼잡도 예측 결과 조회 REST API.

실행: `uv run uvicorn busan_metro_congestion.api.app:app`
문서: 서버 실행 후 `/docs` (Swagger UI), `/openapi.json`
명세: `docs/API_SPEC.md`
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Query

from busan_metro_congestion.api.errors import ApiError, register_error_handlers
from busan_metro_congestion.api.predictions import (
    DEFAULT_PREDICTIONS_PATH,
    PredictionStore,
    PredictionStoreUnavailableError,
    kst_naive_to_aware,
    utc_naive_to_kst,
)
from busan_metro_congestion.api.schemas import (
    ErrorResponse,
    HealthResponse,
    HourlyPrediction,
    ModelInfo,
    PredictionResponse,
    Station,
)
from busan_metro_congestion.api.stations import STATION_MASTER_PATH, StationRepository

ALL_HOURS = list(range(1, 25))
VALID_LINE_NUMS = range(1, 5)


def _responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    return {status: {"model": ErrorResponse} for status in statuses}


def create_app(
    *,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    station_master_path: Path = STATION_MASTER_PATH,
) -> FastAPI:
    stations = StationRepository(station_master_path)
    store = PredictionStore(predictions_path)

    app = FastAPI(
        title="부산 지하철 혼잡도 예측 결과 조회 API",
        version="1",
        description=(
            "배치 추론이 사전 적재한 예측 결과에서 역·날짜·시간대별 예측 승객 수를 "
            "조회합니다. 시간은 KST입니다."
        ),
    )
    register_error_handlers(app)
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/stations", response_model=list[Station], responses=_responses(422))
    def list_stations(
        line_num: Annotated[
            int | None, Query(description="호선 번호 1~4. 생략하면 전체")
        ] = None,
    ) -> list[Station]:
        if line_num is not None and line_num not in VALID_LINE_NUMS:
            raise ApiError(
                422,
                "INVALID_LINE",
                f"line_num은 1부터 4 사이여야 합니다. 받은 값: {line_num}",
            )
        return stations.all(line_num)

    @router.get("/model-info", response_model=ModelInfo, responses=_responses(503))
    def model_info() -> ModelInfo:
        return ModelInfo(**dataclasses.asdict(store.info()))

    @router.get(
        "/predictions",
        response_model=PredictionResponse,
        responses=_responses(404, 422, 503),
    )
    def predictions(
        station_id: Annotated[int, Query(description="역 번호")],
        date: Annotated[dt.date, Query(description="조회 운행일 (YYYY-MM-DD, KST)")],
        hour: Annotated[
            list[int] | None,
            Query(description="시간대 1~24, 반복 가능. 생략하면 1~24 전체"),
        ] = None,
    ) -> PredictionResponse:
        hours = sorted(set(hour)) if hour else ALL_HOURS
        invalid = [h for h in hours if not 1 <= h <= 24]
        if invalid:
            raise ApiError(
                422,
                "INVALID_HOUR",
                f"hour는 1부터 24 사이여야 합니다. 받은 값: {invalid}",
            )

        station = stations.get(station_id)
        if station is None:
            raise ApiError(
                404,
                "STATION_NOT_FOUND",
                f"역 번호 {station_id}을(를) 찾을 수 없습니다.",
            )

        info = store.info()
        rows = store.query(station_id, date)
        if rows.is_empty():
            raise ApiError(
                422,
                "DATE_OUT_OF_RANGE",
                f"{date.isoformat()}은(는) 예측 가능 기간"
                f"({info.available_from.isoformat()} ~ "
                f"{info.available_to.isoformat()})을 벗어났습니다.",
            )

        by_hour = {int(r["target_hour"]): r for r in rows.iter_rows(named=True)}
        missing = [h for h in hours if h not in by_hour]
        if missing:
            raise PredictionStoreUnavailableError(
                f"역 {station_id} {date.isoformat()} 예측에 시간대가 없습니다: {missing}"
            )

        selected = [by_hour[h] for h in hours]
        latest = max(selected, key=lambda r: r["generated_at"])
        return PredictionResponse(
            **station.model_dump(),
            date=date,
            model_version=str(latest["model_version"]),
            generated_at=utc_naive_to_kst(latest["generated_at"]),
            predictions=[
                HourlyPrediction(
                    hour=int(r["target_hour"]),
                    target_datetime=kst_naive_to_aware(r["target_datetime"]),
                    predicted_passengers=round(float(r["predicted_passengers"]), 1),
                )
                for r in selected
            ],
        )

    app.include_router(router)
    return app


app = create_app()
