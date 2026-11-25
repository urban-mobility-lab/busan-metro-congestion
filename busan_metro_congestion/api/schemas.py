"""API 요청/응답 스키마."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class Station(BaseModel):
    station_id: int = Field(description="역 번호")
    station_name: str = Field(description="역 이름")
    line_num: int = Field(description="호선 번호 (1~4)")
    lat: float = Field(description="위도 (WGS84)")
    lng: float = Field(description="경도 (WGS84)")


class HourlyPrediction(BaseModel):
    hour: int = Field(
        ge=1,
        le=24,
        description="시간대 (KST). 8은 08~09시, 24는 다음 날 00~01시",
    )
    target_datetime: dt.datetime = Field(
        description="시간대 시작 시각 (KST, 24시는 익일 00:00)"
    )
    predicted_passengers: float = Field(ge=0, description="예측 승객 수")


class PredictionResponse(Station):
    date: dt.date = Field(description="조회 운행일 (KST)")
    model_version: str = Field(description="모델 버전")
    generated_at: dt.datetime = Field(description="배치 생성 시각 (KST)")
    predictions: list[HourlyPrediction]


class ModelInfo(BaseModel):
    model_version: str = Field(description="모델 버전")
    generated_at: dt.datetime = Field(description="배치 생성 시각 (KST)")
    available_from: dt.date = Field(description="예측 가능 기간 시작일")
    available_to: dt.date = Field(description="예측 가능 기간 종료일")
    station_count: int = Field(description="저장소의 고유 역 수")
    row_count: int = Field(description="저장소 전체 행 수")


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    code: str = Field(description="오류 코드")
    message: str = Field(description="사용자에게 보여줄 한국어 메시지")
