"""
전날 17시 예측 시나리오의 과거 수요 Feature 생성 모듈

D일 입력에는 D-2일까지의 실적만 사용합니다. D-1일 실적은 예측 시점(D-1일 17시)에
아직 집계되지 않았다고 가정합니다.
- 주간 Feature: 7일 이상 지난 실적만 사용하며, 실적 공개가 일주일 늦어도 만들 수 있습니다.
- 최근 Feature: D-2일까지의 실적을 사용합니다.
"""

from __future__ import annotations

import polars as pl

TARGET_COL = "accumulated_passengers"
HISTORY_WEEKLY_FEATURES = [
    "lag_7d",
    "lag_14d",
    "lag_21d",
    "lag_28d",
    "mean_4w_same_hour",
    "lag_7d_is_holiday",
]
HISTORY_RECENT_FEATURES = [
    "lag_2d",
    "lag_2d_is_holiday",
    "mean_7d_same_hour",
    "total_7d",
    "trend_7d",
]
HISTORY_FEATURES = HISTORY_WEEKLY_FEATURES + HISTORY_RECENT_FEATURES
HISTORY_WARMUP_DAYS = 29
LATEST_ACTUAL_OFFSET_DAYS = 2


def _shift_date(days: int) -> pl.Expr:
    return (pl.col("date") + pl.duration(days=days)).alias("date")


def add_history_features(
    df: pl.DataFrame, target_col: str = TARGET_COL
) -> pl.DataFrame:
    """target_col의 과거 실적으로 HISTORY_FEATURES를 계산해 추가합니다."""
    keys = ["station_id", "date", "hour"]
    value = pl.col(target_col).cast(pl.Float64)

    result = df
    for days in (LATEST_ACTUAL_OFFSET_DAYS, 7, 14, 21, 28):
        lagged = df.select(
            "station_id", _shift_date(days), "hour", value.alias(f"lag_{days}d")
        )
        result = result.join(lagged, on=keys, how="left")

    holidays = df.select("date", pl.col("is_holiday").cast(pl.Float32)).unique("date")
    for days in (LATEST_ACTUAL_OFFSET_DAYS, 7):
        result = result.join(
            holidays.select(
                _shift_date(days), pl.col("is_holiday").alias(f"lag_{days}d_is_holiday")
            ),
            on="date",
            how="left",
        )

    same_hour = df.sort("date").select(
        "station_id",
        _shift_date(LATEST_ACTUAL_OFFSET_DAYS),
        "hour",
        value.rolling_mean_by(pl.col("date"), window_size="7d")
        .over(["station_id", "hour"])
        .alias("mean_7d_same_hour"),
    )
    daily = (
        df.group_by(["station_id", "date"])
        .agg(value.sum().alias("_total"))
        .sort("date")
        .with_columns(
            pl.col("_total")
            .rolling_sum_by(pl.col("date"), window_size="7d")
            .over("station_id")
            .alias("total_7d")
        )
    )
    previous_week = daily.select(
        "station_id", _shift_date(7), pl.col("total_7d").alias("_previous_7d")
    )
    totals = (
        daily.join(previous_week, on=["station_id", "date"], how="left")
        .with_columns(
            pl.when(pl.col("_previous_7d") > 0)
            .then(pl.col("total_7d") / pl.col("_previous_7d"))
            .alias("trend_7d")
        )
        .select(
            "station_id",
            _shift_date(LATEST_ACTUAL_OFFSET_DAYS),
            "total_7d",
            "trend_7d",
        )
    )
    return (
        result.join(same_hour, on=keys, how="left")
        .join(totals, on=["station_id", "date"], how="left")
        .with_columns(
            pl.mean_horizontal("lag_7d", "lag_14d", "lag_21d", "lag_28d").alias(
                "mean_4w_same_hour"
            ),
            *[pl.col(c).cast(pl.Float32) for c in ("total_7d", "trend_7d")],
        )
    )


def drop_history_warmup(df: pl.DataFrame) -> pl.DataFrame:
    """과거 실적이 모자란 데이터 시작 구간을 모든 모델의 학습에서 동일하게 제외합니다."""
    first_day = df["date"].min()
    return df.filter(
        pl.col("date") >= first_day + pl.duration(days=HISTORY_WARMUP_DAYS)
    )
