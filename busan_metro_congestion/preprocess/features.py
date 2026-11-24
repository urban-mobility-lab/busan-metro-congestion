"""
지하철, 기상, 축제, 달력 데이터 통합 피처 엔지니어링 파이프라인 모듈

마스터 피처 테이블(data/features/metro_features.parquet)을 생성합니다.
1. 지하철(subway_processed)을 공간·시간 앵커축(station_id, datetime, date, hour)으로 설정
2. 114개 역사 격자 매핑(station_weather_grid) 및 기상(weather_processed) 1:1 Left Join
3. 문화축제·대규모 전시(festival_processed) 1:1 Left Join (반경 1.5km 충격량 피처 4종)
4. 대한민국 11개년 달력 마스터(calendar_master) 1:1 Left Join (공휴일/전야/계절 및 삼각함수 주기)
5. 시간대 주기성(hour_sin, hour_cos) 및 베이지안 5-Fold OOF 타깃 인코딩(station_target_encoded) 결합
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from busan_metro_congestion.config import (
    FEATURES_DATA_DIR,
    PROCESSED_DATA_DIR,
    REFERENCE_DATA_DIR,
    set_seed,
)
from busan_metro_congestion.preprocess.festival import (
    build_festival_processed,
)
from busan_metro_congestion.preprocess.subway import (
    StationTargetEncoder,
    add_cyclical_time_features,
    build_subway_processed,
    process_subway_data,
)
from busan_metro_congestion.preprocess.weather import (
    process_weather_data,
)

logger = logging.getLogger("preprocess_features")

MASTER_FEATURE_COLUMNS = [
    # 식별 및 키
    "station_id",
    "line_num",
    "datetime",
    "date",
    "hour",
    # 타깃
    "accumulated_passengers",
    # 역사 기저 피처
    "station_target_encoded",
    # 기상 피처 (단기예보 10종)
    "lead_time_hours",
    "temperature",
    "humidity",
    "precipitation",
    "pop",
    "rainfall_type",
    "sky_condition",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    # 축제·전시 피처 (4종)
    "nearest_event_rank",
    "max_dist_weight",
    "max_scale_weight",
    "event_impact_score",
    # 달력 및 시간 주기 피처 (10종)
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "season",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
]


def run_feature_pipeline(
    subway_path: Path | str | None = None,
    weather_path: Path | str | None = None,
    festival_path: Path | str | None = None,
    calendar_path: Path | str | None = None,
    station_grid_path: Path | str | None = None,
    station_master_path: Path | str | None = None,
    output_path: Path | str | None = None,
    force_reload: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """
    모든 데이터 소스를 통합 결합하여 data/features/metro_features.parquet를 생성합니다.

    인자:
        subway_path: 지하철 전처리 데이터 경로.
        weather_path: 기상 전처리 데이터 경로.
        festival_path: 축제 전처리 데이터 경로.
        calendar_path: 달력 마스터 데이터 경로.
        station_grid_path: 역사별 기상 격자 매핑 경로.
        station_master_path: 역사 마스터 데이터 경로.
        output_path: 최종 저장 경로 (기본값: FEATURES_DATA_DIR / 'metro_features.parquet').
        force_reload: True이면 기존 캐시 파일 존재 여부와 무관하게 전처리 재생성.

    반환값:
        정제 및 결측치가 완벽히 보정된 마스터 피처 Polars DataFrame.
    """
    target_out = (
        Path(output_path)
        if output_path is not None
        else FEATURES_DATA_DIR / "metro_features.parquet"
    )
    if not force_reload and target_out.exists():
        logger.info(f"기존 마스터 피처 캐시 파일을 즉시 로드합니다: {target_out}")
        return pl.read_parquet(target_out)

    set_seed(seed)
    logger.info("통합 피처 파이프라인 실행 시작")

    # -------------------------------------------------------------
    # 1. 공간·시간 앵커 (subway_processed) 준비
    # -------------------------------------------------------------
    df_subway: pl.DataFrame
    if subway_path is not None:
        p_sub = Path(subway_path)
        if p_sub.suffix == ".parquet":
            df_subway = pl.read_parquet(p_sub)
        else:
            df_subway = process_subway_data(
                p_sub, station_master_path=station_master_path
            )
    else:
        processed_sub = PROCESSED_DATA_DIR / "subway_processed.parquet"
        if processed_sub.exists():
            df_subway = pl.read_parquet(processed_sub)
        else:
            df_subway = build_subway_processed(station_master_path=station_master_path)

    # 컬럼 표준화: line_no -> line_num
    if "line_no" in df_subway.columns and "line_num" not in df_subway.columns:
        df_subway = df_subway.rename({"line_no": "line_num"})

    if "line_num" not in df_subway.columns:
        # station_master에서 매핑
        s_master_p = (
            Path(station_master_path)
            if station_master_path
            else REFERENCE_DATA_DIR / "station_master.parquet"
        )
        if s_master_p.exists():
            stn_m = pl.read_parquet(s_master_p).select(
                [pl.col("station_id"), pl.col("line_num").cast(pl.Int8)]
            )
            df_subway = df_subway.join(stn_m, on="station_id", how="left")
        else:
            df_subway = df_subway.with_columns(
                pl.lit(1, dtype=pl.Int8).alias("line_num")
            )

    df_subway = df_subway.with_columns(
        pl.col("station_id").cast(pl.Int64),
        pl.col("line_num").cast(pl.Int8),
        pl.col("date").cast(pl.Date),
        pl.col("hour").cast(pl.Int8),
        pl.col("datetime").cast(pl.Datetime),
        pl.col("accumulated_passengers").cast(pl.Float32),
    )
    n_expected_rows = len(df_subway)

    # 5-Fold OOF 타깃 인코딩 컬럼 생성
    if "station_target_encoded" not in df_subway.columns:
        logger.info(
            "지하철 데이터에 5-Fold OOF 역사 타깃 인코딩(station_target_encoded)을 생성합니다."
        )
        encoder = StationTargetEncoder(m=10.0, n_splits=5, seed=seed)
        df_subway = encoder.fit_transform(
            df_subway,
            station_col="station_id",
            target_col="accumulated_passengers",
        )

    # -------------------------------------------------------------
    # 2. 기상 격자 매핑 및 기상 피처 결합
    # -------------------------------------------------------------
    if station_grid_path is None:
        station_grid_path = REFERENCE_DATA_DIR / "station_weather_grid.parquet"
    df_grid = (
        pl.read_parquet(station_grid_path)
        .select(
            [
                pl.col("station_id").cast(pl.Int64),
                pl.col("nx").cast(pl.Int64),
                pl.col("ny").cast(pl.Int64),
            ]
        )
        .unique(subset=["station_id"], keep="first")
    )

    df_subway_grid = df_subway.join(df_grid, on="station_id", how="left")

    # 기상 전처리 데이터 로드
    df_weather: pl.DataFrame
    if weather_path is not None:
        p_w = Path(weather_path)
        if p_w.suffix == ".parquet":
            df_weather = pl.read_parquet(p_w)
        else:
            df_weather = process_weather_data(p_w, track="forecast")
    else:
        processed_w = PROCESSED_DATA_DIR / "weather_processed.parquet"
        if processed_w.exists():
            df_weather = pl.read_parquet(processed_w)
        else:
            df_weather = process_weather_data(track="forecast", save_output=True)

    # 기상 데이터 (nx, ny, datetime) 기준 1:1 Left Join (중복 키 방어)
    weather_cols_to_join = [
        "nx",
        "ny",
        "datetime",
        "lead_time_hours",
        "temperature",
        "humidity",
        "precipitation",
        "pop",
        "rainfall_type",
        "sky_condition",
        "wind_speed",
        "wind_dir_sin",
        "wind_dir_cos",
    ]
    df_weather_sel = df_weather.select(
        [c for c in weather_cols_to_join if c in df_weather.columns]
    ).unique(subset=["nx", "ny", "datetime"], keep="first")

    df_with_weather = df_subway_grid.join(
        df_weather_sel, on=["nx", "ny", "datetime"], how="left"
    )

    # 기상 결측치 안전 보정 (기상 매칭 범위 밖일 경우 05시/17시 발표 기준 1~12시간 주기 및 시간대 평균 대치)
    fallback_lead_time = (((pl.col("hour").cast(pl.Int16) - 6) % 12) + 1).cast(pl.Int16)
    df_with_weather = df_with_weather.with_columns(
        pl.col("lead_time_hours").fill_null(fallback_lead_time).cast(pl.Int16),
        pl.col("temperature")
        .fill_null(pl.col("temperature").mean().over("hour"))
        .fill_null(15.0)
        .cast(pl.Float32),
        pl.col("humidity")
        .fill_null(pl.col("humidity").mean().over("hour"))
        .fill_null(60.0)
        .cast(pl.Float32),
        pl.col("precipitation").fill_null(0.0).cast(pl.Float32),
        pl.col("pop").fill_null(0.0).cast(pl.Float32),
        pl.col("rainfall_type").fill_null(0).cast(pl.Int8),
        pl.col("sky_condition").fill_null(1).cast(pl.Int8),
        pl.col("wind_speed")
        .fill_null(pl.col("wind_speed").mean().over("hour"))
        .fill_null(2.0)
        .cast(pl.Float32),
        pl.col("wind_dir_sin").fill_null(0.0).cast(pl.Float32),
        pl.col("wind_dir_cos").fill_null(1.0).cast(pl.Float32),
    )

    # -------------------------------------------------------------
    # 3. 문화축제·대규모 전시 피처 결합
    # -------------------------------------------------------------
    df_festival: pl.DataFrame
    if festival_path is not None:
        df_festival = pl.read_parquet(festival_path)
    else:
        proc_fest = PROCESSED_DATA_DIR / "festival_processed.parquet"
        if proc_fest.exists():
            df_festival = pl.read_parquet(proc_fest)
        else:
            try:
                df_festival = build_festival_processed(
                    station_master_path=station_master_path
                )
            except Exception:
                logger.warning(
                    "축제 전처리 파일 생성 불가: 기본 0값 피처로 대체합니다."
                )
                df_festival = pl.DataFrame(
                    schema={
                        "station_id": pl.Int64,
                        "date": pl.Date,
                        "nearest_event_rank": pl.Int8,
                        "max_dist_weight": pl.Float32,
                        "max_scale_weight": pl.Float32,
                        "event_impact_score": pl.Float32,
                    }
                )

    fest_join_cols = [
        "station_id",
        "date",
        "nearest_event_rank",
        "max_dist_weight",
        "max_scale_weight",
        "event_impact_score",
    ]
    df_fest_sel = df_festival.select(
        [c for c in fest_join_cols if c in df_festival.columns]
    ).unique(subset=["station_id", "date"], keep="first")

    df_with_fest = df_with_weather.join(
        df_fest_sel, on=["station_id", "date"], how="left"
    )

    df_with_fest = df_with_fest.with_columns(
        pl.col("nearest_event_rank").fill_null(0).cast(pl.Int8),
        pl.col("max_dist_weight").fill_null(0.0).cast(pl.Float32),
        pl.col("max_scale_weight").fill_null(0.0).cast(pl.Float32),
        pl.col("event_impact_score").fill_null(0.0).cast(pl.Float32),
    )

    # -------------------------------------------------------------
    # 4. 대한민국 달력 및 공휴일 마스터 결합
    # -------------------------------------------------------------
    df_cal: pl.DataFrame
    if calendar_path is not None:
        df_cal = pl.read_parquet(calendar_path)
    else:
        cal_master_p = REFERENCE_DATA_DIR / "calendar_master.parquet"
        if not cal_master_p.exists():
            raise FileNotFoundError(
                f"달력 마스터 참조 파일을 찾을 수 없습니다: {cal_master_p}. "
                "사전에 'python scripts/build_calendar_master.py'를 실행하여 참조 테이블을 구축하세요."
            )
        df_cal = pl.read_parquet(cal_master_p)

    df_cal = df_cal.unique(subset=["date"], keep="first")
    df_with_cal = df_with_fest.join(df_cal, on="date", how="left")

    df_with_cal = df_with_cal.with_columns(
        pl.col("is_weekend").fill_null(False).cast(pl.Boolean),
        pl.col("is_holiday").fill_null(False).cast(pl.Boolean),
        pl.col("is_holiday_eve").fill_null(False).cast(pl.Boolean),
        pl.col("season").fill_null(1).cast(pl.Int8),
        pl.col("month_sin").fill_null(0.0).cast(pl.Float32),
        pl.col("month_cos").fill_null(0.0).cast(pl.Float32),
        pl.col("weekday_sin").fill_null(0.0).cast(pl.Float32),
        pl.col("weekday_cos").fill_null(0.0).cast(pl.Float32),
    )

    # -------------------------------------------------------------
    # 5. 시간대 주기성 피처 (hour_sin, hour_cos)
    # -------------------------------------------------------------
    df_final = add_cyclical_time_features(
        df_with_cal, hour_col="hour", include_weekday=False
    )

    # -------------------------------------------------------------
    # 6. 최종 스키마 정렬, 무결성 검증 및 저장
    # -------------------------------------------------------------
    df_result = df_final.select(MASTER_FEATURE_COLUMNS).sort(["station_id", "datetime"])

    # 무결성 검증: 행 수 보존 및 타깃 결측치 전무 확인
    if len(df_result) != n_expected_rows:
        raise ValueError(
            f"시공간 조인 무결성 오류: 원본 지하철 행 수({n_expected_rows:,})와 최종 마스터 피처 행 수({len(df_result):,})가 일치하지 않습니다."
        )
    null_targets = df_result["accumulated_passengers"].is_null().sum()
    if null_targets > 0:
        raise ValueError(
            f"타깃 컬럼('accumulated_passengers')에 결측치가 {null_targets:,}건 존재합니다."
        )

    out_p = target_out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_result.write_parquet(out_p)
    logger.info(
        f"마스터 피처 파이프라인 완료: {out_p} ({len(df_result):,}행, {len(df_result.columns)}컬럼)"
    )

    return df_result
