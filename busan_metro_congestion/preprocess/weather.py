"""
기상 데이터 전처리 모듈 (기상청 단기예보)

단일 진입점(process_weather_data)을 통해 기상청 단기예보(getVilageFcst) 데이터를 정제하여
기온, 습도, 강수량, 강수확률, 하늘상태, 강수형태, 풍속, 풍향(13개 표준 컬럼)을 산출합니다.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import polars as pl

from busan_metro_congestion.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger("preprocess_weather")

EXPECTED_WEATHER_COLUMNS = [
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


def _parse_precipitation_expr(col_name: str = "PCP") -> pl.Expr:
    """
    기상청 PCP 문자열을 Float32 강수량(mm)으로 변환하는 Polars 표현식을 반환합니다.

    규칙:
    - '강수없음', '0', '0.0mm', '0mm', '0.0' -> 0.0
    - '1mm 미만' -> 0.5
    - '30.0~50.0mm' -> 40.0
    - '50.0mm 이상' -> 50.0
    - 일반 실수 + 'mm' (예: '5.0mm') -> 5.0
    """
    s = pl.col(col_name).cast(pl.Utf8).str.strip_chars()
    return (
        pl.when(s.is_in(["강수없음", "0", "0.0mm", "0mm", "0.0"]))
        .then(pl.lit(0.0, dtype=pl.Float32))
        .when(s == "1mm 미만")
        .then(pl.lit(0.5, dtype=pl.Float32))
        .when(s == "30.0~50.0mm")
        .then(pl.lit(40.0, dtype=pl.Float32))
        .when(s == "50.0mm 이상")
        .then(pl.lit(50.0, dtype=pl.Float32))
        .otherwise(s.str.extract(r"([\d\.]+)", 1).cast(pl.Float32, strict=False))
        .fill_null(0.0)
        .alias("precipitation")
    )


def _load_forecast_raw(file_path: Path | str | None = None) -> pl.DataFrame:
    """단기예보 원시 Parquet 파일들을 탐색하여 단일 DataFrame으로 로드합니다."""
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            return pl.read_parquet(p)
        if p.is_dir():
            files = sorted(p.glob("**/*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"지정된 디렉터리에 단기예보 Parquet 파일이 없습니다: {p}"
                )
            return pl.read_parquet(files)
        raise FileNotFoundError(f"지정된 기상 데이터 경로가 존재하지 않습니다: {p}")

    # file_path가 None인 경우: raw 05:00 및 17:00 디렉터리 탐색
    raw_0500_dir = RAW_DATA_DIR / "weather" / "forecast" / "daily"
    raw_1700_dir = RAW_DATA_DIR / "weather" / "forecast_1700" / "daily"
    files: list[Path] = []
    if raw_0500_dir.exists():
        files.extend(sorted(raw_0500_dir.glob("**/*.parquet")))
    if raw_1700_dir.exists():
        files.extend(sorted(raw_1700_dir.glob("**/*.parquet")))

    if not files:
        raise FileNotFoundError(
            f"단기예보 원시 Parquet 파일이 존재하지 않습니다: {raw_0500_dir}, {raw_1700_dir}"
        )

    logger.info(f"단기예보 원시 Parquet {len(files)}개 파일 로드 시작")
    return pl.read_parquet(files)


def _process_forecast_weather(file_path: Path | str | None = None) -> pl.DataFrame:
    """
    기상청 단기예보(getVilageFcst) 데이터를 정제하여 13개 표준 컬럼의 DataFrame을 생성합니다.

    핵심 단계:
    1. 원천 데이터 로드 및 컬럼명 표준화 (camelCase / snake_case 대응)
    2. base_datetime, datetime, lead_time_hours 계산 및 lead_time_hours >= 0 필터링
    3. category 피벗 (TMP, REH, POP, PCP, PTY, SKY, WSD, VEC)
    4. 12시간 갱신 Point-in-Time 결합: (nx, ny, datetime) 기준 최소 lead_time_hours 단일화
    5. 변수별 파싱, 타입 캐스팅 및 결측치 보정 (forward_fill 및 시간대별 격자 평균)
    """
    df_raw = _load_forecast_raw(file_path)
    if df_raw.is_empty():
        return pl.DataFrame(
            schema={
                "nx": pl.Int64,
                "ny": pl.Int64,
                "datetime": pl.Datetime,
                "lead_time_hours": pl.Int16,
                "temperature": pl.Float32,
                "humidity": pl.Float32,
                "precipitation": pl.Float32,
                "pop": pl.Float32,
                "rainfall_type": pl.Int8,
                "sky_condition": pl.Int8,
                "wind_speed": pl.Float32,
                "wind_dir_sin": pl.Float32,
                "wind_dir_cos": pl.Float32,
            }
        )

    # 1. 컬럼명 표준화
    rename_dict = {}
    col_mapping = {
        "baseDate": "base_date",
        "baseTime": "base_time",
        "fcstDate": "fcst_date",
        "fcstTime": "fcst_time",
        "fcstValue": "fcst_value",
    }
    for orig, target in col_mapping.items():
        if orig in df_raw.columns:
            rename_dict[orig] = target
    if rename_dict:
        df_raw = df_raw.rename(rename_dict)

    required_cols = {
        "base_date",
        "base_time",
        "fcst_date",
        "fcst_time",
        "category",
        "fcst_value",
        "nx",
        "ny",
    }
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"단기예보 필수 컬럼이 누락되었습니다: {missing}")

    # 2. 타임스탬프 및 lead_time_hours 산출
    df_ts = (
        df_raw.with_columns(
            pl.col("base_date").cast(pl.Utf8).str.pad_start(8, "0"),
            pl.col("base_time").cast(pl.Utf8).str.pad_start(4, "0"),
            pl.col("fcst_date").cast(pl.Utf8).str.pad_start(8, "0"),
            pl.col("fcst_time").cast(pl.Utf8).str.pad_start(4, "0"),
            pl.col("nx").cast(pl.Int64),
            pl.col("ny").cast(pl.Int64),
        )
        .with_columns(
            pl.concat_str([pl.col("base_date"), pl.col("base_time")])
            .str.to_datetime("%Y%m%d%H%M")
            .alias("base_datetime"),
            pl.concat_str([pl.col("fcst_date"), pl.col("fcst_time")])
            .str.to_datetime("%Y%m%d%H%M")
            .alias("datetime"),
        )
        .with_columns(
            ((pl.col("datetime") - pl.col("base_datetime")).dt.total_minutes() / 60)
            .cast(pl.Int16, strict=False)
            .alias("lead_time_hours")
        )
        .filter(pl.col("lead_time_hours") >= 0)
    )

    # 3. category 피벗 대상 카테고리 필터링
    target_categories = [
        "TMP",
        "REH",
        "POP",
        "PCP",
        "PTY",
        "SKY",
        "WSD",
        "VEC",
    ]
    df_filtered = df_ts.filter(pl.col("category").is_in(target_categories))

    # 동일 (nx, ny, base_datetime, datetime, lead_time_hours, category) 중복 방어
    df_dedup_cat = df_filtered.unique(
        subset=[
            "nx",
            "ny",
            "base_datetime",
            "datetime",
            "lead_time_hours",
            "category",
        ],
        keep="first",
    )

    df_pivoted = df_dedup_cat.pivot(
        on="category",
        index=["nx", "ny", "base_datetime", "datetime", "lead_time_hours"],
        values="fcst_value",
    )

    # 누락된 카테고리 컬럼 생성 방어
    for cat in target_categories:
        if cat not in df_pivoted.columns:
            df_pivoted = df_pivoted.with_columns(pl.lit(None, dtype=pl.Utf8).alias(cat))

    # 4. Point-in-Time 중복 제거: (nx, ny, datetime)별로 lead_time_hours가 가장 작은 단일 레코드 선택
    df_latest = df_pivoted.sort(["nx", "ny", "datetime", "lead_time_hours"]).unique(
        subset=["nx", "ny", "datetime"], keep="first"
    )

    # 5. 변수별 파싱 및 타입 캐스팅
    vec_rad_expr = pl.col("VEC").cast(pl.Float32, strict=False) * (math.pi / 180.0)

    df_parsed = df_latest.with_columns(
        pl.col("TMP").cast(pl.Float32, strict=False).alias("temperature"),
        pl.col("REH").cast(pl.Float32, strict=False).clip(0.0, 100.0).alias("humidity"),
        (pl.col("POP").cast(pl.Float32, strict=False) / 100.0)
        .clip(0.0, 1.0)
        .alias("pop"),
        _parse_precipitation_expr("PCP"),
        pl.col("PTY").cast(pl.Int8, strict=False).alias("rainfall_type"),
        pl.col("SKY").cast(pl.Int8, strict=False).alias("sky_condition"),
        pl.col("WSD").cast(pl.Float32, strict=False).alias("wind_speed"),
        vec_rad_expr.sin().alias("wind_dir_sin"),
        vec_rad_expr.cos().alias("wind_dir_cos"),
    )

    # 6. 결측치 보정 (격자별 forward_fill 후 전체 시간대별 평균 대치)
    df_sorted = df_parsed.sort(["nx", "ny", "datetime"])

    # 격자 내 forward_fill 및 backward_fill
    fill_cols = [
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
    df_ffill = df_sorted.with_columns(
        [pl.col(c).forward_fill().backward_fill().over(["nx", "ny"]) for c in fill_cols]
    )

    # 시간대별(hour) 평균 보정 (격자 전체에 값이 없는 극단적 결측 대응)
    df_with_hour = df_ffill.with_columns(pl.col("datetime").dt.hour().alias("hour"))

    hour_impute_exprs = [
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
    ]

    df_imputed = df_with_hour.with_columns(hour_impute_exprs)

    # 최종 컬럼 선별 및 스키마 정렬
    df_final = df_imputed.select(EXPECTED_WEATHER_COLUMNS).sort(
        ["nx", "ny", "datetime"]
    )
    return df_final


def process_weather_data(
    file_path: Path | str | None = None,
    track: str = "forecast",
    save_output: bool = False,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    기상 데이터 전처리 단일 진입점

    인자:
        file_path: 원시 Parquet 경로 또는 디렉터리
        track: 기상 트랙 ("forecast" 단기예보 지원)
        save_output: True일 경우 data/processed/weather_processed.parquet에 저장
        output_path: 저장 파일 경로 (기본값: PROCESSED_DATA_DIR / 'weather_processed.parquet')

    반환값:
        정제된 기상 Polars DataFrame (13개 표준 컬럼)
    """
    if track != "forecast":
        raise ValueError(
            f"지원하지 않는 기상 트랙입니다: '{track}'. ('forecast'만 지원)"
        )

    df = _process_forecast_weather(file_path)

    if save_output and not df.is_empty():
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "weather_processed.parquet"
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_p)
        logger.info(f"기상 전처리 데이터 저장 완료: {out_p} ({len(df)}행)")

    return df
