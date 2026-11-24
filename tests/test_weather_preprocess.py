"""
기상 데이터 전처리 모듈 단위 테스트
"""

from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import SAMPLES_DIR
from busan_metro_congestion.preprocess import weather as weather_module
from busan_metro_congestion.preprocess.weather import (
    EXPECTED_WEATHER_COLUMNS,
    _parse_precipitation_expr,
    process_weather_data,
)


def test_process_forecast_weather_from_sample():
    """샘플 데이터를 통해 단기예보 전처리 및 13개 컬럼 스키마를 검증합니다."""
    sample_path = SAMPLES_DIR / "weather" / "weather_forecast_sample.parquet"
    assert sample_path.exists(), f"샘플 파일이 없습니다: {sample_path}"

    df = process_weather_data(file_path=sample_path, track="forecast")

    # 1. 스키마 및 컬럼 목록 검증
    assert df.columns == EXPECTED_WEATHER_COLUMNS
    assert len(df) == 728

    # 2. (nx, ny, datetime) 고유성 검증 (Point-in-Time 단일화 확인)
    n_unique_keys = df.select(["nx", "ny", "datetime"]).n_unique()
    assert n_unique_keys == 728
    assert len(df) == n_unique_keys

    # 3. lead_time_hours >= 0 검증
    assert df["lead_time_hours"].min() >= 0

    # 4. 결측치 전무 검증
    for col in EXPECTED_WEATHER_COLUMNS:
        assert df[col].is_null().sum() == 0, f"컬럼 {col}에 결측치가 존재합니다."

    # 5. 수치 범위 및 도메인 물리량 불변식 검증
    assert (df["pop"] >= 0.0).all() and (df["pop"] <= 1.0).all()
    assert (df["humidity"] >= 0.0).all() and (df["humidity"] <= 100.0).all()
    assert (df["precipitation"] >= 0.0).all()
    assert (df["temperature"] >= -30.0).all() and (df["temperature"] <= 45.0).all()
    assert (df["wind_speed"] >= 0.0).all()
    assert (df["wind_dir_sin"] >= -1.0).all() and (df["wind_dir_sin"] <= 1.0).all()
    assert (df["wind_dir_cos"] >= -1.0).all() and (df["wind_dir_cos"] <= 1.0).all()
    assert (
        (df["wind_dir_sin"] ** 2 + df["wind_dir_cos"] ** 2 - 1.0).abs() < 1e-4
    ).all()
    assert set(df["rainfall_type"].unique().to_list()).issubset({0, 1, 2, 3, 4})
    assert set(df["sky_condition"].unique().to_list()).issubset({1, 3, 4})


def test_parse_precipitation_edge_cases():
    """강수량(PCP) 특수 문자열 파싱 규칙을 검증합니다."""
    mock_df = pl.DataFrame(
        {
            "PCP": [
                "강수없음",
                "0",
                "0.0mm",
                "1mm 미만",
                "30.0~50.0mm",
                "50.0mm 이상",
                "12.5mm",
                "3.0mm",
                None,
            ]
        }
    )
    res = mock_df.select(_parse_precipitation_expr("PCP"))["precipitation"]

    expected = [0.0, 0.0, 0.0, 0.5, 40.0, 50.0, 12.5, 3.0, 0.0]
    for r, e in zip(res.to_list(), expected):
        assert pytest.approx(r, abs=1e-4) == e


def test_point_in_time_dedup_prefers_minimum_lead_time():
    """동일 (nx, ny, datetime)에 여러 발표 시각이 존재할 때 최신(lead_time_hours 최소) 데이터가 선택되는지 검증합니다."""
    # 동일한 2023-06-01 18:00 예보에 대해:
    # 05:00 발표 (lead_time_hours = 13, TMP = 20)
    # 17:00 발표 (lead_time_hours = 1, TMP = 25)
    mock_data = pl.DataFrame(
        {
            "base_date": ["20230601", "20230601"],
            "base_time": ["0500", "1700"],
            "category": ["TMP", "TMP"],
            "fcst_date": ["20230601", "20230601"],
            "fcst_time": ["1800", "1800"],
            "fcst_value": ["20", "25"],
            "nx": [96, 96],
            "ny": [73, 73],
        }
    )

    # 임시 디렉터리 없이 인메모리 DataFrame 테스트를 위해 파일 저장 대신 단위 파싱 흐름 확인
    # weather.py는 file_path를 인자로 받으므로 가상 파일로 테스트
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        mock_data.write_parquet(tmp_path)
        df_out = process_weather_data(file_path=tmp_path, track="forecast")
        assert len(df_out) == 1
        assert df_out["lead_time_hours"][0] == 1
        assert df_out["temperature"][0] == pytest.approx(25.0, abs=1e-4)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@pytest.mark.parametrize("invalid_track", ["satellite", "observation"])
def test_invalid_track_raises_value_error(invalid_track: str):
    """지원하지 않는 트랙명을 전달했을 때 ValueError가 발생하는지 검증합니다."""
    with pytest.raises(ValueError, match="지원하지 않는 기상 트랙입니다"):
        process_weather_data(track=invalid_track)


def test_process_weather_data_missing_raw_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """원시 단기예보 파일이 없을 때 샘플로 암묵 폴백하지 않고 FileNotFoundError를 발생시키는지 검증합니다."""
    monkeypatch.setattr(weather_module, "RAW_DATA_DIR", tmp_path / "empty_raw")

    with pytest.raises(
        FileNotFoundError, match="단기예보 원시 Parquet 파일이 존재하지 않습니다"
    ):
        process_weather_data(track="forecast")

    with pytest.raises(
        FileNotFoundError, match="지정된 기상 데이터 경로가 존재하지 않습니다"
    ):
        process_weather_data(file_path=tmp_path / "missing.parquet", track="forecast")
