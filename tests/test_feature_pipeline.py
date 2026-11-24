"""
통합 피처 파이프라인 단위 테스트
"""

from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import SAMPLES_DIR
from busan_metro_congestion.preprocess import features as features_module
from busan_metro_congestion.preprocess.features import (
    MASTER_FEATURE_COLUMNS,
    run_feature_pipeline,
)
from busan_metro_congestion.preprocess.festival import (
    build_festival_interim,
    build_festival_processed,
)
from busan_metro_congestion.preprocess.subway import process_subway_data
from busan_metro_congestion.preprocess.weather import process_weather_data


@pytest.fixture(scope="module")
def sample_pipeline_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """data/samples/ 기반의 격리된 전처리 입력 경로 픽스처."""
    base_dir = tmp_path_factory.mktemp("feature_pipeline_samples")

    weather_df = process_weather_data(
        file_path=SAMPLES_DIR / "weather" / "weather_forecast_sample.parquet",
        track="forecast",
    )
    weather_path = base_dir / "weather_processed_sample.parquet"
    weather_df.write_parquet(weather_path)

    fest_interim = build_festival_interim(
        raw_festival_dir=SAMPLES_DIR / "festival",
        output_path=base_dir / "festival_interim_sample.parquet",
    )
    festival_path = base_dir / "festival_processed_sample.parquet"
    build_festival_processed(interim_df=fest_interim, output_path=festival_path)

    return {
        "subway_path": SAMPLES_DIR / "subway" / "subway_2023_sample.csv",
        "weather_path": weather_path,
        "festival_path": festival_path,
    }


def test_run_feature_pipeline_from_samples(
    tmp_path: Path, sample_pipeline_inputs: dict[str, Path]
):
    """샘플 입력 기반 마스터 피처 테이블 생성 및 스키마/결측치 전무를 검증합니다."""
    out_file = tmp_path / "metro_features_sample.parquet"

    df = run_feature_pipeline(
        subway_path=sample_pipeline_inputs["subway_path"],
        weather_path=sample_pipeline_inputs["weather_path"],
        festival_path=sample_pipeline_inputs["festival_path"],
        output_path=out_file,
    )

    # 1. 파일 생성 및 행 수 검증 (5개 역 * 2일 * 24시간 = 240행 완전 보존 검증)
    assert out_file.exists()
    assert len(df) == 240
    assert df.select(["datetime", "station_id"]).n_unique() == 240
    # 물리적 정렬 순서 검증: (station_id, datetime) 기준 정렬 보장 (시계열 연속성 및 LSTM 윈도우 보존 계약)
    assert df.equals(df.sort(["station_id", "datetime"]))

    # 2. 스키마 컬럼 검증
    assert df.columns == MASTER_FEATURE_COLUMNS

    # 3. 결측치 전무 검증
    for col in MASTER_FEATURE_COLUMNS:
        null_count = df[col].is_null().sum()
        assert null_count == 0, f"컬럼 '{col}'에 결측치가 {null_count}건 존재합니다."

    # 4. 주기적 인코딩 수치 범위 및 원주 불변식 검증 (결측 0.0 Fallback 누출 차단)
    assert (df["hour_sin"] >= -1.0).all() and (df["hour_sin"] <= 1.0).all()
    assert (df["hour_cos"] >= -1.0).all() and (df["hour_cos"] <= 1.0).all()
    assert (df["month_sin"] >= -1.0).all() and (df["month_sin"] <= 1.0).all()
    assert (df["month_cos"] >= -1.0).all() and (df["month_cos"] <= 1.0).all()
    assert (df["weekday_sin"] >= -1.0).all() and (df["weekday_sin"] <= 1.0).all()
    assert (df["weekday_cos"] >= -1.0).all() and (df["weekday_cos"] <= 1.0).all()
    assert ((df["hour_sin"] ** 2 + df["hour_cos"] ** 2 - 1.0).abs() < 1e-4).all()
    assert ((df["month_sin"] ** 2 + df["month_cos"] ** 2 - 1.0).abs() < 1e-4).all()
    assert ((df["weekday_sin"] ** 2 + df["weekday_cos"] ** 2 - 1.0).abs() < 1e-4).all()

    # 5. 달력 매핑, 타깃 인코딩 및 기상 피처 검증
    # 2023년 6월 샘플이 결측 Fallback(1)이 아닌 정상 여름(2)으로 매핑되었는지 검증
    assert (df["season"] == 2).all()
    assert (df["station_target_encoded"] > 0).all()
    assert (df["lead_time_hours"] >= 0).all()
    assert set(df["nearest_event_rank"].unique().to_list()).issubset({0, 1, 2, 3})


def test_run_feature_pipeline_force_reload(
    tmp_path: Path, sample_pipeline_inputs: dict[str, Path]
):
    """force_reload=False 시 기존 캐시 파일을 재사용하고, True 시 재생성함을 검증합니다."""
    out_file = tmp_path / "cache_test.parquet"

    # 1. 첫 실행: 240행 마스터 피처 생성
    df1 = run_feature_pipeline(
        subway_path=sample_pipeline_inputs["subway_path"],
        weather_path=sample_pipeline_inputs["weather_path"],
        festival_path=sample_pipeline_inputs["festival_path"],
        output_path=out_file,
    )
    assert out_file.exists()
    assert len(df1) == 240

    # 2. 캐시 파일을 1행 센티넬로 교체한 뒤 force_reload=False 호출 -> 캐시 그대로 반환
    sentinel_df = df1.head(1)
    sentinel_df.write_parquet(out_file)
    mtime_before = out_file.stat().st_mtime_ns

    df_cached = run_feature_pipeline(
        subway_path=sample_pipeline_inputs["subway_path"],
        weather_path=sample_pipeline_inputs["weather_path"],
        festival_path=sample_pipeline_inputs["festival_path"],
        output_path=out_file,
        force_reload=False,
    )
    assert len(df_cached) == 1
    assert out_file.stat().st_mtime_ns == mtime_before

    # 3. force_reload=True 호출 -> 캐시를 무시하고 원본 입력으로 240행 재생성
    df_reloaded = run_feature_pipeline(
        subway_path=sample_pipeline_inputs["subway_path"],
        weather_path=sample_pipeline_inputs["weather_path"],
        festival_path=sample_pipeline_inputs["festival_path"],
        output_path=out_file,
        force_reload=True,
    )
    assert len(df_reloaded) == 240
    assert len(pl.read_parquet(out_file)) == 240


def test_run_feature_pipeline_deduplication_safeguard(
    tmp_path: Path, sample_pipeline_inputs: dict[str, Path]
):
    """실제 조인 키가 일치하는 중복 기상·축제 데이터가 유입되어도 행 증식 없이 1:1 결합되는지 검증합니다."""
    from datetime import date, datetime

    # subway_2023_sample.csv의 서면역(station_id=119 -> nx=98, ny=75)과 실제로 일치하는 시점(2023-06-01 08:00)
    dup_weather = pl.DataFrame(
        {
            "nx": [98, 98],
            "ny": [75, 75],
            "datetime": [
                datetime(2023, 6, 1, 8, 0),
                datetime(2023, 6, 1, 8, 0),
            ],
            "lead_time_hours": [3, 3],
            "temperature": [23.5, 29.5],
            "humidity": [55.0, 80.0],
            "precipitation": [1.5, 9.0],
            "pop": [0.4, 0.9],
            "rainfall_type": [1, 1],
            "sky_condition": [3, 4],
            "wind_speed": [2.0, 5.0],
            "wind_dir_sin": [0.0, 0.0],
            "wind_dir_cos": [1.0, 1.0],
        }
    )
    dup_weather_path = tmp_path / "dup_weather.parquet"
    dup_weather.write_parquet(dup_weather_path)

    # 서면역(119), 2023-06-01에 실제로 매칭되는 중복 축제 레코드
    dup_festival = pl.DataFrame(
        {
            "station_id": [119, 119],
            "date": [date(2023, 6, 1), date(2023, 6, 1)],
            "nearest_event_rank": [3, 1],
            "max_dist_weight": [0.8, 0.4],
            "max_scale_weight": [4.5, 2.0],
            "event_impact_score": [3.6, 0.8],
        }
    )
    dup_festival_path = tmp_path / "dup_festival.parquet"
    dup_festival.write_parquet(dup_festival_path)

    out_file = tmp_path / "dedup_result.parquet"

    df = run_feature_pipeline(
        subway_path=sample_pipeline_inputs["subway_path"],
        weather_path=dup_weather_path,
        festival_path=dup_festival_path,
        output_path=out_file,
    )

    # 중복 키가 실제로 매칭되었음에도 원본 240행과 키 유일성이 엄격히 보존되어야 함
    assert len(df) == 240
    assert df.select(["datetime", "station_id"]).n_unique() == 240
    assert df["accumulated_passengers"].is_null().sum() == 0

    # 실제 조인 적중 검증 (기본 fallback 15.0 / 0 이 아닌 주입된 기상·축제 피처 반영 확인)
    matched_row = df.filter(
        (pl.col("station_id") == 119)
        & (pl.col("datetime") == datetime(2023, 6, 1, 8, 0))
    ).to_dicts()[0]
    assert matched_row["temperature"] in (pytest.approx(23.5), pytest.approx(29.5))
    assert matched_row["precipitation"] in (pytest.approx(1.5), pytest.approx(9.0))
    assert matched_row["nearest_event_rank"] in (1, 3)
    assert matched_row["event_impact_score"] > 0.0


def test_run_feature_pipeline_missing_calendar_raises(
    tmp_path: Path, sample_pipeline_inputs: dict[str, Path]
):
    """지정된 달력 참조 파일이 존재하지 않을 때 FileNotFoundError가 발생하는지 검증합니다."""
    with pytest.raises(FileNotFoundError):
        run_feature_pipeline(
            subway_path=sample_pipeline_inputs["subway_path"],
            weather_path=sample_pipeline_inputs["weather_path"],
            festival_path=sample_pipeline_inputs["festival_path"],
            calendar_path=tmp_path / "non_existent_calendar.parquet",
            output_path=tmp_path / "out.parquet",
        )


def test_run_feature_pipeline_auto_build_invokes_processed_builders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_pipeline_inputs: dict[str, Path],
):
    """data/processed/가 비어 있을 때 subway/weather/festival 자동 빌드 경로가 일관되게 호출되는지 검증합니다."""
    empty_processed_dir = tmp_path / "processed"
    empty_processed_dir.mkdir()
    monkeypatch.setattr(features_module, "PROCESSED_DATA_DIR", empty_processed_dir)

    called: dict[str, object] = {}

    def fake_build_subway_processed(
        station_master_path: Path | str | None = None,
    ) -> pl.DataFrame:
        called["subway"] = station_master_path
        return process_subway_data(
            file_path=sample_pipeline_inputs["subway_path"],
            station_master_path=station_master_path,
        )

    def fake_process_weather_data(
        file_path: Path | str | None = None,
        track: str = "forecast",
        save_output: bool = False,
        output_path: Path | str | None = None,
    ) -> pl.DataFrame:
        called["weather"] = {"track": track, "save_output": save_output}
        return pl.read_parquet(sample_pipeline_inputs["weather_path"])

    def fake_build_festival_processed(
        station_master_path: Path | str | None = None,
    ) -> pl.DataFrame:
        called["festival"] = True
        return pl.read_parquet(sample_pipeline_inputs["festival_path"])

    monkeypatch.setattr(
        features_module, "build_subway_processed", fake_build_subway_processed
    )
    monkeypatch.setattr(
        features_module, "process_weather_data", fake_process_weather_data
    )
    monkeypatch.setattr(
        features_module, "build_festival_processed", fake_build_festival_processed
    )

    out_file = tmp_path / "auto_built_features.parquet"
    df = run_feature_pipeline(output_path=out_file, force_reload=True)

    assert out_file.exists()
    assert len(df) == 240
    assert "subway" in called
    assert called["weather"] == {"track": "forecast", "save_output": True}
    assert called["festival"] is True
