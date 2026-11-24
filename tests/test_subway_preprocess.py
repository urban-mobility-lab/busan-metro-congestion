"""
지하철 승하차 데이터 정제 및 StationTargetEncoder 단위 테스트

주의:
- AGENTS.md 규칙에 따라 data/samples/의 샘플 파일만 사용하여 CI 환경에서 실행 가능해야 합니다.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import REFERENCE_DATA_DIR, SAMPLES_DIR
from busan_metro_congestion.preprocess import subway as subway_module
from busan_metro_congestion.preprocess.subway import (
    StationTargetEncoder,
    SubwayProfileImputer,
    add_cyclical_time_features,
    build_subway_interim,
    build_subway_processed,
    clean_subway_data,
    detect_csv_encoding,
    extract_hour_from_slot,
    process_subway_data,
)

SUBWAY_SAMPLES_DIR = SAMPLES_DIR / "subway"


def test_detect_csv_encoding() -> None:
    path_2023 = SUBWAY_SAMPLES_DIR / "subway_2023_sample.csv"
    path_2025 = SUBWAY_SAMPLES_DIR / "subway_2025_sample.csv"

    assert detect_csv_encoding(path_2023) == "cp949"
    assert detect_csv_encoding(path_2025) == "utf-8-sig"


def test_extract_hour_from_slot() -> None:
    assert extract_hour_from_slot("01시-02시") == 1
    assert extract_hour_from_slot("01시~02시") == 1
    assert extract_hour_from_slot("09시-10시") == 9
    assert extract_hour_from_slot("23시-24시") == 23
    assert extract_hour_from_slot("24시-01시") == 24


def test_clean_subway_data_deduplication() -> None:
    dup_file = SUBWAY_SAMPLES_DIR / "subway_duplicate_sample.csv"
    assert dup_file.exists(), "subway_duplicate_sample.csv 파일이 누락되었습니다."

    # 원본 파일은 6행 (서면 하차 2개, 사상 승차 2개 포함)
    raw_df = pl.read_csv(dup_file, encoding="cp949")
    assert len(raw_df) == 6

    cleaned = clean_subway_data(raw_df)
    # 중복 정제 후 4개 고유 (역번호, 년월일, 구분)만 남아야 함
    assert len(cleaned) == 4

    # 서면(119) 하차의 경우 합계 39,157 행이 남고 합계 26 행은 탈락해야 함
    seomyeon_out = cleaned.filter(
        (pl.col("역번호") == 119) & (pl.col("구분") == "하차")
    )
    assert len(seomyeon_out) == 1
    assert seomyeon_out["합계"][0] == 39157

    # 사상(227) 승차의 경우 합계 21,080 행이 남아야 함
    sasang_in = cleaned.filter((pl.col("역번호") == 227) & (pl.col("구분") == "승차"))
    assert len(sasang_in) == 1
    assert sasang_in["합계"][0] == 21080


def test_clean_subway_data_null_filtering() -> None:
    dummy_df = pl.DataFrame(
        {
            "역번호": [119, None, 227],
            "년월일": ["2023-06-01", "2023-06-01", "2023-06-01"],
            "구분": ["승차", "승차", "하차"],
            "합계": [1000, 0, 500],
        }
    )
    cleaned = clean_subway_data(dummy_df)
    assert len(cleaned) == 2
    assert cleaned["역번호"].to_list() == [119, 227]


def test_process_subway_data_2023_sample() -> None:
    sample_file = SUBWAY_SAMPLES_DIR / "subway_2023_sample.csv"
    station_master = REFERENCE_DATA_DIR / "station_master.parquet"

    df = process_subway_data(
        file_path=sample_file,
        station_master_path=station_master,
    )

    # 5개 역 * 2일 * 24시간 = 240행
    assert len(df) == 240

    expected_cols = [
        "datetime",
        "date",
        "hour",
        "line_no",
        "station_id",
        "station_name",
        "passengers_in",
        "passengers_out",
        "accumulated_passengers",
    ]
    assert df.columns == expected_cols

    # 타입 검증
    assert df.schema["datetime"] == pl.Datetime
    assert df.schema["date"] == pl.Date
    assert df.schema["hour"] == pl.Int8
    assert df.schema["line_no"] == pl.Int8
    assert df.schema["station_id"] == pl.Int64
    assert df.schema["station_name"] == pl.String
    assert df.schema["passengers_in"] == pl.Int32
    assert df.schema["passengers_out"] == pl.Int32
    assert df.schema["accumulated_passengers"] == pl.Int32

    # 통행량 일관성 검증 (in + out == accumulated)
    traffic_diff = df.filter(
        pl.col("accumulated_passengers")
        != (pl.col("passengers_in") + pl.col("passengers_out"))
    )
    assert len(traffic_diff) == 0

    # 자정(hour=24) 타임스탬프 검증: 익일 00:00:00
    midnight_rows = df.filter(pl.col("hour") == 24)
    assert len(midnight_rows) == 10  # 5개 역 * 2일
    for r in midnight_rows.iter_rows(named=True):
        # datetime의 날짜는 date의 다음날이고 시각은 00시여야 함
        assert r["datetime"].hour == 0
        assert r["datetime"].day == r["date"].day + 1


def test_process_subway_data_2025_utf8_bom() -> None:
    sample_file = SUBWAY_SAMPLES_DIR / "subway_2025_sample.csv"
    station_master = REFERENCE_DATA_DIR / "station_master.parquet"

    df = process_subway_data(
        file_path=sample_file,
        station_master_path=station_master,
    )

    # 5개 역 * 1일 * 24시간 = 120행
    assert len(df) == 120
    assert df["station_id"].n_unique() == 5
    assert df["passengers_in"].is_null().sum() == 0
    assert df["passengers_out"].is_null().sum() == 0


def test_process_subway_data_date_filtering() -> None:
    sample_file = SUBWAY_SAMPLES_DIR / "subway_2023_sample.csv"
    station_master = REFERENCE_DATA_DIR / "station_master.parquet"

    # 2023-06-02 하루만 필터링
    df_filtered = process_subway_data(
        file_path=sample_file,
        start_date="2023-06-02",
        end_date="2023-06-02",
        station_master_path=station_master,
    )

    assert len(df_filtered) == 120  # 5개 역 * 1일 * 24시간
    assert df_filtered["date"].n_unique() == 1
    assert str(df_filtered["date"][0]) == "2023-06-02"


def test_station_target_encoder_bayesian_m_estimate(tmp_path: Path) -> None:
    sample_file = SUBWAY_SAMPLES_DIR / "subway_2023_sample.csv"
    df = process_subway_data(file_path=sample_file)

    encoder = StationTargetEncoder(m=10.0, n_splits=5, seed=42)
    df_encoded = encoder.fit_transform(
        df, station_col="station_id", target_col="accumulated_passengers"
    )

    assert "station_target_encoded" in df_encoded.columns
    assert df_encoded["station_target_encoded"].is_null().sum() == 0
    assert len(df_encoded) == len(df)

    # 서빙용 맵에 5개 역이 모두 존재하는지 확인
    assert len(encoder.station_map) == 5
    assert encoder.global_mean > 0

    # 수동 공식 검증: S_i = (n_i * y_bar_i + m * y_global) / (n_i + m)
    stn_95_df = df.filter(pl.col("station_id") == 95)
    n_95 = len(stn_95_df)
    y_bar_95 = stn_95_df["accumulated_passengers"].mean()
    global_mean = df["accumulated_passengers"].mean()
    expected_s_95 = (n_95 * y_bar_95 + 10.0 * global_mean) / (n_95 + 10.0)

    actual_s_95 = encoder.station_map[95]
    assert pytest.approx(actual_s_95, rel=1e-4) == expected_s_95

    # 매핑 파일 영속화 및 복원 검증
    map_path = tmp_path / "station_target_map_test.parquet"
    encoder.save_mapping(map_path)
    assert map_path.exists()

    loaded_encoder = StationTargetEncoder.load_mapping(map_path, m=10.0)
    assert pytest.approx(loaded_encoder.global_mean, rel=1e-4) == encoder.global_mean
    assert len(loaded_encoder.station_map) == len(encoder.station_map)
    assert pytest.approx(loaded_encoder.station_map[95], rel=1e-4) == actual_s_95

    # transform 일치성 및 미관측 역사 글로벌 평균 대치 검증
    test_df = pl.DataFrame(
        {
            "station_id": [95, 999],  # 999는 미관측 역사
            "accumulated_passengers": [100, 200],
        }
    )
    transformed = loaded_encoder.transform(test_df, station_col="station_id")
    assert (
        pytest.approx(transformed["station_target_encoded"][0], rel=1e-4) == actual_s_95
    )
    # 999번 역은 글로벌 평균으로 대치되어야 함
    assert (
        pytest.approx(transformed["station_target_encoded"][1], rel=1e-4)
        == encoder.global_mean
    )


def test_station_target_encoder_date_block_split() -> None:
    """날짜 블록 기반 OOF 분할 시 동일 날짜 단일 Fold 할당 및 자기 날짜 누출 차단을 검증합니다."""
    from datetime import date, datetime

    # 10일간 역 101·102에 시간대 2개씩 배치하되, 10일차에 극단적 이상치(100,000)를 주입
    dates = [date(2023, 6, 1 + i) for i in range(10)]
    records = []
    for day_idx, d in enumerate(dates):
        for hour in (8, 18):
            val_101 = 100_000.0 if day_idx == 9 else float((day_idx + 1) * 10)
            val_102 = 200.0 + float(day_idx)
            records.append(
                {
                    "date": d,
                    "datetime": datetime(d.year, d.month, d.day, hour),
                    "station_id": 101,
                    "accumulated_passengers": val_101,
                }
            )
            records.append(
                {
                    "date": d,
                    "datetime": datetime(d.year, d.month, d.day, hour),
                    "station_id": 102,
                    "accumulated_passengers": val_102,
                }
            )
    df = pl.DataFrame(records)

    encoder = StationTargetEncoder(m=5.0, n_splits=5, seed=42, split_strategy="date")
    encoded = encoder.fit_transform(df)

    assert "station_target_encoded" in encoded.columns
    assert len(encoded) == len(df)
    assert encoded["station_target_encoded"].is_null().sum() == 0

    # 1. 동일 (station_id, date) 내 모든 시간대는 반드시 동일 Fold로 할당되어 인코딩 값이 정확히 일치해야 함
    per_day_unique = encoded.group_by(["station_id", "date"]).agg(
        pl.col("station_target_encoded").n_unique().alias("n_uniq")
    )
    assert (per_day_unique["n_uniq"] == 1).all()

    # 2. 이상치(100,000)가 있는 10일차(2023-06-10)의 OOF 인코딩 값은 자기 날짜의 이상치를 배제해야 함
    outlier_day_val = encoded.filter(
        (pl.col("station_id") == 101) & (pl.col("date") == date(2023, 6, 10))
    )["station_target_encoded"][0]
    assert outlier_day_val < 500.0
    # 반면 서빙용 전체 평균 맵(station_map)에는 이상치가 반영되어 훨씬 큰 값을 가져야 함
    assert encoder.station_map[101] > 5000.0


def test_subway_profile_imputer_hierarchical() -> None:
    """SubwayProfileImputer의 L1(역·요일·시) -> L2(역·시) -> L3(전체·시) 3단계 계층 보정 및 통행량 재계산을 검증합니다."""
    from datetime import date

    # 2023-06-05(월, weekday=1), 2023-06-06(화, weekday=2), 2023-06-07(수, weekday=3)
    train_df = pl.DataFrame(
        {
            "date": [
                date(2023, 6, 5),
                date(2023, 6, 6),
                date(2023, 6, 7),
            ],
            "hour": pl.Series([8, 8, 8], dtype=pl.Int8),
            "station_id": pl.Series([119, 119, 113], dtype=pl.Int64),
            "passengers_in": pl.Series([100, 300, 500], dtype=pl.Int32),
            "passengers_out": pl.Series([200, 400, 600], dtype=pl.Int32),
            "accumulated_passengers": pl.Series([300, 700, 1100], dtype=pl.Int32),
        }
    )

    imputer = SubwayProfileImputer(strategy="median").fit(train_df)
    assert imputer.profile_l1 is not None
    assert imputer.profile_l2 is not None
    assert imputer.profile_l3 is not None

    # 3가지 결측 케이스:
    # - row 0 (L1): 역 119, 월요일, 8시 -> L1 중앙값 100
    # - row 1 (L2): 역 119, 수요일(미관측 요일), 8시 -> L2(역 119, 8시) 중앙값 median(100, 300)=200
    # - row 2 (L3): 역 999(미관측 역사), 수요일, 8시 -> L3(전체 8시) 중앙값 median(100, 300, 500)=300
    test_missing = pl.DataFrame(
        {
            "date": [date(2023, 6, 5), date(2023, 6, 7), date(2023, 6, 7)],
            "hour": pl.Series([8, 8, 8], dtype=pl.Int8),
            "station_id": pl.Series([119, 119, 999], dtype=pl.Int64),
            "passengers_in": pl.Series([None, None, None], dtype=pl.Int32),
            "passengers_out": pl.Series([50, 50, 50], dtype=pl.Int32),
            "accumulated_passengers": pl.Series([None, None, None], dtype=pl.Int32),
        }
    )

    imputed = imputer.transform(test_missing)
    assert len(imputed) == 3
    assert imputed["passengers_in"].is_null().sum() == 0
    assert imputed["accumulated_passengers"].is_null().sum() == 0

    assert imputed["passengers_in"].to_list() == [100, 200, 300]
    assert imputed["accumulated_passengers"].to_list() == [150, 250, 350]


def test_add_cyclical_time_features_continuity() -> None:
    # 23시 -> 24시(0시) -> 1시 불연속성 해소 및 주기적 연속성 검증
    dummy_df = pl.DataFrame(
        {
            "hour": [23, 24, 1],
            "date": ["2023-06-01", "2023-06-01", "2023-06-01"],
        }
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))

    encoded = add_cyclical_time_features(dummy_df, hour_col="hour", date_col="date")

    assert "hour_sin" in encoded.columns
    assert "hour_cos" in encoded.columns
    assert "weekday_sin" in encoded.columns
    assert "weekday_cos" in encoded.columns

    row_23 = encoded.filter(pl.col("hour") == 23).to_dicts()[0]
    row_24 = encoded.filter(pl.col("hour") == 24).to_dicts()[0]
    row_01 = encoded.filter(pl.col("hour") == 1).to_dicts()[0]

    # 24시는 0시(자정)와 동일하므로 sin=0, cos=1
    assert pytest.approx(row_24["hour_sin"], abs=1e-5) == 0.0
    assert pytest.approx(row_24["hour_cos"], abs=1e-5) == 1.0

    # 23시와 1시는 cos가 동일하고 sin이 대칭
    assert pytest.approx(row_23["hour_cos"], abs=1e-4) == row_01["hour_cos"]
    assert pytest.approx(row_23["hour_sin"], abs=1e-4) == -row_01["hour_sin"]

    # 23시와 24시 사이의 거리 == 24시와 1시 사이의 거리 (원주 상에서 완벽한 연속)
    dist_23_24 = (row_23["hour_sin"] - row_24["hour_sin"]) ** 2 + (
        row_23["hour_cos"] - row_24["hour_cos"]
    ) ** 2
    dist_24_01 = (row_24["hour_sin"] - row_01["hour_sin"]) ** 2 + (
        row_24["hour_cos"] - row_01["hour_cos"]
    ) ** 2
    assert pytest.approx(dist_23_24, rel=1e-4) == dist_24_01


def test_build_subway_2stage_pipeline(tmp_path: Path) -> None:
    # CI 샘플 디렉터리를 대상으로 interim -> processed 2단계 파이프라인 구동 검증
    interim_file = tmp_path / "subway_interim_test.parquet"
    processed_file = tmp_path / "subway_processed_test.parquet"
    station_master = REFERENCE_DATA_DIR / "station_master.parquet"

    # 1단계: build_subway_interim
    df_interim = build_subway_interim(
        raw_dir=SUBWAY_SAMPLES_DIR,
        output_path=interim_file,
    )
    assert interim_file.exists()
    assert len(df_interim) == 408
    assert df_interim.select(["datetime", "station_id"]).n_unique() == 408
    assert "datetime" in df_interim.columns
    assert "station_id" in df_interim.columns
    assert "passengers_in" in df_interim.columns
    assert "passengers_out" in df_interim.columns

    # 2단계: build_subway_processed
    df_processed = build_subway_processed(
        interim_path=interim_file,
        output_path=processed_file,
        start_date="2023-06-01",
        end_date="2025-01-01",
        station_master_path=station_master,
    )
    assert processed_file.exists()
    assert len(df_processed) == 408
    assert df_processed.select(["datetime", "station_id"]).n_unique() == 408
    assert "line_no" in df_processed.columns
    assert "station_name" in df_processed.columns
    assert "accumulated_passengers" in df_processed.columns
    assert df_processed["accumulated_passengers"].is_null().sum() == 0
    assert (df_processed["accumulated_passengers"] >= 0).all()


def test_process_subway_data_missing_raw_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """원시 지하철 CSV가 없을 때 샘플로 암묵 폴백하지 않고 FileNotFoundError를 발생시키는지 검증합니다."""
    empty_raw_root = tmp_path / "empty_raw"
    (empty_raw_root / "subway").mkdir(parents=True)
    monkeypatch.setattr(subway_module, "RAW_DATA_DIR", empty_raw_root)

    with pytest.raises(
        FileNotFoundError, match="지하철 원시 CSV 파일이 존재하지 않습니다"
    ):
        process_subway_data()


def test_build_subway_interim_missing_raw_raises(tmp_path: Path) -> None:
    """빈 원시 디렉터리로 build_subway_interim 호출 시 FileNotFoundError를 발생시키는지 검증합니다."""
    empty_dir = tmp_path / "empty_subway"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="경로에 승하차 CSV 파일이 없습니다"):
        build_subway_interim(
            raw_dir=empty_dir,
            output_path=tmp_path / "subway_interim.parquet",
        )
