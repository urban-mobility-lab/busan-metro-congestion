from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import PROJECT_ROOT
from scripts.build_station_master import (
    build_station_master,
    extract_gugun,
    extract_line_number,
    normalize_station_name,
)


def test_normalize_station_name() -> None:
    assert normalize_station_name("다대포항역") == "다대포항"
    assert normalize_station_name("부산역") == "부산역"
    assert normalize_station_name("서면역(1호선)") == "서면"
    assert normalize_station_name("서면역(2호선)") == "서면"
    assert normalize_station_name("다대포해수욕장") == "다대포해수욕장"
    assert normalize_station_name("국제금융센터부산은행역") == "국제금융센터부산은행"


def test_extract_gugun() -> None:
    assert extract_gugun("부산광역시 사하구 다대로 지하 692") == "사하구"
    assert extract_gugun("부산광역시 부산진구 중앙대로 지하 730") == "부산진구"
    assert extract_gugun("경상남도 양산시 동면 호포로 67") == "양산시"
    assert extract_gugun("서울특별시 중구 태평로") == ""


def test_extract_line_number() -> None:
    assert extract_line_number("부산도시철도 1호선") == 1
    assert extract_line_number("부산도시철도 2호선") == 2
    assert extract_line_number("부산도시철도 3호선") == 3
    assert extract_line_number("부산도시철도 4호선") == 4
    assert extract_line_number("동해선") == 0


def test_build_station_master_with_sample(tmp_path: Path) -> None:
    lines = [
        "역번호\t역사명\t노선번호\t노선명\t영문역사명\t한자역사명\t환승역구분\t환승노선번호\t환승노선명\t역위도\t역경도\t운영기관명\t역사도로명주소\t역사전화번호\t데이터기준일자",
        "95\t다대포해수욕장\tS2601\t부산도시철도 1호선\tDadaepo Beach\t多大浦海水浴場\t일반역\t\t\t35.04867\t128.9641\t부산교통공사\t부산광역시 사하구 다대로 지하 692\t051-678-6195\t2021-02-26",
        "96\t다대포항역\tS2601\t부산도시철도 1호선\tDadaepo Harbor\t多大浦港\t일반역\t\t\t35.05782\t128.9713\t부산교통공사\t부산광역시 사하구 다대로 지하 548\t051-678-6196\t2021-02-26",
        "113\t부산역\tS2601\t부산도시철도 1호선\tBusan Station\t釜山\t일반역\t\t\t35.11522\t129.0422\t부산교통공사\t부산광역시 동구 중앙대로 지하 206\t051-678-6113\t2021-02-26",
        "119\t서면역(1호선)\tS2601\t부산도시철도 1호선\tSeomyeon\t西面\t환승역\tS2602\t부산도시철도 2호선\t35.15828\t129.0594\t부산교통공사\t부산광역시 부산진구 중앙대로 지하 730\t051-678-6119\t2021-02-26",
        "239\t호포역\tS2602\t부산도시철도 2호선\tHopo\t湖浦\t일반역\t\t\t35.28154\t129.0118\t부산교통공사\t경상남도 양산시 동면 호포로 67\t051-678-6239\t2021-02-26",
    ]
    sample_csv = tmp_path / "station_info_sample.csv"
    sample_csv.write_bytes("\n".join(lines).encode("utf-16"))

    output_parquet = tmp_path / "test_station_master.parquet"
    df = build_station_master(input_path=sample_csv, output_path=output_parquet)

    assert output_parquet.exists()
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5

    expected_cols = [
        "station_id",
        "station_name",
        "raw_station_name",
        "line_num",
        "line_name",
        "lat",
        "lng",
        "gugun",
        "address",
    ]
    assert df.columns == expected_cols

    # Types check
    assert df.schema["station_id"] == pl.Int64
    assert df.schema["lat"] == pl.Float64
    assert df.schema["lng"] == pl.Float64
    assert df.schema["line_num"] == pl.Int8

    # Specific rows check
    row_95 = df.filter(pl.col("station_id") == 95).to_dicts()[0]
    assert row_95["station_name"] == "다대포해수욕장"
    assert row_95["gugun"] == "사하구"
    assert pytest.approx(row_95["lat"], 0.001) == 35.04867
    assert pytest.approx(row_95["lng"], 0.001) == 128.9641

    row_113 = df.filter(pl.col("station_id") == 113).to_dicts()[0]
    assert row_113["station_name"] == "부산역"

    row_239 = df.filter(pl.col("station_id") == 239).to_dicts()[0]
    assert row_239["station_name"] == "호포"
    assert row_239["gugun"] == "양산시"
    assert row_239["line_num"] == 2


def test_reference_station_master_file() -> None:
    ref_parquet = PROJECT_ROOT / "data" / "reference" / "station_master.parquet"
    assert ref_parquet.exists()

    df_ref = pl.read_parquet(ref_parquet)
    assert len(df_ref) == 114
    assert df_ref["station_id"].n_unique() == 114
    assert df_ref["lat"].is_null().sum() == 0
    assert df_ref["lng"].is_null().sum() == 0

    # Latitude / Longitude valid bounding box (Busan / Yangsan)
    assert (df_ref["lat"] >= 35.0).all()
    assert (df_ref["lat"] <= 35.4).all()
    assert (df_ref["lng"] >= 128.8).all()
    assert (df_ref["lng"] <= 129.3).all()
