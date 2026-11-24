from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import PROJECT_ROOT
from scripts.build_station_weather_grid import (
    build_station_weather_grid,
    latlon_to_grid,
)


def test_latlon_to_grid_known_locations() -> None:
    # 서울특별시청 기준점 검증 (60, 127)
    assert latlon_to_grid(37.563569, 126.980008) == (60, 127)
    # 부산광역시청 기준점 검증 (98, 76)
    assert latlon_to_grid(35.177019, 129.076953) == (98, 76)
    # 다대포해수욕장역 (96, 73)
    assert latlon_to_grid(35.04867, 128.9641) == (96, 73)
    # 서면역 (98, 75)
    assert latlon_to_grid(35.15828, 129.0594) == (98, 75)
    # 호포역 (97, 78)
    assert latlon_to_grid(35.28154, 129.0118) == (97, 78)


def test_build_station_weather_grid_with_sample(tmp_path: Path) -> None:
    sample_master = pl.DataFrame(
        {
            "station_id": [95, 119, 239],
            "station_name": ["다대포해수욕장", "서면", "호포"],
            "lat": [35.04867, 35.15828, 35.28154],
            "lng": [128.9641, 129.0594, 129.0118],
        }
    )
    input_parquet = tmp_path / "sample_master.parquet"
    output_parquet = tmp_path / "sample_weather_grid.parquet"
    sample_master.write_parquet(input_parquet)

    df_grid = build_station_weather_grid(
        input_path=input_parquet, output_path=output_parquet
    )

    assert output_parquet.exists()
    assert len(df_grid) == 3
    assert df_grid.columns == ["station_id", "station_name", "nx", "ny"]
    assert df_grid.schema["station_id"] == pl.Int64
    assert df_grid.schema["nx"] == pl.Int64
    assert df_grid.schema["ny"] == pl.Int64

    rows = df_grid.to_dicts()
    assert (rows[0]["nx"], rows[0]["ny"]) == (96, 73)
    assert (rows[1]["nx"], rows[1]["ny"]) == (98, 75)
    assert (rows[2]["nx"], rows[2]["ny"]) == (97, 78)


def test_build_station_weather_grid_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_station_weather_grid(
            input_path=tmp_path / "non_existent.parquet",
            output_path=tmp_path / "out.parquet",
        )


def test_reference_station_weather_grid_file() -> None:
    ref_parquet = PROJECT_ROOT / "data" / "reference" / "station_weather_grid.parquet"
    assert ref_parquet.exists()

    df_ref = pl.read_parquet(ref_parquet)
    assert len(df_ref) == 114
    assert df_ref["station_id"].n_unique() == 114
    assert df_ref["nx"].is_null().sum() == 0
    assert df_ref["ny"].is_null().sum() == 0
    assert df_ref.select(["nx", "ny"]).n_unique() == 19

    # 부산/양산 격자 범위 검증
    assert (df_ref["nx"] >= 95).all() and (df_ref["nx"] <= 101).all()
    assert (df_ref["ny"] >= 72).all() and (df_ref["ny"] <= 80).all()
