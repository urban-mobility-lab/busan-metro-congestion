"""
부산도시철도 역사 좌표를 기상청 단기예보 격자(nx, ny)로 변환하여
매핑 Parquet 파일을 생성하는 스크립트

입력: data/reference/station_master.parquet
출력: data/reference/station_weather_grid.parquet (114개 역사 1:1 격자 매핑)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from busan_metro_congestion.config import PROJECT_ROOT


def latlon_to_grid(lat: float, lon: float) -> tuple[int, int]:
    """
    기상청 단기예보 공식 LCC(Lambert Conformal Conic) 투영 변환 수식
    위도(lat), 경도(lon)를 입력받아 단기예보 격자 좌표 (nx, ny)를 반환합니다.
    """
    re_km = 6371.00877  # 지구 반경(km)
    grid_km = 5.0  # 격자 간격(km)
    slat1_deg = 30.0  # 투영 위도1(degree)
    slat2_deg = 60.0  # 투영 위도2(degree)
    olon_deg = 126.0  # 기준점 경도(degree)
    olat_deg = 38.0  # 기준점 위도(degree)
    xo = 43.0  # 기준점 X좌표(GRID)
    yo = 136.0  # 기준점 Y좌표(GRID)

    degrad = math.pi / 180.0
    re = re_km / grid_km
    slat1 = slat1_deg * degrad
    slat2 = slat2_deg * degrad
    olon = olon_deg * degrad
    olat = olat_deg * degrad

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)

    ra = math.tan(math.pi * 0.25 + lat * degrad * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * degrad - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    nx = math.floor(ra * math.sin(theta) + xo + 0.5)
    ny = math.floor(ro - ra * math.cos(theta) + yo + 0.5)
    return nx, ny


def build_station_weather_grid(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """
    station_master.parquet의 역사별 위경도를 기상청 격자(nx, ny)로 변환하여
    station_weather_grid.parquet로 저장합니다.
    """
    if input_path is None:
        input_path = PROJECT_ROOT / "data" / "reference" / "station_master.parquet"
    if output_path is None:
        output_path = (
            PROJECT_ROOT / "data" / "reference" / "station_weather_grid.parquet"
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"역사 마스터 Parquet 파일을 찾을 수 없습니다: {input_path}"
        )

    df_master = pl.read_parquet(input_path)

    required_cols = ["station_id", "station_name", "lat", "lng"]
    for col in required_cols:
        if col not in df_master.columns:
            raise KeyError(
                f"필수 컬럼이 누락되었습니다: '{col}' (보유 컬럼: {df_master.columns})"
            )

    nx_list: list[int] = []
    ny_list: list[int] = []
    for lat, lng in zip(df_master["lat"], df_master["lng"]):
        nx, ny = latlon_to_grid(float(lat), float(lng))
        nx_list.append(nx)
        ny_list.append(ny)

    df_grid = (
        df_master.with_columns(
            [
                pl.Series("nx", nx_list, dtype=pl.Int64),
                pl.Series("ny", ny_list, dtype=pl.Int64),
            ]
        )
        .select(["station_id", "station_name", "nx", "ny"])
        .sort("station_id")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_grid.write_parquet(output_path)

    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path

    n_unique_grids = df_grid.select(["nx", "ny"]).n_unique()
    print(
        f"[성공] 총 {len(df_grid)}개 역사 기상 격자 매핑 완료 "
        f"(고유 격자 {n_unique_grids}개) -> {display_path}"
    )

    return df_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="부산 도시철도 역사별 기상청 단기예보 격자(nx, ny) 매핑 빌더"
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="입력 station_master.parquet 경로"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="출력 station_weather_grid.parquet 경로",
    )
    args = parser.parse_args()

    build_station_weather_grid(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
