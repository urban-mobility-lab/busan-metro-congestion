"""
부산도시철도 역사정보 원시 데이터 정제 및 마스터 Parquet 생성 스크립트

입력: data/raw/subway/부산교통공사_도시철도역사정보_20210226.csv (UTF-16 LE TSV)
출력: data/reference/station_master.parquet (114개 역사 표준 마스터)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from busan_metro_congestion.config import PROJECT_ROOT


def normalize_station_name(name: str) -> str:
    """
    역사명 정규화 함수:
    - 괄호 내 부가정보 제거 (예: '서면역(1호선)' -> '서면역')
    - '부산역'을 제외하고 끝의 '역' 접미사 제거 (예: '다대포항역' -> '다대포항', '부산역' -> '부산역')
    """
    cleaned = re.sub(r"\(.*?\)", "", name).strip()
    if cleaned != "부산역" and cleaned.endswith("역") and len(cleaned) > 2:
        cleaned = cleaned[:-1]
    return cleaned


def extract_gugun(address: str) -> str:
    """
    도로명주소에서 구/군/시 행정구역 추출 (모델링 범주형 피처 및 EDA 보조 메타데이터):
    예: '부산광역시 사하구 다대로...' -> '사하구'
        '경상남도 양산시 물금읍...' -> '양산시'
    """
    match = re.search(r"(?:부산광역시|경상남도)\s+(\S+[구군시])", address)
    return match.group(1) if match else ""


def extract_line_number(line_name: str) -> int:
    """노선명에서 호선 번호 정수 추출 (예: '부산도시철도 1호선' -> 1)"""
    match = re.search(r"(\d+)호선", line_name)
    return int(match.group(1)) if match else 0


def build_station_master(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """
    도시철도 역사정보 원시 TSV(UTF-16)를 읽어 정규화된 station_master DataFrame을 생성하고
    Parquet로 저장합니다.
    """
    if input_path is None:
        input_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "subway"
            / "부산교통공사_도시철도역사정보_20210226.csv"
        )
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "reference" / "station_master.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"원시 역사 정보 파일을 찾을 수 없습니다: {input_path}")

    # 1. UTF-16 디코딩 및 TSV 파싱
    content = input_path.read_bytes().decode("utf-16")
    df_raw = pl.read_csv(content.encode("utf-8"), separator="\t")

    # 2. 필수 컬럼 유효성 검증
    required_cols = ["역번호", "역사명", "노선명", "역위도", "역경도", "역사도로명주소"]
    for col in required_cols:
        if col not in df_raw.columns:
            raise KeyError(
                f"필수 컬럼이 누락되었습니다: '{col}' (보유 컬럼: {df_raw.columns})"
            )

    # 3. 정규화 및 파생 컬럼 생성
    norm_names = [normalize_station_name(str(name)) for name in df_raw["역사명"]]
    guguns = [extract_gugun(str(addr)) for addr in df_raw["역사도로명주소"]]
    line_nums = [extract_line_number(str(line)) for line in df_raw["노선명"]]
    line_clean_names = [f"{n}호선" if n > 0 else "기타" for n in line_nums]

    df_master = (
        df_raw.with_columns(
            [
                pl.col("역번호").cast(pl.Int64).alias("station_id"),
                pl.Series("station_name", norm_names),
                pl.col("역사명").alias("raw_station_name"),
                pl.Series("line_num", line_nums).cast(pl.Int8),
                pl.Series("line_name", line_clean_names),
                pl.col("역위도").cast(pl.Float64).alias("lat"),
                pl.col("역경도").cast(pl.Float64).alias("lng"),
                pl.Series("gugun", guguns),
                pl.col("역사도로명주소").alias("address"),
            ]
        )
        .select(
            [
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
        )
        .sort("station_id")
    )

    # 4. Parquet 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_master.write_parquet(output_path)
    try:
        display_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = output_path
    print(f"[성공] 총 {len(df_master)}개 역사 정제 완료 -> {display_path}")

    return df_master


def main() -> None:
    parser = argparse.ArgumentParser(
        description="부산 도시철도 역사 마스터 Parquet 빌더"
    )
    parser.add_argument("--input", type=Path, default=None, help="원시 CSV 경로")
    parser.add_argument("--output", type=Path, default=None, help="출력 Parquet 경로")
    args = parser.parse_args()

    build_station_master(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
