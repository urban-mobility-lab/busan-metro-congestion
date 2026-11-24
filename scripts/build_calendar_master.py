"""
한국천문연구원 특일 정보(getRestDeInfo) 원천 JSON 기반 달력 및 공휴일 마스터 빌더 스크립트

기능:
1. data/raw/calendar/*.json(KASI getRestDeInfo 수집본)에서 isHoliday=='Y' 공휴일 날짜를 추출·중복 제거.
2. 지정된 기간(기본: 2016-01-01 ~ 2026-12-31, 총 4,018일) 전수 일자 시퀀스 생성.
3. 주말(is_weekend), 공휴일(is_holiday), 공휴일 전야(is_holiday_eve),
   계절(season: 1봄, 2여름, 3가을, 4겨울) 및 삼각함수 주기(month_sin/cos, weekday_sin/cos) 산출.
4. 산출물: data/reference/calendar_master.parquet 저장.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from busan_metro_congestion.config import RAW_DATA_DIR, REFERENCE_DATA_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_calendar_master")


def _extract_items_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """KASI getRestDeInfo JSON 페이로드에서 공휴일 항목 리스트를 안전하게 추출합니다."""
    body = payload.get("response", {}).get("body", {})
    if not isinstance(body, dict):
        return []

    items_node = body.get("items")
    if not isinstance(items_node, dict):
        return []

    raw_items = items_node.get("item")
    if raw_items is None:
        return []
    if isinstance(raw_items, dict):
        return [raw_items]
    if isinstance(raw_items, list):
        return [it for it in raw_items if isinstance(it, dict)]
    return []


def _parse_locdate(locdate: Any) -> date | None:
    """KASI locdate(예: 20250505 또는 '20250505')를 date 객체로 변환합니다."""
    if locdate is None:
        return None
    s = str(locdate).strip()
    if len(s) != 8 or not s.isdigit():
        return None
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def load_holidays_from_raw_dir(raw_calendar_dir: Path | str) -> set[date]:
    """
    원천 달력 디렉터리의 KASI getRestDeInfo JSON 파일들을 모두 읽어
    isHoliday == 'Y'인 고유 공휴일 날짜 집합을 반환합니다.
    """
    raw_dir = Path(raw_calendar_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"공휴일 원천 디렉터리가 존재하지 않습니다: {raw_dir}. "
            "사전에 'uv run python scripts/fetch_calendar_holidays_api.py'를 실행하세요."
        )

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"공휴일 원천 JSON 파일이 존재하지 않습니다: {raw_dir}. "
            "사전에 'uv run python scripts/fetch_calendar_holidays_api.py'를 실행하세요."
        )

    holidays: set[date] = set()
    for jf in json_files:
        payload = json.loads(jf.read_text(encoding="utf-8"))
        for item in _extract_items_from_payload(payload):
            if str(item.get("isHoliday", "")).strip().upper() != "Y":
                continue
            parsed = _parse_locdate(item.get("locdate"))
            if parsed is not None:
                holidays.add(parsed)

    logger.info(
        f"원천 JSON {len(json_files)}개 파일에서 고유 공휴일 {len(holidays)}건 로드 완료 ({raw_dir})"
    )
    return holidays


def build_calendar_master(
    start_date: date = date(2016, 1, 1),
    end_date: date = date(2026, 12, 31),
    raw_calendar_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    지정된 날짜 범위의 달력 및 공휴일 마스터 DataFrame을 생성합니다.

    컬럼 구성:
    - date: Date
    - is_weekend: Boolean (토·일요일)
    - is_holiday: Boolean (법정/대체/임시공휴일 및 선거일)
    - is_holiday_eve: Boolean (익일이 is_holiday인 날)
    - season: Int8 (1: 봄 3~5월, 2: 여름 6~8월, 3: 가을 9~11월, 4: 겨울 12~2월)
    - month_sin, month_cos: Float32
    - weekday_sin, weekday_cos: Float32
    """
    if start_date > end_date:
        raise ValueError(
            f"시작일({start_date})은 종료일({end_date})보다 클 수 없습니다."
        )

    raw_dir = (
        Path(raw_calendar_dir)
        if raw_calendar_dir is not None
        else RAW_DATA_DIR / "calendar"
    )
    holidays = load_holidays_from_raw_dir(raw_dir)
    holiday_dates = sorted(holidays)

    logger.info(f"달력 마스터 생성 시작: {start_date} ~ {end_date}")

    # 1. 일자 시퀀스 생성
    df_dates = pl.date_range(
        start=start_date,
        end=end_date,
        interval="1d",
        eager=True,
    ).to_frame("date")

    # 2. 날짜 기본 속성 및 주말, 공휴일 플래그 생성
    df_base = (
        df_dates.with_columns(
            pl.col("date").dt.weekday().alias("weekday"),  # 월: 1 ~ 일: 7
            pl.col("date").dt.month().alias("month"),
            pl.col("date").is_in(holiday_dates).alias("is_holiday"),
        )
        .with_columns(
            pl.col("weekday").is_in([6, 7]).alias("is_weekend"),
        )
        .sort("date")
    )

    # 3. is_holiday_eve (다음 날이 공휴일인지 여부, lead(is_holiday, 1))
    df_with_eve = df_base.with_columns(
        pl.col("is_holiday").shift(-1).fill_null(False).alias("is_holiday_eve")
    )

    # 4. season (1: 봄 3~5월, 2: 여름 6~8월, 3: 가을 9~11월, 4: 겨울 12~2월)
    season_expr = (
        pl.when(pl.col("month").is_in([3, 4, 5]))
        .then(pl.lit(1, dtype=pl.Int8))
        .when(pl.col("month").is_in([6, 7, 8]))
        .then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col("month").is_in([9, 10, 11]))
        .then(pl.lit(3, dtype=pl.Int8))
        .otherwise(pl.lit(4, dtype=pl.Int8))
        .alias("season")
    )

    # 5. 주기적 인코딩 (month_sin/cos, weekday_sin/cos)
    month_rad = pl.col("month").cast(pl.Float32) * (2.0 * math.pi / 12.0)
    weekday_rad = pl.col("weekday").cast(pl.Float32) * (2.0 * math.pi / 7.0)

    df_master = df_with_eve.with_columns(
        season_expr,
        month_rad.sin().alias("month_sin"),
        month_rad.cos().alias("month_cos"),
        weekday_rad.sin().alias("weekday_sin"),
        weekday_rad.cos().alias("weekday_cos"),
    ).select(
        [
            "date",
            "is_weekend",
            "is_holiday",
            "is_holiday_eve",
            "season",
            "month_sin",
            "month_cos",
            "weekday_sin",
            "weekday_cos",
        ]
    )

    if output_path is None:
        output_path = REFERENCE_DATA_DIR / "calendar_master.parquet"
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_master.write_parquet(out_p)
    logger.info(f"달력 마스터 저장 완료: {out_p} ({len(df_master)}일자)")

    return df_master


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KASI 특일 정보 JSON 기반 대한민국 달력 마스터(calendar_master.parquet) 빌더"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="달력 시작일 (YYYY-MM-DD, 기본값: 2016-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-12-31",
        help="달력 종료일 (YYYY-MM-DD, 기본값: 2026-12-31)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="공휴일 원천 JSON 디렉터리 (기본값: data/raw/calendar)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 Parquet 경로 (기본값: data/reference/calendar_master.parquet)",
    )
    return parser


def main(argv: list[str] | None = None) -> pl.DataFrame:
    args = build_parser().parse_args(argv)
    return build_calendar_master(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        raw_calendar_dir=args.raw_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
