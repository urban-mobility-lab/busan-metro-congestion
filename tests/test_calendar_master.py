"""
달력 및 공휴일 마스터 빌더 단위 테스트
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from busan_metro_congestion.config import SAMPLES_DIR
from scripts.build_calendar_master import (
    build_calendar_master,
    infer_calendar_year_bounds,
    load_holidays_from_raw_dir,
    main,
)

CALENDAR_SAMPLES_DIR = SAMPLES_DIR / "calendar"


def test_build_calendar_master_range_and_schema(tmp_path: Path) -> None:
    """샘플 KASI JSON 기반 2개년(2024~2025, 731일) 일자 생성 및 스키마를 검증합니다."""
    out_file = tmp_path / "calendar_master_test.parquet"
    df = build_calendar_master(
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        raw_calendar_dir=CALENDAR_SAMPLES_DIR,
        output_path=out_file,
    )

    # 1. 731일 확인 (366(2024 윤년) + 365(2025 평년))
    assert len(df) == 731
    assert out_file.exists()

    # 2. 컬럼 및 스키마 검증
    expected_cols = [
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
    assert df.columns == expected_cols
    assert df.schema["date"] == pl.Date
    assert df.schema["is_weekend"] == pl.Boolean
    assert df.schema["is_holiday"] == pl.Boolean
    assert df.schema["is_holiday_eve"] == pl.Boolean
    assert df.schema["season"] == pl.Int8
    assert df.schema["month_sin"] == pl.Float32
    assert df.schema["month_cos"] == pl.Float32
    assert df.schema["weekday_sin"] == pl.Float32
    assert df.schema["weekday_cos"] == pl.Float32

    # 3. 결측치 전무 검증
    for col in expected_cols:
        assert df[col].is_null().sum() == 0, f"컬럼 {col}에 결측치가 존재합니다."

    # 4. 삼각함수 주기 피처 주기성 및 기준점 값 검증
    # 12월(2*pi) -> sin=0, cos=1 / 3월(pi/2) -> sin=1, cos=0
    dec_row = df.filter(pl.col("date") == date(2024, 12, 1))
    mar_row = df.filter(pl.col("date") == date(2024, 3, 1))
    assert pytest.approx(dec_row["month_sin"][0], abs=1e-5) == 0.0
    assert pytest.approx(dec_row["month_cos"][0], abs=1e-5) == 1.0
    assert pytest.approx(mar_row["month_sin"][0], abs=1e-5) == 1.0
    assert pytest.approx(mar_row["month_cos"][0], abs=1e-5) == 0.0
    # 일요일(weekday=7 -> 2*pi) -> sin=0, cos=1 / 월요일(weekday=1)과 토요일(weekday=6)은 cos 대칭
    sun_row = df.filter(pl.col("date") == date(2024, 6, 2))
    mon_row = df.filter(pl.col("date") == date(2024, 6, 3))
    sat_row = df.filter(pl.col("date") == date(2024, 6, 1))
    assert pytest.approx(sun_row["weekday_sin"][0], abs=1e-5) == 0.0
    assert pytest.approx(sun_row["weekday_cos"][0], abs=1e-5) == 1.0
    assert (
        pytest.approx(mon_row["weekday_cos"][0], abs=1e-5) == sat_row["weekday_cos"][0]
    )
    assert (
        pytest.approx(mon_row["weekday_sin"][0], abs=1e-5) == -sat_row["weekday_sin"][0]
    )


def test_major_holidays_dedup_and_non_holiday_filtering(tmp_path: Path) -> None:
    """선거일·임시공휴일 반영, 중복 locdate 중복 제거, isHoliday='N' 제외를 검증합니다."""
    holidays = load_holidays_from_raw_dir(CALENDAR_SAMPLES_DIR)
    # 샘플 11건 중 20250505 중복 1건 제거 + 20250717(isHoliday='N') 1건 제외 -> 고유 9건
    assert len(holidays) == 9
    assert date(2025, 7, 17) not in holidays  # 제헌절(isHoliday="N") 제외

    out_file = tmp_path / "cal_test.parquet"
    df = build_calendar_master(
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        raw_calendar_dir=CALENDAR_SAMPLES_DIR,
        output_path=out_file,
    )

    check_holidays = [
        "2024-04-10",  # 제22대 총선
        "2024-10-01",  # 국군의 날 임시공휴일
        "2025-01-27",  # 설 연휴 임시공휴일
        "2025-05-05",  # 어린이날 + 부처님오신날 중복일
        "2025-05-06",  # 대체공휴일
        "2025-06-03",  # 제21대 대선
    ]
    for h_str in check_holidays:
        row = df.filter(pl.col("date") == date.fromisoformat(h_str))
        assert len(row) == 1
        assert row["is_holiday"][0] is True, f"{h_str}이 공휴일로 지정되지 않았습니다."

    # 제헌절(isHoliday="N")은 공휴일이 아니어야 함
    constitution_day = df.filter(pl.col("date") == date(2025, 7, 17))
    assert constitution_day["is_holiday"][0] is False


def test_holiday_eve_season_and_weekend(tmp_path: Path) -> None:
    """공휴일 전야(is_holiday_eve), 계절(season), 주말(is_weekend) 판별 로직을 검증합니다."""
    out_file = tmp_path / "cal_logic.parquet"
    df = build_calendar_master(
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        raw_calendar_dir=CALENDAR_SAMPLES_DIR,
        output_path=out_file,
    )

    # 2024-04-10(총선) 전날 2024-04-09 -> is_holiday_eve == True
    assert df.filter(pl.col("date") == date(2024, 4, 9))["is_holiday_eve"][0] is True
    assert df.filter(pl.col("date") == date(2024, 4, 11))["is_holiday_eve"][0] is False

    # 계절 검증: 3월(봄=1), 7월(여름=2), 10월(가을=3), 1월(겨울=4)
    assert df.filter(pl.col("date") == date(2024, 3, 15))["season"][0] == 1
    assert df.filter(pl.col("date") == date(2024, 7, 15))["season"][0] == 2
    assert df.filter(pl.col("date") == date(2024, 10, 15))["season"][0] == 3
    assert df.filter(pl.col("date") == date(2024, 1, 15))["season"][0] == 4

    # 주말 검증: 2024-06-01(토), 2024-06-02(일), 2024-06-03(월)
    assert df.filter(pl.col("date") == date(2024, 6, 1))["is_weekend"][0] is True
    assert df.filter(pl.col("date") == date(2024, 6, 2))["is_weekend"][0] is True
    assert df.filter(pl.col("date") == date(2024, 6, 3))["is_weekend"][0] is False


def test_load_holidays_single_dict_and_empty_items(tmp_path: Path) -> None:
    """단건 dict JSON과 공휴일 0건(items='') JSON이 함께 있을 때 정상 로드되는지 검증합니다."""
    (tmp_path / "holidays_2025.json").write_text(
        json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                    "body": {
                        "items": {
                            "item": {
                                "dateKind": "01",
                                "dateName": "삼일절",
                                "isHoliday": "Y",
                                "locdate": "20250301",
                                "seq": 1,
                            }
                        },
                        "totalCount": 1,
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "holidays_2026.json").write_text(
        json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                    "body": {"items": "", "totalCount": 0},
                }
            }
        ),
        encoding="utf-8",
    )

    holidays = load_holidays_from_raw_dir(tmp_path)
    assert holidays == {date(2025, 3, 1)}


def test_build_calendar_master_missing_raw_and_invalid_dates(tmp_path: Path) -> None:
    """원천 JSON 부재 시 FileNotFoundError, 날짜 역전 시 ValueError 발생을 검증합니다."""
    empty_dir = tmp_path / "empty_calendar"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="원천 JSON 파일이 존재하지 않습니다"):
        build_calendar_master(
            raw_calendar_dir=empty_dir,
            output_path=tmp_path / "out.parquet",
        )

    with pytest.raises(FileNotFoundError, match="원천 디렉터리가 존재하지 않습니다"):
        build_calendar_master(
            raw_calendar_dir=tmp_path / "non_existent",
            output_path=tmp_path / "out.parquet",
        )

    with pytest.raises(ValueError, match="시작일"):
        build_calendar_master(
            start_date=date(2026, 1, 1),
            end_date=date(2025, 1, 1),
            raw_calendar_dir=CALENDAR_SAMPLES_DIR,
            output_path=tmp_path / "out.parquet",
        )


def test_build_calendar_master_cli_and_default_full_range(tmp_path: Path) -> None:
    """CLI main()의 날짜 인자 생략 시 기본 11개년(2016~2026, 4,018일) 생성 동작을 검증합니다."""
    out_file = tmp_path / "cli_calendar.parquet"
    df = main(
        [
            "--raw-dir",
            str(CALENDAR_SAMPLES_DIR),
            "--output",
            str(out_file),
        ]
    )
    assert out_file.exists()
    assert len(df) == 4018
    assert df["date"].min() == date(2016, 1, 1)
    assert df["date"].max() == date(2026, 12, 31)


def test_build_calendar_master_auto_expand_range(tmp_path: Path) -> None:
    """auto_expand_range=True/False, 과거·미래 양방향 확장, CLI --auto-expand, 날짜 역전 예외를 검증합니다."""
    raw_dir = tmp_path / "calendar"
    raw_dir.mkdir()
    (raw_dir / "holidays_2028.json").write_text(
        json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                    "body": {
                        "items": {
                            "item": [
                                {
                                    "dateKind": "01",
                                    "dateName": "신정",
                                    "isHoliday": "Y",
                                    "locdate": 20280101,
                                    "seq": 1,
                                }
                            ]
                        },
                        "totalCount": 1,
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 1. auto_expand_range=False(기본값)일 때는 미래 연도 JSON이 있어도 지정 범위 유지
    df_fixed = build_calendar_master(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        raw_calendar_dir=raw_dir,
        output_path=tmp_path / "fixed_calendar.parquet",
        auto_expand_range=False,
    )
    assert df_fixed["date"].min() == date(2026, 1, 1)
    assert df_fixed["date"].max() == date(2026, 12, 31)

    # 2. auto_expand_range=True일 때 미래 연도로 자동 확장
    out_file = tmp_path / "expanded_calendar.parquet"
    df = build_calendar_master(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        raw_calendar_dir=raw_dir,
        output_path=out_file,
        auto_expand_range=True,
    )
    assert df["date"].min() == date(2026, 1, 1)
    assert df["date"].max() == date(2028, 12, 31)
    assert df.filter(pl.col("date") == date(2028, 1, 1))["is_holiday"][0] is True

    # 3. 과거 연도 JSON 추가 + CLI --auto-expand 플래그로 양방향 확장
    (raw_dir / "holidays_2024.json").write_text(
        json.dumps(
            {
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                    "body": {"items": "", "totalCount": 0},
                }
            }
        ),
        encoding="utf-8",
    )
    cli_out = tmp_path / "cli_expanded.parquet"
    df_cli = main(
        [
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-12-31",
            "--raw-dir",
            str(raw_dir),
            "--output",
            str(cli_out),
            "--auto-expand",
        ]
    )
    assert df_cli["date"].min() == date(2024, 1, 1)
    assert df_cli["date"].max() == date(2028, 12, 31)

    # 4. 날짜 역전 입력은 auto_expand_range=True여도 반드시 ValueError 발생
    with pytest.raises(ValueError, match="시작일"):
        build_calendar_master(
            start_date=date(2026, 1, 1),
            end_date=date(2025, 1, 1),
            raw_calendar_dir=raw_dir,
            output_path=tmp_path / "invalid.parquet",
            auto_expand_range=True,
        )


def test_infer_calendar_year_bounds_edge_cases(tmp_path: Path) -> None:
    """infer_calendar_year_bounds의 빈/미존재 디렉터리, 파일명 없는 locdate 추출, 손상 JSON 무시 동작을 검증합니다."""
    # 1. 미존재 디렉터리 -> 기본값 반환
    assert infer_calendar_year_bounds(
        tmp_path / "missing", default_start_year=2020, default_end_year=2025
    ) == (2020, 2025)

    raw_dir = tmp_path / "raw_bounds"
    raw_dir.mkdir()
    # 2. 빈 디렉터리 -> 기본값 반환
    assert infer_calendar_year_bounds(
        raw_dir, default_start_year=2020, default_end_year=2025
    ) == (2020, 2025)

    # 3. 파일명에 연도가 없어도 내부 locdate에서 연도 추출, 빈 items 파일명으로 종료 연도 확장, 손상 JSON은 안전 스킵
    (raw_dir / "extra_custom.json").write_text(
        json.dumps(
            {
                "response": {
                    "body": {
                        "items": {
                            "item": [
                                {"locdate": 20180505, "isHoliday": "Y"},
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "holidays_2029.json").write_text(
        json.dumps({"response": {"body": {"items": "", "totalCount": 0}}}),
        encoding="utf-8",
    )
    (raw_dir / "corrupted.json").write_text("{invalid json", encoding="utf-8")

    assert infer_calendar_year_bounds(
        raw_dir, default_start_year=2020, default_end_year=2025
    ) == (2018, 2029)
