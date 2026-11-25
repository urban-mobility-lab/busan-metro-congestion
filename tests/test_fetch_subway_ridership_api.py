"""
부산교통공사 시간대별 승하차인원(odcloud) 단발 수집기 단위 테스트
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from busan_metro_congestion.config import REFERENCE_DATA_DIR
from busan_metro_congestion.preprocess.subway import build_subway_interim
from scripts.fetch_subway_ridership_api import (
    HOUR_SLOT_COLUMNS,
    ODCLOUD_API_BASE,
    ODCLOUD_OAS_URL,
    AuthenticationError,
    QuotaExceededError,
    SubwayEndpoint,
    extract_subway_endpoints_from_oas,
    fetch_subway_ridership,
    fetch_subway_uddi_rows,
    parse_summary_snapshot_date,
    rows_to_subway_dataframe,
    validate_subway_raw_df,
)


def _make_mock_response(
    payload: dict[str, Any] | None = None,
    status_code: int = 200,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if payload is not None:
        resp.json.return_value = payload
    else:
        resp.json.side_effect = ValueError("Invalid JSON")
    resp.raise_for_status.return_value = None
    return resp


def _make_subway_row(
    station_id: int = 119,
    station_name: str = "서면",
    ymd: str = "2026-01-01",
    weekday: str = "목",
    gubun: str = "승차",
    base_val: int = 100,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "역번호": station_id,
        "역명": station_name,
        "년월일": ymd,
        "요일": weekday,
        "구분": gubun,
        "합계": base_val * 24,
    }
    for col in HOUR_SLOT_COLUMNS:
        row[col] = base_val
    return row


def _make_valid_model_def() -> dict[str, Any]:
    props: dict[str, Any] = {
        "역번호": {"type": "integer"},
        "역명": {"type": "string"},
        "년월일": {"type": "string"},
        "요일": {"type": "string"},
        "구분": {"type": "string"},
        "합계": {"type": "integer"},
    }
    for col in HOUR_SLOT_COLUMNS:
        props[col] = {"type": "integer"}
    return {"type": "object", "properties": props}


def _make_oas_spec(entries: list[tuple[str, str, bool]]) -> dict[str, Any]:
    """(uddi, summary, valid_model) 목록으로 odcloud OAS 스펙을 생성합니다."""
    paths: dict[str, Any] = {}
    defs: dict[str, Any] = {}
    for uddi, summary, valid_model in entries:
        path = f"/3057229/v1/{uddi}"
        paths[path] = {
            "get": {
                "summary": summary,
                "parameters": [
                    {"name": "page"},
                    {"name": "perPage"},
                    {"name": "returnType"},
                ],
            }
        }
        if valid_model:
            defs[f"{uddi}_model"] = _make_valid_model_def()
        else:
            defs[f"{uddi}_model"] = {
                "type": "object",
                "properties": {"<!DOCTYPE html>": {"type": "string"}},
            }
    return {
        "host": "api.odcloud.kr",
        "basePath": "/api",
        "paths": paths,
        "definitions": defs,
    }


def test_parse_summary_snapshot_date_formats() -> None:
    assert parse_summary_snapshot_date("부산교통공사_시간대별 승하차인원_20261031") == (
        2026,
        date(2026, 10, 31),
        True,
    )
    assert parse_summary_snapshot_date(
        "부산교통공사_시간대별 승하차인원_10/01/2021"
    ) == (2021, date(2021, 10, 1), True)
    assert parse_summary_snapshot_date("부산교통공사 시간대별 승하차인원(2018년)") == (
        2018,
        date(2018, 12, 31),
        False,
    )
    assert parse_summary_snapshot_date("무관한 텍스트") is None


def test_extract_subway_endpoints_selects_latest_per_year_and_filters_corrupt() -> None:
    """연도별 최신 스냅샷 1개만 선별하고 비정상 OAS 스키마를 제외하는지 검증합니다."""
    spec = _make_oas_spec(
        [
            ("uddi:bad-2018", "부산교통공사 시간대별 승하차인원(2018년)", False),
            ("uddi:2024-apr", "부산교통공사_시간대별 승하차인원_20240430", True),
            ("uddi:2024-sep", "부산교통공사_시간대별 승하차인원_20240930", True),
            ("uddi:2024-dec", "부산교통공사_시간대별 승하차인원_20241231", True),
            ("uddi:2025-dec", "부산교통공사_시간대별 승하차인원_20251231", True),
            ("uddi:2026-aug", "부산교통공사_시간대별 승하차인원_20260831", True),
            ("uddi:2026-oct", "부산교통공사_시간대별 승하차인원_20261031", True),
        ]
    )

    endpoints = extract_subway_endpoints_from_oas(
        spec, start_year=2024, end_year=2026, latest_per_year=True
    )
    assert [e.year for e in endpoints] == [2024, 2025, 2026]
    assert [e.uddi for e in endpoints] == [
        "uddi:2024-dec",
        "uddi:2025-dec",
        "uddi:2026-oct",
    ]
    assert endpoints[2].snapshot_date == date(2026, 10, 31)

    all_eps = extract_subway_endpoints_from_oas(
        spec, start_year=2018, end_year=2026, latest_per_year=True
    )
    assert all(e.uddi != "uddi:bad-2018" for e in all_eps)


def test_fetch_subway_ridership_stateless_resume_and_force(tmp_path: Path) -> None:
    """단발 수집기의 페이지네이션, 기존 파일 스킵, force 덮어쓰기 동작을 검증합니다."""
    oas_spec = _make_oas_spec(
        [
            ("uddi:2025-dec", "부산교통공사_시간대별 승하차인원_20251231", True),
            ("uddi:2026-oct", "부산교통공사_시간대별 승하차인원_20261031", True),
        ]
    )
    datasets = {
        "uddi:2025-dec": [
            _make_subway_row(119, "서면", "2025-12-31", "수", "승차", 120),
            _make_subway_row(119, "서면", "2025-12-31", "수", "하차", 110),
        ],
        "uddi:2026-oct": [
            _make_subway_row(119, "서면", "2026-10-31", "토", "승차", 210),
            _make_subway_row(119, "서면", "2026-10-31", "토", "하차", 200),
            _make_subway_row(227, "사상", "2026-10-31", "토", "승차", 90),
            _make_subway_row(227, "사상", "2026-10-31", "토", "하차", 80),
        ],
    }
    data_calls: list[str] = []

    def fake_get(
        url: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> MagicMock:
        params = params or {}
        if url == ODCLOUD_OAS_URL:
            return _make_mock_response(payload=oas_spec)
        uddi = url.rstrip("/").split("/")[-1]
        data_calls.append(uddi)
        page = int(params.get("page", 1))
        per_page = int(params.get("perPage", 1000))
        rows = datasets[uddi]
        start = (page - 1) * per_page
        chunk = rows[start : start + per_page]
        return _make_mock_response(
            payload={
                "page": page,
                "perPage": per_page,
                "totalCount": len(rows),
                "currentCount": len(chunk),
                "matchCount": len(rows),
                "data": chunk,
            }
        )

    # 1. 최초 실행: 2025(1페이지), 2026(2페이지) 수집
    files1 = fetch_subway_ridership(
        api_key="test-key",
        start_year=2025,
        output_dir=tmp_path,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        per_page=2,
        delay_sec=0.0,
        http_get=fake_get,
    )
    assert len(files1) == 2
    assert (tmp_path / "subway_ridership_2025.parquet").exists()
    assert (tmp_path / "subway_ridership_2026.parquet").exists()
    assert data_calls == ["uddi:2025-dec", "uddi:2026-oct", "uddi:2026-oct"]

    # 2. 재실행(force=False): 기존 파일 스킵으로 데이터 API 미호출
    data_calls.clear()
    fetch_subway_ridership(
        api_key="test-key",
        start_year=2025,
        output_dir=tmp_path,
        per_page=2,
        delay_sec=0.0,
        force=False,
        http_get=fake_get,
    )
    assert data_calls == []

    # 3. 강제 재수집(force=True)
    fetch_subway_ridership(
        api_key="test-key",
        start_year=2025,
        output_dir=tmp_path,
        per_page=2,
        delay_sec=0.0,
        force=True,
        http_get=fake_get,
    )
    assert len(data_calls) == 3

    # 4. 연도 역전 예외
    with pytest.raises(ValueError, match="시작 연도"):
        fetch_subway_ridership(
            api_key="test-key",
            start_year=2026,
            end_year=2025,
            output_dir=tmp_path,
            http_get=fake_get,
        )


@pytest.mark.parametrize(
    ("status_code", "expected_exc"),
    [
        (429, QuotaExceededError),
        (401, AuthenticationError),
    ],
)
def test_fetch_subway_uddi_fail_fast_on_quota_and_auth(
    status_code: int,
    expected_exc: type[Exception],
) -> None:
    """HTTP 429/401 응답 시 재시도 없이 즉시 예외가 발생하는지 검증합니다."""
    calls = 0

    def fake_get(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal calls
        calls += 1
        return _make_mock_response(payload={}, status_code=status_code)

    ep = SubwayEndpoint(
        path="/3057229/v1/uddi:test",
        uddi="uddi:test",
        summary="부산교통공사_시간대별 승하차인원_20261031",
        year=2026,
        snapshot_date=date(2026, 10, 31),
    )
    with pytest.raises(expected_exc):
        fetch_subway_uddi_rows(
            api_key="test-key",
            endpoint=ep,
            api_base=ODCLOUD_API_BASE,
            max_retries=3,
            http_get=fake_get,
        )
    assert calls == 1


def test_validate_subway_raw_df_rejects_corrupted_data() -> None:
    """빈 데이터, 음수 인원, 승차/하차 누락, 미매칭 역번호를 검증 게이트가 차단하는지 확인합니다."""
    with pytest.raises(ValueError, match="비어 있습니다"):
        validate_subway_raw_df(rows_to_subway_dataframe([]))

    bad_neg = _make_subway_row(119, "서면", "2026-01-01", "목", "승차", 10)
    bad_neg["08시-09시"] = -5
    df_neg = rows_to_subway_dataframe(
        [bad_neg, _make_subway_row(119, "서면", "2026-01-01", "목", "하차", 10)]
    )
    with pytest.raises(ValueError, match="음수 인원"):
        validate_subway_raw_df(df_neg)

    df_only_in = rows_to_subway_dataframe(
        [_make_subway_row(119, "서면", "2026-01-01", "목", "승차", 10)]
    )
    with pytest.raises(ValueError, match="승차/하차"):
        validate_subway_raw_df(df_only_in)

    df_unknown = rows_to_subway_dataframe(
        [
            _make_subway_row(999999, "미등록역", "2026-01-01", "목", "승차", 10),
            _make_subway_row(999999, "미등록역", "2026-01-01", "목", "하차", 10),
        ]
    )
    with pytest.raises(ValueError, match="일치하는 역번호가 0건"):
        validate_subway_raw_df(
            df_unknown,
            station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        )


def test_build_subway_interim_merges_csv_and_api_parquet_without_duplicates(
    tmp_path: Path,
) -> None:
    """
    기존 CSV와 API 수집 Parquet가 동일 디렉터리에 공존할 때
    build_subway_interim이 중복 없이 최신 API Parquet 값으로 통합하는지 검증합니다.
    """
    raw_dir = tmp_path / "raw_subway"
    raw_dir.mkdir()

    df_old = rows_to_subway_dataframe(
        [
            _make_subway_row(119, "서면", "2026-08-31", "월", "승차", 100),
            _make_subway_row(119, "서면", "2026-08-31", "월", "하차", 100),
        ]
    )
    df_old.write_csv(raw_dir / "부산교통공사_시간대별 승하차인원_20260831.csv")

    df_new = rows_to_subway_dataframe(
        [
            _make_subway_row(119, "서면", "2026-08-31", "월", "승차", 150),
            _make_subway_row(119, "서면", "2026-08-31", "월", "하차", 150),
            _make_subway_row(119, "서면", "2026-10-31", "토", "승차", 200),
            _make_subway_row(119, "서면", "2026-10-31", "토", "하차", 200),
        ]
    )
    df_new.write_parquet(raw_dir / "subway_ridership_2026.parquet")

    interim_out = tmp_path / "subway_interim.parquet"
    df_interim = build_subway_interim(raw_dir=raw_dir, output_path=interim_out)

    assert len(df_interim) == 48
    assert df_interim.select(["datetime", "station_id"]).n_unique() == 48

    aug_vals = df_interim.filter(pl.col("date") == date(2026, 8, 31))[
        "passengers_in"
    ].unique()
    assert aug_vals.to_list() == [150]
