"""
운영용 공공데이터(지하철 신규 UUID 감지 · KASI 특일 해시 감지 · 달력 마스터 자동 갱신) 동기화 단위 테스트
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

import scripts.sync_operational_data as sync_mod
from busan_metro_congestion.config import REFERENCE_DATA_DIR
from scripts.fetch_subway_ridership_api import HOUR_SLOT_COLUMNS, ODCLOUD_OAS_URL
from scripts.sync_operational_data import (
    compute_holiday_signature,
    main,
    sync_calendar_holidays,
    sync_operational_data,
    sync_subway_ridership,
)


def _make_mock_response(
    payload: dict[str, Any],
    status_code: int = 200,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = json.dumps(payload, ensure_ascii=False)
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _make_kasi_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
            "body": {
                "items": {"item": items} if items else "",
                "numOfRows": 100,
                "pageNo": 1,
                "totalCount": len(items),
            },
        }
    }


def _make_subway_row(station_id: int, ymd: str, gubun: str, val: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "역번호": station_id,
        "역명": "서면",
        "년월일": ymd,
        "요일": "목",
        "구분": gubun,
        "합계": val * 24,
    }
    for col in HOUR_SLOT_COLUMNS:
        row[col] = val
    return row


def _make_oas_spec(entries: list[tuple[str, str]]) -> dict[str, Any]:
    props: dict[str, Any] = {
        "역번호": {"type": "integer"},
        "년월일": {"type": "string"},
        "구분": {"type": "string"},
    }
    paths: dict[str, Any] = {}
    defs: dict[str, Any] = {}
    for uddi, summary in entries:
        paths[f"/3057229/v1/{uddi}"] = {"get": {"summary": summary, "parameters": []}}
        defs[f"{uddi}_model"] = {"type": "object", "properties": props}
    return {"paths": paths, "definitions": defs}


def test_sync_subway_ridership_uuid_detection_and_in_place_update(
    tmp_path: Path,
) -> None:
    """
    sync_subway_ridership의 핵심 변경 감지 시나리오를 검증합니다:
    1) 최초 신규 UUID 다운로드
    2) 동일 UUID 재실행 스킵 (확정 연도 0콜, 미확정 연도 perPage=1 메타 1콜)
    3) 신규 UUID 등장 시 해당 연도만 다운로드·원자적 교체
    4) dry-run 시 상태/파일 무부작용
    """
    out_dir = tmp_path / "subway"
    state_path = tmp_path / "_state" / "ingestion_state.json"

    current_oas = _make_oas_spec(
        [
            ("uddi:2025-dec", "부산교통공사_시간대별 승하차인원_20251231"),
            ("uddi:2026-aug", "부산교통공사_시간대별 승하차인원_20260831"),
        ]
    )
    uddi_datasets: dict[str, list[dict[str, Any]]] = {
        "uddi:2025-dec": [
            _make_subway_row(119, "2025-12-31", "승차", 120),
            _make_subway_row(119, "2025-12-31", "하차", 110),
        ],
        "uddi:2026-aug": [
            _make_subway_row(119, "2026-08-31", "승차", 150),
            _make_subway_row(119, "2026-08-31", "하차", 140),
            _make_subway_row(227, "2026-08-31", "승차", 90),
            _make_subway_row(227, "2026-08-31", "하차", 80),
        ],
        "uddi:2026-oct": [
            _make_subway_row(119, "2026-08-31", "승차", 155),
            _make_subway_row(119, "2026-08-31", "하차", 145),
            _make_subway_row(119, "2026-10-31", "승차", 210),
            _make_subway_row(119, "2026-10-31", "하차", 200),
        ],
    }
    call_log: list[tuple[str, int, int]] = []

    def fake_get(
        url: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> MagicMock:
        params = params or {}
        if url == ODCLOUD_OAS_URL:
            return _make_mock_response(current_oas)
        assert params.get("serviceKey") == "mock-key"
        uddi = url.rstrip("/").split("/")[-1]
        page = int(params.get("page", 1))
        per_page = int(params.get("perPage", 1000))
        call_log.append((uddi, page, per_page))

        rows = uddi_datasets[uddi]
        start = (page - 1) * per_page
        chunk = rows[start : start + per_page]
        return _make_mock_response(
            {
                "page": page,
                "perPage": per_page,
                "totalCount": len(rows),
                "currentCount": len(chunk),
                "matchCount": len(rows),
                "data": chunk,
            }
        )

    # 1. 최초 실행: 2025, 2026 신규 UUID 다운로드
    res1 = sync_subway_ridership(
        api_key="mock-key",
        start_year=2025,
        output_dir=out_dir,
        state_path=state_path,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        per_page=2,
        delay_sec=0.0,
        http_get=fake_get,
    )
    assert res1["changed"] is True
    assert [d["year"] for d in res1["downloaded"]] == [2025, 2026]
    assert (out_dir / "subway_ridership_2025.parquet").exists()
    assert (out_dir / "subway_ridership_2026.parquet").exists()

    # 2. 동일 상태 재실행: 2025는 0콜 스킵, 2026은 perPage=1 메타 확인 후 스킵
    call_log.clear()
    res2 = sync_subway_ridership(
        api_key="mock-key",
        start_year=2025,
        output_dir=out_dir,
        state_path=state_path,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        per_page=2,
        delay_sec=0.0,
        http_get=fake_get,
    )
    assert res2["changed"] is False
    assert [s["year"] for s in res2["skipped"]] == [2025, 2026]
    assert call_log == [("uddi:2026-aug", 1, 1)]

    # 3. 2026 신규 UUID 등장 -> 2025 스킵, 2026만 자동 다운로드 및 상태 갱신
    current_oas = _make_oas_spec(
        [
            ("uddi:2025-dec", "부산교통공사_시간대별 승하차인원_20251231"),
            ("uddi:2026-aug", "부산교통공사_시간대별 승하차인원_20260831"),
            ("uddi:2026-oct", "부산교통공사_시간대별 승하차인원_20261031"),
        ]
    )
    res3 = sync_subway_ridership(
        api_key="mock-key",
        start_year=2025,
        output_dir=out_dir,
        state_path=state_path,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        per_page=2,
        delay_sec=0.0,
        http_get=fake_get,
    )
    assert res3["changed"] is True
    assert [d["year"] for d in res3["downloaded"]] == [2026]
    assert res3["downloaded"][0]["uddi"] == "uddi:2026-oct"
    assert res3["downloaded"][0]["max_date"] == "2026-10-31"
    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["subway"]["years"]["2026"]["uddi"] == "uddi:2026-oct"

    # 4. Dry-run
    res4 = sync_subway_ridership(
        api_key="mock-key",
        start_year=2025,
        output_dir=out_dir,
        state_path=state_path,
        force=True,
        dry_run=True,
        http_get=fake_get,
    )
    assert res4["dry_run"] is True
    assert res4["changed"] is False
    assert [p["year"] for p in res4["planned"]] == [2025, 2026]


def test_sync_operational_data_end_to_end_incremental_and_holiday_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    통합 운영 동기화 CLI의 3단계 시나리오를 오프라인 모킹으로 검증합니다:
    1) 최초 실행: 지하철 신규 UUID 수집 + KASI 특일 수집 + calendar_master 자동 확장 빌드
    2) 무변경 재실행: 확정 연도·동일 UUID 스킵, 공휴일 해시 동일 시 마스터 재빌드 생략
    3) 임시공휴일 추가: 해시 변경 감지 -> JSON 교체 -> calendar_master 자동 재빌드 반영
    """
    subway_dir = tmp_path / "raw" / "subway"
    cal_dir = tmp_path / "raw" / "calendar"
    cal_master = tmp_path / "reference" / "calendar_master.parquet"
    state_path = tmp_path / "raw" / "_state" / "ingestion_state.json"
    manifest_path = tmp_path / "manifest.json"

    kasi_store: dict[int, list[dict[str, Any]]] = {
        2025: [
            {
                "dateKind": "01",
                "dateName": "신정",
                "isHoliday": "Y",
                "locdate": 20250101,
                "seq": 1,
            }
        ],
        2026: [
            {
                "dateKind": "01",
                "dateName": "신정",
                "isHoliday": "Y",
                "locdate": 20260101,
                "seq": 1,
            }
        ],
        2027: [
            {
                "dateKind": "01",
                "dateName": "삼일절",
                "isHoliday": "Y",
                "locdate": 20270301,
                "seq": 1,
            }
        ],
        2028: [],
    }
    kasi_called_years: list[int] = []

    def fake_fetch_kasi(api_key: str, year: int, **kwargs: Any) -> dict[str, Any]:
        kasi_called_years.append(year)
        return _make_kasi_payload(kasi_store.get(year, []))

    monkeypatch.setattr(sync_mod, "fetch_holidays_for_year", fake_fetch_kasi)

    oas_spec = _make_oas_spec(
        [("uddi:2026-oct", "부산교통공사_시간대별 승하차인원_20261031")]
    )
    subway_rows = [
        _make_subway_row(119, "2026-10-31", "승차", 120),
        _make_subway_row(119, "2026-10-31", "하차", 110),
    ]

    def fake_subway_get(
        url: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> MagicMock:
        if url == ODCLOUD_OAS_URL:
            return _make_mock_response(oas_spec)
        return _make_mock_response(
            {
                "page": 1,
                "perPage": 1000,
                "totalCount": len(subway_rows),
                "currentCount": len(subway_rows),
                "matchCount": len(subway_rows),
                "data": subway_rows,
            }
        )

    # 1. 최초 실행 (기준일 2026-11-25 -> 달력 2025~2028 자동 확장)
    m1 = sync_operational_data(
        api_key="test-key",
        target="all",
        mode="incremental",
        subway_start_year=2026,
        calendar_start_year=2025,
        reference_today=date(2026, 11, 25),
        subway_output_dir=subway_dir,
        calendar_output_dir=cal_dir,
        calendar_master_path=cal_master,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        state_path=state_path,
        manifest_path=manifest_path,
        delay=0.0,
        http_get=fake_subway_get,
    )
    assert m1["status"] == "ok"
    assert m1["any_data_changed"] is True
    assert m1["calendar"]["master_rebuilt"] is True
    assert m1["calendar"]["changed_years"] == [2025, 2026, 2027, 2028]
    assert m1["subway"]["changed"] is True
    assert cal_master.exists()
    assert manifest_path.exists()

    df_master_1 = pl.read_parquet(cal_master)
    assert df_master_1["date"].min() == date(2025, 1, 1)
    assert df_master_1["date"].max() == date(2028, 12, 31)

    # 2. 무변경 재실행
    kasi_called_years.clear()
    m2 = sync_operational_data(
        api_key="test-key",
        target="all",
        mode="incremental",
        subway_start_year=2026,
        calendar_start_year=2025,
        reference_today=date(2026, 11, 25),
        subway_output_dir=subway_dir,
        calendar_output_dir=cal_dir,
        calendar_master_path=cal_master,
        station_master_path=REFERENCE_DATA_DIR / "station_master.parquet",
        state_path=state_path,
        delay=0.0,
        http_get=fake_subway_get,
    )
    assert kasi_called_years == [2026, 2027, 2028]
    assert m2["calendar"]["skipped_years"] == [2025]
    assert m2["calendar"]["changed"] is False
    assert m2["calendar"]["master_rebuilt"] is False
    assert m2["subway"]["changed"] is False
    assert m2["any_data_changed"] is False

    # 3. 2026년 임시공휴일(2026-10-02) 사후 지정 -> 자동 감지 및 calendar_master 반영
    kasi_store[2026].append(
        {
            "dateKind": "01",
            "dateName": "임시공휴일",
            "isHoliday": "Y",
            "locdate": 20261002,
            "seq": 2,
        }
    )
    m3 = sync_operational_data(
        api_key="test-key",
        target="calendar",
        mode="incremental",
        calendar_start_year=2025,
        reference_today=date(2026, 11, 25),
        calendar_output_dir=cal_dir,
        calendar_master_path=cal_master,
        state_path=state_path,
        delay=0.0,
    )
    assert m3["calendar"]["changed_years"] == [2026]
    assert m3["calendar"]["master_rebuilt"] is True
    assert m3["any_data_changed"] is True

    df_master_3 = pl.read_parquet(cal_master)
    assert (
        df_master_3.filter(pl.col("date") == date(2026, 10, 2))["is_holiday"][0] is True
    )
    assert (
        df_master_3.filter(pl.col("date") == date(2026, 10, 1))["is_holiday_eve"][0]
        is True
    )


def test_sync_operational_data_backfill_dry_run_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """backfill 모드, dry-run 무부작용, 잘못된 target/mode 예외 및 CLI main()을 검증합니다."""
    with pytest.raises(ValueError, match="지원하지 않는 target"):
        sync_operational_data(api_key="k", target="weather")

    with pytest.raises(ValueError, match="지원하지 않는 mode"):
        sync_operational_data(api_key="k", mode="streaming")

    cal_dir = tmp_path / "cal"
    cal_master = tmp_path / "calendar_master.parquet"
    state_path = tmp_path / "state.json"

    monkeypatch.setattr(sync_mod, "get_api_key", lambda: "mock-key")
    monkeypatch.setattr(
        sync_mod,
        "fetch_holidays_for_year",
        lambda api_key, year, **kwargs: _make_kasi_payload(
            [
                {
                    "dateKind": "01",
                    "dateName": "신정",
                    "isHoliday": "Y",
                    "locdate": int(f"{year}0101"),
                    "seq": 1,
                }
            ]
        ),
    )

    dry_res = main(
        [
            "--target",
            "calendar",
            "--mode",
            "backfill",
            "--start-year",
            "2024",
            "--end-year",
            "2025",
            "--calendar-dir",
            str(cal_dir),
            "--calendar-master",
            str(cal_master),
            "--state-path",
            str(state_path),
            "--delay",
            "0",
            "--dry-run",
        ]
    )
    assert dry_res["dry_run"] is True
    assert dry_res["calendar"]["planned_years"] == [2024, 2025]
    assert not cal_master.exists()
    assert not state_path.exists()

    backfill_res = main(
        [
            "--target",
            "calendar",
            "--mode",
            "backfill",
            "--start-year",
            "2024",
            "--end-year",
            "2025",
            "--calendar-dir",
            str(cal_dir),
            "--calendar-master",
            str(cal_master),
            "--state-path",
            str(state_path),
            "--delay",
            "0",
        ]
    )
    assert backfill_res["calendar"]["fetched_years"] == [2024, 2025]
    assert backfill_res["calendar"]["master_rebuilt"] is True
    assert cal_master.exists()
    df_bf = pl.read_parquet(cal_master)
    assert df_bf["date"].min() == date(2024, 1, 1)
    assert df_bf["date"].max() == date(2025, 12, 31)


def test_compute_holiday_signature_normalization_and_change_detection() -> None:
    """공휴일 해시가 항목 순서·공백·대소문자·seq에 불변이며 실질 변경(명칭·여부·추가)을 감지하는지 검증합니다."""
    payload_a = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "dateKind": "01",
                            "dateName": "삼일절",
                            "isHoliday": "Y",
                            "locdate": 20260301,
                            "seq": 1,
                        },
                        {
                            "dateKind": "01",
                            "dateName": "신정",
                            "isHoliday": "Y",
                            "locdate": 20260101,
                            "seq": 1,
                        },
                    ]
                }
            }
        }
    }
    payload_reordered = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "dateKind": " 01 ",
                            "dateName": " 신정 ",
                            "isHoliday": "y",
                            "locdate": "20260101",
                            "seq": 99,
                        },
                        {
                            "dateKind": "01",
                            "dateName": "삼일절",
                            "isHoliday": "Y",
                            "locdate": 20260301,
                            "seq": 2,
                        },
                    ]
                }
            }
        }
    }
    payload_status_changed = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "dateKind": "01",
                            "dateName": "삼일절",
                            "isHoliday": "N",
                            "locdate": 20260301,
                            "seq": 1,
                        },
                        {
                            "dateKind": "01",
                            "dateName": "신정",
                            "isHoliday": "Y",
                            "locdate": 20260101,
                            "seq": 1,
                        },
                    ]
                }
            }
        }
    }
    payload_name_changed = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "dateKind": "01",
                            "dateName": "3·1절",
                            "isHoliday": "Y",
                            "locdate": 20260301,
                            "seq": 1,
                        },
                        {
                            "dateKind": "01",
                            "dateName": "신정",
                            "isHoliday": "Y",
                            "locdate": 20260101,
                            "seq": 1,
                        },
                    ]
                }
            }
        }
    }

    sig_a = compute_holiday_signature(payload_a)
    assert sig_a == compute_holiday_signature(payload_reordered)
    assert sig_a != compute_holiday_signature(payload_status_changed)
    assert sig_a != compute_holiday_signature(payload_name_changed)


def test_sync_calendar_holidays_dry_run_force_unchanged_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """sync_calendar_holidays의 연도 역전, dry-run 무생성, force 시 무변경 미기록, 혼합 스킵/계획을 검증합니다."""
    with pytest.raises(ValueError, match="시작 연도"):
        sync_calendar_holidays(
            api_key="k", start_year=2027, end_year=2026, output_dir=tmp_path
        )

    out_dir = tmp_path / "cal_sync"
    fetched_years: list[int] = []
    store: dict[int, list[dict[str, Any]]] = {
        2025: [
            {
                "dateKind": "01",
                "dateName": "신정",
                "isHoliday": "Y",
                "locdate": 20250101,
                "seq": 1,
            }
        ],
        2026: [
            {
                "dateKind": "01",
                "dateName": "신정",
                "isHoliday": "Y",
                "locdate": 20260101,
                "seq": 1,
            }
        ],
    }

    def fake_fetch(api_key: str, year: int, **kwargs: Any) -> dict[str, Any]:
        fetched_years.append(year)
        return _make_kasi_payload(store[year])

    monkeypatch.setattr(sync_mod, "fetch_holidays_for_year", fake_fetch)

    dry1 = sync_calendar_holidays(
        api_key="k",
        start_year=2025,
        end_year=2026,
        output_dir=out_dir,
        dry_run=True,
        delay=0.0,
    )
    assert not out_dir.exists()
    assert fetched_years == []
    assert dry1["planned_years"] == [2025, 2026]
    assert dry1["changed"] is False

    s1 = sync_calendar_holidays(
        api_key="k",
        start_year=2025,
        end_year=2026,
        output_dir=out_dir,
        delay=0.0,
    )
    assert s1["changed"] is True
    assert s1["changed_years"] == [2025, 2026]
    assert s1["unchanged_years"] == []

    writes: list[Path] = []
    orig_write = sync_mod.atomic_write_json

    def spy_write(path: Path | str, payload: Any) -> Path:
        writes.append(Path(path))
        return orig_write(path, payload)

    monkeypatch.setattr(sync_mod, "atomic_write_json", spy_write)
    fetched_years.clear()

    s2 = sync_calendar_holidays(
        api_key="k",
        start_year=2025,
        end_year=2026,
        output_dir=out_dir,
        force=True,
        delay=0.0,
    )
    assert fetched_years == [2025, 2026]
    assert s2["unchanged_years"] == [2025, 2026]
    assert s2["changed_years"] == []
    assert s2["changed"] is False
    assert writes == []

    fetched_years.clear()
    dry2 = sync_calendar_holidays(
        api_key="k",
        start_year=2025,
        end_year=2026,
        output_dir=out_dir,
        refresh_years={2026},
        dry_run=True,
        delay=0.0,
    )
    assert fetched_years == []
    assert dry2["skipped_years"] == [2025]
    assert dry2["planned_years"] == [2026]
    assert "2025" in dry2["year_signatures"]
