"""
한국천문연구원 특일 정보(getRestDeInfo) 공휴일 수집기 단위 테스트
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import scripts.fetch_calendar_holidays_api as fetcher_mod
from scripts.fetch_calendar_holidays_api import (
    QuotaExceededError,
    extract_holiday_items,
    fetch_calendar_holidays,
    fetch_holidays_for_year,
    main,
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


def test_extract_holiday_items_edge_cases() -> None:
    """빈 문자열 items, 단건 dict, 복수 list 등 KASI 응답 변형을 안전하게 정규화하는지 검증합니다."""
    # 1. 공휴일 0건: items=""
    empty_payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
            "body": {"items": "", "totalCount": 0},
        }
    }
    assert extract_holiday_items(empty_payload) == []

    # 2. 단건 응답: item이 단일 dict
    single_payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
            "body": {
                "items": {
                    "item": {
                        "dateKind": "01",
                        "dateName": "삼일절",
                        "isHoliday": "Y",
                        "locdate": 20250301,
                        "seq": 1,
                    }
                },
                "totalCount": 1,
            },
        }
    }
    single_items = extract_holiday_items(single_payload)
    assert len(single_items) == 1
    assert single_items[0]["locdate"] == 20250301

    # 3. 복수 응답: 중복 locdate 및 isHoliday="N" 포함
    multi_payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
            "body": {
                "items": {
                    "item": [
                        {"dateName": "어린이날", "isHoliday": "Y", "locdate": 20250505},
                        {
                            "dateName": "부처님오신날",
                            "isHoliday": "Y",
                            "locdate": 20250505,
                        },
                        {"dateName": "제헌절", "isHoliday": "N", "locdate": 20250717},
                    ]
                },
                "totalCount": 3,
            },
        }
    }
    multi_items = extract_holiday_items(multi_payload)
    assert len(multi_items) == 3


def test_fetch_holidays_for_year_params_and_single_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """요청 파라미터(ServiceKey, solYear, _type=json, numOfRows) 및 단건 dict 정규화를 검증합니다."""
    captured: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> MagicMock:
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _make_mock_response(
            payload={
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                    "body": {
                        "items": {
                            "item": {
                                "dateKind": "01",
                                "dateName": "1월1일",
                                "isHoliday": "Y",
                                "locdate": 20250101,
                                "seq": 1,
                            }
                        },
                        "numOfRows": 100,
                        "pageNo": 1,
                        "totalCount": 1,
                    },
                }
            }
        )

    monkeypatch.setattr(fetcher_mod.requests, "get", fake_get)

    res = fetch_holidays_for_year(api_key="test-key", year=2025)
    assert captured["params"]["ServiceKey"] == "test-key"
    assert captured["params"]["solYear"] == "2025"
    assert captured["params"]["_type"] == "json"
    assert captured["params"]["numOfRows"] == 100
    assert "solMonth" not in captured["params"]

    items = res["response"]["body"]["items"]["item"]
    assert isinstance(items, list)
    assert len(items) == 1
    assert res["response"]["body"]["totalCount"] == 1


def test_fetch_holidays_for_year_empty_and_nodata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """items='' 또는 resultCode='03'(NODATA) 응답을 오류 없이 0건으로 처리하는지 검증합니다."""
    monkeypatch.setattr(
        fetcher_mod.requests,
        "get",
        lambda *args, **kwargs: _make_mock_response(
            payload={
                "response": {
                    "header": {"resultCode": "03", "resultMsg": "NODATA_ERROR"},
                    "body": {"items": "", "totalCount": 0},
                }
            }
        ),
    )

    res = fetch_holidays_for_year(api_key="test-key", year=2030)
    assert res["response"]["body"]["totalCount"] == 0
    assert res["response"]["body"]["items"] == ""


@pytest.mark.parametrize(
    "mock_resp",
    [
        _make_mock_response(payload={}, status_code=429),
        _make_mock_response(
            payload={
                "response": {
                    "header": {
                        "resultCode": "22",
                        "resultMsg": "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
                    }
                }
            }
        ),
        _make_mock_response(
            payload=None,
            text="<OpenAPI_ServiceResponse><cmmMsgHeader><returnReasonCode>22</returnReasonCode></cmmMsgHeader></OpenAPI_ServiceResponse>",
        ),
    ],
)
def test_fetch_holidays_quota_exceeded_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    mock_resp: MagicMock,
) -> None:
    """HTTP 429, JSON 코드 22, XML 오류코드 22 응답 시 재시도 없이 즉시 QuotaExceededError가 발생하는지 검증합니다."""
    call_count = 0

    def fake_get(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return mock_resp

    monkeypatch.setattr(fetcher_mod.requests, "get", fake_get)

    with pytest.raises(QuotaExceededError):
        fetch_holidays_for_year(api_key="test-key", year=2025, max_retries=3)

    assert call_count == 1


def test_fetch_calendar_holidays_resume_and_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """기완료 연도 파일 스킵(Resume) 및 force=True 덮어쓰기 동작을 검증합니다."""
    called_years: list[int] = []

    def fake_fetch(api_key: str, year: int, **kwargs: Any) -> dict[str, Any]:
        called_years.append(year)
        return {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                "body": {
                    "items": {
                        "item": [
                            {
                                "dateKind": "01",
                                "dateName": "신정",
                                "isHoliday": "Y",
                                "locdate": int(f"{year}0101"),
                                "seq": 1,
                            }
                        ]
                    },
                    "numOfRows": 100,
                    "pageNo": 1,
                    "totalCount": 1,
                },
            }
        }

    monkeypatch.setattr(fetcher_mod, "fetch_holidays_for_year", fake_fetch)

    # 1. 최초 수집: 2024, 2025 모두 호출
    files1 = fetch_calendar_holidays(
        api_key="test-key",
        start_year=2024,
        end_year=2025,
        output_dir=tmp_path,
        delay=0.0,
    )
    assert called_years == [2024, 2025]
    assert len(files1) == 2
    assert (tmp_path / "holidays_2024.json").exists()
    assert (tmp_path / "holidays_2025.json").exists()

    # 2. 재실행(force=False): 기존 연도는 모두 스킵
    called_years.clear()
    fetch_calendar_holidays(
        api_key="test-key",
        start_year=2024,
        end_year=2025,
        output_dir=tmp_path,
        force=False,
        delay=0.0,
    )
    assert called_years == []

    # 3. 강제 재수집(force=True): 전 연도 재호출 및 덮어쓰기
    fetch_calendar_holidays(
        api_key="test-key",
        start_year=2024,
        end_year=2025,
        output_dir=tmp_path,
        force=True,
        delay=0.0,
    )
    assert called_years == [2024, 2025]
    saved = json.loads((tmp_path / "holidays_2025.json").read_text(encoding="utf-8"))
    assert saved["response"]["body"]["totalCount"] == 1


def test_fetch_calendar_holidays_invalid_year_range_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """연도 역전 예외 및 CLI main() 엔트리포인트 동작을 검증합니다."""
    with pytest.raises(ValueError, match="시작 연도"):
        fetch_calendar_holidays(
            api_key="test-key",
            start_year=2026,
            end_year=2025,
            output_dir=tmp_path,
        )

    monkeypatch.setattr(fetcher_mod, "get_api_key", lambda: "mock-key")
    monkeypatch.setattr(
        fetcher_mod,
        "fetch_holidays_for_year",
        lambda api_key, year, **kwargs: {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                "body": {"items": "", "numOfRows": 100, "pageNo": 1, "totalCount": 0},
            }
        },
    )

    result_files = main(
        [
            "--start-year",
            "2025",
            "--end-year",
            "2025",
            "--output-dir",
            str(tmp_path),
            "--delay",
            "0",
        ]
    )
    assert result_files == [tmp_path / "holidays_2025.json"]
    assert result_files[0].exists()
