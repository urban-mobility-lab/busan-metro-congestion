"""
운영용 공공데이터 OpenAPI 수집 공통 유틸리티(scripts/ingestion_utils.py) 단위 테스트
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
import requests

import scripts.ingestion_utils as utils_mod
from scripts.ingestion_utils import (
    AuthenticationError,
    QuotaExceededError,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    canonical_sha256,
    load_ingestion_state,
    parse_json_response,
    request_json_with_retry,
    save_ingestion_state,
)


def _make_response(
    payload: Any = None,
    status_code: int = 200,
    text: str = "",
    json_error: Exception | None = None,
    http_error: Exception | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_error is not None:
        resp.json.side_effect = json_error
    else:
        resp.json.return_value = payload
    if http_error is not None:
        resp.raise_for_status.side_effect = http_error
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.mark.parametrize(
    ("xml_body", "expected_exc"),
    [
        (
            "<OpenAPI_ServiceResponse><cmmMsgHeader>"
            "<errMsg>SERVICE ERROR</errMsg>"
            "<returnAuthMsg>LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR</returnAuthMsg>"
            "<returnReasonCode>22</returnReasonCode>"
            "</cmmMsgHeader></OpenAPI_ServiceResponse>",
            QuotaExceededError,
        ),
        (
            "<OpenAPI_ServiceResponse><cmmMsgHeader>"
            "<returnReasonCode>22</returnReasonCode>"
            "</cmmMsgHeader></OpenAPI_ServiceResponse>",
            QuotaExceededError,
        ),
        (
            "<OpenAPI_ServiceResponse><cmmMsgHeader>"
            "<returnAuthMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</returnAuthMsg>"
            "<returnReasonCode>30</returnReasonCode>"
            "</cmmMsgHeader></OpenAPI_ServiceResponse>",
            AuthenticationError,
        ),
        (
            "<OpenAPI_ServiceResponse><cmmMsgHeader>"
            "<returnReasonCode>31</returnReasonCode>"
            "</cmmMsgHeader></OpenAPI_ServiceResponse>",
            AuthenticationError,
        ),
    ],
)
def test_parse_json_response_detects_xml_quota_and_auth_errors(
    xml_body: str,
    expected_exc: type[Exception],
) -> None:
    """XML 오류 본문(코드 22/30/31)을 QuotaExceededError/AuthenticationError로 정확히 분류하는지 검증합니다."""
    resp = _make_response(
        text=xml_body,
        json_error=ValueError("Not JSON"),
    )
    with pytest.raises(expected_exc):
        parse_json_response(resp)


def test_parse_json_response_rejects_html_and_non_dict_json() -> None:
    """일반 비JSON(HTML) 본문과 dict가 아닌 JSON 응답을 RuntimeError로 거부하는지 검증합니다."""
    html_resp = _make_response(
        text="<html><body>502 Bad Gateway</body></html>",
        json_error=ValueError("Invalid JSON"),
    )
    with pytest.raises(RuntimeError, match="JSON 파싱 실패"):
        parse_json_response(html_resp)

    list_resp = _make_response(payload=[1, 2, 3])
    with pytest.raises(RuntimeError, match="예상하지 못한 JSON 응답 형식"):
        parse_json_response(list_resp)


def test_request_json_with_retry_recovers_after_transient_failures() -> None:
    """일시적 네트워크 오류 및 HTTP 5xx 발생 시 재시도 후 정상 응답을 복구하는지 검증합니다."""
    attempts = 0
    captured_kwargs: list[dict[str, Any]] = []

    def flaky_get(url: str, **kwargs: Any) -> MagicMock:
        nonlocal attempts
        attempts += 1
        captured_kwargs.append(kwargs)
        if attempts == 1:
            raise requests.ConnectionError("temporary network glitch")
        if attempts == 2:
            return _make_response(
                status_code=503,
                http_error=requests.HTTPError("503 Service Unavailable"),
            )
        return _make_response(payload={"ok": True, "attempt": attempts})

    result = request_json_with_retry(
        url="https://example.test/api",
        params={"page": 1},
        headers={"X-Trace": "t-1"},
        max_retries=3,
        timeout=10,
        backoff_sec=0.0,
        http_get=flaky_get,
    )
    assert result == {"ok": True, "attempt": 3}
    assert attempts == 3
    assert captured_kwargs[0] == {
        "params": {"page": 1},
        "timeout": 10,
        "headers": {"X-Trace": "t-1"},
    }


def test_request_json_with_retry_raises_after_exhausting_attempts() -> None:
    """최대 재시도 횟수를 모두 소진하면 RuntimeError를 발생시키는지 검증합니다."""
    attempts = 0

    def always_timeout(url: str, **kwargs: Any) -> MagicMock:
        nonlocal attempts
        attempts += 1
        raise requests.Timeout("read timed out")

    with pytest.raises(RuntimeError, match="3회 재시도 실패"):
        request_json_with_retry(
            url="https://example.test/slow",
            max_retries=3,
            backoff_sec=0.0,
            http_get=always_timeout,
        )
    assert attempts == 3


@pytest.mark.parametrize(
    ("xml_body", "expected_exc"),
    [
        ("<returnReasonCode>22</returnReasonCode>", QuotaExceededError),
        ("<returnReasonCode>30</returnReasonCode>", AuthenticationError),
    ],
)
def test_request_json_with_retry_fails_fast_on_xml_quota_and_auth(
    xml_body: str,
    expected_exc: type[Exception],
) -> None:
    """HTTP 200 본문에 XML 트래픽 초과/인증 오류가 담긴 경우에도 재시도 없이 즉시 실패하는지 검증합니다."""
    attempts = 0

    def xml_error_get(url: str, **kwargs: Any) -> MagicMock:
        nonlocal attempts
        attempts += 1
        return _make_response(
            status_code=200,
            text=xml_body,
            json_error=ValueError("XML body"),
        )

    with pytest.raises(expected_exc):
        request_json_with_retry(
            url="https://example.test/xml",
            max_retries=3,
            backoff_sec=0.0,
            http_get=xml_error_get,
        )
    assert attempts == 1


def test_atomic_write_text_and_json_preserve_existing_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """원자적 텍스트/JSON 교체 중 예외 발생 시 기존 파일이 유지되고 임시 파일이 정리되는지 검증합니다."""
    target = tmp_path / "data.json"
    atomic_write_json(target, {"version": 1, "name": "부산"})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "version": 1,
        "name": "부산",
    }
    assert list(tmp_path.glob("*.tmp.*")) == []

    def failing_replace(src: Path | str, dst: Path | str) -> None:
        raise OSError("simulated disk error during os.replace")

    monkeypatch.setattr(utils_mod.os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated disk error"):
        atomic_write_text(target, "CORRUPTED")

    # 기존 원본 파일 보존 및 임시 파일 정리 확인
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "version": 1,
        "name": "부산",
    }
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_parquet_preserves_existing_and_cleans_temp_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """원자적 Parquet 교체 실패 시 기존 Parquet이 손상되지 않고 임시 파일이 삭제되는지 검증합니다."""
    target = tmp_path / "sample.parquet"
    df_v1 = pl.DataFrame({"station_id": [119, 227], "passengers": [100, 200]})
    atomic_write_parquet(df_v1, target)
    assert pl.read_parquet(target).to_dict(as_series=False) == {
        "station_id": [119, 227],
        "passengers": [100, 200],
    }
    assert list(tmp_path.glob("*.tmp.*")) == []

    def failing_replace(src: Path | str, dst: Path | str) -> None:
        raise OSError("simulated parquet replace failure")

    monkeypatch.setattr(utils_mod.os, "replace", failing_replace)

    df_v2 = pl.DataFrame({"station_id": [999], "passengers": [0]})
    with pytest.raises(OSError, match="simulated parquet replace failure"):
        atomic_write_parquet(df_v2, target)

    assert pl.read_parquet(target).to_dict(as_series=False) == {
        "station_id": [119, 227],
        "passengers": [100, 200],
    }
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_canonical_sha256_is_deterministic_across_key_order() -> None:
    """dict 키 순서가 달라도 동일 SHA-256을 산출하고 값 변경 시 해시가 달라지는지 검증합니다."""
    obj_a = {"b": [1, 2], "a": {"이름": "서면", "code": 119}}
    obj_b = {"a": {"code": 119, "이름": "서면"}, "b": [1, 2]}
    obj_c = {"a": {"code": 119, "이름": "사상"}, "b": [1, 2]}

    assert canonical_sha256(obj_a) == canonical_sha256(obj_b)
    assert canonical_sha256(obj_a) != canonical_sha256(obj_c)


def test_load_and_save_ingestion_state_defaults_normalization_and_corruption(
    tmp_path: Path,
) -> None:
    """상태 파일 부재, 부분 누락 스키마 정규화, 저장/복원, 손상 JSON 거부 동작을 검증합니다."""
    state_path = tmp_path / "_state" / "ingestion_state.json"

    # 1. 부재 시 기본 구조 반환
    empty = load_ingestion_state(state_path)
    assert empty == {"version": 1, "subway": {"years": {}}, "calendar": {"years": {}}}

    # 2. 일부 키가 누락된 부분 상태 파일 정규화
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"subway": {}}), encoding="utf-8")
    normalized = load_ingestion_state(state_path)
    assert normalized["version"] == 1
    assert normalized["subway"]["years"] == {}
    assert normalized["calendar"]["years"] == {}

    # 3. 저장 및 재로드 정합성
    normalized["subway"]["years"]["2026"] = {"uddi": "uddi:2026-oct"}
    save_ingestion_state(normalized, state_path)
    reloaded = load_ingestion_state(state_path)
    assert reloaded["subway"]["years"]["2026"]["uddi"] == "uddi:2026-oct"

    # 4. dict가 아닌 손상 JSON 거부
    state_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    with pytest.raises(ValueError, match="손상된 수집 상태 파일"):
        load_ingestion_state(state_path)
