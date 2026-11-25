"""
운영용 공공데이터 OpenAPI 수집 공통 유틸리티

제공 기능:
1. 공통 예외 계층: QuotaExceededError(트래픽 한도 초과), AuthenticationError(인증키 오류)
2. 재시도 및 백오프 기반 HTTP JSON 요청 유틸(request_json_with_retry)
3. 원자적(Atomic) 파일 쓰기 유틸(atomic_write_text, atomic_write_json, atomic_write_parquet)
4. 결정론적 콘텐츠 해시(canonical_sha256) 및 수집 워터마크/UUID 상태 관리(load/save_ingestion_state)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import requests

from busan_metro_congestion.config import RAW_DATA_DIR

DEFAULT_STATE_PATH = RAW_DATA_DIR / "_state" / "ingestion_state.json"


class QuotaExceededError(RuntimeError):
    """공공데이터포털 일일 트래픽 한도(HTTP 429 또는 오류코드 22) 도달 예외"""


class AuthenticationError(RuntimeError):
    """공공데이터포털 인증키(HTTP 401 또는 인증 실패) 오류 예외"""


def parse_json_response(resp: requests.Response) -> dict[str, Any]:
    """
    HTTP 응답을 JSON dict로 파싱하며, 공공데이터포털 XML 트래픽 초과/인증 오류를 감지합니다.
    """
    try:
        data = resp.json()
    except ValueError as exc:
        body_text = resp.text or ""
        if (
            "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR" in body_text
            or "<returnReasonCode>22</returnReasonCode>" in body_text
        ):
            raise QuotaExceededError(
                "공공데이터포털 일일 트래픽 한도 초과 (XML 오류코드 22)"
            ) from exc
        if (
            "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in body_text
            or "<returnReasonCode>30</returnReasonCode>" in body_text
            or "<returnReasonCode>31</returnReasonCode>" in body_text
        ):
            raise AuthenticationError(
                "공공데이터포털 인증키가 등록되지 않았거나 만료되었습니다."
            ) from exc
        snippet = body_text[:200].replace("\n", " ")
        raise RuntimeError(f"JSON 파싱 실패 (비정상 응답 본문): {snippet}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"예상하지 못한 JSON 응답 형식입니다: {type(data).__name__}")
    return data


def request_json_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    timeout: int = 25,
    backoff_sec: float = 1.0,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """
    HTTP GET 요청 후 JSON dict를 반환합니다.

    - HTTP 429 / XML 코드 22: 재시도 없이 즉시 QuotaExceededError 발생
    - HTTP 401 / XML 인증 오류: 재시도 없이 즉시 AuthenticationError 발생
    - 일시적 네트워크/5xx 오류: max_retries 회 지수 선형 백오프 재시도
    """
    getter = http_get if http_get is not None else requests.get
    attempts = max(1, int(max_retries))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            kwargs: dict[str, Any] = {"params": params, "timeout": timeout}
            if headers is not None:
                kwargs["headers"] = headers
            resp = getter(url, **kwargs)
            if resp.status_code == 429:
                raise QuotaExceededError(
                    "공공데이터포털 일일 트래픽 한도(HTTP 429 Too Many Requests)에 도달했습니다."
                )
            if resp.status_code == 401:
                raise AuthenticationError(
                    "공공데이터포털 API 인증에 실패했습니다(HTTP 401). PUBLIC_DATA_PORTAL_KEY를 확인하세요."
                )
            resp.raise_for_status()
            return parse_json_response(resp)
        except (QuotaExceededError, AuthenticationError):
            raise
        except Exception as exc:
            last_error = exc
            if attempt < attempts and backoff_sec > 0:
                time.sleep(backoff_sec * attempt)

    raise RuntimeError(
        f"HTTP 요청 실패 ({url}, {attempts}회 재시도 실패): {last_error}"
    )


def _make_temp_path(target_path: Path) -> Path:
    """동일 디렉터리 내 원자적 교체용 임시 파일 경로를 생성합니다."""
    return target_path.with_name(
        f".{target_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )


def atomic_write_text(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """임시 파일에 기록한 뒤 os.replace()로 대상 텍스트 파일을 원자적으로 교체합니다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _make_temp_path(target)
    try:
        tmp_path.write_text(content, encoding=encoding)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    """JSON 객체를 UTF-8 텍스트로 직렬화하여 원자적으로 저장합니다."""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return atomic_write_text(path, serialized, encoding="utf-8")


def atomic_write_parquet(df: pl.DataFrame, path: Path | str) -> Path:
    """Polars DataFrame을 임시 Parquet에 기록한 뒤 os.replace()로 원자적으로 교체합니다."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _make_temp_path(target)
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def canonical_sha256(obj: Any) -> str:
    """JSON 직렬화 가능한 객체의 결정론적 SHA-256 해시를 반환합니다."""
    canonical = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_ingestion_state(state_path: Path | str | None = None) -> dict[str, Any]:
    """수집 상태(워터마크·UUID·콘텐츠 해시) JSON을 로드합니다. 부재 시 빈 상태를 반환합니다."""
    path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    if not path.exists():
        return {"version": 1, "subway": {"years": {}}, "calendar": {"years": {}}}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"손상된 수집 상태 파일 형식입니다: {path}")

    data.setdefault("version", 1)
    data.setdefault("subway", {})
    data["subway"].setdefault("years", {})
    data.setdefault("calendar", {})
    data["calendar"].setdefault("years", {})
    return data


def save_ingestion_state(
    state: dict[str, Any],
    state_path: Path | str | None = None,
) -> Path:
    """수집 상태 JSON을 원자적으로 저장합니다."""
    path = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    return atomic_write_json(path, state)
