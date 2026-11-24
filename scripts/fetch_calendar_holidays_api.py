"""
한국천문연구원 특일 정보제공 서비스(SpcdeInfoService/getRestDeInfo) 연도별 공휴일 수집 스크립트

- 공식 활용가이드(v1.4) 명세 준수:
  - 엔드포인트: https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo
  - 요청 파라미터: ServiceKey, solYear, (선택) solMonth, _type=json, numOfRows=100, pageNo
- 방어 처리:
  1. 공휴일 0건 응답 시 items=""(빈 문자열) 정규화
  2. 단건 응답 시 items.item이 단일 dict로 반환되는 케이스 리스트 정규화
  3. 일일 트래픽 한도 초과(HTTP 429, resultCode=22, XML 에러 응답) 즉시 감지 및 중단
  4. 기완료 연도 파일 자동 스킵(Resume) 및 --force 강제 재수집 지원 (임시·대체공휴일 사후 반영)
- 출력: data/raw/calendar/holidays_{year}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import requests

from busan_metro_congestion.config import RAW_DATA_DIR, get_api_key

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_calendar_holidays_api")

KASI_REST_DE_URL = (
    "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"
)


class QuotaExceededError(RuntimeError):
    """공공데이터포털 일일 트래픽 한도(HTTP 429 또는 오류코드 22) 도달 예외"""


def extract_holiday_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    KASI getRestDeInfo JSON 응답에서 공휴일 item 리스트를 안전하게 추출합니다.

    - 공휴일이 0건이면 body.items가 빈 문자열("") 또는 None으로 반환되므로 빈 리스트를 반환합니다.
    - 공휴일이 1건이면 body.items.item이 단일 dict로 반환되므로 리스트로 감싸 반환합니다.
    """
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


def _parse_json_or_raise(resp: requests.Response) -> dict[str, Any]:
    """응답 본문을 JSON으로 파싱하며, 공공데이터포털 XML 오류 응답 시 명확한 예외를 발생시킵니다."""
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
        snippet = body_text[:200].replace("\n", " ")
        raise RuntimeError(f"JSON 파싱 실패 (비정상 응답 본문): {snippet}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"예상하지 못한 JSON 응답 형식입니다: {type(data).__name__}")
    return data


def fetch_holidays_for_year(
    api_key: str,
    year: int,
    sol_month: int | None = None,
    num_of_rows: int = 100,
    max_retries: int = 3,
    timeout: int = 25,
) -> dict[str, Any]:
    """
    지정된 연도(및 선택 월)의 대한민국 공휴일 목록을 KASI getRestDeInfo API로 수집합니다.
    """
    all_items: list[dict[str, Any]] = []
    page_no = 1
    total_count = 0

    while True:
        params: dict[str, Any] = {
            "ServiceKey": api_key,
            "solYear": str(year),
            "_type": "json",
            "numOfRows": num_of_rows,
            "pageNo": page_no,
        }
        if sol_month is not None:
            params["solMonth"] = f"{int(sol_month):02d}"

        payload: dict[str, Any] | None = None
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(KASI_REST_DE_URL, params=params, timeout=timeout)
                if resp.status_code == 429:
                    raise QuotaExceededError(
                        "공공데이터포털 일일 트래픽 한도(HTTP 429 Too Many Requests)에 도달했습니다."
                    )
                resp.raise_for_status()
                payload = _parse_json_or_raise(resp)

                header = payload.get("response", {}).get("header", {})
                code = str(header.get("resultCode", ""))
                msg = str(header.get("resultMsg", ""))

                if code == "00":
                    break
                if code in ("03", "NODATA_ERROR"):
                    payload = None
                    break
                if code in (
                    "22",
                    "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
                ):
                    raise QuotaExceededError(
                        f"공공데이터포털 일일 트래픽 한도 초과 ({code}): {msg}"
                    )
                raise RuntimeError(f"KASI OpenAPI 오류 ({code}): {msg}")

            except QuotaExceededError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)
        else:
            raise RuntimeError(
                f"KASI 공휴일 API 호출 실패 (year={year}, page={page_no}, {max_retries}회 재시도 실패): {last_error}"
            )

        if payload is None:
            break

        body = payload.get("response", {}).get("body", {})
        if page_no == 1:
            total_count = int(body.get("totalCount", 0) or 0)

        page_items = extract_holiday_items(payload)
        if not page_items:
            break

        all_items.extend(page_items)
        total_pages = max(1, math.ceil(total_count / num_of_rows)) if total_count else 1
        if page_no >= total_pages or len(all_items) >= total_count:
            break
        page_no += 1

    return {
        "response": {
            "header": {
                "resultCode": "00",
                "resultMsg": "NORMAL SERVICE.",
            },
            "body": {
                "items": {"item": all_items} if all_items else "",
                "numOfRows": num_of_rows,
                "pageNo": 1,
                "totalCount": len(all_items),
            },
        }
    }


def fetch_calendar_holidays(
    api_key: str,
    start_year: int = 2016,
    end_year: int = 2026,
    output_dir: Path | str | None = None,
    force: bool = False,
    delay: float = 0.2,
) -> list[Path]:
    """
    지정된 연도 구간의 공휴일 데이터를 연도별 JSON(holidays_{year}.json)으로 수집·저장합니다.
    """
    if start_year > end_year:
        raise ValueError(
            f"시작 연도({start_year})는 종료 연도({end_year})보다 클 수 없습니다."
        )

    out_dir = Path(output_dir) if output_dir is not None else RAW_DATA_DIR / "calendar"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    years = list(range(start_year, end_year + 1))

    for idx, year in enumerate(years):
        out_path = out_dir / f"holidays_{year}.json"
        if out_path.exists() and not force:
            logger.info(f"[스킵] 이미 수집된 연도 파일 존재: {out_path}")
            saved_files.append(out_path)
            continue

        data = fetch_holidays_for_year(api_key=api_key, year=year)
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total = data["response"]["body"]["totalCount"]
        logger.info(f"[저장] {year}년 공휴일 {total}건 -> {out_path}")
        saved_files.append(out_path)

        if delay > 0 and idx < len(years) - 1:
            time.sleep(delay)

    return saved_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="한국천문연구원 특일 정보(getRestDeInfo) 연도별 공휴일 수집기"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="수집 시작 연도 (기본값: 2016)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="수집 종료 연도 (기본값: 2026)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="저장 디렉터리 (기본값: data/raw/calendar)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 연도별 JSON 파일이 있어도 강제로 다시 수집하여 덮어씀",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="연도별 요청 간 대기 시간(초, 기본값: 0.2)",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    api_key = get_api_key()
    return fetch_calendar_holidays(
        api_key=api_key,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        force=args.force,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
