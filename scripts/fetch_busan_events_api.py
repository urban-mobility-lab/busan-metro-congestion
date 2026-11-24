"""
부산광역시 공공데이터 오픈API(부산축제정보 서비스 및 전시 목록 서비스) 일괄 수집 스크립트

출력:
- data/raw/festival/busan_festival_kr.json (FestivalService/getFestivalKr)
- data/raw/festival/busan_culture_exhibit.json (BusanCultureExhibitService/getBusanCultureExhibit)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import requests

from busan_metro_congestion.config import RAW_DATA_DIR, get_api_key

FESTIVAL_API_URL = "https://apis.data.go.kr/6260000/FestivalService/getFestivalKr"
EXHIBIT_API_URL = (
    "https://apis.data.go.kr/6260000/BusanCultureExhibitService/getBusanCultureExhibit"
)


def fetch_paginated_items(
    url: str,
    root_key: str,
    api_key: str,
    num_of_rows: int = 1000,
    timeout: int = 25,
) -> dict[str, Any]:
    """
    공공데이터포털 JSON OpenAPI를 페이지네이션하여 전체 항목(item)을 수집합니다.
    """
    all_items: list[dict[str, Any]] = []
    page_no = 1
    total_count = 0

    while True:
        resp = requests.get(
            url,
            params={
                "serviceKey": api_key,
                "numOfRows": num_of_rows,
                "pageNo": page_no,
                "resultType": "json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        body = payload.get(root_key, {})

        header = body.get("header", {})
        code = str(header.get("code", "00"))
        if code != "00":
            msg = header.get("message", "Unknown API Error")
            raise RuntimeError(f"OpenAPI 응답 오류 ({code}): {msg}")

        if page_no == 1:
            total_count = int(body.get("totalCount", 0))

        items = body.get("item", [])
        if isinstance(items, dict):
            items = [items]
        if not items:
            break

        all_items.extend(items)
        total_pages = max(1, math.ceil(total_count / num_of_rows))
        if page_no >= total_pages or len(all_items) >= total_count:
            break
        page_no += 1

    return {
        root_key: {
            "header": {"code": "00", "message": "NORMAL_CODE"},
            "totalCount": len(all_items),
            "item": all_items,
        }
    }


def fetch_busan_festival_kr(
    api_key: str,
    output_path: Path | None = None,
) -> Path:
    """부산광역시 부산축제정보 서비스 전체 데이터를 수집하여 JSON으로 저장합니다."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "festival" / "busan_festival_kr.json"

    data = fetch_paginated_items(
        url=FESTIVAL_API_URL,
        root_key="getFestivalKr",
        api_key=api_key,
        num_of_rows=100,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    count = data["getFestivalKr"]["totalCount"]
    print(f"[성공] 부산축제정보 {count}건 저장 완료 -> {output_path}")
    return output_path


def fetch_busan_culture_exhibit(
    api_key: str,
    output_path: Path | None = None,
) -> Path:
    """부산광역시 전시 목록 서비스 전체 데이터를 수집하여 JSON으로 저장합니다."""
    if output_path is None:
        output_path = RAW_DATA_DIR / "festival" / "busan_culture_exhibit.json"

    data = fetch_paginated_items(
        url=EXHIBIT_API_URL,
        root_key="getBusanCultureExhibit",
        api_key=api_key,
        num_of_rows=1000,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    count = data["getBusanCultureExhibit"]["totalCount"]
    print(f"[성공] 부산전시목록 {count}건 저장 완료 -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="부산광역시 축제 및 전시 목록 OpenAPI 일괄 수집기"
    )
    parser.add_argument(
        "--target",
        choices=["all", "festival", "exhibit"],
        default="all",
        help="수집 대상 API 선택 (기본값: all)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    if args.target in ("all", "festival"):
        fetch_busan_festival_kr(api_key=api_key)
    if args.target in ("all", "exhibit"):
        fetch_busan_culture_exhibit(api_key=api_key)


if __name__ == "__main__":
    main()
