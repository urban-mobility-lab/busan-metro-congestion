"""
부산교통공사 시간대별 승하차인원(공공데이터포털 odcloud) OpenAPI 단발 수집 스크립트

역할(Stateless Fetcher):
1. OAS 스펙(https://infuser.odcloud.kr/oas/docs?namespace=3057229/v1)에서
   전체 엔드포인트(uddi:<uuid>)를 파싱하고 연도별 최신 스냅샷 엔드포인트를 선별합니다.
2. 지정된 연도 구간의 승하차 데이터를 페이지네이션으로 수집하고,
   Polars DataFrame(30개 표준 원천 컬럼) 정규화 및 스키마·역사 마스터 검증을 수행합니다.
3. 기완료 연도 파일이 존재하면 스킵하며(--force 지정 시 강제 재수집),
   정기 자동화용 UUID 변경 감지 및 워터마크 상태 관리는 sync_operational_data.py에서 담당합니다.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from busan_metro_congestion.config import (  # noqa: E402
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
    get_api_key,
)
from scripts.ingestion_utils import (  # noqa: E402
    AuthenticationError,
    QuotaExceededError,
    atomic_write_parquet,
    request_json_with_retry,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fetch_subway_ridership_api")

ODCLOUD_OAS_URL = "https://infuser.odcloud.kr/oas/docs"
SUBWAY_OAS_NAMESPACE = "3057229/v1"
ODCLOUD_API_BASE = "https://api.odcloud.kr/api"

HOUR_SLOT_COLUMNS: list[str] = [f"{h:02d}시-{(h % 24) + 1:02d}시" for h in range(1, 25)]
BASE_SUBWAY_COLUMNS: list[str] = ["역번호", "역명", "년월일", "요일", "구분", "합계"]
STANDARD_SUBWAY_COLUMNS: list[str] = BASE_SUBWAY_COLUMNS + HOUR_SLOT_COLUMNS
REQUIRED_SUBWAY_COLUMNS: set[str] = {"역번호", "년월일", "구분", *HOUR_SLOT_COLUMNS}


@dataclass(frozen=True)
class SubwayEndpoint:
    """odcloud OAS에서 탐색된 부산교통공사 승하차인원 단일 스냅샷 엔드포인트"""

    path: str
    uddi: str
    summary: str
    year: int
    snapshot_date: date
    has_explicit_date: bool = True


def parse_summary_snapshot_date(summary: str) -> tuple[int, date, bool] | None:
    """
    OAS summary 문자열에서 연도와 스냅샷 기준일을 추출합니다.

    지원 예시:
    - '부산교통공사_시간대별 승하차인원_20261031' -> (2026, date(2026, 10, 31), True)
    - '부산교통공사_시간대별 승하차인원_10/01/2021' -> (2021, date(2021, 10, 1), True)
    - '부산교통공사 시간대별 승하차인원(2018년)' -> (2018, date(2018, 12, 31), False)
    """
    text = str(summary or "").strip()
    if not text:
        return None

    # 1. YYYYMMDD (표준 포맷)
    m_ymd = re.search(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", text)
    if m_ymd:
        y, m, d = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
        try:
            return y, date(y, m, d), True
        except ValueError:
            return None

    # 2. MM/DD/YYYY (예: 10/01/2021)
    m_mdy = re.search(r"(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(20\d{2})", text)
    if m_mdy:
        m, d, y = int(m_mdy.group(1)), int(m_mdy.group(2)), int(m_mdy.group(3))
        try:
            return y, date(y, m, d), True
        except ValueError:
            return None

    # 3. 연도만 명시된 경우 (예: (2018년))
    m_yr = re.search(r"(20\d{2})\s*년?", text)
    if m_yr:
        y = int(m_yr.group(1))
        return y, date(y, 12, 31), False

    return None


def _is_valid_oas_model(spec: dict[str, Any], uddi: str) -> bool:
    """OAS definitions에 모델 스키마가 있는 경우 정상 승하차 스키마인지 확인합니다."""
    defs = spec.get("definitions")
    if not isinstance(defs, dict):
        return True
    model = defs.get(f"{uddi}_model")
    if not isinstance(model, dict):
        return True
    props = model.get("properties")
    if not isinstance(props, dict) or not props:
        return True
    prop_keys = set(props.keys())
    return "역번호" in prop_keys and "년월일" in prop_keys and "구분" in prop_keys


def extract_subway_endpoints_from_oas(
    spec: dict[str, Any],
    start_year: int | None = 2023,
    end_year: int | None = None,
    latest_per_year: bool = True,
) -> list[SubwayEndpoint]:
    """
    odcloud OAS(Swagger) JSON 스펙에서 유효한 승하차 엔드포인트 목록을 추출합니다.

    latest_per_year=True인 경우 동일 연도 내 중간 스냅샷(예: 20240430, 20240930)을 제외하고
    가장 최신 스냅샷(예: 20241231) 1개만 대표로 선별합니다.
    """
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("OAS 스펙에 유효한 'paths' 객체가 없습니다.")

    candidates: list[SubwayEndpoint] = []
    for path_str, detail in paths.items():
        if not isinstance(detail, dict):
            continue
        get_op = detail.get("get")
        if not isinstance(get_op, dict):
            continue

        uddi_part = str(path_str).rstrip("/").split("/")[-1]
        if not uddi_part.startswith("uddi:"):
            continue
        if not _is_valid_oas_model(spec, uddi_part):
            logger.debug(f"[제외] 비정상 OAS 모델 스키마 엔드포인트 스킵: {path_str}")
            continue

        summary = str(get_op.get("summary") or "").strip()
        parsed = parse_summary_snapshot_date(summary)
        if parsed is None:
            continue
        yr, snap_dt, has_explicit = parsed

        if start_year is not None and yr < start_year:
            continue
        if end_year is not None and yr > end_year:
            continue

        candidates.append(
            SubwayEndpoint(
                path=str(path_str),
                uddi=uddi_part,
                summary=summary,
                year=yr,
                snapshot_date=snap_dt,
                has_explicit_date=has_explicit,
            )
        )

    if not latest_per_year:
        return sorted(candidates, key=lambda e: (e.year, e.snapshot_date, e.uddi))

    by_year: dict[int, SubwayEndpoint] = {}
    for ep in candidates:
        existing = by_year.get(ep.year)
        if existing is None:
            by_year[ep.year] = ep
            continue
        if (ep.snapshot_date, ep.has_explicit_date, ep.uddi) > (
            existing.snapshot_date,
            existing.has_explicit_date,
            existing.uddi,
        ):
            by_year[ep.year] = ep

    return [by_year[y] for y in sorted(by_year)]


def discover_subway_endpoints(
    oas_url: str = ODCLOUD_OAS_URL,
    namespace: str = SUBWAY_OAS_NAMESPACE,
    start_year: int | None = 2023,
    end_year: int | None = None,
    latest_per_year: bool = True,
    max_retries: int = 3,
    timeout: int = 25,
    http_get: Callable[..., Any] | None = None,
) -> list[SubwayEndpoint]:
    """공공데이터포털 odcloud OAS 문서를 조회하여 대상 연도별 최신 uddi 엔드포인트를 반환합니다."""
    getter = http_get if http_get is not None else requests.get
    spec = request_json_with_retry(
        url=oas_url,
        params={"namespace": namespace},
        max_retries=max_retries,
        timeout=timeout,
        http_get=getter,
    )
    return extract_subway_endpoints_from_oas(
        spec=spec,
        start_year=start_year,
        end_year=end_year,
        latest_per_year=latest_per_year,
    )


def fetch_subway_endpoint_total_count(
    api_key: str,
    endpoint: SubwayEndpoint,
    api_base: str = ODCLOUD_API_BASE,
    max_retries: int = 3,
    timeout: int = 25,
    http_get: Callable[..., Any] | None = None,
) -> int:
    """perPage=1 메타 호출로 해당 uddi 엔드포인트의 전체 레코드 수(totalCount)를 조회합니다."""
    getter = http_get if http_get is not None else requests.get
    url = f"{api_base.rstrip('/')}/{endpoint.path.lstrip('/')}"
    payload = request_json_with_retry(
        url=url,
        params={
            "serviceKey": api_key,
            "page": 1,
            "perPage": 1,
            "returnType": "JSON",
        },
        max_retries=max_retries,
        timeout=timeout,
        http_get=getter,
    )
    return int(payload.get("totalCount", payload.get("matchCount", 0)) or 0)


def fetch_subway_uddi_rows(
    api_key: str,
    endpoint: SubwayEndpoint,
    per_page: int = 1000,
    api_base: str = ODCLOUD_API_BASE,
    delay_sec: float = 0.05,
    max_retries: int = 3,
    timeout: int = 25,
    http_get: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    단일 uddi 엔드포인트를 페이지네이션으로 전수 수집합니다.

    반환: (수집된 레코드 리스트, API가 보고한 totalCount)
    """
    if per_page <= 0:
        raise ValueError(f"per_page는 양수여야 합니다: {per_page}")

    getter = http_get if http_get is not None else requests.get
    url = f"{api_base.rstrip('/')}/{endpoint.path.lstrip('/')}"

    all_rows: list[dict[str, Any]] = []
    page = 1
    total_count = 0

    while True:
        payload = request_json_with_retry(
            url=url,
            params={
                "serviceKey": api_key,
                "page": page,
                "perPage": per_page,
                "returnType": "JSON",
            },
            max_retries=max_retries,
            timeout=timeout,
            http_get=getter,
        )

        if page == 1:
            total_count = int(
                payload.get("totalCount", payload.get("matchCount", 0)) or 0
            )

        rows = payload.get("data", [])
        if not isinstance(rows, list):
            raise RuntimeError(
                f"예상하지 못한 odcloud 응답 data 형식({endpoint.uddi}, page={page}): {type(rows).__name__}"
            )
        if not rows:
            break

        all_rows.extend(r for r in rows if isinstance(r, dict))

        total_pages = max(1, math.ceil(total_count / per_page)) if total_count else page
        if page >= total_pages or (total_count > 0 and len(all_rows) >= total_count):
            break

        page += 1
        if delay_sec > 0:
            time.sleep(delay_sec)

    return all_rows, total_count


def _to_int_or_none(val: Any) -> int | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).replace(",", "").strip()
    if not s:
        return None
    return int(float(s))


def rows_to_subway_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """odcloud JSON 레코드 리스트를 subway.py 호환 30컬럼 표준 Polars DataFrame으로 변환합니다."""
    schema: dict[str, pl.DataType] = {
        "역번호": pl.Int64,
        "역명": pl.String,
        "년월일": pl.String,
        "요일": pl.String,
        "구분": pl.String,
        "합계": pl.Int64,
        **{col: pl.Int64 for col in HOUR_SLOT_COLUMNS},
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "역번호": _to_int_or_none(row.get("역번호")),
            "역명": "" if row.get("역명") is None else str(row.get("역명")).strip(),
            "년월일": ""
            if row.get("년월일") is None
            else str(row.get("년월일")).strip(),
            "요일": "" if row.get("요일") is None else str(row.get("요일")).strip(),
            "구분": "" if row.get("구분") is None else str(row.get("구분")).strip(),
            "합계": _to_int_or_none(row.get("합계")),
        }
        for col in HOUR_SLOT_COLUMNS:
            val = row.get(col)
            if val is None:
                alt_col = col.replace("-", "~")
                val = row.get(alt_col)
            item[col] = _to_int_or_none(val)
        normalized.append(item)

    return pl.DataFrame(normalized, schema=schema)


def validate_subway_raw_df(
    df: pl.DataFrame,
    station_master_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    수집된 지하철 원천 DataFrame의 스키마·값 범위·날짜·역사 마스터 정합성을 검증합니다.

    검증 실패 시 ValueError를 발생시켜 손상 데이터의 저장을 차단합니다.
    """
    if len(df) == 0:
        raise ValueError("수집된 지하철 승하차 데이터가 비어 있습니다(0행).")

    missing_cols = sorted(REQUIRED_SUBWAY_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError(f"필수 지하철 컬럼이 누락되었습니다: {missing_cols}")

    non_null_stations = df.filter(pl.col("역번호").is_not_null())
    if len(non_null_stations) == 0:
        raise ValueError("유효한 역번호가 포함된 행이 없습니다.")

    gubun_set = set(non_null_stations["구분"].drop_nulls().unique().to_list())
    if not {"승차", "하차"}.issubset(gubun_set):
        raise ValueError(f"'구분' 컬럼에 승차/하차가 모두 존재해야 합니다: {gubun_set}")

    try:
        parsed_dates = (
            non_null_stations["년월일"]
            .cast(pl.String)
            .str.replace_all("-", "")
            .str.to_date("%Y%m%d", strict=True)
        )
    except Exception as exc:
        raise ValueError(
            f"'년월일' 컬럼에 유효하지 않은 날짜 형식이 포함되어 있습니다: {exc}"
        ) from exc

    for col in HOUR_SLOT_COLUMNS:
        neg_cnt = int((non_null_stations[col].fill_null(0) < 0).sum())
        if neg_cnt > 0:
            raise ValueError(
                f"시간대 컬럼 '{col}'에 음수 인원({neg_cnt}건)이 감지되었습니다."
            )

    matched_stations: int | None = None
    master_p = (
        Path(station_master_path)
        if station_master_path is not None
        else REFERENCE_DATA_DIR / "station_master.parquet"
    )
    if master_p.exists():
        master_ids = set(
            pl.read_parquet(master_p)["station_id"].cast(pl.Int64).to_list()
        )
        observed_ids = set(
            non_null_stations["역번호"].cast(pl.Int64).unique().to_list()
        )
        matched_stations = len(observed_ids & master_ids)
        if matched_stations == 0:
            raise ValueError(f"역사 마스터({master_p})와 일치하는 역번호가 0건입니다.")

    min_date = parsed_dates.min()
    max_date = parsed_dates.max()
    return {
        "row_count": len(df),
        "valid_row_count": len(non_null_stations),
        "min_date": min_date.isoformat()
        if isinstance(min_date, date)
        else str(min_date),
        "max_date": max_date.isoformat()
        if isinstance(max_date, date)
        else str(max_date),
        "matched_stations": matched_stations,
    }


def fetch_subway_ridership(
    api_key: str,
    start_year: int = 2023,
    end_year: int | None = None,
    output_dir: Path | str | None = None,
    station_master_path: Path | str | None = None,
    oas_url: str = ODCLOUD_OAS_URL,
    namespace: str = SUBWAY_OAS_NAMESPACE,
    api_base: str = ODCLOUD_API_BASE,
    per_page: int = 1000,
    delay_sec: float = 0.05,
    max_retries: int = 3,
    timeout: int = 25,
    force: bool = False,
    http_get: Callable[..., Any] | None = None,
) -> list[Path]:
    """
    지정된 연도 구간의 지하철 승하차 데이터를 연도별 Parquet(subway_ridership_{year}.parquet)로
    단발 수집·저장합니다 (상태 파일 비의존, 기존재 파일 스킵 및 force 덮어쓰기 지원).
    """
    if end_year is not None and start_year > end_year:
        raise ValueError(
            f"시작 연도({start_year})는 종료 연도({end_year})보다 클 수 없습니다."
        )

    out_dir = Path(output_dir) if output_dir is not None else RAW_DATA_DIR / "subway"
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = discover_subway_endpoints(
        oas_url=oas_url,
        namespace=namespace,
        start_year=start_year,
        end_year=end_year,
        latest_per_year=True,
        max_retries=max_retries,
        timeout=timeout,
        http_get=http_get,
    )

    saved_files: list[Path] = []
    for ep in endpoints:
        out_path = out_dir / f"subway_ridership_{ep.year}.parquet"
        if out_path.exists() and not force:
            logger.info(f"[스킵] 이미 수집된 연도 파일 존재: {out_path}")
            saved_files.append(out_path)
            continue

        logger.info(f"[수집] {ep.year}년 {ep.summary} ({ep.uddi}) 다운로드 시작...")
        rows, _ = fetch_subway_uddi_rows(
            api_key=api_key,
            endpoint=ep,
            per_page=per_page,
            api_base=api_base,
            delay_sec=delay_sec,
            max_retries=max_retries,
            timeout=timeout,
            http_get=http_get,
        )
        df = rows_to_subway_dataframe(rows)
        val_info = validate_subway_raw_df(df, station_master_path=station_master_path)
        atomic_write_parquet(df, out_path)
        logger.info(
            f"[저장] {ep.year}년 {val_info['row_count']:,}행 "
            f"({val_info['min_date']} ~ {val_info['max_date']}) -> {out_path}"
        )
        saved_files.append(out_path)

    return saved_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="부산교통공사 시간대별 승하차인원(odcloud) 연도별 단발 수집기"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2023,
        help="수집 시작 연도 (기본값: 2023)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="수집 종료 연도 (미지정 시 OAS 최신 연도까지 전체)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="저장 디렉터리 (기본값: data/raw/subway)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=1000,
        help="페이지당 요청 건수 (기본값: 1000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="페이지 요청 간 대기 시간(초, 기본값: 0.05)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 연도별 Parquet 파일이 있어도 강제로 다시 수집하여 덮어씀",
    )
    return parser


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_parser().parse_args(argv)
    api_key = get_api_key()
    return fetch_subway_ridership(
        api_key=api_key,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        per_page=args.per_page,
        delay_sec=args.delay,
        force=args.force,
    )


__all__ = [
    "AuthenticationError",
    "HOUR_SLOT_COLUMNS",
    "ODCLOUD_API_BASE",
    "ODCLOUD_OAS_URL",
    "QuotaExceededError",
    "STANDARD_SUBWAY_COLUMNS",
    "SUBWAY_OAS_NAMESPACE",
    "SubwayEndpoint",
    "discover_subway_endpoints",
    "extract_subway_endpoints_from_oas",
    "fetch_subway_endpoint_total_count",
    "fetch_subway_ridership",
    "fetch_subway_uddi_rows",
    "parse_summary_snapshot_date",
    "rows_to_subway_dataframe",
    "validate_subway_raw_df",
]


if __name__ == "__main__":
    main()
