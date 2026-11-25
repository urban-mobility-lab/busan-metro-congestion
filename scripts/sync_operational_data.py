"""
운영용 공공데이터(부산교통공사 승하차인원 · KASI 특일 및 달력 마스터) 정기 자동화 동기화 CLI

책임 범위(Stateful Operational Sync):
- 스케줄러에 비의존적인 단발·멱등(idempotent) 실행 및 백필(backfill) 지원
- 지하철 승하차: odcloud OAS 스펙 동적 탐색 -> 상태 파일(ingestion_state.json)과 비교하여
  신규 uddi(UUID) 또는 당해 연도 totalCount 변경 시에만 fetch_subway_ridership_api를 통해 수집·원자적 교체
- KASI 특일: 과거 확정 연도는 스킵하고 당해/차년도/차차년도의 임시·대체공휴일 변경을 해시로 감지하여
  변경 발생 시 data/reference/calendar_master.parquet를 자동 확장·원자적 갱신
- 실행 결과를 기계 판독 가능한 Run Manifest JSON으로 반환/기록하여 후속 오케스트레이션과 연계
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from busan_metro_congestion.config import (  # noqa: E402
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
    get_api_key,
)
from scripts.build_calendar_master import build_calendar_master  # noqa: E402
from scripts.fetch_calendar_holidays_api import (  # noqa: E402
    extract_holiday_items,
    fetch_holidays_for_year,
)
from scripts.fetch_subway_ridership_api import (  # noqa: E402
    ODCLOUD_API_BASE,
    ODCLOUD_OAS_URL,
    SUBWAY_OAS_NAMESPACE,
    SubwayEndpoint,
    discover_subway_endpoints,
    fetch_subway_endpoint_total_count,
    fetch_subway_uddi_rows,
    rows_to_subway_dataframe,
    validate_subway_raw_df,
)
from scripts.ingestion_utils import (  # noqa: E402
    DEFAULT_STATE_PATH,
    atomic_write_json,
    atomic_write_parquet,
    canonical_sha256,
    load_ingestion_state,
    save_ingestion_state,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("sync_operational_data")


# =====================================================================
# 1. KASI 특일 변경 감지 및 동기화
# =====================================================================


def compute_holiday_signature(payload: dict[str, Any]) -> str:
    """
    KASI getRestDeInfo JSON 페이로드의 정규화된 공휴일 목록 SHA-256 해시를 산출합니다.
    임시공휴일·대체공휴일 추가 또는 속성 변경 감지에 사용합니다.
    """
    items = extract_holiday_items(payload)
    normalized = sorted(
        (
            str(it.get("locdate", "")).strip(),
            str(it.get("isHoliday", "")).strip().upper(),
            str(it.get("dateName", "")).strip(),
            str(it.get("dateKind", "")).strip(),
        )
        for it in items
    )
    return canonical_sha256(normalized)


def sync_calendar_holidays(
    api_key: str,
    start_year: int = 2016,
    end_year: int = 2026,
    output_dir: Path | str | None = None,
    refresh_years: set[int] | None = None,
    force: bool = False,
    delay: float = 0.2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    KASI 특일 멱등 동기화 엔진.

    - 기완료 과거 연도는 스킵(Resume)
    - refresh_years(당해·차년도·차차년도) 또는 force=True 대상은 재조회 후
      정규화 해시(compute_holiday_signature) 비교로 변경 시에만 원자적 교체
    """
    if start_year > end_year:
        raise ValueError(
            f"시작 연도({start_year})는 종료 연도({end_year})보다 클 수 없습니다."
        )

    out_dir = Path(output_dir) if output_dir is not None else RAW_DATA_DIR / "calendar"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    refresh_set = set(refresh_years or ())
    years = list(range(start_year, end_year + 1))

    fetched_years: list[int] = []
    changed_years: list[int] = []
    unchanged_years: list[int] = []
    skipped_years: list[int] = []
    planned_years: list[int] = []
    year_signatures: dict[str, str] = {}

    for idx, year in enumerate(years):
        out_path = out_dir / f"holidays_{year}.json"
        exists = out_path.exists()
        must_fetch = force or (not exists) or (year in refresh_set)

        if not must_fetch:
            logger.info(f"[스킵] 확정 연도 공휴일 파일 존재: {out_path}")
            skipped_years.append(year)
            existing_payload = json.loads(out_path.read_text(encoding="utf-8"))
            year_signatures[str(year)] = compute_holiday_signature(existing_payload)
            continue

        if dry_run:
            logger.info(
                f"[Dry-Run] 공휴일 조회 예정: {year}년 "
                f"(기존파일={'있음' if exists else '없음'})"
            )
            planned_years.append(year)
            continue

        new_data = fetch_holidays_for_year(api_key=api_key, year=year)
        fetched_years.append(year)
        new_sig = compute_holiday_signature(new_data)
        year_signatures[str(year)] = new_sig

        old_sig: str | None = None
        if exists:
            old_payload = json.loads(out_path.read_text(encoding="utf-8"))
            old_sig = compute_holiday_signature(old_payload)

        if old_sig == new_sig and exists:
            logger.info(f"[유지] {year}년 공휴일 변경 없음 (재기록 생략)")
            unchanged_years.append(year)
        else:
            atomic_write_json(out_path, new_data)
            total = new_data["response"]["body"]["totalCount"]
            status_tag = "신규" if not exists else "갱신"
            logger.info(f"[{status_tag}] {year}년 공휴일 {total}건 -> {out_path}")
            changed_years.append(year)

        if delay > 0 and idx < len(years) - 1:
            time.sleep(delay)

    return {
        "start_year": start_year,
        "end_year": end_year,
        "fetched_years": fetched_years,
        "changed_years": changed_years,
        "unchanged_years": unchanged_years,
        "skipped_years": skipped_years,
        "planned_years": planned_years,
        "year_signatures": year_signatures,
        "changed": len(changed_years) > 0,
        "dry_run": dry_run,
    }


# =====================================================================
# 2. 지하철 승하차 신규 UUID / totalCount 변경 감지 및 동기화
# =====================================================================


def _should_download_subway_endpoint(
    endpoint: SubwayEndpoint,
    year_state: dict[str, Any] | None,
    output_file: Path,
    force: bool,
    check_in_place_update: bool,
    api_key: str,
    api_base: str,
    max_retries: int,
    timeout: int,
    http_get: Callable[..., Any] | None,
) -> tuple[bool, str]:
    """신규 UUID 여부, 로컬 파일 부재, 미확정 연도 totalCount 변경 여부를 판별합니다."""
    if force:
        return True, "forced"
    if year_state is None:
        return True, "new_year_uddi"
    prev_uddi = str(year_state.get("uddi") or "")
    if prev_uddi != endpoint.uddi:
        return True, "new_uddi"
    if not output_file.exists():
        return True, "missing_local_file"

    is_partial_year = endpoint.snapshot_date < date(endpoint.year, 12, 31)
    if check_in_place_update and is_partial_year:
        remote_total = fetch_subway_endpoint_total_count(
            api_key=api_key,
            endpoint=endpoint,
            api_base=api_base,
            max_retries=max_retries,
            timeout=timeout,
            http_get=http_get,
        )
        prev_total = int(year_state.get("total_count", -1))
        if remote_total != prev_total:
            return True, f"total_count_changed({prev_total}->{remote_total})"

    return False, "already_collected_uddi"


def sync_subway_ridership(
    api_key: str,
    start_year: int = 2023,
    end_year: int | None = None,
    output_dir: Path | str | None = None,
    state_path: Path | str | None = None,
    station_master_path: Path | str | None = None,
    oas_url: str = ODCLOUD_OAS_URL,
    namespace: str = SUBWAY_OAS_NAMESPACE,
    api_base: str = ODCLOUD_API_BASE,
    per_page: int = 1000,
    delay_sec: float = 0.05,
    max_retries: int = 3,
    timeout: int = 25,
    force: bool = False,
    dry_run: bool = False,
    check_in_place_update: bool = True,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """
    OAS에서 연도별 최신 uddi를 탐색하고, 상태 파일에 없는 신규 UUID(또는 totalCount 변경)만
    선별 수집·검증·원자적 저장합니다.
    """
    if end_year is not None and start_year > end_year:
        raise ValueError(
            f"시작 연도({start_year})는 종료 연도({end_year})보다 클 수 없습니다."
        )

    out_dir = Path(output_dir) if output_dir is not None else RAW_DATA_DIR / "subway"
    state_file = Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    state = load_ingestion_state(state_file)
    years_state: dict[str, Any] = state["subway"]["years"]

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

    downloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []

    for ep in endpoints:
        year_key = str(ep.year)
        output_file = out_dir / f"subway_ridership_{ep.year}.parquet"
        should_dl, reason = _should_download_subway_endpoint(
            endpoint=ep,
            year_state=years_state.get(year_key),
            output_file=output_file,
            force=force,
            check_in_place_update=check_in_place_update,
            api_key=api_key,
            api_base=api_base,
            max_retries=max_retries,
            timeout=timeout,
            http_get=http_get,
        )

        if not should_dl:
            logger.info(
                f"[스킵] {ep.year}년 최신 uddi({ep.uddi}, {ep.snapshot_date}) 이미 수집됨"
            )
            skipped.append(
                {
                    "year": ep.year,
                    "uddi": ep.uddi,
                    "snapshot_date": ep.snapshot_date.isoformat(),
                    "reason": reason,
                }
            )
            continue

        if dry_run:
            logger.info(
                f"[Dry-Run] 수집 예정: {ep.year}년 {ep.uddi} ({ep.snapshot_date}, 사유: {reason})"
            )
            planned.append(
                {
                    "year": ep.year,
                    "uddi": ep.uddi,
                    "summary": ep.summary,
                    "snapshot_date": ep.snapshot_date.isoformat(),
                    "reason": reason,
                    "target_file": str(output_file),
                }
            )
            continue

        logger.info(
            f"[수집] {ep.year}년 {ep.summary} ({ep.uddi}, 사유: {reason}) 다운로드 시작..."
        )
        rows, total_count = fetch_subway_uddi_rows(
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

        atomic_write_parquet(df, output_file)

        signature = canonical_sha256(
            {
                "uddi": ep.uddi,
                "row_count": val_info["row_count"],
                "min_date": val_info["min_date"],
                "max_date": val_info["max_date"],
            }
        )
        years_state[year_key] = {
            "uddi": ep.uddi,
            "path": ep.path,
            "summary": ep.summary,
            "snapshot_date": ep.snapshot_date.isoformat(),
            "total_count": total_count,
            "row_count": val_info["row_count"],
            "min_date": val_info["min_date"],
            "max_date": val_info["max_date"],
            "matched_stations": val_info["matched_stations"],
            "file": output_file.name,
            "content_sha256": signature,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_ingestion_state(state, state_file)

        logger.info(
            f"[저장] {ep.year}년 {val_info['row_count']:,}행 "
            f"({val_info['min_date']} ~ {val_info['max_date']}) -> {output_file}"
        )
        downloaded.append(
            {
                "year": ep.year,
                "uddi": ep.uddi,
                "snapshot_date": ep.snapshot_date.isoformat(),
                "reason": reason,
                "file": str(output_file),
                **val_info,
            }
        )

    return {
        "discovered_count": len(endpoints),
        "downloaded": downloaded,
        "skipped": skipped,
        "planned": planned,
        "changed": len(downloaded) > 0,
        "dry_run": dry_run,
    }


# =====================================================================
# 3. 운영 데이터 통합 동기화 진입점
# =====================================================================


def sync_operational_data(
    api_key: str,
    target: str = "all",
    mode: str = "incremental",
    subway_start_year: int = 2023,
    subway_end_year: int | None = None,
    calendar_start_year: int = 2016,
    calendar_end_year: int | None = None,
    reference_today: date | None = None,
    subway_output_dir: Path | str | None = None,
    calendar_output_dir: Path | str | None = None,
    calendar_master_path: Path | str | None = None,
    station_master_path: Path | str | None = None,
    state_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    force: bool = False,
    rebuild_master: bool = False,
    dry_run: bool = False,
    delay: float = 0.05,
    per_page: int = 1000,
    oas_url: str = ODCLOUD_OAS_URL,
    namespace: str = SUBWAY_OAS_NAMESPACE,
    api_base: str = ODCLOUD_API_BASE,
    http_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """
    운영용 지하철 승하차(신규 UUID 감지) 및 KASI 특일(해시 변경 감지 + 달력 마스터 자동 갱신)을
    단발·멱등 실행합니다.
    """
    if target not in ("all", "subway", "calendar"):
        raise ValueError(
            f"지원하지 않는 target입니다: '{target}'. ('all', 'subway', 'calendar' 중 선택)"
        )
    if mode not in ("incremental", "backfill"):
        raise ValueError(
            f"지원하지 않는 mode입니다: '{mode}'. ('incremental', 'backfill' 중 선택)"
        )

    today = reference_today if reference_today is not None else date.today()
    effective_state_path = (
        Path(state_path) if state_path is not None else DEFAULT_STATE_PATH
    )
    effective_cal_dir = (
        Path(calendar_output_dir)
        if calendar_output_dir is not None
        else RAW_DATA_DIR / "calendar"
    )
    effective_cal_master = (
        Path(calendar_master_path)
        if calendar_master_path is not None
        else REFERENCE_DATA_DIR / "calendar_master.parquet"
    )

    manifest: dict[str, Any] = {
        "status": "ok",
        "target": target,
        "mode": mode,
        "dry_run": dry_run,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "subway": None,
        "calendar": None,
        "any_data_changed": False,
    }

    # 1. KASI 특일 동기화 + calendar_master.parquet 자동 갱신
    if target in ("all", "calendar"):
        if mode == "incremental":
            eff_cal_end = (
                calendar_end_year if calendar_end_year is not None else today.year + 2
            )
            refresh_years = {today.year, today.year + 1, today.year + 2}
        else:
            eff_cal_end = (
                calendar_end_year if calendar_end_year is not None else today.year + 1
            )
            refresh_years = set()

        cal_summary = sync_calendar_holidays(
            api_key=api_key,
            start_year=calendar_start_year,
            end_year=eff_cal_end,
            output_dir=effective_cal_dir,
            refresh_years=refresh_years,
            force=force,
            delay=delay,
            dry_run=dry_run,
        )

        need_master_rebuild = (
            cal_summary["changed"]
            or rebuild_master
            or (not effective_cal_master.exists())
        )
        master_rebuilt = False
        master_rows: int | None = None
        master_date_range: dict[str, str] | None = None

        if need_master_rebuild and not dry_run:
            df_master = build_calendar_master(
                start_date=date(calendar_start_year, 1, 1),
                end_date=date(eff_cal_end, 12, 31),
                raw_calendar_dir=effective_cal_dir,
                output_path=effective_cal_master,
                auto_expand_range=True,
            )
            master_rebuilt = True
            master_rows = len(df_master)
            min_d = df_master["date"].min()
            max_d = df_master["date"].max()
            master_date_range = {
                "min_date": min_d.isoformat()
                if isinstance(min_d, date)
                else str(min_d),
                "max_date": max_d.isoformat()
                if isinstance(max_d, date)
                else str(max_d),
            }

            state = load_ingestion_state(effective_state_path)
            for yr_key, sig in cal_summary["year_signatures"].items():
                state["calendar"]["years"][yr_key] = {
                    "signature": sig,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            state["calendar"]["master_file"] = str(effective_cal_master)
            state["calendar"]["master_rows"] = master_rows
            state["calendar"]["master_date_range"] = master_date_range
            save_ingestion_state(state, effective_state_path)

        cal_summary["master_rebuilt"] = master_rebuilt
        cal_summary["master_rebuild_planned"] = bool(need_master_rebuild and dry_run)
        cal_summary["master_path"] = str(effective_cal_master)
        cal_summary["master_rows"] = master_rows
        cal_summary["master_date_range"] = master_date_range

        manifest["calendar"] = cal_summary
        if cal_summary["changed"] or master_rebuilt:
            manifest["any_data_changed"] = True

    # 2. 부산교통공사 승하차인원(odcloud) 신규 UUID 감지 및 동기화
    if target in ("all", "subway"):
        subway_summary = sync_subway_ridership(
            api_key=api_key,
            start_year=subway_start_year,
            end_year=subway_end_year,
            output_dir=subway_output_dir,
            state_path=effective_state_path,
            station_master_path=station_master_path,
            oas_url=oas_url,
            namespace=namespace,
            api_base=api_base,
            per_page=per_page,
            delay_sec=delay,
            force=force,
            dry_run=dry_run,
            check_in_place_update=(mode == "incremental"),
            http_get=http_get,
        )
        manifest["subway"] = subway_summary
        if subway_summary["changed"]:
            manifest["any_data_changed"] = True

    if manifest_path is not None and not dry_run:
        atomic_write_json(manifest_path, manifest)
        logger.info(f"실행 매니페스트 저장 완료: {manifest_path}")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="운영용 공공데이터(지하철 승하차 신규 UUID · KASI 특일 · 달력 마스터) 정기 동기화 CLI"
    )
    parser.add_argument(
        "--target",
        choices=["all", "subway", "calendar"],
        default="all",
        help="동기화 대상 선택 (기본값: all)",
    )
    parser.add_argument(
        "--mode",
        choices=["incremental", "backfill"],
        default="incremental",
        help="실행 모드 ('incremental': 정기 증분·변경 감지, 'backfill': 구간 적재)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="공통 시작 연도 지정 (미지정 시 지하철=2023, 달력=2016)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="공통 종료 연도 지정 (미지정 시 지하철=OAS 최신, 달력=당해+2년)",
    )
    parser.add_argument(
        "--subway-start-year",
        type=int,
        default=None,
        help="지하철 수집 시작 연도 (기본값: 2023)",
    )
    parser.add_argument(
        "--subway-end-year",
        type=int,
        default=None,
        help="지하철 수집 종료 연도",
    )
    parser.add_argument(
        "--calendar-start-year",
        type=int,
        default=None,
        help="달력 수집 시작 연도 (기본값: 2016)",
    )
    parser.add_argument(
        "--calendar-end-year",
        type=int,
        default=None,
        help="달력 수집 종료 연도",
    )
    parser.add_argument(
        "--subway-dir",
        type=str,
        default=None,
        help="지하철 원천 저장 디렉터리 (기본값: data/raw/subway)",
    )
    parser.add_argument(
        "--calendar-dir",
        type=str,
        default=None,
        help="달력 원천 JSON 디렉터리 (기본값: data/raw/calendar)",
    )
    parser.add_argument(
        "--calendar-master",
        type=str,
        default=None,
        help="달력 마스터 Parquet 경로 (기본값: data/reference/calendar_master.parquet)",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="동기화 상태 JSON 경로 (기본값: data/raw/_state/ingestion_state.json)",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="실행 결과 매니페스트 JSON 저장 경로 (선택)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=1000,
        help="지하철 odcloud API 페이지당 요청 건수 (기본값: 1000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="API 요청 간 대기 시간(초, 기본값: 0.05)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기수집 UUID·연도 파일이 있어도 강제 재수집",
    )
    parser.add_argument(
        "--rebuild-master",
        action="store_true",
        help="공휴일 변경 여부와 무관하게 calendar_master.parquet 강제 재빌드",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="신규 UUID·변경 연도 탐지만 수행하고 다운로드 및 파일 쓰기 생략",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="실행 결과 매니페스트를 표준 출력(stdout)으로 JSON 출력",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    api_key = get_api_key()

    sub_start = (
        args.subway_start_year
        if args.subway_start_year is not None
        else (args.start_year if args.start_year is not None else 2023)
    )
    sub_end = (
        args.subway_end_year if args.subway_end_year is not None else args.end_year
    )
    cal_start = (
        args.calendar_start_year
        if args.calendar_start_year is not None
        else (args.start_year if args.start_year is not None else 2016)
    )
    cal_end = (
        args.calendar_end_year if args.calendar_end_year is not None else args.end_year
    )

    manifest = sync_operational_data(
        api_key=api_key,
        target=args.target,
        mode=args.mode,
        subway_start_year=sub_start,
        subway_end_year=sub_end,
        calendar_start_year=cal_start,
        calendar_end_year=cal_end,
        subway_output_dir=args.subway_dir,
        calendar_output_dir=args.calendar_dir,
        calendar_master_path=args.calendar_master,
        state_path=args.state_path,
        manifest_path=args.manifest_path,
        force=args.force,
        rebuild_master=args.rebuild_master,
        dry_run=args.dry_run,
        delay=args.delay,
        per_page=args.per_page,
    )
    if args.json_summary:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
