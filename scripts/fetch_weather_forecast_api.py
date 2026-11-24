"""
기상청 단기예보(getVilageFcst) OpenAPI 일괄 수집 스크립트

- 고유 격자(19개)만 선별 호출하여 중복 호출 완전 배제 (114개 역사 1:1 매핑 지원)
- 날짜 오름차순(가장 오래된 날 2023-06-01부터) 순차 수집
- 멀티스레드(ThreadPoolExecutor) 병렬 수집 지원: 기본 5개 워커로 수집 속도 약 4배 향상
- 체크포인트 및 이어받기(Resume):
  - 격자 단위 임시 캐시(.cache/) 및 일별 Parquet(daily/) 기반으로 중단 시점부터 즉시 재개
  - 이미 완료된 날짜/격자는 API 호출 없이 자동 스킵
- 1회 호출당 numOfRows=1000으로 해당 발표 시점의 3~4일치(72~96시간) 전체 예보 일괄 수집
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import requests

from busan_metro_congestion.config import PROJECT_ROOT, get_api_key

VILAGE_FCST_URL = (
    "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
)


def load_unique_grids(grid_path: Path | None = None) -> list[tuple[int, int]]:
    """station_weather_grid.parquet에서 고유 (nx, ny) 격자 목록을 오름차순으로 로드합니다."""
    if grid_path is None:
        grid_path = PROJECT_ROOT / "data" / "reference" / "station_weather_grid.parquet"
    if not grid_path.exists():
        raise FileNotFoundError(f"격자 참조 파일을 찾을 수 없습니다: {grid_path}")

    df_grid = pl.read_parquet(grid_path)
    unique_rows = (
        df_grid.select(["nx", "ny"]).unique().sort(["nx", "ny"]).iter_rows(named=True)
    )
    return [(int(r["nx"]), int(r["ny"])) for r in unique_rows]


class QuotaExceededError(RuntimeError):
    """일일 트래픽 한도(429 Too Many Requests 또는 서비스 요청 제한 초과) 도달 예외"""

    pass


def fetch_vilage_forecast_grid(
    api_key: str,
    base_date: str,
    base_time: str,
    nx: int,
    ny: int,
    max_retries: int = 3,
    timeout: int = 25,
) -> list[dict[str, Any]]:
    """
    단일 격자 (nx, ny)의 단기예보(getVilageFcst) 1회 발표분을 수집합니다.
    numOfRows=1000으로 요청하여 3~4일치(72~96시간) 전체 예보(800~900여건)를 단 1콜로 반환받습니다.
    """
    params = {
        "serviceKey": api_key,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(VILAGE_FCST_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                raise QuotaExceededError(
                    "공공데이터포털 일일 트래픽 한도(429 Too Many Requests)에 도달했습니다."
                )
            resp.raise_for_status()
            payload = resp.json()

            header = payload.get("response", {}).get("header", {})
            code = str(header.get("resultCode", ""))
            msg = header.get("resultMsg", "")

            if code == "00":
                body = payload.get("response", {}).get("body", {})
                items = body.get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]
                return items
            elif code in ("03", "NODATA_ERROR"):
                return []
            elif code in ("22", "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR"):
                raise QuotaExceededError(
                    f"공공데이터포털 일일 트래픽 한도 초과 ({code}): {msg}"
                )
            else:
                raise RuntimeError(f"OpenAPI 오류 ({code}): {msg}")

        except QuotaExceededError:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    raise RuntimeError(
        f"API 호출 실패 ({base_date} {base_time} nx={nx} ny={ny}, {max_retries}회 재시도 실패): {last_error}"
    )


def items_to_polars(items: list[dict[str, Any]]) -> pl.DataFrame:
    """API 응답 항목 리스트를 표준 스키마의 Polars DataFrame으로 변환합니다."""
    if not items:
        return pl.DataFrame(
            schema={
                "base_date": pl.String,
                "base_time": pl.String,
                "category": pl.String,
                "fcst_date": pl.String,
                "fcst_time": pl.String,
                "fcst_value": pl.String,
                "nx": pl.Int64,
                "ny": pl.Int64,
            }
        )

    return pl.DataFrame(
        [
            {
                "base_date": str(it.get("baseDate", "")),
                "base_time": str(it.get("baseTime", "")),
                "category": str(it.get("category", "")),
                "fcst_date": str(it.get("fcstDate", "")),
                "fcst_time": str(it.get("fcstTime", "")),
                "fcst_value": str(it.get("fcstValue", "")),
                "nx": int(it.get("nx", 0)),
                "ny": int(it.get("ny", 0)),
            }
            for it in items
        ],
        schema={
            "base_date": pl.String,
            "base_time": pl.String,
            "category": pl.String,
            "fcst_date": pl.String,
            "fcst_time": pl.String,
            "fcst_value": pl.String,
            "nx": pl.Int64,
            "ny": pl.Int64,
        },
    )


def generate_date_range(start_date: str, end_date: str) -> list[str]:
    """YYYY-MM-DD 형식의 시작일과 종료일 사이의 날짜 목록(YYYYMMDD)을 오름차순으로 생성합니다."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError(
            f"시작일({start_date})이 종료일({end_date})보다 늦을 수 없습니다."
        )

    dates: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        dates.append(curr.strftime("%Y%m%d"))
        curr += timedelta(days=1)
    return dates


def _fetch_and_cache_grid(
    api_key: str,
    base_date: str,
    base_time: str,
    nx: int,
    ny: int,
    cache_file: Path,
) -> pl.DataFrame:
    """단일 격자를 호출하고 임시 캐시 Parquet로 저장한 뒤 Polars DataFrame을 반환합니다."""
    items = fetch_vilage_forecast_grid(
        api_key=api_key,
        base_date=base_date,
        base_time=base_time,
        nx=nx,
        ny=ny,
    )
    df_grid = items_to_polars(items)
    df_grid.write_parquet(cache_file)
    return df_grid


def collect_forecast_data(
    api_key: str,
    start_date: str = "2023-06-01",
    end_date: str = "2026-05-31",
    base_time: str = "0500",
    output_dir: Path | None = None,
    grid_path: Path | None = None,
    delay_sec: float = 0.05,
    max_calls: int | None = None,
    workers: int = 5,
) -> dict[str, int]:
    """
    멀티스레드 및 체크포인트 기반 단기예보 일괄 수집 엔진

    - 19개 고유 격자에 대해 가장 오래된 날부터 순서대로 수집
    - 멀티스레드(workers)로 격자들을 병렬 수집하여 수집 속도 대폭 단축
    - 이미 완료된 일별 Parquet 또는 격자별 캐시가 있으면 즉시 스킵 (Resume)
    - 19개 격자가 모두 모이면 일별 Parquet로 병합 저장하고 임시 캐시 정리
    """
    if output_dir is None:
        if base_time == "0500":
            output_dir = PROJECT_ROOT / "data" / "raw" / "weather" / "forecast"
        else:
            output_dir = (
                PROJECT_ROOT / "data" / "raw" / "weather" / f"forecast_{base_time}"
            )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    daily_dir = output_dir / "daily"
    cache_dir = output_dir / ".cache"
    daily_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_dates = generate_date_range(start_date, end_date)
    unique_grids = load_unique_grids(grid_path)

    total_days = len(target_dates)
    total_grids = len(unique_grids)
    call_count = 0
    call_lock = threading.Lock()
    skipped_days = 0
    completed_days = 0

    print(
        f"[시작] 수집 기간: {start_date} ~ {end_date} (총 {total_days}일, {total_grids}개 고유 격자, 발표시각: {base_time}, 워커: {workers}개)"
    )

    for day_idx, d_str in enumerate(target_dates, start=1):
        if max_calls is not None and call_count >= max_calls:
            print(
                f"[중단] 최대 호출 한도({max_calls}회)에 도달하여 수집을 안전하게 일시 중단합니다."
            )
            break

        ym = d_str[:6]
        month_daily_dir = daily_dir / ym
        month_daily_dir.mkdir(parents=True, exist_ok=True)
        final_daily_file = month_daily_dir / f"forecast_{d_str}_{base_time}.parquet"

        # 1. 일별 최종 파일이 이미 존재하면 해당 일자 19개 격자 전수 스킵
        if final_daily_file.exists():
            skipped_days += 1
            continue

        # 2. 격자별 캐시 확인 및 필요한 격자만 병렬 수집
        day_dfs: list[pl.DataFrame] = []
        grids_to_fetch: list[tuple[int, int, Path]] = []

        for nx, ny in unique_grids:
            grid_cache_file = cache_dir / f"grid_{d_str}_{base_time}_{nx}_{ny}.parquet"
            if grid_cache_file.exists():
                df_grid = pl.read_parquet(grid_cache_file)
                day_dfs.append(df_grid)
            else:
                grids_to_fetch.append((nx, ny, grid_cache_file))

        if grids_to_fetch:
            # 최대 호출 한도 고려
            if max_calls is not None:
                remaining_quota = max_calls - call_count
                if remaining_quota <= 0:
                    break
                grids_to_fetch = grids_to_fetch[:remaining_quota]

            quota_exceeded = False
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                future_to_grid = {
                    executor.submit(
                        _fetch_and_cache_grid,
                        api_key,
                        d_str,
                        base_time,
                        nx,
                        ny,
                        cache_file,
                    ): (nx, ny)
                    for nx, ny, cache_file in grids_to_fetch
                }

                for future in as_completed(future_to_grid):
                    try:
                        df_grid = future.result()
                        day_dfs.append(df_grid)
                        with call_lock:
                            call_count += 1
                        if delay_sec > 0:
                            time.sleep(delay_sec)
                    except QuotaExceededError as e:
                        print(
                            f"\n[안내] {e}\n"
                            f"현재까지 수집된 {completed_days}일치의 데이터는 안전하게 보존되었습니다.\n"
                            f"공공데이터포털 트래픽은 매일 자정(00:00)에 자동 리셋되므로, 자정 이후 다시 실행하시면 중단 지점부터 즉시 이어서 수집됩니다."
                        )
                        quota_exceeded = True
                        break

            if quota_exceeded:
                break

        # 3. 19개 격자가 모두 모였으면 일별 최종 Parquet로 병합하고 임시 캐시 삭제
        if len(day_dfs) == total_grids:
            df_daily = pl.concat(day_dfs).sort(
                ["fcst_date", "fcst_time", "category", "nx", "ny"]
            )
            df_daily.write_parquet(final_daily_file)

            # 임시 캐시 정리
            for nx, ny in unique_grids:
                grid_cache_file = (
                    cache_dir / f"grid_{d_str}_{base_time}_{nx}_{ny}.parquet"
                )
                if grid_cache_file.exists():
                    grid_cache_file.unlink()

            completed_days += 1
            if day_idx % 5 == 0 or day_idx == total_days:
                print(
                    f"[{day_idx}/{total_days}일 완료] {d_str} ({base_time}) "
                    f"-> 저장: {final_daily_file.name} ({len(df_daily):,}행) | 누적 호출: {call_count}회"
                )

    print(
        f"[종료] 완료된 일수: {completed_days}일 | 기완료 스킵: {skipped_days}일 | 실제 호출: {call_count}회"
    )
    return {
        "call_count": call_count,
        "completed_days": completed_days,
        "skipped_days": skipped_days,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기상청 단기예보(getVilageFcst) 19개 고유 격자 병렬 체크포인트 일괄 수집기"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-06-01",
        help="수집 시작일 (YYYY-MM-DD, 기본값: 2023-06-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-05-31",
        help="수집 종료일 (YYYY-MM-DD, 기본값: 2026-05-31)",
    )
    parser.add_argument(
        "--base-time",
        type=str,
        default="0500",
        help="발표시각 (예: 0500, 1700 / 기본값: 0500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="출력 루트 디렉터리 (기본값: data/raw/weather/forecast)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="API 호출 간 대기 시간(초, 기본값: 0.05)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="병렬 다운로드 워커(스레드) 수 (기본값: 5)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="단일 실행당 최대 API 호출 횟수 제한 (기본값: 무제한)",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    collect_forecast_data(
        api_key=api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        base_time=args.base_time,
        output_dir=args.output_dir,
        delay_sec=args.delay,
        workers=args.workers,
        max_calls=args.max_calls,
    )


if __name__ == "__main__":
    main()
