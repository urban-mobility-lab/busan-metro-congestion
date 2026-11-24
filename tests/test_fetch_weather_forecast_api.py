from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from scripts.fetch_weather_forecast_api import (
    collect_forecast_data,
    fetch_vilage_forecast_grid,
    generate_date_range,
    items_to_polars,
    load_unique_grids,
)


def test_generate_date_range() -> None:
    dates = generate_date_range("2023-06-01", "2023-06-03")
    assert dates == ["20230601", "20230602", "20230603"]

    with pytest.raises(ValueError, match="시작일.*늦을 수 없습니다"):
        generate_date_range("2023-06-05", "2023-06-01")


def test_load_unique_grids(tmp_path: Path) -> None:
    grid_parquet = tmp_path / "mock_grid.parquet"
    df_grid = pl.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "station_name": ["역A", "역B", "역C", "역D"],
            "nx": [98, 98, 99, 99],
            "ny": [76, 76, 75, 75],
        }
    )
    df_grid.write_parquet(grid_parquet)

    grids = load_unique_grids(grid_path=grid_parquet)
    assert grids == [(98, 76), (99, 75)]


def test_items_to_polars_empty() -> None:
    df = items_to_polars([])
    assert len(df) == 0
    assert "category" in df.columns
    assert "fcst_value" in df.columns


def test_items_to_polars_valid() -> None:
    items = [
        {
            "baseDate": "20230601",
            "baseTime": "0500",
            "category": "TMP",
            "fcstDate": "20230601",
            "fcstTime": "0600",
            "fcstValue": "16",
            "nx": 98,
            "ny": 76,
        }
    ]
    df = items_to_polars(items)
    assert len(df) == 1
    assert df["base_date"][0] == "20230601"
    assert df["category"][0] == "TMP"
    assert df["fcst_value"][0] == "16"
    assert df["nx"][0] == 98
    assert df["ny"][0] == 76


def test_fetch_vilage_forecast_grid_success() -> None:
    mock_payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {
                "totalCount": 1,
                "items": {
                    "item": [
                        {
                            "baseDate": "20230601",
                            "baseTime": "0500",
                            "category": "TMP",
                            "fcstDate": "20230601",
                            "fcstTime": "0600",
                            "fcstValue": "16",
                            "nx": 98,
                            "ny": 76,
                        }
                    ]
                },
            },
        }
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload

    with patch(
        "scripts.fetch_weather_forecast_api.requests.get",
        return_value=mock_resp,
    ) as mock_get:
        items = fetch_vilage_forecast_grid(
            api_key="test_key",
            base_date="20230601",
            base_time="0500",
            nx=98,
            ny=76,
        )

    assert mock_get.call_count == 1
    assert len(items) == 1
    assert items[0]["category"] == "TMP"


def test_fetch_vilage_forecast_grid_nodata() -> None:
    mock_payload = {
        "response": {
            "header": {"resultCode": "03", "resultMsg": "NODATA_ERROR"},
            "body": {},
        }
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_payload

    with patch(
        "scripts.fetch_weather_forecast_api.requests.get",
        return_value=mock_resp,
    ):
        items = fetch_vilage_forecast_grid(
            api_key="test_key",
            base_date="20230601",
            base_time="0500",
            nx=98,
            ny=76,
        )

    assert items == []


def test_fetch_vilage_forecast_grid_error_retry() -> None:
    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 200
    mock_err_resp.json.return_value = {
        "response": {
            "header": {
                "resultCode": "30",
                "resultMsg": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR",
            }
        }
    }

    # 1. max_retries=3 모두 실패 시 정확히 3회 시도 후 예외 발생 검증
    with (
        patch(
            "scripts.fetch_weather_forecast_api.requests.get",
            return_value=mock_err_resp,
        ) as mock_get,
        patch("scripts.fetch_weather_forecast_api.time.sleep") as mock_sleep,
    ):
        with pytest.raises(RuntimeError, match="OpenAPI 오류"):
            fetch_vilage_forecast_grid(
                api_key="bad_key",
                base_date="20230601",
                base_time="0500",
                nx=98,
                ny=76,
                max_retries=3,
            )

    assert mock_get.call_count == 3
    assert mock_sleep.call_count == 2

    # 2. 1차 실패 후 2차 재시도에서 성공하는 복구 경로 검증
    mock_ok_resp = MagicMock()
    mock_ok_resp.status_code = 200
    mock_ok_resp.json.return_value = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {"items": {"item": [{"category": "TMP", "fcstValue": "20"}]}},
        }
    }
    with (
        patch(
            "scripts.fetch_weather_forecast_api.requests.get",
            side_effect=[mock_err_resp, mock_ok_resp],
        ) as mock_get_recover,
        patch("scripts.fetch_weather_forecast_api.time.sleep") as mock_sleep_recover,
    ):
        items = fetch_vilage_forecast_grid(
            api_key="key",
            base_date="20230601",
            base_time="0500",
            nx=98,
            ny=76,
            max_retries=3,
        )

    assert mock_get_recover.call_count == 2
    assert mock_sleep_recover.call_count == 1
    assert len(items) == 1
    assert items[0]["category"] == "TMP"


def test_fetch_vilage_forecast_grid_quota_exceeded() -> None:
    from scripts.fetch_weather_forecast_api import QuotaExceededError

    # 1. HTTP 429 응답 시 max_retries=3이어도 재시도 없이 1회 호출 후 즉시 중단 검증
    mock_429 = MagicMock()
    mock_429.status_code = 429

    with patch(
        "scripts.fetch_weather_forecast_api.requests.get",
        return_value=mock_429,
    ) as mock_get_429:
        with pytest.raises(QuotaExceededError, match="일일 트래픽 한도"):
            fetch_vilage_forecast_grid(
                api_key="key",
                base_date="20241023",
                base_time="0500",
                nx=98,
                ny=76,
                max_retries=3,
            )
    assert mock_get_429.call_count == 1

    # 2. JSON resultCode="22" 응답 시에도 재시도 없이 1회 호출 후 즉시 중단 검증
    mock_code22 = MagicMock()
    mock_code22.status_code = 200
    mock_code22.json.return_value = {
        "response": {
            "header": {
                "resultCode": "22",
                "resultMsg": "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS_ERROR",
            }
        }
    }
    with patch(
        "scripts.fetch_weather_forecast_api.requests.get",
        return_value=mock_code22,
    ) as mock_get_22:
        with pytest.raises(QuotaExceededError, match="일일 트래픽 한도 초과"):
            fetch_vilage_forecast_grid(
                api_key="key",
                base_date="20241023",
                base_time="0500",
                nx=98,
                ny=76,
                max_retries=3,
            )
    assert mock_get_22.call_count == 1


def test_collect_forecast_data_checkpoint_and_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "forecast"
    grid_parquet = tmp_path / "mock_grid.parquet"
    df_grid = pl.DataFrame(
        {
            "station_id": [1, 2],
            "station_name": ["역A", "역B"],
            "nx": [98, 99],
            "ny": [76, 75],
        }
    )
    df_grid.write_parquet(grid_parquet)

    # 1. First run: grid (98, 76) is already in cache
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df_cached = items_to_polars(
        [
            {
                "baseDate": "20230601",
                "baseTime": "0500",
                "category": "TMP",
                "fcstDate": "20230601",
                "fcstTime": "0600",
                "fcstValue": "16",
                "nx": 98,
                "ny": 76,
            }
        ]
    )
    df_cached.write_parquet(cache_dir / "grid_20230601_0500_98_76.parquet")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {
                "totalCount": 1,
                "items": {
                    "item": [
                        {
                            "baseDate": "20230601",
                            "baseTime": "0500",
                            "category": "POP",
                            "fcstDate": "20230601",
                            "fcstTime": "0600",
                            "fcstValue": "20",
                            "nx": 99,
                            "ny": 75,
                        }
                    ]
                },
            },
        }
    }

    with patch(
        "scripts.fetch_weather_forecast_api.requests.get",
        return_value=mock_resp,
    ) as mock_get:
        res = collect_forecast_data(
            api_key="test",
            start_date="2023-06-01",
            end_date="2023-06-01",
            base_time="0500",
            output_dir=out_dir,
            grid_path=grid_parquet,
            delay_sec=0.0,
        )

    # Only 1 call made (for grid 99, 75) because 98, 76 was in cache
    assert mock_get.call_count == 1
    assert res["call_count"] == 1
    assert res["completed_days"] == 1
    assert res["skipped_days"] == 0

    # Verify final daily file was generated
    daily_file = out_dir / "daily" / "202306" / "forecast_20230601_0500.parquet"
    assert daily_file.exists()
    df_daily = pl.read_parquet(daily_file)
    assert len(df_daily) == 2
    assert sorted(df_daily["category"].to_list()) == ["POP", "TMP"]

    # 2. Second run: daily file already exists -> should skip completely without calling API
    with patch("scripts.fetch_weather_forecast_api.requests.get") as mock_get_again:
        res2 = collect_forecast_data(
            api_key="test",
            start_date="2023-06-01",
            end_date="2023-06-01",
            base_time="0500",
            output_dir=out_dir,
            grid_path=grid_parquet,
            delay_sec=0.0,
        )

    assert mock_get_again.call_count == 0
    assert res2["call_count"] == 0
    assert res2["skipped_days"] == 1
