"""사전 추론 예측 결과 조회 API 테스트.

`data/samples/predictions/predictions_sample.parquet`(4개 역 × 2일 × 24시간)와
`data/reference/station_master.parquet`만 사용한다.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from busan_metro_congestion.api.app import create_app
from busan_metro_congestion.api.stations import STATION_MASTER_PATH
from busan_metro_congestion.config import SAMPLES_DIR
from busan_metro_congestion.modeling.predict import PREDICTION_COLUMNS

SAMPLE_PATH = SAMPLES_DIR / "predictions" / "predictions_sample.parquet"
SAMPLE = pl.read_parquet(SAMPLE_PATH)
STATION_ID = 119
SAMPLE_DATE = "2026-12-01"


def _client(predictions_path: Path) -> TestClient:
    app = create_app(predictions_path=predictions_path)
    return TestClient(app, raise_server_exceptions=False)


def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"code", "message"}
    assert body["code"] == code
    assert body["message"]
    return body


def _bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    target = tmp_path / "predictions.parquet"
    shutil.copyfile(SAMPLE_PATH, target)
    return target


@pytest.fixture
def client(store_path: Path) -> TestClient:
    return _client(store_path)


def test_health_ignores_store_state(tmp_path: Path) -> None:
    response = _client(tmp_path / "missing.parquet").get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stations_returns_master_sorted(client: TestClient) -> None:
    body = client.get("/api/v1/stations").json()
    master = pl.read_parquet(STATION_MASTER_PATH)
    assert len(body) == len(master)
    assert [s["station_id"] for s in body] == sorted(master["station_id"].to_list())
    assert set(body[0]) == {"station_id", "station_name", "line_num", "lat", "lng"}


@pytest.mark.parametrize("line_num", [1, 2, 3, 4])
def test_stations_filter_by_line(client: TestClient, line_num: int) -> None:
    body = client.get("/api/v1/stations", params={"line_num": line_num}).json()
    assert body
    assert {s["line_num"] for s in body} == {line_num}


@pytest.mark.parametrize("line_num", [0, 5])
def test_stations_invalid_line(client: TestClient, line_num: int) -> None:
    response = client.get("/api/v1/stations", params={"line_num": line_num})
    _assert_error(response, 422, "INVALID_LINE")


def test_model_info_summarizes_store(client: TestClient) -> None:
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_version": "0.1.0-set1-sample",
        "generated_at": "2026-11-25T19:00:00+09:00",
        "available_from": "2026-12-01",
        "available_to": "2026-12-05",
        "station_count": 4,
        "row_count": len(SAMPLE),
    }


def test_predictions_all_hours(client: TestClient) -> None:
    response = client.get(
        "/api/v1/predictions", params={"station_id": STATION_ID, "date": SAMPLE_DATE}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["station_id"] == STATION_ID
    assert body["station_name"] == "서면"
    assert body["line_num"] == 1
    assert body["date"] == SAMPLE_DATE
    assert body["model_version"] == "0.1.0-set1-sample"
    assert body["generated_at"] == "2026-11-25T19:00:00+09:00"
    assert [p["hour"] for p in body["predictions"]] == list(range(1, 25))

    expected = SAMPLE.filter(
        (pl.col("station_id") == STATION_ID)
        & (pl.col("target_date") == dt.date.fromisoformat(SAMPLE_DATE))
    ).sort("target_hour")
    assert [p["predicted_passengers"] for p in body["predictions"]] == [
        round(v, 1) for v in expected["predicted_passengers"].to_list()
    ]


def test_predictions_selected_hours_sorted_and_deduplicated(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/predictions",
        params={"station_id": STATION_ID, "date": SAMPLE_DATE, "hour": [24, 8, 8]},
    )
    assert response.status_code == 200
    assert response.json()["predictions"] == [
        {
            "hour": 8,
            "target_datetime": "2026-12-01T08:00:00+09:00",
            "predicted_passengers": 4195.6,
        },
        {
            "hour": 24,
            "target_datetime": "2026-12-02T00:00:00+09:00",
            "predicted_passengers": 168.0,
        },
    ]


def test_response_fields_cover_prediction_schema(client: TestClient) -> None:
    body = client.get(
        "/api/v1/predictions",
        params={"station_id": 203, "date": "2026-12-05", "hour": 24},
    ).json()
    hourly = body["predictions"][0]
    row = SAMPLE.filter(
        (pl.col("station_id") == 203)
        & (pl.col("target_date") == dt.date(2026, 12, 5))
        & (pl.col("target_hour") == 24)
    ).row(0, named=True)
    kst = dt.timezone(dt.timedelta(hours=9))

    from_response = {
        "station_id": body["station_id"],
        "station_name": body["station_name"],
        "line_num": body["line_num"],
        "target_date": dt.date.fromisoformat(body["date"]),
        "target_hour": hourly["hour"],
        "target_datetime": dt.datetime.fromisoformat(hourly["target_datetime"]),
        "predicted_passengers": hourly["predicted_passengers"],
        "model_version": body["model_version"],
        "generated_at": dt.datetime.fromisoformat(body["generated_at"]),
    }
    from_store = {
        **row,
        "target_datetime": row["target_datetime"].replace(tzinfo=kst),
        "predicted_passengers": round(row["predicted_passengers"], 1),
        "generated_at": row["generated_at"].replace(tzinfo=dt.timezone.utc),
    }
    assert list(from_response) == PREDICTION_COLUMNS
    assert from_response == from_store
    assert from_response["target_datetime"] == dt.datetime(2026, 12, 6, tzinfo=kst)


def test_predictions_unknown_station(client: TestClient) -> None:
    response = client.get(
        "/api/v1/predictions", params={"station_id": 999, "date": SAMPLE_DATE}
    )
    body = _assert_error(response, 404, "STATION_NOT_FOUND")
    assert "999" in body["message"]


@pytest.mark.parametrize(
    ("station_id", "date"),
    [(STATION_ID, "2027-01-01"), (STATION_ID, "2026-12-03"), (101, SAMPLE_DATE)],
    ids=["after-range", "gap-inside-range", "station-not-in-store"],
)
def test_predictions_date_out_of_range(
    client: TestClient, station_id: int, date: str
) -> None:
    response = client.get(
        "/api/v1/predictions", params={"station_id": station_id, "date": date}
    )
    body = _assert_error(response, 422, "DATE_OUT_OF_RANGE")
    assert "2026-12-01 ~ 2026-12-05" in body["message"]


@pytest.mark.parametrize("hour", [0, 25])
def test_predictions_invalid_hour(client: TestClient, hour: int) -> None:
    response = client.get(
        "/api/v1/predictions",
        params={"station_id": STATION_ID, "date": SAMPLE_DATE, "hour": hour},
    )
    body = _assert_error(response, 422, "INVALID_HOUR")
    assert str(hour) in body["message"]


@pytest.mark.parametrize(
    "params",
    [
        {"date": SAMPLE_DATE},
        {"station_id": STATION_ID},
        {"station_id": STATION_ID, "date": "2026-12-32"},
        {"station_id": "abc", "date": SAMPLE_DATE},
    ],
    ids=["missing-station", "missing-date", "bad-date", "bad-station"],
)
def test_predictions_invalid_request(client: TestClient, params: dict) -> None:
    response = client.get("/api/v1/predictions", params=params)
    _assert_error(response, 422, "INVALID_REQUEST")


def test_missing_store_returns_not_ready(tmp_path: Path) -> None:
    client = _client(tmp_path / "missing.parquet")
    _assert_error(client.get("/api/v1/model-info"), 503, "PREDICTION_NOT_READY")
    response = client.get(
        "/api/v1/predictions", params={"station_id": STATION_ID, "date": SAMPLE_DATE}
    )
    _assert_error(response, 503, "PREDICTION_NOT_READY")


@pytest.mark.parametrize(
    "broken",
    [SAMPLE.head(0), SAMPLE.drop("model_version")],
    ids=["empty", "missing-column"],
)
def test_invalid_store_returns_not_ready(tmp_path: Path, broken: pl.DataFrame) -> None:
    path = tmp_path / "predictions.parquet"
    broken.write_parquet(path)
    _assert_error(_client(path).get("/api/v1/model-info"), 503, "PREDICTION_NOT_READY")


def test_missing_hour_in_store_returns_not_ready(tmp_path: Path) -> None:
    path = tmp_path / "predictions.parquet"
    SAMPLE.filter(
        ~((pl.col("station_id") == STATION_ID) & (pl.col("target_hour") == 8))
    ).write_parquet(path)
    client = _client(path)

    response = client.get(
        "/api/v1/predictions",
        params={"station_id": STATION_ID, "date": SAMPLE_DATE, "hour": 8},
    )
    _assert_error(response, 503, "PREDICTION_NOT_READY")

    ok = client.get(
        "/api/v1/predictions",
        params={"station_id": STATION_ID, "date": SAMPLE_DATE, "hour": 9},
    )
    assert ok.status_code == 200


def test_store_recovers_after_file_appears(tmp_path: Path) -> None:
    path = tmp_path / "predictions.parquet"
    client = _client(path)
    _assert_error(client.get("/api/v1/model-info"), 503, "PREDICTION_NOT_READY")

    shutil.copyfile(SAMPLE_PATH, path)
    assert client.get("/api/v1/model-info").status_code == 200


def test_store_reloads_when_file_replaced(client: TestClient, store_path: Path) -> None:
    assert client.get("/api/v1/model-info").json()["row_count"] == len(SAMPLE)

    replaced = SAMPLE.filter(pl.col("station_id") == STATION_ID).with_columns(
        pl.lit("0.2.0-set1", dtype=pl.String).alias("model_version")
    )
    replaced.write_parquet(store_path)
    _bump_mtime(store_path)

    body = client.get("/api/v1/model-info").json()
    assert body["model_version"] == "0.2.0-set1"
    assert body["station_count"] == 1
    assert body["row_count"] == len(replaced)


def test_unknown_path_and_method(client: TestClient) -> None:
    _assert_error(client.get("/api/v1/unknown"), 404, "INVALID_REQUEST")
    _assert_error(client.post("/api/v1/predictions"), 405, "INVALID_REQUEST")
