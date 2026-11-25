"""배치 추론 예측 결과 저장소(`predictions.parquet`) 조회.

저장소는 `modeling.predict.run_batch_inference()`가 원자적으로 교체하는 9컬럼 Parquet이다.
첫 요청 시 로드해 메모리에 두고, 파일의 수정 시각·크기가 바뀌면 다음 요청에서 다시 읽는다.
로드나 스키마 검증에 실패한 상태는 캐시하지 않는다.
"""

from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from busan_metro_congestion.modeling.predict import (
    DEFAULT_PREDICTIONS_PATH,
    validate_predictions_df,
)

KST = dt.timezone(dt.timedelta(hours=9), "KST")


class PredictionStoreUnavailableError(RuntimeError):
    """저장소 파일 부재, 빈 저장소, 스키마 계약 위반 등으로 조회할 수 없는 상태."""


@dataclass(frozen=True)
class StoreInfo:
    model_version: str
    generated_at: dt.datetime
    available_from: dt.date
    available_to: dt.date
    station_count: int
    row_count: int


@dataclass(frozen=True)
class _Snapshot:
    signature: tuple[int, int]
    df: pl.DataFrame
    info: StoreInfo


def utc_naive_to_kst(value: dt.datetime) -> dt.datetime:
    """저장소의 naive UTC 시각을 KST aware 시각으로 바꾼다."""
    return value.replace(tzinfo=dt.timezone.utc).astimezone(KST)


def kst_naive_to_aware(value: dt.datetime) -> dt.datetime:
    """저장소의 naive KST 시각(`target_datetime`)에 KST 오프셋을 붙인다."""
    return value.replace(tzinfo=KST)


class PredictionStore:
    def __init__(self, path: Path = DEFAULT_PREDICTIONS_PATH) -> None:
        self._path = Path(path)
        self._snapshot: _Snapshot | None = None
        self._lock = threading.Lock()

    def _current(self) -> _Snapshot:
        try:
            stat = self._path.stat()
        except OSError as exc:
            raise PredictionStoreUnavailableError(
                f"예측 결과 저장소를 찾을 수 없습니다: {self._path}"
            ) from exc
        signature = (stat.st_mtime_ns, stat.st_size)

        with self._lock:
            if self._snapshot is not None and self._snapshot.signature == signature:
                return self._snapshot
            try:
                df = pl.read_parquet(self._path)
                validate_predictions_df(df)
            except Exception as exc:
                self._snapshot = None
                raise PredictionStoreUnavailableError(
                    f"예측 결과 저장소를 불러올 수 없습니다: {exc}"
                ) from exc
            self._snapshot = _Snapshot(signature, df, _summarize(df))
            return self._snapshot

    def info(self) -> StoreInfo:
        return self._current().info

    def query(self, station_id: int, date: dt.date) -> pl.DataFrame:
        """한 역·운행일의 행을 `target_hour` 오름차순으로 돌려준다. 없으면 0행."""
        return (
            self._current()
            .df.filter(
                (pl.col("station_id") == station_id) & (pl.col("target_date") == date)
            )
            .sort("target_hour")
        )


def _summarize(df: pl.DataFrame) -> StoreInfo:
    latest = df.sort("generated_at").row(-1, named=True)
    return StoreInfo(
        model_version=str(latest["model_version"]),
        generated_at=utc_naive_to_kst(latest["generated_at"]),
        available_from=df["target_date"].min(),  # type: ignore[arg-type]
        available_to=df["target_date"].max(),  # type: ignore[arg-type]
        station_count=int(df["station_id"].n_unique()),
        row_count=len(df),
    )
