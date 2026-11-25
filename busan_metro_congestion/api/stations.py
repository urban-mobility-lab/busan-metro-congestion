"""역 목록 (`data/reference/station_master.parquet`) 조회."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from busan_metro_congestion.api.schemas import Station
from busan_metro_congestion.config import REFERENCE_DATA_DIR

STATION_MASTER_PATH = REFERENCE_DATA_DIR / "station_master.parquet"


class StationRepository:
    def __init__(self, path: Path = STATION_MASTER_PATH) -> None:
        df = pl.read_parquet(path, columns=list(Station.model_fields)).sort(
            "station_id"
        )
        self._stations = [Station(**row) for row in df.to_dicts()]
        self._by_id = {s.station_id: s for s in self._stations}

    def all(self, line_num: int | None = None) -> list[Station]:
        if line_num is None:
            return list(self._stations)
        return [s for s in self._stations if s.line_num == line_num]

    def get(self, station_id: int) -> Station | None:
        return self._by_id.get(station_id)
