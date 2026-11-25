"""
부산교통공사 승하차 데이터 정제 및 K-Fold OOF Target Encoding 모듈

주요 기능:
1. 원시 승하차 데이터(2021.07~2026.08)의 인코딩 감지, 결측치 필터링 및 중복 레코드 클렌징.
2. 24개 시간대 Unpivot(Melt) 및 물리적 datetime(자정 운행 익일 00:00 처리) 생성.
3. 114개 역사 매핑 및 시간대별 승하차 통행량(accumulated_passengers) 산출.
4. 베이지안 평활화(M-estimate, m=10.0)가 적용된 5-Fold OOF 역사 타깃 인코더(StationTargetEncoder).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from busan_metro_congestion.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REFERENCE_DATA_DIR,
)

logger = logging.getLogger("preprocess_subway")

# 24개 표준 시간대 컬럼 정의 (01시-02시 ~ 24시-01시)
HOUR_SLOT_PATTERN = re.compile(r"^(\d{2})시[~-](\d{2})시$")


def detect_csv_encoding(file_path: Path | str) -> str:
    """CSV 파일의 인코딩(utf-8-sig 또는 cp949)을 감지합니다."""
    path = Path(file_path)
    sample_bytes = path.read_bytes()[:4096]
    if sample_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        sample_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def read_subway_csv(file_path: Path | str) -> pl.DataFrame:
    """인코딩을 자동 감지하여 지하철 승하차 원시 CSV를 로드합니다."""
    path = Path(file_path)
    enc = detect_csv_encoding(path)
    return pl.read_csv(path, encoding=enc)


def clean_subway_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    원천 데이터의 결측 역사 및 중복 이상치 레코드를 정제합니다.

    1. 역번호가 null인 빈 행 제거 (2015년 등 공백 행 방어)
    2. (역번호, 년월일, 구분) 기준 중복 레코드 충돌 시 합계가 최대인 정상 행 보존
    """
    if "역번호" not in df.columns:
        raise ValueError("필수 컬럼 '역번호'가 누락되었습니다.")

    # 1. Null 역번호 제거 및 Int64 캐스팅
    df_clean = df.filter(pl.col("역번호").is_not_null()).with_columns(
        pl.col("역번호").cast(pl.Int64)
    )

    # 2. 중복 충돌 시 합계 기준 최상위 정상 행 보존 (2024-07-01 서면·사상 중복 레코드 클렌징)
    sort_cols = ["역번호", "년월일", "구분"]
    desc_flags = [False, False, False]
    if "합계" in df_clean.columns:
        sort_cols.append("합계")
        desc_flags.append(True)

    df_dedup = (
        df_clean.sort(sort_cols, descending=desc_flags)
        .unique(subset=["역번호", "년월일", "구분"], keep="first")
        .sort(["역번호", "년월일", "구분"])
    )
    return df_dedup


def normalize_subway_date(df: pl.DataFrame) -> pl.DataFrame:
    """'년월일' 컬럼을 일관된 Date 타입('date')으로 변환합니다."""
    return df.with_columns(
        pl.col("년월일")
        .cast(pl.Utf8)
        .str.replace_all("-", "")
        .str.to_date("%Y%m%d")
        .alias("date")
    )


def extract_hour_from_slot(slot_name: str) -> int:
    """
    시간대 슬롯명('01시-02시', '24시-01시' 등)에서 시작 시각(1~24)을 추출합니다.
    - 01시-02시 -> 1
    - 23시-24시 -> 23
    - 24시-01시 -> 24 (심야 운행)
    """
    match = HOUR_SLOT_PATTERN.match(slot_name.strip())
    if match:
        start_h = int(match.group(1))
        return 24 if start_h == 0 else start_h
    nums = re.findall(r"\d+", slot_name)
    if nums:
        h = int(nums[0])
        return 24 if h == 0 else h
    raise ValueError(f"유효하지 않은 시간대 슬롯 형식입니다: {slot_name}")


def unpivot_subway_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    24개 시간대 컬럼을 Unpivot(Melt)하여 승차 및 하차 인원을 정렬하고,
    물리적 타임스탬프(datetime)를 생성합니다.
    """
    # 시간대 컬럼 식별
    hour_cols = [c for c in df.columns if "시" in c]
    if not hour_cols:
        raise ValueError("시간대 컬럼을 찾을 수 없습니다.")

    # 1. Unpivot
    melted = df.unpivot(
        index=["역번호", "date", "구분"],
        on=hour_cols,
        variable_name="시간대",
        value_name="인원",
    )

    # 2. 시간대 정수 변환 (1~24)
    hour_mapping = {c: extract_hour_from_slot(c) for c in hour_cols}
    melted = melted.with_columns(
        pl.col("시간대")
        .replace_strict(hour_mapping, default=None, return_dtype=pl.Int8)
        .alias("hour")
    ).filter(pl.col("hour").is_not_null())

    # 3. 승차 / 하차 분리 및 조인
    df_in = melted.filter(pl.col("구분") == "승차").select(
        [
            pl.col("역번호").alias("station_id"),
            pl.col("date"),
            pl.col("hour"),
            pl.col("인원").cast(pl.Int32).alias("passengers_in"),
        ]
    )
    df_out = melted.filter(pl.col("구분") == "하차").select(
        [
            pl.col("역번호").alias("station_id"),
            pl.col("date"),
            pl.col("hour"),
            pl.col("인원").cast(pl.Int32).alias("passengers_out"),
        ]
    )

    # Outer join으로 누락 방지 (결측치는 null로 유지하여 Imputer로 보정 가능하게 함)
    merged = df_in.join(
        df_out, on=["station_id", "date", "hour"], how="full", coalesce=True
    )

    # 4. 물리적 datetime 생성
    # - 1~23시: date + hour
    # - 24시: date + 1일, 00:00:00 (익일 자정 정시 타임스탬프)
    merged = merged.with_columns(
        pl.when(pl.col("hour") == 24)
        .then(pl.col("date").dt.offset_by("1d").cast(pl.Datetime))
        .otherwise(pl.col("date").cast(pl.Datetime) + pl.duration(hours=pl.col("hour")))
        .alias("datetime")
    )

    return merged


def join_station_master(
    df: pl.DataFrame,
    station_master_path: Path | str | None = None,
) -> pl.DataFrame:
    """114개 역사 마스터(station_master.parquet)와 조인하여 표준 노선번호 및 역사명을 결합합니다."""
    if station_master_path is None:
        master_file = REFERENCE_DATA_DIR / "station_master.parquet"
    else:
        master_file = Path(station_master_path)

    if not master_file.exists():
        raise FileNotFoundError(f"역사 마스터 파일이 존재하지 않습니다: {master_file}")

    stn_master = pl.read_parquet(master_file).select(
        [
            pl.col("station_id"),
            pl.col("line_num").cast(pl.Int8).alias("line_no"),
            pl.col("station_name"),
        ]
    )

    joined = df.join(stn_master, on="station_id", how="inner")
    return joined


def calculate_traffic(df: pl.DataFrame) -> pl.DataFrame:
    """시간대별 역사 총 통행량(accumulated_passengers = in + out)을 산출합니다."""
    return df.with_columns(
        (pl.col("passengers_in") + pl.col("passengers_out"))
        .cast(pl.Int32)
        .alias("accumulated_passengers")
    )


def process_subway_data(
    file_path: Path | str | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    station_master_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    단일 CSV 또는 기본 경로의 지하철 승하차 데이터를 정제하여 표준 스키마의 Polars DataFrame으로 반환합니다.

    반환 컬럼:
    [datetime, date, hour, line_no, station_id, station_name,
     passengers_in, passengers_out, accumulated_passengers]
    """
    if file_path is None:
        raw_dir = RAW_DATA_DIR / "subway"
        raw_files = sorted(raw_dir.glob("*.csv"))
        raw_files = [f for f in raw_files if "역사정보" not in f.name]
        if not raw_files:
            raise FileNotFoundError(
                f"지하철 원시 CSV 파일이 존재하지 않습니다: {raw_dir}"
            )
        file_path = raw_files[-1]

    logger.info(f"지하철 승하차 데이터 로드: {file_path}")
    raw_df = read_subway_csv(file_path)

    # 1. 정제 및 날짜 정규화
    cleaned_df = clean_subway_data(raw_df)
    dated_df = normalize_subway_date(cleaned_df)

    # 2. 기간 필터링
    if start_date is not None:
        s_dt = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(start_date, str)
            else start_date
        )
        dated_df = dated_df.filter(pl.col("date") >= s_dt)
    if end_date is not None:
        e_dt = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            if isinstance(end_date, str)
            else end_date
        )
        dated_df = dated_df.filter(pl.col("date") <= e_dt)

    # 3. Unpivot 및 역사 마스터 조인
    unpivoted_df = unpivot_subway_data(dated_df)
    joined_df = join_station_master(unpivoted_df, station_master_path)

    # 4. 통행량 산출
    result_df = calculate_traffic(joined_df)

    # 5. 스키마 정렬 및 정렬 순서 정의
    ordered_cols = [
        "datetime",
        "date",
        "hour",
        "line_no",
        "station_id",
        "station_name",
        "passengers_in",
        "passengers_out",
        "accumulated_passengers",
    ]
    return result_df.select(ordered_cols).sort(["datetime", "station_id"])


def build_subway_interim(
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """
    [1단계: raw -> interim]
    원시 승하차 CSV 파일들을 로드하여 인코딩 정규화,
    결측/중복 레코드 클렌징, 24개 시간대 Unpivot(Melt)을 수행하고
    data/interim/subway/subway_interim.parquet에 표준 시계열로 저장합니다.

    반환 컬럼:
    [datetime, date, hour, station_id, passengers_in, passengers_out]
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR / "subway"
    else:
        raw_dir = Path(raw_dir)

    if output_path is None:
        output_file = INTERIM_DATA_DIR / "subway" / "subway_interim.parquet"
    else:
        output_file = Path(output_path)

    csv_files = sorted(raw_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "역사정보" not in f.name]
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    source_files = csv_files + parquet_files

    if not source_files:
        raise FileNotFoundError(f"경로에 승하차 CSV 파일이 없습니다: {raw_dir}")

    dfs: list[pl.DataFrame] = []
    for f in source_files:
        logger.info(f"Interim 변환 중: {f.name}")
        raw_df = (
            pl.read_parquet(f) if f.suffix.lower() == ".parquet" else read_subway_csv(f)
        )
        cleaned_df = clean_subway_data(raw_df)
        dated_df = normalize_subway_date(cleaned_df)
        unpivoted_df = unpivot_subway_data(dated_df)
        dfs.append(unpivoted_df)

    # 파일 간 중복 시 후순위 파일(최신 연도 CSV 또는 API 수집 Parquet) 우선 보존
    combined = (
        pl.concat(dfs, how="vertical_relaxed")
        .unique(subset=["datetime", "station_id"], keep="last", maintain_order=True)
        .sort(["datetime", "station_id"])
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_file)
    logger.info(f"subway_interim 저장 완료: {output_file} (총 {len(combined):,}행)")
    return combined


def build_subway_processed(
    interim_path: Path | str | None = None,
    output_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
    start_date: str | None = "2023-06-01",
    end_date: str | None = "2026-08-31",
    station_master_path: Path | str | None = None,
    force_rebuild: bool = False,
) -> pl.DataFrame:
    """
    [2단계: interim -> processed]
    subway_interim.parquet를 로드(부재 또는 force_rebuild=True 시 자동 빌드)하여
    기간 필터링(start_date/end_date가 None이면 해당 경계 미제한),
    station_master.parquet 조인(114개 역사 매핑) 및 통행량(accumulated_passengers)을 산출하여
    data/processed/subway_processed.parquet에 저장합니다.

    반환 컬럼:
    [datetime, date, hour, line_no, station_id, station_name,
     passengers_in, passengers_out, accumulated_passengers]
    """
    if interim_path is None:
        interim_file = INTERIM_DATA_DIR / "subway" / "subway_interim.parquet"
    else:
        interim_file = Path(interim_path)

    if force_rebuild or not interim_file.exists():
        logger.info(
            f"interim 파일을 원시 데이터에서 빌드합니다: {interim_file} (force_rebuild={force_rebuild})"
        )
        interim_df = build_subway_interim(raw_dir=raw_dir, output_path=interim_file)
    else:
        interim_df = pl.read_parquet(interim_file)

    if output_path is None:
        output_file = PROCESSED_DATA_DIR / "subway_processed.parquet"
    else:
        output_file = Path(output_path)

    # 1. 기간 필터링 (start_date / end_date가 None이면 해당 경계 무제한)
    filtered = interim_df
    if start_date is not None:
        s_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        filtered = filtered.filter(pl.col("date") >= s_dt)
    if end_date is not None:
        e_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        filtered = filtered.filter(pl.col("date") <= e_dt)
    if len(filtered) == 0:
        raise ValueError(
            f"지정된 기간({start_date or '*'} ~ {end_date or '*'})에 해당하는 데이터가 없습니다."
        )

    # 2. 역사 마스터 조인 (114개 역)
    joined = join_station_master(filtered, station_master_path)

    # 3. 통행량 산출
    result = calculate_traffic(joined)

    ordered_cols = [
        "datetime",
        "date",
        "hour",
        "line_no",
        "station_id",
        "station_name",
        "passengers_in",
        "passengers_out",
        "accumulated_passengers",
    ]
    final_df = result.select(ordered_cols).sort(["datetime", "station_id"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(output_file)
    logger.info(f"subway_processed 저장 완료: {output_file} (총 {len(final_df):,}행)")
    return final_df


# =====================================================================
# 베이지안 평활화 기반 5-Fold OOF 역사 타깃 인코더
# =====================================================================


class StationTargetEncoder:
    """
    K-Fold OOF Target Encoder.

    - 5-Fold OOF(Out-of-Fold) 분할로 자기 참조 누수(Data Leakage) 차단.
    - 베이지안 평활화 (M-estimate, m=10.0):
      S_i = (n_i * y_i_bar + m * y_global_bar) / (n_i + m)
    - 전체 학습셋 기준 최종 매핑 테이블 빌드 및 artifacts 영속화.
    - 추론/서빙 시 O(1) Left Join 및 미관측 역사 글로벌 평균(y_global_bar) 대치.
    """

    def __init__(
        self,
        m: float = 10.0,
        n_splits: int = 5,
        seed: int = 42,
        split_strategy: str = "date",
    ) -> None:
        self.m = float(m)
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.split_strategy = str(split_strategy)
        self.global_mean: float = 0.0
        self.station_map: dict[int, float] = {}

    def _compute_smoothed_means(
        self,
        df: pl.DataFrame,
        station_col: str,
        target_col: str,
    ) -> tuple[dict[int, float], float]:
        """주어진 데이터에 대해 베이지안 평활화된 역사별 타깃 평균과 글로벌 평균을 계산합니다."""
        if len(df) == 0:
            return {}, 0.0

        global_mean = float(df[target_col].mean() or 0.0)

        # 역사별 n_i 및 y_bar_i 계산
        agg_df = df.group_by(station_col).agg(
            [
                pl.len().alias("count"),
                pl.col(target_col).mean().alias("mean"),
            ]
        )

        mapping: dict[int, float] = {}
        for row in agg_df.iter_rows(named=True):
            stn_id = int(row[station_col])
            n_i = float(row["count"])
            y_bar_i = float(row["mean"])
            smoothed = (n_i * y_bar_i + self.m * global_mean) / (n_i + self.m)
            mapping[stn_id] = float(smoothed)

        return mapping, global_mean

    def fit_transform(
        self,
        df: pl.DataFrame,
        station_col: str = "station_id",
        target_col: str = "accumulated_passengers",
    ) -> pl.DataFrame:
        """
        학습 데이터에 대해 5-Fold OOF 타깃 인코딩을 적용하여 정보 누수 없이
        'station_target_encoded' 컬럼을 생성하고, 서빙용 매핑 테이블을 빌드합니다.
        split_strategy='date'인 경우 날짜 단위 블록 분할로 시계열 누출을 차단합니다.
        """
        n_rows = len(df)
        if n_rows == 0:
            return df.with_columns(
                pl.lit(None, dtype=pl.Float32).alias("station_target_encoded")
            )

        # 0. 입력 데이터프레임 결정론적 사전 정렬
        sort_keys = [c for c in ["datetime", station_col] if c in df.columns]
        if sort_keys:
            df = df.sort(sort_keys)

        # 1. 전체 학습 데이터 기준 최종 매핑 테이블 산출 (서빙용)
        self.station_map, self.global_mean = self._compute_smoothed_means(
            df, station_col, target_col
        )

        # 2. 5-Fold OOF 분할 (날짜 단위 블록 분할 지원)
        rng = np.random.default_rng(self.seed)
        if self.split_strategy == "date" and "date" in df.columns:
            unique_dates = df["date"].unique().sort()
            n_dates = len(unique_dates)
            if n_dates >= self.n_splits:
                date_folds = rng.integers(0, self.n_splits, size=n_dates)
                date_fold_map = dict(zip(unique_dates.to_list(), date_folds))
                fold_indices = [date_fold_map.get(d, 0) for d in df["date"].to_list()]
            else:
                fold_indices = rng.integers(0, self.n_splits, size=n_rows)
        else:
            fold_indices = rng.integers(0, self.n_splits, size=n_rows)

        df_with_fold = df.with_columns(pl.Series("_fold", fold_indices, dtype=pl.Int32))

        oof_dfs: list[pl.DataFrame] = []
        for k in range(self.n_splits):
            train_fold = df_with_fold.filter(pl.col("_fold") != k)
            val_fold = df_with_fold.filter(pl.col("_fold") == k)

            fold_map, fold_global_mean = self._compute_smoothed_means(
                train_fold, station_col, target_col
            )

            # val_fold에 fold_map 적용
            encoded_vals = [
                fold_map.get(int(sid), fold_global_mean)
                for sid in val_fold[station_col].to_list()
            ]
            val_encoded = val_fold.with_columns(
                pl.Series(
                    "station_target_encoded",
                    encoded_vals,
                    dtype=pl.Float32,
                )
            )
            oof_dfs.append(val_encoded)

        result = pl.concat(oof_dfs).drop("_fold")
        if sort_keys:
            result = result.sort(sort_keys)
        return result

    def transform(
        self,
        df: pl.DataFrame,
        station_col: str = "station_id",
    ) -> pl.DataFrame:
        """
        검증/테스트/추론 데이터에 대해 사전 빌드된 매핑 테이블을 O(1) 매핑하고,
        미관측 역사는 글로벌 평균(global_mean)으로 수렴 대치합니다.
        """
        encoded_vals = [
            self.station_map.get(int(sid), self.global_mean)
            for sid in df[station_col].to_list()
        ]
        return df.with_columns(
            pl.Series(
                "station_target_encoded",
                encoded_vals,
                dtype=pl.Float32,
            )
        )

    def save_mapping(self, output_path: Path | str) -> None:
        """
        매핑 딕셔너리와 글로벌 평균을 Parquet 파일로 영속화합니다.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {"station_id": sid, "target_encoded": val}
            for sid, val in self.station_map.items()
        ]
        # 글로벌 평균을 station_id=0 특수 행으로 함께 보존
        rows.append({"station_id": 0, "target_encoded": self.global_mean})

        map_df = pl.DataFrame(
            rows,
            schema={"station_id": pl.Int64, "target_encoded": pl.Float32},
        )
        map_df.write_parquet(path)
        logger.info(f"StationTargetEncoder 매핑 테이블 저장 완료: {path}")

    @classmethod
    def load_mapping(
        cls, input_path: Path | str, m: float = 10.0
    ) -> StationTargetEncoder:
        """영속화된 Parquet 파일로부터 인코더를 복원합니다."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"매핑 파일이 존재하지 않습니다: {path}")

        map_df = pl.read_parquet(path)
        encoder = cls(m=m)

        global_row = map_df.filter(pl.col("station_id") == 0)
        if len(global_row) > 0:
            encoder.global_mean = float(global_row["target_encoded"][0])

        stn_rows = map_df.filter(pl.col("station_id") != 0)
        encoder.station_map = {
            int(r["station_id"]): float(r["target_encoded"])
            for r in stn_rows.iter_rows(named=True)
        }
        return encoder


# =====================================================================
# 시계열 요일·시간대 프로파일 결측치 보정기 (시계열 누수 방지 준수)
# =====================================================================


class SubwayProfileImputer:
    """
    학습 데이터의 시간 범위(Train Time Range)만을 기준으로
    역사별·요일별·시간대별 정상 통행량 프로파일을 학습하고 결측치를 보정하는 시계열 Imputer.

    보정 계층 (Hierarchical Backoff):
    1. Level 1 (최우선): 역사(station_id) × 요일(weekday, 1~7) × 시간대(hour, 1~24) 중앙값
    2. Level 2 (보조): 역사(station_id) × 시간대(hour) 중앙값
    3. Level 3 (글로벌): 전체 역사 시간대(hour) 중앙값
    """

    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy
        self.profile_l1: pl.DataFrame | None = None
        self.profile_l2: pl.DataFrame | None = None
        self.profile_l3: pl.DataFrame | None = None

    def fit(self, train_df: pl.DataFrame) -> SubwayProfileImputer:
        """
        오직 학습 기간 데이터만을 사용하여 보정 기준 프로파일을 산출합니다 (Data Leakage 원천 차단).
        """
        if len(train_df) == 0:
            raise ValueError("학습 데이터가 비어 있어 프로파일을 생성할 수 없습니다.")

        df_feat = train_df.with_columns(
            pl.col("date").dt.weekday().cast(pl.Int8).alias("_weekday")
        )

        # Level 1: (station_id, weekday, hour)
        self.profile_l1 = df_feat.group_by(["station_id", "_weekday", "hour"]).agg(
            [
                pl.col("passengers_in")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l1_in"),
                pl.col("passengers_out")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l1_out"),
            ]
        )

        # Level 2: (station_id, hour)
        self.profile_l2 = df_feat.group_by(["station_id", "hour"]).agg(
            [
                pl.col("passengers_in")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l2_in"),
                pl.col("passengers_out")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l2_out"),
            ]
        )

        # Level 3: (hour)
        self.profile_l3 = df_feat.group_by("hour").agg(
            [
                pl.col("passengers_in")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l3_in"),
                pl.col("passengers_out")
                .drop_nulls()
                .median()
                .round(0)
                .cast(pl.Int32)
                .alias("_imp_l3_out"),
            ]
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        학습 데이터에서 산출된 프로파일을 기반으로 결측치를 계층적으로 보정합니다.
        (행을 삭제하지 않고 통계적 정상치로 보정)
        """
        if (
            self.profile_l1 is None
            or self.profile_l2 is None
            or self.profile_l3 is None
        ):
            raise RuntimeError(
                "Imputer가 학습되지 않았습니다. 먼저 fit()을 호출하세요."
            )

        df_work = df.with_columns(
            pl.col("date").dt.weekday().cast(pl.Int8).alias("_weekday")
        )

        # 계층적 Join
        joined = (
            df_work.join(
                self.profile_l1,
                on=["station_id", "_weekday", "hour"],
                how="left",
            )
            .join(self.profile_l2, on=["station_id", "hour"], how="left")
            .join(self.profile_l3, on="hour", how="left")
        )

        # 결측치 Coalesce 보정
        imputed_in = pl.coalesce(
            [
                pl.col("passengers_in"),
                pl.col("_imp_l1_in"),
                pl.col("_imp_l2_in"),
                pl.col("_imp_l3_in"),
                pl.lit(0, dtype=pl.Int32),
            ]
        ).cast(pl.Int32)

        imputed_out = pl.coalesce(
            [
                pl.col("passengers_out"),
                pl.col("_imp_l1_out"),
                pl.col("_imp_l2_out"),
                pl.col("_imp_l3_out"),
                pl.lit(0, dtype=pl.Int32),
            ]
        ).cast(pl.Int32)

        result = joined.with_columns(
            [
                imputed_in.alias("passengers_in"),
                imputed_out.alias("passengers_out"),
            ]
        ).with_columns(
            (pl.col("passengers_in") + pl.col("passengers_out"))
            .cast(pl.Int32)
            .alias("accumulated_passengers")
        )

        # 임시 컬럼 제거
        drop_cols = [
            c for c in result.columns if c.startswith("_imp_") or c == "_weekday"
        ]
        return result.drop(drop_cols)


# =====================================================================
# 시간대 및 요일 주기성 삼각함수 인코딩 (Cyclical Encoding)
# =====================================================================


def add_cyclical_time_features(
    df: pl.DataFrame,
    hour_col: str = "hour",
    date_col: str = "date",
    include_weekday: bool = True,
) -> pl.DataFrame:
    """
    시간대 및 요일에 대해 삼각함수 기반 Cyclical Encoding을 수행합니다.

    시간대(1~24시)는 24시-01시(자정 0시)와 01시-02시(1시) 사이의 연속성을 보존하기 위해
    hour_sin, hour_cos를 산출하여 수치적 불연속성(Discontinuity)을 제거합니다:
        hour_rad = 2π * (hour % 24) / 24.0
        hour_sin = sin(hour_rad), hour_cos = cos(hour_rad)

    요일(1=월~7=일) 또한 7일 주기의 연속 곡선을 형성합니다:
        weekday_rad = 2π * (weekday - 1) / 7.0
        weekday_sin = sin(weekday_rad), weekday_cos = cos(weekday_rad)
    """
    two_pi = 2.0 * np.pi
    exprs = [
        ((pl.col(hour_col) % 24) * (two_pi / 24.0))
        .sin()
        .cast(pl.Float32)
        .alias("hour_sin"),
        ((pl.col(hour_col) % 24) * (two_pi / 24.0))
        .cos()
        .cast(pl.Float32)
        .alias("hour_cos"),
    ]

    if include_weekday and date_col in df.columns:
        weekday_expr = pl.col(date_col).dt.weekday().cast(pl.Float32)
        exprs.extend(
            [
                ((weekday_expr - 1.0) * (two_pi / 7.0))
                .sin()
                .cast(pl.Float32)
                .alias("weekday_sin"),
                ((weekday_expr - 1.0) * (two_pi / 7.0))
                .cos()
                .cast(pl.Float32)
                .alias("weekday_cos"),
            ]
        )

    return df.with_columns(exprs)
