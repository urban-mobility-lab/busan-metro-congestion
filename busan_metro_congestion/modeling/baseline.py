"""
Fast Track 머신러닝 베이스라인 모델 (LightGBM & Random Forest)

기상청 단기예보(12시간 주기) 기준 2-Track 예측 호라이즌 아키텍처를 지원합니다.
- Track 1: 단기 예측 모델 (horizon="short", <= 12h) -> 기상 피처 8종 + 기저/축제/달력 17종 (총 25개)
- Track 2: 중·장기 예측 모델 (horizon="long", > 12h) -> Weather-Free (기상 배제 17개 피처)
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor

# 1. 지하철 식별 및 고유 기저 피처 (3종)
SUBWAY_BASE_FEATURES = [
    "station_target_encoded",
    "line_num",
    "station_id",
]

# 2. 문화축제 및 대규모 전시 피처 (4종)
FESTIVAL_FEATURES = [
    "nearest_event_rank",
    "max_dist_weight",
    "max_scale_weight",
    "event_impact_score",
]

# 3. 달력 및 시계열 주기성 피처 (10종)
CALENDAR_TEMPORAL_FEATURES = [
    "season",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
]

# 공통 기본 피처: 식별/기저(3) + 축제(4) + 달력/주기(10) = 17종
BASE_HORIZON_FEATURES = (
    SUBWAY_BASE_FEATURES + FESTIVAL_FEATURES + CALENDAR_TEMPORAL_FEATURES
)

# 단기 예측 기상 피처 (8종: lead_time_hours + 핵심 기상 7종)
WEATHER_SHORT_FEATURES = [
    "lead_time_hours",
    "precipitation",
    "pop",
    "temperature",
    "rainfall_type",
    "humidity",
    "wind_speed",
    "sky_condition",
]

SHORT_HORIZON_FEATURES = BASE_HORIZON_FEATURES + WEATHER_SHORT_FEATURES
LONG_HORIZON_FEATURES = BASE_HORIZON_FEATURES  # Weather-Free


def add_years(d: date, years: int) -> date:
    """
    윤년(2월 29일)을 고려하여 date에 정확히 N년을 더합니다.
    - 2월 29일 윤일에서 평년으로 갈 경우 2월 28일로 처리
    """
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return date(d.year + years, 2, 28)


def add_months(d: date, months: int) -> date:
    """월 단위로 정확히 N개월을 더합니다 (말일 초과 방지)."""
    year = d.year + (d.month + months - 1) // 12
    month = (d.month + months - 1) % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def calculate_end_date(start_date: str | date, years: int = 1) -> str:
    """
    시작일로부터 윤년을 고려한 N년 기간([start, end])의 종료일을 반환합니다.
    예:
        2023-06-01 -> 2024-05-31 (1년, 366일)
        2024-03-01 -> 2025-02-28 (1년, 365일)
        2024-02-29 -> 2025-02-28 (1년, 366일)
    """
    s_d = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
    if s_d.month == 2 and s_d.day == 29:
        target_year = s_d.year + years
        e_d = date(target_year, 2, 28)
    else:
        e_d = add_years(s_d, years) - timedelta(days=1)
    return e_d.isoformat()


def validate_date_interval_yearly(
    start_str: str,
    end_str: str,
    label: str = "date",
    max_years: int = 10,
) -> int:
    """
    종료일이 시작일 기준 1년 단위(1년, 2년 등 정수 년, 윤년 고려)인지 검증합니다.

    반환값:
        int: 기간(년수 N)
    발생 예외:
        ValueError: 1년 단위가 아니거나 종료일이 시작일 이전인 경우
    """
    s_d = date.fromisoformat(start_str)
    e_d = date.fromisoformat(end_str)

    if e_d < s_d:
        raise ValueError(
            f"{label} 종료일({end_str})은 시작일({start_str})보다 앞설 수 없습니다."
        )

    for k in range(1, max_years + 1):
        expected_end = date.fromisoformat(calculate_end_date(s_d, k))
        if e_d == expected_end:
            return k

    rec_1y = calculate_end_date(s_d, 1)
    raise ValueError(
        f"{label} 종료일({end_str})은 시작일({start_str}) 기준 1년 단위(윤년 고려)여야 합니다. "
        f"(권장 1년 종료일: {rec_1y})"
    )


def generate_rolling_tracks(
    n_tracks: int = 3,
    shift_months: int = 1,
    base_train_start: str = "2023-06-01",
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    설정된 Track 수(최대 3) 및 시프트 개월 수에 따라
    3-Track 롤링 슬라이딩 윈도우 및 Holdout Test 날짜 범위를 동적으로 생성합니다.
    """
    if n_tracks < 1 or n_tracks > 3:
        raise ValueError(f"Track 수는 1 이상 3 이하여야 합니다: {n_tracks}")
    if shift_months < 1:
        raise ValueError(
            f"슬라이딩 시프트 개월 수는 1 이상이어야 합니다: {shift_months}"
        )

    base_start_d = date.fromisoformat(base_train_start)
    tracks = []

    for i in range(n_tracks):
        tr_s = add_months(base_start_d, i * shift_months)
        tr_e = add_months(tr_s, 12) - timedelta(days=1)
        val_s = tr_e + timedelta(days=1)
        val_e = add_months(val_s, 12) - timedelta(days=1)

        tracks.append(
            {
                "name": f"Track {i + 1}",
                "train_start": tr_s.isoformat(),
                "train_end": tr_e.isoformat(),
                "val_start": val_s.isoformat(),
                "val_end": val_e.isoformat(),
            }
        )

    last_val_end = date.fromisoformat(tracks[-1]["val_end"])
    test_s = last_val_end + timedelta(days=1)
    test_e = add_months(test_s, 12) - timedelta(days=1)
    ho_tr_s = tracks[-1]["train_start"]
    ho_tr_e = last_val_end.isoformat()

    holdout_dict = {
        "train_start": ho_tr_s,
        "train_end": ho_tr_e,
        "test_start": test_s.isoformat(),
        "test_end": test_e.isoformat(),
    }

    return tracks, holdout_dict


# 기본 아키텍처 롤링 Walk-Forward 및 최종 Hold-out 일정
ROLLING_SPLIT_TRACKS, HOLDOUT_TEST_PERIOD = generate_rolling_tracks(
    n_tracks=3, shift_months=1, base_train_start="2023-06-01"
)


def get_feature_columns(horizon: str = "short") -> list[str]:
    """호라이즌 트랙('short' 또는 'long')에 따른 피처 컬럼 목록을 반환합니다."""
    if horizon == "short":
        return list(SHORT_HORIZON_FEATURES)
    elif horizon == "long":
        return list(LONG_HORIZON_FEATURES)
    else:
        raise ValueError(
            f"지원하지 않는 예측 호라이즌입니다: '{horizon}'. ('short', 'long' 중 선택)"
        )


def validate_short_horizon_lead_time(
    data: pl.DataFrame | np.ndarray,
    feature_cols: list[str] | None = None,
    max_lead_time: int = 12,
) -> None:
    """
    단기 예측 모델(short horizon) 입력 데이터의 lead_time_hours 유효성을 검증합니다.

    기상청 단기예보(12시간 주기 갱신)에 기반한 단기 모델은 1~12시간 이내의 최신 예보만을
    정상 입력으로 수용합니다. 12시간을 초과하는 데이터(예: 4일 뒤 등 장기 예측)는
    OOD(Out-of-Distribution) 왜곡을 방지하기 위해 단기 모델 인입을 차단하며,
    날씨 피처가 배제된 Weather-Free 장기 모델(long)을 사용해야 합니다.

    인자:
        data: 검증 대상 Polars DataFrame 또는 NumPy 배열
        feature_cols: data가 NumPy 배열일 때 컬럼명 매핑 리스트
        max_lead_time: 허용되는 최대 선행 시간 (기본값: 12)

    발생 예외:
        ValueError: lead_time_hours > max_lead_time 이거나 결측치/음수가 발견된 경우
    """
    if isinstance(data, pl.DataFrame):
        if "lead_time_hours" not in data.columns:
            return
        col = data["lead_time_hours"]
        if col.null_count() > 0:
            raise ValueError(
                "단기 모델 입력의 lead_time_hours에 결측치가 존재합니다. "
                "기상 예보 부재 시 임의 0값 대치는 금지되며 Weather-Free 장기 모델(long)을 사용해야 합니다."
            )
        max_val = col.max()
        min_val = col.min()
        if max_val is not None and max_val > max_lead_time:
            raise ValueError(
                f"단기 예측 모델(short)은 최대 {max_lead_time}시간 이내의 기상 예보만 허용합니다. "
                f"초과 값(최대 {max_val}시간)이 발견되었습니다. 12시간 초과(4일 뒤 등) 장기 예측은 "
                "Weather-Free 장기 모델(long)을 사용해야 합니다."
            )
        if min_val is not None and min_val < 0:
            raise ValueError(f"lead_time_hours는 음수일 수 없습니다: {min_val}")

    elif isinstance(data, np.ndarray):
        if feature_cols is None or "lead_time_hours" not in feature_cols:
            return
        idx = feature_cols.index("lead_time_hours")
        lead_arr = data[:, idx]
        if np.isnan(lead_arr).any():
            raise ValueError(
                "단기 모델 입력의 lead_time_hours에 NaN 결측치가 존재합니다. "
                "Weather-Free 장기 모델(long)을 사용해야 합니다."
            )
        max_val = float(np.nanmax(lead_arr))
        min_val = float(np.nanmin(lead_arr))
        if max_val > max_lead_time:
            raise ValueError(
                f"단기 예측 모델(short)은 최대 {max_lead_time}시간 이내의 기상 예보만 허용합니다. "
                f"초과 값(최대 {max_val:.1f}시간)이 발견되었습니다. Weather-Free 장기 모델(long)을 사용해야 합니다."
            )
        if min_val < 0:
            raise ValueError(f"lead_time_hours는 음수일 수 없습니다: {min_val:.1f}")


def split_time_series(
    df: pl.DataFrame,
    test_ratio: float = 0.2,
    date_col: str = "datetime",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    시계열 순서를 보존하여 시간 누출(Data Leakage) 없이 Train과 Test로 분할합니다.
    시간 기준점(split_time)을 계산하여 필터링함으로써, (station_id, datetime) 등의
    역사별 물리적 정렬 순서를 훼손하지 않고 모든 역사에 대해 공평하게 기간을 분할합니다.
    """
    if len(df) == 0:
        return df, df

    unique_times = df[date_col].unique().sort()
    n_times = len(unique_times)
    split_idx = int(n_times * (1.0 - test_ratio))
    split_idx = max(0, min(split_idx, n_times - 1))
    split_time = unique_times[split_idx]

    train_df = df.filter(pl.col(date_col) < split_time)
    test_df = df.filter(pl.col(date_col) >= split_time)
    return train_df, test_df


def split_by_date_range(
    df: pl.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    date_col: str = "date",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    엄밀한 날짜 범위로 Train과 Test/Val을 분할합니다.
    """
    s_tr = pl.lit(train_start).str.to_date("%Y-%m-%d")
    e_tr = pl.lit(train_end).str.to_date("%Y-%m-%d")
    s_val = pl.lit(val_start).str.to_date("%Y-%m-%d")
    e_val = pl.lit(val_end).str.to_date("%Y-%m-%d")

    d_expr = (
        pl.col(date_col).str.to_date("%Y-%m-%d")
        if df[date_col].dtype in (pl.String, pl.Utf8)
        else (
            pl.col(date_col).cast(pl.Date)
            if df[date_col].dtype != pl.Date
            else pl.col(date_col)
        )
    )

    train_df = df.filter((d_expr >= s_tr) & (d_expr <= e_tr))
    val_df = df.filter((d_expr >= s_val) & (d_expr <= e_val))
    return train_df, val_df


def split_rolling_time_series(
    df: pl.DataFrame,
    strategy: str = "rolling",
    date_col: str = "date",
    n_tracks: int = 3,
    shift_months: int = 1,
    base_train_start: str = "2023-06-01",
) -> dict[str, Any]:
    """
    롤링 슬라이딩 윈도우 분할을 수행합니다.

    인자:
        df: 마스터 피처 DataFrame
        strategy: 분할 전략 ('rolling')
        date_col: 날짜 컬럼명
        n_tracks: 롤링 Track 수 (최대 3, 기본: 3)
        shift_months: Track 간 슬라이딩 개월 수 (기본: 1)
        base_train_start: 기준 시작일 (기본: '2023-06-01')
    """
    if strategy != "rolling":
        raise ValueError(
            f"지원하지 않는 분할 전략입니다: '{strategy}'. ('rolling'만 지원)"
        )

    tracks, holdout_info = generate_rolling_tracks(
        n_tracks=n_tracks,
        shift_months=shift_months,
        base_train_start=base_train_start,
    )

    actual_date_col = date_col if date_col in df.columns else "datetime"

    splits = []
    for track in tracks:
        tr_df, val_df = split_by_date_range(
            df,
            train_start=track["train_start"],
            train_end=track["train_end"],
            val_start=track["val_start"],
            val_end=track["val_end"],
            date_col=actual_date_col,
        )
        splits.append(
            {
                "name": track["name"],
                "train_df": tr_df,
                "val_df": val_df,
            }
        )

    ho_tr, ho_te = split_by_date_range(
        df,
        train_start=holdout_info["train_start"],
        train_end=holdout_info["train_end"],
        val_start=holdout_info["test_start"],
        val_end=holdout_info["test_end"],
        date_col=actual_date_col,
    )
    return {
        "strategy": "rolling",
        "validation_splits": splits,
        "holdout_test": {"train_df": ho_tr, "test_df": ho_te},
        "tracks_info": tracks,
        "holdout_info": holdout_info,
    }


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> lgb.LGBMRegressor:
    """LightGBM 회귀 모델 학습 (초저지연 실시간 Fast Track)"""
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


class SafeRandomForestRegressor(RandomForestRegressor):
    """
    대규모 행렬(수백만 행, 수십 개 피처) 추론 시 scikit-learn C-SIMD 내부 메모리 스핀락 및
    float64/float32 변환 과정의 무한 행(hang)을 방지하는 안전 Random Forest 회귀 모델.

    - fit 단계: 입력 데이터를 np.ascontiguousarray(..., dtype=np.float32)로 사전 변환하여 메모리 정렬 보장
    - predict 단계: 입력 데이터를 float32 연속 메모리로 변환하고, chunk_size 단위로 분할 추론하여 메모리 폭주 방지
    """

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> SafeRandomForestRegressor:
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        y_c = np.ascontiguousarray(y, dtype=np.float32)
        super().fit(X_c, y_c, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: Any,
        chunk_size: int = 100_000,
    ) -> np.ndarray:
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        n_samples = len(X_c)
        if n_samples <= chunk_size or chunk_size <= 0:
            return super().predict(X_c).astype(np.float32)

        preds = []
        for i in range(0, n_samples, chunk_size):
            preds.append(super().predict(X_c[i : i + chunk_size]))
        return np.concatenate(preds).astype(np.float32)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> SafeRandomForestRegressor:
    """SafeRandomForestRegressor 회귀 모델 학습 (float32 연속 메모리 정렬 및 청크 추론 보장)"""
    model = SafeRandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model
