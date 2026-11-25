"""
Set 1(13종: 지하철 기저 + 달력·특일·주기성) 운영 배치 추론 및 산출물 스키마 모듈

주요 책임:
1. 예측 산출물 스키마 계약(PREDICTION_SCHEMA, 9컬럼) 정의 및 무결성 검증(validate_predictions_df)
2. 동일 디렉터리 임시 파일 + os.replace() 기반 원자적 Parquet/JSON 교체
3. data/reference/{station_master,calendar_master}.parquet와 StationTargetEncoder 매핑 기반
   Set 1 미래 피처 생성 (target_hour 1~24, 심야 24시 -> 익일 00:00:00 경계 보존)
4. 학습 체크포인트·스케일러·타깃 인코더·매니페스트 엄격 로드 (부재 시 FileNotFoundError)
5. 스케줄러 비의존적 단독·멱등 배치 추론 실행 (run_batch_inference)
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import torch

from busan_metro_congestion.config import (
    ARTIFACTS_DIR,
    DEVICE,
    REFERENCE_DATA_DIR,
    set_seed,
)
from busan_metro_congestion.modeling.baseline import OPERATIONAL_FEATURES
from busan_metro_congestion.modeling.ensemble import (
    blend_accurate_track,
    blend_fast_track,
)
from busan_metro_congestion.modeling.lstm_net import MetroLSTM, MetroLSTMEstimator
from busan_metro_congestion.preprocess.subway import (
    StationTargetEncoder,
    add_cyclical_time_features,
)

logger = logging.getLogger("modeling_predict")

KST = timezone(timedelta(hours=9))

OPERATIONAL_FEATURE_SET = "subway_calendar"
SUPPORTED_TRACKS = ("accurate", "fast")
VALID_LINE_NUMS = frozenset({1, 2, 3, 4})

DEFAULT_CHECKPOINT_DIR = ARTIFACTS_DIR / "baseline" / "checkpoints"
DEFAULT_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions" / "predictions.parquet"

PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "station_id": pl.Int64,
    "station_name": pl.String,
    "line_num": pl.Int8,
    "target_date": pl.Date,
    "target_hour": pl.Int8,
    "target_datetime": pl.Datetime("us"),
    "predicted_passengers": pl.Float32,
    "model_version": pl.String,
    "generated_at": pl.Datetime("us"),
}
PREDICTION_COLUMNS: list[str] = list(PREDICTION_SCHEMA.keys())


def get_checkpoint_paths(
    checkpoint_dir: Path | str,
) -> dict[str, Path]:
    """운영(Set 1) 체크포인트·스케일러·타깃 인코더·매니페스트 표준 파일 경로를 반환합니다."""
    ckpt_dir = Path(checkpoint_dir)
    return {
        "lgbm": ckpt_dir / "lgbm.txt",
        "rf": ckpt_dir / "rf.joblib",
        "lstm": ckpt_dir / "lstm.pt",
        "scaler": ckpt_dir / "scaler.joblib",
        "target_encoder": ckpt_dir / "station_target_encoder.parquet",
        "manifest": ckpt_dir / "model_manifest.json",
    }


def atomic_write_parquet(df: pl.DataFrame, output_path: Path | str) -> Path:
    """동일 디렉터리 임시 파일을 경유해 os.replace()로 Parquet를 원자적 교체합니다."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}_",
        suffix=".tmp.parquet",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(payload: dict[str, Any], output_path: Path | str) -> Path:
    """동일 디렉터리 임시 파일을 경유해 os.replace()로 JSON을 원자적 교체합니다."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}_",
        suffix=".tmp.json",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def build_target_datetime_expr(
    date_col: str = "target_date",
    hour_col: str = "target_hour",
) -> pl.Expr:
    """
    부산 지하철 시간대 규약(1~23시 -> 당일 HH:00:00, 24시 -> 익일 00:00:00)에 따른
    물리적 target_datetime(Datetime("us")) Polars 표현식을 반환합니다.
    """
    return (
        pl.when(pl.col(hour_col) == 24)
        .then(pl.col(date_col).dt.offset_by("1d").cast(pl.Datetime("us")))
        .otherwise(
            pl.col(date_col).cast(pl.Datetime("us"))
            + pl.duration(hours=pl.col(hour_col).cast(pl.Int64))
        )
    )


def validate_predictions_df(df: pl.DataFrame) -> None:
    """
    배치 추론 산출물 DataFrame이 PREDICTION_SCHEMA 및 도메인 불변식을 만족하는지 검증합니다.

    검증 항목:
    1. 1행 이상 존재
    2. 컬럼 구성·순서 및 Polars 타입 완전 일치
    3. 전 컬럼 결측치(null) 0건
    4. (station_id, target_date, target_hour) 복합 키 유일성
    5. line_num ∈ {1, 2, 3, 4}, target_hour ∈ [1, 24]
    6. target_datetime과 (target_date, target_hour)의 심야 24시 포함 정합성
    7. predicted_passengers 비음수(>= 0.0) 및 유한성(NaN/Inf 금지)
    8. model_version 비공백 문자열
    """
    if len(df) == 0:
        raise ValueError("예측 산출물 DataFrame이 비어 있습니다(0행).")

    if df.columns != PREDICTION_COLUMNS:
        raise ValueError(
            f"예측 산출물 컬럼이 스키마 계약과 일치하지 않습니다. "
            f"기대={PREDICTION_COLUMNS}, 실제={df.columns}"
        )

    for col_name, expected_dtype in PREDICTION_SCHEMA.items():
        actual_dtype = df.schema[col_name]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"컬럼 '{col_name}'의 타입이 일치하지 않습니다: "
                f"기대={expected_dtype}, 실제={actual_dtype}"
            )

    null_counts = {
        col: int(df[col].null_count())
        for col in PREDICTION_COLUMNS
        if df[col].null_count() > 0
    }
    if null_counts:
        raise ValueError(f"예측 산출물에 결측치가 존재합니다: {null_counts}")

    unique_keys = df.select(["station_id", "target_date", "target_hour"]).n_unique()
    if unique_keys != len(df):
        raise ValueError(
            f"(station_id, target_date, target_hour) 중복 키가 발견되었습니다: "
            f"전체={len(df):,}행, 고유={unique_keys:,}행"
        )

    lines = set(df["line_num"].unique().to_list())
    if not lines.issubset(VALID_LINE_NUMS):
        raise ValueError(
            f"허용되지 않는 line_num이 포함되었습니다: {sorted(lines)} (허용: 1~4)"
        )

    min_h = df["target_hour"].min()
    max_h = df["target_hour"].max()
    if min_h is None or max_h is None or int(min_h) < 1 or int(max_h) > 24:
        raise ValueError(
            f"target_hour는 1~24 범위여야 합니다: min={min_h}, max={max_h}"
        )

    dt_mismatch = df.filter(
        pl.col("target_datetime")
        != build_target_datetime_expr("target_date", "target_hour")
    )
    if len(dt_mismatch) > 0:
        sample_row = (
            dt_mismatch.select(
                ["station_id", "target_date", "target_hour", "target_datetime"]
            )
            .head(1)
            .to_dicts()[0]
        )
        raise ValueError(
            f"target_datetime이 (target_date, target_hour) 규약과 불일치합니다 "
            f"({len(dt_mismatch):,}건, 예: {sample_row})"
        )

    pred_series = df["predicted_passengers"]
    if not bool(pred_series.is_finite().all()):
        raise ValueError("predicted_passengers에 NaN 또는 무한대(Inf) 값이 존재합니다.")
    min_pred = pred_series.min()
    if min_pred is not None and float(min_pred) < 0.0:
        raise ValueError(
            f"predicted_passengers는 음수일 수 없습니다: min={float(min_pred):.4f}"
        )

    empty_ver = df.filter(
        pl.col("model_version").str.strip_chars().str.len_chars() == 0
    )
    if len(empty_ver) > 0:
        raise ValueError("model_version은 비어 있거나 공백 문자열일 수 없습니다.")


def save_predictions_atomic(
    df: pl.DataFrame,
    output_path: Path | str = DEFAULT_PREDICTIONS_PATH,
) -> Path:
    """예측 산출물 스키마 검증을 통과한 뒤 임시 파일 + os.replace()로 원자적 저장합니다."""
    validate_predictions_df(df)
    saved = atomic_write_parquet(df, output_path)
    logger.info(f"예측 산출물 원자적 저장 완료: {saved} ({len(df):,}행)")
    return saved


def _parse_date_arg(value: date | str | None, label: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"{label} 날짜 형식이 올바르지 않습니다(YYYY-MM-DD 필요): '{value}'"
            ) from exc
    raise ValueError(f"지원하지 않는 {label} 타입입니다: {type(value).__name__}")


def resolve_prediction_date_range(
    calendar_df: pl.DataFrame,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    days: int | None = None,
    reference_today: date | None = None,
) -> tuple[date, date]:
    """
    calendar_master.parquet 범위와 CLI 날짜 옵션으로 최종 예측 대상 구간([start_date, end_date])을 결정합니다.

    규칙:
    - start_date 미지정 시 실행 시점 KST 당일 사용
    - end_date와 days는 상호 배타
    - end_date·days 모두 미지정 시 실행 PC의 calendar_master.parquet 말일까지 전체 추론
    - 요청 구간의 모든 날짜가 calendar_master.parquet에 존재해야 함 (결측 시 ValueError)
    """
    if len(calendar_df) == 0 or "date" not in calendar_df.columns:
        raise ValueError(
            "calendar_master.parquet가 비어 있거나 'date' 컬럼이 없습니다."
        )

    if end_date is not None and days is not None:
        raise ValueError("end_date와 days는 동시에 지정할 수 없습니다.")
    if days is not None and days < 1:
        raise ValueError(f"days는 1 이상의 정수여야 합니다: {days}")

    cal_min_raw = calendar_df["date"].min()
    cal_max_raw = calendar_df["date"].max()
    if not isinstance(cal_min_raw, date) or not isinstance(cal_max_raw, date):
        raise ValueError(
            "calendar_master.parquet의 'date' 컬럼이 유효한 Date 타입이 아닙니다."
        )

    eff_start = _parse_date_arg(start_date, "start_date")
    if eff_start is None:
        eff_start = (
            reference_today if reference_today is not None else datetime.now(KST).date()
        )

    eff_end = _parse_date_arg(end_date, "end_date")
    if eff_end is None:
        if days is not None:
            eff_end = eff_start + timedelta(days=days - 1)
        else:
            eff_end = cal_max_raw

    if eff_start > eff_end:
        if end_date is None and days is None and eff_start > cal_max_raw:
            raise ValueError(
                f"시작일({eff_start.isoformat()})이 calendar_master 말일({cal_max_raw.isoformat()})을 "
                "초과했습니다. 'uv run python scripts/sync_operational_data.py --target calendar'로 "
                "달력 마스터를 최신화하세요."
            )
        raise ValueError(
            f"종료일({eff_end.isoformat()})은 시작일({eff_start.isoformat()})보다 앞설 수 없습니다."
        )

    expected_days = (eff_end - eff_start).days + 1
    matched_days = (
        calendar_df.filter((pl.col("date") >= eff_start) & (pl.col("date") <= eff_end))
        .select("date")
        .n_unique()
    )
    if matched_days != expected_days:
        raise ValueError(
            f"요청 기간({eff_start.isoformat()} ~ {eff_end.isoformat()}, {expected_days}일) 중 "
            f"달력 마스터에 수록되지 않은 날짜가 {expected_days - matched_days}일 존재합니다 "
            f"(수록 범위: {cal_min_raw.isoformat()} ~ {cal_max_raw.isoformat()}). "
            "결측 날짜에 대한 임의 대치는 금지됩니다."
        )

    return eff_start, eff_end


def _load_reference_tables(
    station_master_path: Path | str | None = None,
    calendar_master_path: Path | str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    stn_path = (
        Path(station_master_path)
        if station_master_path is not None
        else REFERENCE_DATA_DIR / "station_master.parquet"
    )
    cal_path = (
        Path(calendar_master_path)
        if calendar_master_path is not None
        else REFERENCE_DATA_DIR / "calendar_master.parquet"
    )

    if not stn_path.exists():
        raise FileNotFoundError(f"역사 마스터 참조 파일을 찾을 수 없습니다: {stn_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"달력 마스터 참조 파일을 찾을 수 없습니다: {cal_path}")

    df_stn = (
        pl.read_parquet(stn_path)
        .select(
            [
                pl.col("station_id").cast(pl.Int64),
                pl.col("station_name").cast(pl.String),
                pl.col("line_num").cast(pl.Int8),
            ]
        )
        .unique(subset=["station_id"], keep="first")
        .sort("station_id")
    )
    if len(df_stn) == 0:
        raise ValueError(f"역사 마스터 파일이 비어 있습니다: {stn_path}")

    df_cal = (
        pl.read_parquet(cal_path)
        .select(
            [
                pl.col("date").cast(pl.Date),
                pl.col("is_weekend").cast(pl.Boolean),
                pl.col("is_holiday").cast(pl.Boolean),
                pl.col("is_holiday_eve").cast(pl.Boolean),
                pl.col("season").cast(pl.Int8),
                pl.col("month_sin").cast(pl.Float32),
                pl.col("month_cos").cast(pl.Float32),
                pl.col("weekday_sin").cast(pl.Float32),
                pl.col("weekday_cos").cast(pl.Float32),
            ]
        )
        .unique(subset=["date"], keep="first")
        .sort("date")
    )
    if len(df_cal) == 0:
        raise ValueError(f"달력 마스터 파일이 비어 있습니다: {cal_path}")

    return df_stn, df_cal


def _build_set1_frame(
    df_stations: pl.DataFrame,
    df_calendar: pl.DataFrame,
    start_date: date,
    end_date: date,
    target_encoder: StationTargetEncoder,
    lookback_hours: int = 0,
) -> pl.DataFrame:
    """
    Set 1(13종) 추론용 시공간 피처 DataFrame을 생성합니다.
    lookback_hours > 0이면 start_date 직전 가용 시간대(최대 lookback_hours개)를
    역사별로 선행 부착하고 '_is_target' 불리언 마스크로 본 예측 구간을 구분합니다.
    """
    lookback_days = math.ceil(lookback_hours / 24) if lookback_hours > 0 else 0
    cal_min = df_calendar["date"].min()
    if not isinstance(cal_min, date):
        raise ValueError("calendar_master의 최소 날짜를 확인할 수 없습니다.")

    grid_start = max(cal_min, start_date - timedelta(days=lookback_days))
    cal_slice = df_calendar.filter(
        (pl.col("date") >= grid_start) & (pl.col("date") <= end_date)
    )

    hours_df = pl.DataFrame(
        {"target_hour": pl.Series(list(range(1, 25)), dtype=pl.Int8)}
    )
    time_grid = (
        cal_slice.join(hours_df, how="cross")
        .with_columns(
            pl.col("date").alias("target_date"),
            build_target_datetime_expr("date", "target_hour").alias("target_datetime"),
            (pl.col("date") >= start_date).alias("_is_target"),
        )
        .sort("target_datetime")
    )

    if lookback_hours > 0:
        lb_grid = time_grid.filter(~pl.col("_is_target")).tail(lookback_hours)
        tgt_grid = time_grid.filter(pl.col("_is_target"))
        time_grid = pl.concat([lb_grid, tgt_grid]).sort("target_datetime")
    else:
        time_grid = time_grid.filter(pl.col("_is_target"))

    time_grid = add_cyclical_time_features(
        time_grid, hour_col="target_hour", include_weekday=False
    )

    # station_target_encoded는 역사별 상수이므로 114개 역사에 먼저 O(1) 매핑 후 cross join
    stations_encoded = target_encoder.transform(df_stations, station_col="station_id")

    full_df = stations_encoded.join(time_grid, how="cross").sort(
        ["station_id", "target_datetime"]
    )
    return full_df


def build_future_features(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    target_encoder: StationTargetEncoder | None = None,
    days: int | None = None,
    station_master_path: Path | str | None = None,
    calendar_master_path: Path | str | None = None,
    target_encoder_path: Path | str | None = None,
    reference_today: date | None = None,
) -> pl.DataFrame:
    """
    Set 1(13종) 운영 배치 추론용 미래 피처 DataFrame을 생성합니다.

    - 역사 마스터(114역) × 대상 날짜 구간 × 24개 시간대(1~24시)
    - target_hour=24는 (target_date + 1일) 00:00:00으로 물리 시각 보존
    - station_target_encoded는 저장된 StationTargetEncoder 매핑으로 O(1) 매핑
    """
    if target_encoder is None:
        if target_encoder_path is None:
            raise ValueError(
                "target_encoder 또는 target_encoder_path 중 하나를 제공해야 합니다."
            )
        target_encoder = StationTargetEncoder.load_mapping(target_encoder_path)

    df_stn, df_cal = _load_reference_tables(
        station_master_path=station_master_path,
        calendar_master_path=calendar_master_path,
    )
    eff_start, eff_end = resolve_prediction_date_range(
        df_cal,
        start_date=start_date,
        end_date=end_date,
        days=days,
        reference_today=reference_today,
    )
    df = _build_set1_frame(
        df_stations=df_stn,
        df_calendar=df_cal,
        start_date=eff_start,
        end_date=eff_end,
        target_encoder=target_encoder,
        lookback_hours=0,
    ).drop("_is_target")
    return df


@dataclass(frozen=True)
class LoadedInferenceArtifacts:
    """배치 추론에 필요한 Set 1 체크포인트·인코더·매니페스트 번들."""

    feature_set: str
    track: str
    model_version: str
    selected_features: list[str]
    manifest: dict[str, Any]
    target_encoder: StationTargetEncoder
    lgbm_booster: lgb.Booster
    rf_model: Any
    weights: dict[str, float]
    lstm_estimator: MetroLSTMEstimator | None = None
    scaler: dict[str, Any] | None = None


def _validate_manifest_contract(
    manifest: dict[str, Any],
    track: str,
) -> tuple[str, dict[str, float]]:
    manifest_feature_set = manifest.get("feature_set")
    if manifest_feature_set != OPERATIONAL_FEATURE_SET:
        raise ValueError(
            f"체크포인트 매니페스트 피처셋이 Set 1('{OPERATIONAL_FEATURE_SET}')이 아닙니다: "
            f"'{manifest_feature_set}'"
        )

    selected_features = manifest.get("selected_features")
    if selected_features != OPERATIONAL_FEATURES:
        raise ValueError(
            "체크포인트 피처 구성이 Set 1(13종) 계약과 일치하지 않습니다. "
            f"기대={OPERATIONAL_FEATURES}, 실제={selected_features}"
        )

    model_version = str(manifest.get("model_version", "")).strip()
    if not model_version:
        raise ValueError("매니페스트에 유효한 'model_version'이 없습니다.")

    weights_root = manifest.get("weights", {})
    track_key = "accurate_track" if track == "accurate" else "fast_track"
    track_weights = weights_root.get(track_key)
    if not isinstance(track_weights, dict):
        raise ValueError(f"매니페스트에 '{track_key}' 앙상블 가중치가 없습니다.")

    required_w_keys = (
        ("w_rf", "w_lgb", "w_lstm") if track == "accurate" else ("w_rf", "w_lgb")
    )
    parsed_weights: dict[str, float] = {}
    for k in required_w_keys:
        if k not in track_weights:
            raise ValueError(
                f"매니페스트 '{track_key}'에 가중치 '{k}'가 누락되었습니다."
            )
        w_val = float(track_weights[k])
        if not math.isfinite(w_val) or w_val < 0.0:
            raise ValueError(f"유효하지 않은 앙상블 가중치입니다: {k}={w_val}")
        parsed_weights[k] = w_val

    weight_sum = sum(parsed_weights.values())
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(
            f"'{track_key}' 앙상블 가중치 합은 1.0이어야 합니다 (현재={weight_sum:.4f})."
        )

    return model_version, parsed_weights


def load_inference_artifacts(
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    track: str = "accurate",
) -> LoadedInferenceArtifacts:
    """
    Set 1 배치 추론용 체크포인트·타깃 인코더·스케일러·매니페스트를 엄격 검증 후 로드합니다.
    필수 파일 중 하나라도 부재하면 암묵적 폴백 없이 FileNotFoundError를 발생시킵니다.
    """
    if track not in SUPPORTED_TRACKS:
        raise ValueError(
            f"지원하지 않는 추론 트랙입니다: '{track}'. ({', '.join(SUPPORTED_TRACKS)} 중 선택)"
        )

    paths = get_checkpoint_paths(checkpoint_dir)
    required_keys = ["manifest", "target_encoder", "lgbm", "rf"]
    if track == "accurate":
        required_keys.extend(["lstm", "scaler"])

    missing = [f"{k}: {paths[k]}" for k in required_keys if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(
            "배치 추론에 필요한 체크포인트 아티팩트가 존재하지 않습니다. "
            "먼저 'uv run busan-metro train'으로 모델을 학습하세요. "
            f"누락 파일: {', '.join(missing)}"
        )

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    model_version, parsed_weights = _validate_manifest_contract(manifest, track=track)

    target_encoder = StationTargetEncoder.load_mapping(paths["target_encoder"])
    lgbm_booster = lgb.Booster(model_file=str(paths["lgbm"]))
    if lgbm_booster.num_feature() != len(OPERATIONAL_FEATURES):
        raise ValueError(
            f"LightGBM 모델 피처 수({lgbm_booster.num_feature()})가 "
            f"Set 1({len(OPERATIONAL_FEATURES)}종)과 일치하지 않습니다."
        )

    rf_model = joblib.load(paths["rf"])
    rf_n_features = getattr(rf_model, "n_features_in_", None)
    if rf_n_features is not None and int(rf_n_features) != len(OPERATIONAL_FEATURES):
        raise ValueError(
            f"Random Forest 모델 피처 수({rf_n_features})가 "
            f"Set 1({len(OPERATIONAL_FEATURES)}종)과 일치하지 않습니다."
        )

    lstm_estimator: MetroLSTMEstimator | None = None
    scaler: dict[str, Any] | None = None
    if track == "accurate":
        scaler_loaded = joblib.load(paths["scaler"])
        expected_scaler_keys = {"mean_x", "std_x", "mean_y", "std_y"}
        if not isinstance(scaler_loaded, dict) or not expected_scaler_keys.issubset(
            scaler_loaded.keys()
        ):
            raise ValueError(
                f"LSTM 스케일러 파일 형식이 올바르지 않습니다: {paths['scaler']}"
            )
        mean_x = np.asarray(scaler_loaded["mean_x"], dtype=np.float32)
        std_x = np.asarray(scaler_loaded["std_x"], dtype=np.float32)
        if mean_x.shape[-1] != len(OPERATIONAL_FEATURES) or std_x.shape[-1] != len(
            OPERATIONAL_FEATURES
        ):
            raise ValueError(
                f"LSTM 스케일러 피처 차원({mean_x.shape[-1]})이 "
                f"Set 1({len(OPERATIONAL_FEATURES)}종)과 일치하지 않습니다."
            )
        scaler = {
            "mean_x": mean_x,
            "std_x": std_x,
            "mean_y": float(scaler_loaded["mean_y"]),
            "std_y": float(scaler_loaded["std_y"]),
        }

        lstm_cfg = manifest.get("lstm_config", {})
        input_dim = int(lstm_cfg.get("input_dim", len(OPERATIONAL_FEATURES)))
        if input_dim != len(OPERATIONAL_FEATURES):
            raise ValueError(
                f"LSTM 매니페스트 input_dim({input_dim})이 "
                f"Set 1({len(OPERATIONAL_FEATURES)}종)과 일치하지 않습니다."
            )

        seq_len = int(lstm_cfg.get("seq_len", 3))
        hidden_dim1 = int(lstm_cfg.get("hidden_dim1", 64))
        hidden_dim2 = int(lstm_cfg.get("hidden_dim2", 32))
        dense_dim = int(lstm_cfg.get("dense_dim", 16))
        dropout_rate = float(lstm_cfg.get("dropout_rate", 0.2))
        batch_size = int(lstm_cfg.get("batch_size", 256))

        lstm_net = MetroLSTM(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dense_dim=dense_dim,
            dropout_rate=dropout_rate,
        )
        state_dict = torch.load(paths["lstm"], map_location=DEVICE, weights_only=True)
        lstm_net.load_state_dict(state_dict)
        lstm_net.to(DEVICE)
        lstm_net.eval()

        est = MetroLSTMEstimator(
            seq_len=seq_len,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dense_dim=dense_dim,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            device=DEVICE,
        )
        est.model_ = lstm_net
        est.n_features_in_ = input_dim
        est.is_fitted_ = True
        lstm_estimator = est

    return LoadedInferenceArtifacts(
        feature_set=OPERATIONAL_FEATURE_SET,
        track=track,
        model_version=model_version,
        selected_features=list(OPERATIONAL_FEATURES),
        manifest=manifest,
        target_encoder=target_encoder,
        lgbm_booster=lgbm_booster,
        rf_model=rf_model,
        weights=parsed_weights,
        lstm_estimator=lstm_estimator,
        scaler=scaler,
    )


def run_batch_inference(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    days: int | None = None,
    track: str = "accurate",
    checkpoint_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    manifest_out: Path | str | None = None,
    station_master_path: Path | str | None = None,
    calendar_master_path: Path | str | None = None,
    reference_today: date | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Set 1(13종) 운영 배치 추론 파이프라인을 단독·멱등 실행합니다.

    1. 체크포인트·인코더·매니페스트 로드 및 피처 계약 검증
    2. station_master + calendar_master 기반 Set 1 미래 피처 생성
    3. Fast Track(RF+LGBM) 또는 Accurate Track(RF+LGBM+LSTM, Lookback 버퍼 적용) 앙상블 추론
    4. PREDICTION_SCHEMA 조립·검증 및 임시 파일 경유 os.replace() 원자적 교체
    """
    set_seed(seed)

    eff_ckpt_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    )
    eff_output_path = (
        Path(output_path) if output_path is not None else DEFAULT_PREDICTIONS_PATH
    )

    artifacts = load_inference_artifacts(
        checkpoint_dir=eff_ckpt_dir,
        track=track,
    )
    df_stn, df_cal = _load_reference_tables(
        station_master_path=station_master_path,
        calendar_master_path=calendar_master_path,
    )
    eff_start, eff_end = resolve_prediction_date_range(
        df_cal,
        start_date=start_date,
        end_date=end_date,
        days=days,
        reference_today=reference_today,
    )

    lookback_hours = (
        artifacts.lstm_estimator.seq_len
        if track == "accurate" and artifacts.lstm_estimator is not None
        else 0
    )

    logger.info("=" * 60)
    logger.info("부산 지하철 혼잡도 Set 1 배치 추론 파이프라인 시작")
    logger.info(f"   - 디바이스: {DEVICE}")
    logger.info(f"   - 추론 트랙: {track} (Set 1, {len(OPERATIONAL_FEATURES)}종 피처)")
    logger.info(f"   - 모델 버전: {artifacts.model_version}")
    logger.info(
        f"   - 예측 기간: {eff_start.isoformat()} ~ {eff_end.isoformat()} ({len(df_stn)}개 역사 × 24시간)"
    )
    logger.info("=" * 60)

    full_df = _build_set1_frame(
        df_stations=df_stn,
        df_calendar=df_cal,
        start_date=eff_start,
        end_date=eff_end,
        target_encoder=artifacts.target_encoder,
        lookback_hours=lookback_hours,
    )
    target_mask = full_df["_is_target"].to_numpy()
    target_df = full_df.filter(pl.col("_is_target"))

    X_target = np.ascontiguousarray(
        target_df.select(OPERATIONAL_FEATURES).to_numpy(), dtype=np.float32
    )
    pred_lgb = np.clip(
        np.asarray(artifacts.lgbm_booster.predict(X_target), dtype=np.float32),
        0.0,
        None,
    )
    pred_rf = np.clip(
        np.asarray(artifacts.rf_model.predict(X_target), dtype=np.float32),
        0.0,
        None,
    )

    if track == "accurate":
        assert artifacts.lstm_estimator is not None and artifacts.scaler is not None
        X_full = np.ascontiguousarray(
            full_df.select(OPERATIONAL_FEATURES).to_numpy(), dtype=np.float32
        )
        stn_full = full_df["station_id"].to_numpy()
        X_full_norm = (X_full - artifacts.scaler["mean_x"]) / artifacts.scaler["std_x"]
        pred_lstm_full_norm = artifacts.lstm_estimator.predict(
            X_full_norm, station_ids=stn_full
        )
        pred_lstm_target = np.clip(
            pred_lstm_full_norm[target_mask] * artifacts.scaler["std_y"]
            + artifacts.scaler["mean_y"],
            0.0,
            None,
        )
        pred_final = np.clip(
            blend_accurate_track(
                pred_rf,
                pred_lgb,
                pred_lstm_target,
                w_rf=artifacts.weights["w_rf"],
                w_lgb=artifacts.weights["w_lgb"],
                w_lstm=artifacts.weights["w_lstm"],
            ),
            0.0,
            None,
        ).astype(np.float32)
    else:
        pred_final = np.clip(
            blend_fast_track(
                pred_rf,
                pred_lgb,
                w_rf=artifacts.weights["w_rf"],
                w_lgb=artifacts.weights["w_lgb"],
            ),
            0.0,
            None,
        ).astype(np.float32)

    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0)
    generated_at_naive_utc = generated_at_utc.replace(tzinfo=None)

    df_predictions = (
        target_df.select(
            [
                pl.col("station_id").cast(pl.Int64),
                pl.col("station_name").cast(pl.String),
                pl.col("line_num").cast(pl.Int8),
                pl.col("target_date").cast(pl.Date),
                pl.col("target_hour").cast(pl.Int8),
                pl.col("target_datetime").cast(pl.Datetime("us")),
            ]
        )
        .with_columns(
            pl.Series("predicted_passengers", pred_final, dtype=pl.Float32),
            pl.lit(artifacts.model_version, dtype=pl.String).alias("model_version"),
            pl.lit(generated_at_naive_utc, dtype=pl.Datetime("us")).alias(
                "generated_at"
            ),
        )
        .select(PREDICTION_COLUMNS)
        .sort(["station_id", "target_date", "target_hour"])
    )

    saved_path = save_predictions_atomic(df_predictions, eff_output_path)

    run_summary: dict[str, Any] = {
        "mode": "predict",
        "status": "ok",
        "feature_set": OPERATIONAL_FEATURE_SET,
        "track": track,
        "model_version": artifacts.model_version,
        "start_date": eff_start.isoformat(),
        "end_date": eff_end.isoformat(),
        "station_count": int(len(df_stn)),
        "row_count": int(len(df_predictions)),
        "output_path": str(saved_path),
        "checkpoint_dir": str(eff_ckpt_dir),
        "generated_at": generated_at_utc.isoformat(),
    }

    if manifest_out is not None:
        manifest_path = atomic_write_json(run_summary, manifest_out)
        run_summary["manifest_out"] = str(manifest_path)
        logger.info(f"배치 추론 실행 매니페스트 저장 완료: {manifest_path}")

    return {
        **run_summary,
        "predictions": df_predictions,
    }
