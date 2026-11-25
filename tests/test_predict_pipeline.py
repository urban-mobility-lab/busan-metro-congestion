"""
Set 1(13종) 운영 배치 추론 파이프라인(modeling/predict.py) 및
새 서버 자립 실행(기상·축제 무의존 train -> predict) 단위·통합 테스트
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import busan_metro_congestion.modeling.predict as predict_mod
import busan_metro_congestion.preprocess.features as features_mod
import busan_metro_congestion.preprocess.subway as subway_mod
from busan_metro_congestion.config import REFERENCE_DATA_DIR, SAMPLES_DIR
from busan_metro_congestion.modeling.baseline import (
    FESTIVAL_FEATURES,
    OPERATIONAL_FEATURES,
    WEATHER_SHORT_FEATURES,
    get_feature_columns,
    train_lightgbm,
    train_random_forest,
)
from busan_metro_congestion.modeling.ensemble import (
    blend_accurate_track,
    blend_fast_track,
)
from busan_metro_congestion.modeling.predict import (
    PREDICTION_COLUMNS,
    PREDICTION_SCHEMA,
    _build_set1_frame,
    atomic_write_json,
    atomic_write_parquet,
    build_future_features,
    get_checkpoint_paths,
    load_inference_artifacts,
    resolve_prediction_date_range,
    run_batch_inference,
    save_predictions_atomic,
    validate_predictions_df,
)
from busan_metro_congestion.preprocess.features import OPERATIONAL_MASTER_COLUMNS
from busan_metro_congestion.preprocess.subway import (
    StationTargetEncoder,
    read_subway_csv,
)
from busan_metro_congestion.run_pipeline import run_pipeline

FAST_OPERATIONAL_CONFIG: dict[str, Any] = {
    "project": {"seed": 42, "version": "0.1.0"},
    "model": {
        "lightgbm": {"n_estimators": 12, "final_n_estimators": 12, "n_jobs": 2},
        "random_forest": {
            "n_estimators": 10,
            "final_n_estimators": 10,
            "tune_max_depth": 5,
            "max_depth": 5,
            "n_jobs": 2,
        },
        "lstm": {
            "seq_len": 2,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dense_dim": 8,
            "tune_epochs": 2,
            "final_epochs": 2,
            "batch_size": 32,
        },
    },
}


def _build_synthetic_raw_subway_parquet(
    dest_path: Path,
    base_dates: list[str],
) -> Path:
    """data/samples/subway/subway_2023_sample.csv 스키마를 복제해 지정 날짜들의 원시 Parquet를 생성합니다."""
    src_df = read_subway_csv(SAMPLES_DIR / "subway" / "subway_2023_sample.csv")
    first_day = src_df["년월일"].min()
    template = src_df.filter(pl.col("년월일") == first_day)
    chunks = [template.with_columns(pl.lit(d).alias("년월일")) for d in base_dates]
    combined = pl.concat(chunks, how="vertical_relaxed")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(dest_path)
    return dest_path


@pytest.fixture(scope="module")
def trained_operational_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """data/samples/subway + data/reference/만으로 Set 1 운영 체크포인트를 1회 학습하는 픽스처."""
    base_dir = tmp_path_factory.mktemp("operational_ckpt_fixture")
    raw_dir = base_dir / "raw" / "subway"
    _build_synthetic_raw_subway_parquet(
        raw_dir / "subway_ridership_2026.parquet",
        base_dates=[
            "2026-10-01",
            "2026-10-02",
            "2026-10-03",
            "2026-10-04",
            "2026-10-05",
        ],
    )
    interim_path = base_dir / "interim" / "subway_interim.parquet"
    processed_path = base_dir / "processed" / "subway_processed.parquet"
    features_path = base_dir / "features" / "metro_features_operational.parquet"
    out_dir = base_dir / "artifacts"

    df_sub = subway_mod.build_subway_processed(
        interim_path=interim_path,
        output_path=processed_path,
        raw_dir=raw_dir,
        start_date=None,
        end_date=None,
        force_rebuild=True,
    )
    features_mod.run_feature_pipeline(
        subway_path=processed_path,
        output_path=features_path,
        force_reload=True,
        include_external=False,
    )
    train_res = run_pipeline(
        mode="train",
        step="train_eval",
        config_dict=FAST_OPERATIONAL_CONFIG,
        output_dir=out_dir,
        features_path=features_path,
    )
    return {
        "base_dir": base_dir,
        "checkpoint_dir": out_dir / "checkpoints",
        "out_dir": out_dir,
        "train_res": train_res,
        "df_sub": df_sub,
    }


def test_operational_features_set1_cross_module_contract():
    """Set 1(operational)이 방어적 복사본을 반환하고 전처리 마스터 스키마(OPERATIONAL_MASTER_COLUMNS)와 정합하는지 검증합니다."""
    op_cols = get_feature_columns("operational")

    assert len(op_cols) == 13
    assert set(op_cols).isdisjoint(WEATHER_SHORT_FEATURES)
    assert set(op_cols).isdisjoint(FESTIVAL_FEATURES)
    assert set(op_cols).issubset(set(OPERATIONAL_MASTER_COLUMNS))

    # 반환 리스트 변경이 모듈 상수를 오염시키지 않아야 함
    op_cols.append("tampered_col")
    assert "tampered_col" not in OPERATIONAL_FEATURES
    assert get_feature_columns("operational") == OPERATIONAL_FEATURES


def test_fresh_server_operational_e2e_without_weather_or_festival(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """
    기상·축제 원천/전처리 파일이 전혀 없는 새 서버 환경을 모사해
    raw/subway(*.parquet, 2026-08-31 이후 날짜 포함) + data/reference만으로
    train(operational) -> predict(accurate/fast)가 완전 자립 실행되는지 검증합니다.
    """
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    out_dir = tmp_path / "artifacts"

    # 기상/축제 디렉터리 미생성 상태 유지 — 지하철 원시 Parquet만 배치 (2026-08-31 이후 날짜 포함)
    _build_synthetic_raw_subway_parquet(
        raw_dir / "subway" / "subway_ridership_2026.parquet",
        base_dates=[
            "2026-11-01",
            "2026-11-02",
            "2026-11-03",
            "2026-11-04",
            "2026-11-05",
        ],
    )

    # 로컬 개발 환경에 실파일이 존재하더라도 기상·축제 전처리기가 호출되지 않음을 강제 검증
    def _forbid_external(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Set 1 운영 경로에서 기상·축제 전처리기가 호출되었습니다.")

    monkeypatch.setattr(subway_mod, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(subway_mod, "INTERIM_DATA_DIR", interim_dir)
    monkeypatch.setattr(subway_mod, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(features_mod, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(features_mod, "FEATURES_DATA_DIR", features_dir)
    monkeypatch.setattr(features_mod, "process_weather_data", _forbid_external)
    monkeypatch.setattr(features_mod, "build_festival_processed", _forbid_external)

    # 1. Set 1 운영 학습 (step='all' — 전처리부터 체크포인트 원자적 교체까지 일괄 실행)
    train_res = run_pipeline(
        mode="train",
        step="all",
        config_dict=FAST_OPERATIONAL_CONFIG,
        output_dir=out_dir,
        features_path=features_dir / "metro_features_operational.parquet",
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_paths = get_checkpoint_paths(ckpt_dir)
    for name, path in ckpt_paths.items():
        assert path.exists() and path.stat().st_size > 0, f"체크포인트 누락: {name}"

    # 2026-08-31 하드코딩 절단 없이 2026-11-01~05 전 기간이 학습에 반영되었는지 검증
    manifest = json.loads(ckpt_paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["feature_set"] == "subway_calendar"
    assert manifest["selected_features"] == OPERATIONAL_FEATURES
    assert manifest["train_date_range"]["min_date"] == "2026-11-01"
    assert manifest["train_date_range"]["max_date"] == "2026-11-05"
    assert manifest["model_version"] == train_res["model_version"]

    # 2. Accurate Track 배치 추론 (멱등 2회 실행, 비퇴화 예측값, 실행 매니페스트 정합 검증)
    pred_path = tmp_path / "predictions" / "predictions.parquet"
    run_manifest_path = tmp_path / "predictions" / "run_manifest.json"
    res_acc_1 = run_batch_inference(
        start_date="2026-12-01",
        end_date="2026-12-02",
        track="accurate",
        checkpoint_dir=ckpt_dir,
        output_path=pred_path,
        manifest_out=run_manifest_path,
    )
    res_acc_2 = run_batch_inference(
        start_date="2026-12-01",
        end_date="2026-12-02",
        track="accurate",
        checkpoint_dir=ckpt_dir,
        output_path=pred_path,
        manifest_out=run_manifest_path,
    )

    assert res_acc_1["row_count"] == 114 * 2 * 24
    assert res_acc_1["model_version"] == train_res["model_version"]
    assert pred_path.exists()
    assert run_manifest_path.exists()

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["mode"] == "predict"
    assert run_manifest["status"] == "ok"
    assert run_manifest["feature_set"] == "subway_calendar"
    assert run_manifest["track"] == "accurate"
    assert run_manifest["model_version"] == train_res["model_version"]
    assert run_manifest["start_date"] == "2026-12-01"
    assert run_manifest["end_date"] == "2026-12-02"
    assert run_manifest["station_count"] == 114
    assert run_manifest["row_count"] == 114 * 2 * 24

    df_disk = pl.read_parquet(pred_path)
    validate_predictions_df(df_disk)
    assert len(df_disk) == 114 * 2 * 24
    assert (df_disk["predicted_passengers"] > 0.0).any()
    assert float(df_disk["predicted_passengers"].std() or 0.0) > 0.0
    assert df_disk["model_version"].unique().to_list() == [train_res["model_version"]]
    np.testing.assert_allclose(
        res_acc_1["predictions"]["predicted_passengers"].to_numpy(),
        res_acc_2["predictions"]["predicted_passengers"].to_numpy(),
        atol=1e-5,
    )

    # 3. 동일 output_path 재추론(기간 변경) 시 원자적 덮어쓰기 및 임시 파일 잔존 없음 검증
    res_overwrite = run_batch_inference(
        start_date="2026-12-05",
        days=1,
        track="fast",
        checkpoint_dir=ckpt_dir,
        output_path=pred_path,
    )
    assert res_overwrite["track"] == "fast"
    assert res_overwrite["row_count"] == 114 * 24
    df_overwritten = pl.read_parquet(pred_path)
    validate_predictions_df(df_overwritten)
    assert df_overwritten["target_date"].unique().to_list() == [date(2026, 12, 5)]
    assert list(pred_path.parent.glob("*.tmp.parquet")) == []


def test_run_batch_inference_ensemble_math_and_lookback_isolation(
    trained_operational_artifacts: dict[str, Any],
    tmp_path: Path,
):
    """
    배치 추론의 앙상블 블렌딩 수치 정확성(RF+LGBM / RF+LGBM+LSTM 역정규화)과
    LSTM Lookback 버퍼 부착 및 calendar_master 최소일(lookback=0) 경계 동작을 검증합니다.
    """
    ckpt_dir = trained_operational_artifacts["checkpoint_dir"]

    # 1. _build_set1_frame의 lookback_hours > 24(2일 경계) 및 시작일=cal_min 경계 검증
    df_stn = pl.read_parquet(REFERENCE_DATA_DIR / "station_master.parquet").head(2)
    df_cal = pl.read_parquet(REFERENCE_DATA_DIR / "calendar_master.parquet")
    artifacts = load_inference_artifacts(ckpt_dir, track="accurate")

    frame_lb = _build_set1_frame(
        df_stations=df_stn,
        df_calendar=df_cal,
        start_date=date(2026, 12, 3),
        end_date=date(2026, 12, 3),
        target_encoder=artifacts.target_encoder,
        lookback_hours=30,
    )
    # 역당 lookback 30시간 + 본 구간 24시간 = 54시간, 2개 역 = 108행
    assert len(frame_lb) == 2 * (30 + 24)
    assert frame_lb.filter(pl.col("_is_target")).height == 2 * 24
    assert frame_lb.filter(~pl.col("_is_target")).height == 2 * 30
    assert frame_lb.equals(frame_lb.sort(["station_id", "target_datetime"]))

    # 2. Accurate Track 수치 일치 검증 (독립 계산 블렌딩 == run_batch_inference 출력)
    out_acc = tmp_path / "pred_acc.parquet"
    res_acc = run_batch_inference(
        start_date="2026-12-02",
        days=1,
        track="accurate",
        checkpoint_dir=ckpt_dir,
        output_path=out_acc,
    )
    df_stn_all = (
        pl.read_parquet(REFERENCE_DATA_DIR / "station_master.parquet")
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
    assert artifacts.lstm_estimator is not None and artifacts.scaler is not None
    full_expected = _build_set1_frame(
        df_stations=df_stn_all,
        df_calendar=df_cal,
        start_date=date(2026, 12, 2),
        end_date=date(2026, 12, 2),
        target_encoder=artifacts.target_encoder,
        lookback_hours=artifacts.lstm_estimator.seq_len,
    )
    mask = full_expected["_is_target"].to_numpy()
    target_expected = full_expected.filter(pl.col("_is_target"))
    X_target = np.ascontiguousarray(
        target_expected.select(OPERATIONAL_FEATURES).to_numpy(), dtype=np.float32
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
    X_full = np.ascontiguousarray(
        full_expected.select(OPERATIONAL_FEATURES).to_numpy(), dtype=np.float32
    )
    X_full_norm = (X_full - artifacts.scaler["mean_x"]) / artifacts.scaler["std_x"]
    pred_lstm_norm = artifacts.lstm_estimator.predict(
        X_full_norm, station_ids=full_expected["station_id"].to_numpy()
    )
    pred_lstm = np.clip(
        pred_lstm_norm[mask] * artifacts.scaler["std_y"] + artifacts.scaler["mean_y"],
        0.0,
        None,
    )
    expected_acc = np.clip(
        blend_accurate_track(
            pred_rf,
            pred_lgb,
            pred_lstm,
            w_rf=artifacts.weights["w_rf"],
            w_lgb=artifacts.weights["w_lgb"],
            w_lstm=artifacts.weights["w_lstm"],
        ),
        0.0,
        None,
    ).astype(np.float32)

    np.testing.assert_allclose(
        res_acc["predictions"]["predicted_passengers"].to_numpy(),
        expected_acc,
        atol=1e-4,
    )

    # 3. Fast Track 수치 일치 검증
    artifacts_fast = load_inference_artifacts(ckpt_dir, track="fast")
    out_fast = tmp_path / "pred_fast.parquet"
    res_fast = run_batch_inference(
        start_date="2026-12-02",
        days=1,
        track="fast",
        checkpoint_dir=ckpt_dir,
        output_path=out_fast,
    )
    expected_fast = np.clip(
        blend_fast_track(
            pred_rf,
            pred_lgb,
            w_rf=artifacts_fast.weights["w_rf"],
            w_lgb=artifacts_fast.weights["w_lgb"],
        ),
        0.0,
        None,
    ).astype(np.float32)
    np.testing.assert_allclose(
        res_fast["predictions"]["predicted_passengers"].to_numpy(),
        expected_fast,
        atol=1e-4,
    )

    # 4. start_date == calendar_master 최소일(선행 lookback=0시간) 경계에서 accurate 트랙 정상 추론 검증
    cal_min = df_cal["date"].min()
    assert isinstance(cal_min, date)
    out_cal_min = tmp_path / "pred_cal_min.parquet"
    res_cal_min = run_batch_inference(
        start_date=cal_min,
        days=1,
        track="accurate",
        checkpoint_dir=ckpt_dir,
        output_path=out_cal_min,
    )
    assert res_cal_min["row_count"] == 114 * 24
    validate_predictions_df(pl.read_parquet(out_cal_min))


def test_build_future_features_hour24_boundary_and_target_encoding(tmp_path: Path):
    """
    Set 1 미래 피처 생성의 핵심 불변식을 검증합니다:
    - target_encoder/target_encoder_path 모두 누락 시 ValueError
    - 114개 역사 × 24개 시간대(1~24) 생성 및 연도 전환 24시 경계 보존
    - 달력 마스터 특일(신정/전야) 및 원주 주기성 불변식 보존
    - StationTargetEncoder O(1) 매핑 및 미관측 역사 global_mean 수렴 대치
    """
    with pytest.raises(ValueError, match="target_encoder"):
        build_future_features(start_date="2026-12-31", days=1)

    encoder = StationTargetEncoder(m=10.0)
    encoder.global_mean = 500.0
    encoder.station_map = {119: 3200.5, 203: 1800.25}
    enc_path = tmp_path / "encoder.parquet"
    encoder.save_mapping(enc_path)

    df = build_future_features(
        start_date="2026-12-31",
        days=2,
        target_encoder_path=enc_path,
    )
    # 인메모리 target_encoder 직접 주입 경로도 동일 결과 생성 검증
    df_mem = build_future_features(
        start_date=date(2026, 12, 31),
        days=2,
        target_encoder=encoder,
    )
    assert df.equals(df_mem)

    assert len(df) == 114 * 2 * 24
    for col in OPERATIONAL_FEATURES:
        assert col in df.columns
        assert df[col].null_count() == 0

    # 주기성 원주 불변식 (sin^2 + cos^2 == 1) 검증
    for prefix in ("hour", "month", "weekday"):
        norm_sq = df[f"{prefix}_sin"] ** 2 + df[f"{prefix}_cos"] ** 2
        assert ((norm_sq - 1.0).abs() < 1e-4).all()

    # 심야 24시 경계 검증: 2026-12-31 24시 -> 2027-01-01 00:00:00 (연도 전환 경계 포함)
    row_h24 = df.filter(
        (pl.col("station_id") == 119)
        & (pl.col("target_date") == date(2026, 12, 31))
        & (pl.col("target_hour") == 24)
    ).to_dicts()[0]
    row_h01 = df.filter(
        (pl.col("station_id") == 119)
        & (pl.col("target_date") == date(2026, 12, 31))
        & (pl.col("target_hour") == 1)
    ).to_dicts()[0]
    row_new_year = df.filter(
        (pl.col("station_id") == 119)
        & (pl.col("target_date") == date(2027, 1, 1))
        & (pl.col("target_hour") == 9)
    ).to_dicts()[0]
    assert row_h24["target_date"] == date(2026, 12, 31)
    assert row_h24["target_datetime"] == datetime(2027, 1, 1, 0, 0, 0)
    assert row_h01["target_datetime"] == datetime(2026, 12, 31, 1, 0, 0)
    assert row_h24["hour_sin"] == pytest.approx(0.0, abs=1e-5)
    assert row_h24["hour_cos"] == pytest.approx(1.0, abs=1e-5)
    # 2027-01-01 신정 공휴일 및 2026-12-31 전야 반영 검증
    assert row_h24["is_holiday_eve"] is True
    assert row_new_year["is_holiday"] is True

    # 관측 역사는 매핑값, 미관측 역사는 global_mean 대치 검증
    assert row_h24["station_target_encoded"] == pytest.approx(3200.5, rel=1e-5)
    unseen_row = df.filter(
        (pl.col("station_id") == 95)
        & (pl.col("target_date") == date(2026, 12, 31))
        & (pl.col("target_hour") == 1)
    ).to_dicts()[0]
    assert unseen_row["station_target_encoded"] == pytest.approx(500.0, rel=1e-5)


def test_resolve_date_range_defaults_to_calendar_master_end_and_edge_cases():
    """날짜 미지정 시 calendar_master 말일 자동 설정, 타입/형식/내부 공백/범위 초과 예외를 검증합니다."""
    cal_df = pl.read_parquet(REFERENCE_DATA_DIR / "calendar_master.parquet")
    cal_min = cal_df["date"].min()
    cal_max = cal_df["date"].max()
    assert isinstance(cal_min, date) and isinstance(cal_max, date)

    # 1. start/end/days 모두 미지정 -> [reference_today, calendar_master 말일]
    s1, e1 = resolve_prediction_date_range(cal_df, reference_today=date(2028, 12, 20))
    assert s1 == date(2028, 12, 20)
    assert e1 == cal_max

    # 2. start_date만 지정 -> [start_date, calendar_master 말일]
    s2, e2 = resolve_prediction_date_range(cal_df, start_date="2028-12-25")
    assert s2 == date(2028, 12, 25)
    assert e2 == cal_max

    # 3. start_date + days 지정 및 date 객체 직접 전달
    s3, e3 = resolve_prediction_date_range(cal_df, start_date="2026-12-01", days=3)
    assert (s3, e3) == (date(2026, 12, 1), date(2026, 12, 3))
    s3_single, e3_single = resolve_prediction_date_range(
        cal_df, start_date=date(2026, 12, 1), days=1
    )
    assert (s3_single, e3_single) == (date(2026, 12, 1), date(2026, 12, 1))

    # 4. end_date와 days 동시 지정 거부
    with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
        resolve_prediction_date_range(
            cal_df, start_date="2026-12-01", end_date="2026-12-03", days=3
        )

    # 5. days < 1 거부
    with pytest.raises(ValueError, match="1 이상의 정수"):
        resolve_prediction_date_range(cal_df, start_date="2026-12-01", days=0)

    # 6. 종료일이 시작일보다 앞섬 거부
    with pytest.raises(ValueError, match="앞설 수 없습니다"):
        resolve_prediction_date_range(
            cal_df, start_date="2026-12-10", end_date="2026-12-01"
        )

    # 7. calendar_master 말일 초과 / 최소일 미만 날짜 요청 거부 (결측 0 대치 금지)
    with pytest.raises(ValueError, match="달력 마스터에 수록되지 않은 날짜"):
        resolve_prediction_date_range(
            cal_df,
            start_date="2028-12-30",
            end_date=(cal_max + timedelta(days=2)).isoformat(),
        )
    with pytest.raises(ValueError, match="달력 마스터에 수록되지 않은 날짜"):
        resolve_prediction_date_range(
            cal_df,
            start_date=(cal_min - timedelta(days=1)).isoformat(),
            end_date=cal_min.isoformat(),
        )

    # 8. calendar_master 내부 날짜 공백(중간 하루 누락) 거부
    gapped_cal = cal_df.filter(pl.col("date") != date(2026, 12, 2))
    with pytest.raises(ValueError, match="달력 마스터에 수록되지 않은 날짜가 1일"):
        resolve_prediction_date_range(
            gapped_cal, start_date="2026-12-01", end_date="2026-12-03"
        )

    # 9. 실행일 자체가 calendar_master 말일 초과 시 안내 예외
    with pytest.raises(ValueError, match="calendar_master 말일"):
        resolve_prediction_date_range(
            cal_df, reference_today=cal_max + timedelta(days=1)
        )

    # 10. 잘못된 날짜 형식/타입 및 비정상 calendar_df 거부
    with pytest.raises(ValueError, match="날짜 형식이 올바르지 않습니다"):
        resolve_prediction_date_range(cal_df, start_date="2026/12/01", days=1)
    with pytest.raises(ValueError, match="지원하지 않는 start_date 타입"):
        resolve_prediction_date_range(
            cal_df,
            start_date=datetime(2026, 12, 1, 0, 0),  # type: ignore[arg-type]
            days=1,
        )
    with pytest.raises(ValueError, match="비어 있거나 'date' 컬럼이 없습니다"):
        resolve_prediction_date_range(cal_df.head(0), start_date="2026-12-01", days=1)
    with pytest.raises(ValueError, match="비어 있거나 'date' 컬럼이 없습니다"):
        resolve_prediction_date_range(
            pl.DataFrame({"other": [1]}), start_date="2026-12-01", days=1
        )
    with pytest.raises(ValueError, match="유효한 Date 타입이 아닙니다"):
        resolve_prediction_date_range(
            pl.DataFrame({"date": ["2026-12-01"]}), start_date="2026-12-01", days=1
        )


@pytest.mark.parametrize(
    "missing_key",
    ["manifest", "target_encoder", "lgbm", "rf", "lstm", "scaler"],
)
def test_load_inference_artifacts_missing_each_file_raises_file_not_found(
    trained_operational_artifacts: dict[str, Any],
    tmp_path: Path,
    missing_key: str,
):
    """체크포인트·스케일러·인코더·매니페스트 중 하나라도 없으면 폴백 없이 FileNotFoundError를 발생시킵니다."""
    ckpt_copy = tmp_path / f"ckpt_missing_{missing_key}"
    shutil.copytree(trained_operational_artifacts["checkpoint_dir"], ckpt_copy)

    paths = get_checkpoint_paths(ckpt_copy)
    paths[missing_key].unlink()

    with pytest.raises(FileNotFoundError, match=missing_key):
        load_inference_artifacts(ckpt_copy, track="accurate")

    # fast 트랙은 lstm/scaler 없이도 정상 로드되며, 나머지 4개 부재 시에는 FileNotFoundError
    if missing_key in ("lstm", "scaler"):
        loaded_fast = load_inference_artifacts(ckpt_copy, track="fast")
        assert loaded_fast.track == "fast"
        assert loaded_fast.lstm_estimator is None
    else:
        with pytest.raises(FileNotFoundError, match=missing_key):
            load_inference_artifacts(ckpt_copy, track="fast")


def test_load_inference_artifacts_contract_and_model_dim_validation(
    trained_operational_artifacts: dict[str, Any],
    tmp_path: Path,
):
    """매니페스트 계약 위반, 모델/스케일러 피처 차원 불일치, 참조 파일 부재·빈 테이블 시 명시 실패를 검증합니다."""
    import joblib

    src_ckpt = trained_operational_artifacts["checkpoint_dir"]

    def _copy_ckpt(name: str) -> tuple[Path, dict[str, Path]]:
        dst = tmp_path / name
        shutil.copytree(src_ckpt, dst)
        return dst, get_checkpoint_paths(dst)

    # 1. 지원하지 않는 추론 트랙 거부
    with pytest.raises(ValueError, match="지원하지 않는 추론 트랙"):
        load_inference_artifacts(src_ckpt, track="medium")

    # 2. 매니페스트 계약 위반 (feature_set / selected_features / model_version / weights)
    bad_cases: list[tuple[str, Any, str]] = [
        ("bad_fset", lambda m: m.update({"feature_set": "short"}), "Set 1"),
        (
            "bad_feats",
            lambda m: m.update(
                {"selected_features": OPERATIONAL_FEATURES + ["temperature"]}
            ),
            r"Set 1\(13종\) 계약과 일치하지 않습니다",
        ),
        ("empty_ver", lambda m: m.update({"model_version": "   "}), "model_version"),
        (
            "missing_weights",
            lambda m: m["weights"].pop("accurate_track"),
            "accurate_track",
        ),
        (
            "missing_w_key",
            lambda m: m["weights"]["accurate_track"].pop("w_lstm"),
            "w_lstm",
        ),
        (
            "neg_weight",
            lambda m: m["weights"]["accurate_track"].update(
                {"w_rf": -0.2, "w_lgb": 0.6, "w_lstm": 0.6}
            ),
            "유효하지 않은 앙상블 가중치",
        ),
        (
            "bad_w_sum",
            lambda m: m["weights"]["accurate_track"].update(
                {"w_rf": 0.5, "w_lgb": 0.5, "w_lstm": 0.5}
            ),
            "가중치 합은 1.0",
        ),
        (
            "bad_lstm_dim",
            lambda m: m["lstm_config"].update({"input_dim": 17}),
            "input_dim",
        ),
    ]
    for case_name, mutator, err_match in bad_cases:
        ckpt_dir, paths = _copy_ckpt(case_name)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        mutator(manifest)
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=err_match):
            load_inference_artifacts(ckpt_dir, track="accurate")

    # 3. 바이너리 모델 피처 수 불일치(매니페스트는 정상이지만 LGBM/RF/Scaler 차원이 Set 1과 불일치)
    rng = np.random.default_rng(0)
    X_wrong = rng.standard_normal((20, 17)).astype(np.float32)
    y_wrong = rng.standard_normal(20).astype(np.float32)

    ckpt_bad_lgb, paths_bad_lgb = _copy_ckpt("bad_lgb_dim")
    bad_lgb = train_lightgbm(X_wrong, y_wrong, n_estimators=5, n_jobs=1)
    bad_lgb.booster_.save_model(str(paths_bad_lgb["lgbm"]))
    with pytest.raises(ValueError, match="LightGBM 모델 피처 수"):
        load_inference_artifacts(ckpt_bad_lgb, track="fast")

    ckpt_bad_rf, paths_bad_rf = _copy_ckpt("bad_rf_dim")
    bad_rf = train_random_forest(
        X_wrong, y_wrong, n_estimators=5, max_depth=3, n_jobs=1
    )
    joblib.dump(bad_rf, paths_bad_rf["rf"])
    with pytest.raises(ValueError, match="Random Forest 모델 피처 수"):
        load_inference_artifacts(ckpt_bad_rf, track="fast")

    ckpt_bad_scaler_keys, paths_bad_sk = _copy_ckpt("bad_scaler_keys")
    joblib.dump({"mean_x": np.zeros(13)}, paths_bad_sk["scaler"])
    with pytest.raises(ValueError, match="스케일러 파일 형식"):
        load_inference_artifacts(ckpt_bad_scaler_keys, track="accurate")

    ckpt_bad_scaler_dim, paths_bad_sd = _copy_ckpt("bad_scaler_dim")
    joblib.dump(
        {
            "mean_x": np.zeros((1, 17), dtype=np.float32),
            "std_x": np.ones((1, 17), dtype=np.float32),
            "mean_y": 0.0,
            "std_y": 1.0,
        },
        paths_bad_sd["scaler"],
    )
    with pytest.raises(ValueError, match="스케일러 피처 차원"):
        load_inference_artifacts(ckpt_bad_scaler_dim, track="accurate")

    # 4. 참조 파일 부재 시 FileNotFoundError 및 빈 참조 파일 시 ValueError
    with pytest.raises(FileNotFoundError, match="역사 마스터"):
        run_batch_inference(
            start_date="2026-12-01",
            days=1,
            checkpoint_dir=src_ckpt,
            output_path=tmp_path / "out.parquet",
            station_master_path=tmp_path / "missing_station.parquet",
        )
    with pytest.raises(FileNotFoundError, match="달력 마스터"):
        run_batch_inference(
            start_date="2026-12-01",
            days=1,
            checkpoint_dir=src_ckpt,
            output_path=tmp_path / "out.parquet",
            calendar_master_path=tmp_path / "missing_calendar.parquet",
        )

    empty_stn = tmp_path / "empty_station.parquet"
    pl.read_parquet(REFERENCE_DATA_DIR / "station_master.parquet").head(
        0
    ).write_parquet(empty_stn)
    with pytest.raises(ValueError, match="역사 마스터 파일이 비어 있습니다"):
        run_batch_inference(
            start_date="2026-12-01",
            days=1,
            checkpoint_dir=src_ckpt,
            output_path=tmp_path / "out.parquet",
            station_master_path=empty_stn,
        )

    empty_cal = tmp_path / "empty_calendar.parquet"
    pl.read_parquet(REFERENCE_DATA_DIR / "calendar_master.parquet").head(
        0
    ).write_parquet(empty_cal)
    with pytest.raises(ValueError, match="달력 마스터 파일이 비어 있습니다"):
        run_batch_inference(
            start_date="2026-12-01",
            days=1,
            checkpoint_dir=src_ckpt,
            output_path=tmp_path / "out.parquet",
            calendar_master_path=empty_cal,
        )


def _make_valid_prediction_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "station_id": [119, 119],
            "station_name": ["서면", "서면"],
            "line_num": [1, 1],
            "target_date": [date(2026, 12, 1), date(2026, 12, 1)],
            "target_hour": [8, 24],
            "target_datetime": [
                datetime(2026, 12, 1, 8, 0, 0),
                datetime(2026, 12, 2, 0, 0, 0),
            ],
            "predicted_passengers": [2500.0, 120.5],
            "model_version": ["0.1.0-operational-test", "0.1.0-operational-test"],
            "generated_at": [
                datetime(2026, 11, 25, 10, 0, 0),
                datetime(2026, 11, 25, 10, 0, 0),
            ],
        },
        schema=PREDICTION_SCHEMA,
    )


def test_validate_predictions_df_invariants_and_failure_paths():
    """예측 산출물 스키마 검증 함수의 정상 통과 및 모든 위반 경로를 검증합니다."""
    valid_df = _make_valid_prediction_df()
    validate_predictions_df(valid_df)

    # 1. 빈 DataFrame
    with pytest.raises(ValueError, match="비어 있습니다"):
        validate_predictions_df(valid_df.head(0))

    # 2. 컬럼 누락 및 순서 불일치
    with pytest.raises(ValueError, match="컬럼이 스키마 계약과 일치하지 않습니다"):
        validate_predictions_df(valid_df.drop("model_version"))
    reordered_cols = list(PREDICTION_COLUMNS)
    reordered_cols[0], reordered_cols[1] = reordered_cols[1], reordered_cols[0]
    with pytest.raises(ValueError, match="컬럼이 스키마 계약과 일치하지 않습니다"):
        validate_predictions_df(valid_df.select(reordered_cols))

    # 3. 타입 불일치 (Float64 != Float32)
    with pytest.raises(ValueError, match="타입이 일치하지 않습니다"):
        validate_predictions_df(
            valid_df.with_columns(pl.col("predicted_passengers").cast(pl.Float64))
        )

    # 4. Null 값 포함
    with pytest.raises(ValueError, match="결측치가 존재합니다"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.when(pl.col("target_hour") == 24)
                .then(None)
                .otherwise(pl.col("station_name"))
                .alias("station_name")
            )
        )

    # 5. 복합 키 중복
    with pytest.raises(ValueError, match="중복 키"):
        validate_predictions_df(pl.concat([valid_df, valid_df.head(1)]))

    # 6. 허용 범위 밖 line_num
    with pytest.raises(ValueError, match="line_num"):
        validate_predictions_df(
            valid_df.with_columns(pl.lit(5, dtype=pl.Int8).alias("line_num"))
        )

    # 7. 허용 범위 밖 target_hour (0 및 25)
    with pytest.raises(ValueError, match="target_hour는 1~24"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.when(pl.col("target_hour") == 8)
                .then(pl.lit(0, dtype=pl.Int8))
                .otherwise(pl.col("target_hour"))
                .alias("target_hour")
            )
        )
    with pytest.raises(ValueError, match="target_hour는 1~24"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.when(pl.col("target_hour") == 24)
                .then(pl.lit(25, dtype=pl.Int8))
                .otherwise(pl.col("target_hour"))
                .alias("target_hour")
            )
        )

    # 8. 심야 24시 및 일반 시간대(1~23시) target_datetime 경계 위반
    with pytest.raises(ValueError, match="target_datetime이"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.when(pl.col("target_hour") == 24)
                .then(pl.lit(datetime(2026, 12, 1, 0, 0, 0), dtype=pl.Datetime("us")))
                .otherwise(pl.col("target_datetime"))
                .alias("target_datetime")
            )
        )
    with pytest.raises(ValueError, match="target_datetime이"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.when(pl.col("target_hour") == 8)
                .then(pl.lit(datetime(2026, 12, 1, 9, 0, 0), dtype=pl.Datetime("us")))
                .otherwise(pl.col("target_datetime"))
                .alias("target_datetime")
            )
        )

    # 9. 비음수 경계(0.0 허용) 및 음수 거부
    validate_predictions_df(
        valid_df.with_columns(
            pl.lit(0.0, dtype=pl.Float32).alias("predicted_passengers")
        )
    )
    with pytest.raises(ValueError, match="음수일 수 없습니다"):
        validate_predictions_df(
            valid_df.with_columns(
                pl.lit(-1.0, dtype=pl.Float32).alias("predicted_passengers")
            )
        )

    # 10. NaN 및 Inf 예측값 거부
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="NaN 또는 무한대"):
            validate_predictions_df(
                valid_df.with_columns(
                    pl.lit(non_finite, dtype=pl.Float32).alias("predicted_passengers")
                )
            )

    # 11. 공백 model_version
    with pytest.raises(ValueError, match="model_version"):
        validate_predictions_df(
            valid_df.with_columns(pl.lit("   ", dtype=pl.String).alias("model_version"))
        )


def test_atomic_write_and_save_predictions_cleanup_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """
    스키마 검증 실패뿐 아니라 실제 디스크 교체(os.replace)·직렬화 실패 시에도
    기존 Parquet/JSON 파일이 손상되지 않고 임시 파일이 완전히 정리되는지 검증합니다.
    """
    target_pq = tmp_path / "predictions.parquet"
    valid_df = _make_valid_prediction_df()
    save_predictions_atomic(valid_df, target_pq)
    original_pq_bytes = target_pq.read_bytes()

    # 1. 스키마 검증 실패 시 기존 파일 보존
    invalid_df = valid_df.with_columns(
        pl.lit(-5.0, dtype=pl.Float32).alias("predicted_passengers")
    )
    with pytest.raises(ValueError, match="음수일 수 없습니다"):
        save_predictions_atomic(invalid_df, target_pq)
    assert target_pq.read_bytes() == original_pq_bytes

    # 2. atomic_write_parquet 내부 os.replace 실패 시 임시 파일 생성 후 정리 및 원본 보존 검증
    replaced_tmp_paths: list[Path] = []

    def _fail_replace(src: Any, dst: Any) -> None:
        src_p = Path(src)
        assert src_p.exists()
        assert src_p.parent == Path(dst).parent
        replaced_tmp_paths.append(src_p)
        raise OSError("disk full simulated")

    monkeypatch.setattr(predict_mod.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="disk full simulated"):
        atomic_write_parquet(valid_df, target_pq)

    assert len(replaced_tmp_paths) == 1
    assert not replaced_tmp_paths[0].exists()
    assert target_pq.read_bytes() == original_pq_bytes
    assert list(tmp_path.glob("*.tmp.parquet")) == []

    # 3. atomic_write_json 실패(os.replace 실패 & 직렬화 불가 TypeError) 시 임시 파일 정리 및 원본 보존
    target_json = tmp_path / "manifest.json"
    monkeypatch.undo()
    atomic_write_json({"status": "ok"}, target_json)
    original_json_text = target_json.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json({"bad": object()}, target_json)
    assert target_json.read_text(encoding="utf-8") == original_json_text
    assert list(tmp_path.glob("*.tmp.json")) == []

    monkeypatch.setattr(predict_mod.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="disk full simulated"):
        atomic_write_json({"status": "new"}, target_json)
    assert target_json.read_text(encoding="utf-8") == original_json_text
    assert list(tmp_path.glob("*.tmp.json")) == []


def test_offline_prediction_sample_matches_schema_contract():
    """data/samples/predictions/predictions_sample.parquet가 9컬럼 스키마·station_master 정합성을 만족하는지 검증합니다."""
    sample_path = SAMPLES_DIR / "predictions" / "predictions_sample.parquet"
    assert sample_path.exists()

    df = pl.read_parquet(sample_path)
    validate_predictions_df(df)
    assert set(df["line_num"].unique().to_list()) == {1, 2, 3, 4}
    assert set(df["target_hour"].unique().to_list()) == set(range(1, 25))
    assert len(df) == 4 * 2 * 24
    assert (df["predicted_passengers"] > 0.0).any()
    assert float(df["predicted_passengers"].std() or 0.0) > 0.0

    # station_id, station_name, line_num이 공식 station_master.parquet와 일치하는지 검증
    stn_master = pl.read_parquet(REFERENCE_DATA_DIR / "station_master.parquet").select(
        [
            pl.col("station_id").cast(pl.Int64),
            pl.col("station_name").cast(pl.String),
            pl.col("line_num").cast(pl.Int8),
        ]
    )
    sample_stations = df.select(["station_id", "station_name", "line_num"]).unique()
    matched = sample_stations.join(
        stn_master, on=["station_id", "station_name", "line_num"], how="inner"
    )
    assert len(matched) == len(sample_stations)

    # Windows 등 tzdata 미설치 환경에서도 naive UTC generated_at의 API 직렬화(to_dicts)가 안전하게 동작하는지 검증
    first_record = df.head(1).to_dicts()[0]
    assert isinstance(first_record["generated_at"], datetime)
    assert first_record["generated_at"].tzinfo is None
