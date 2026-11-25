"""
Set 1(13종: 지하철 기저 + 달력·특일·주기성) 운영 E2E 배치 학습 및 추론 파이프라인 실행 진입점
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
import yaml

from busan_metro_congestion.config import (
    ARTIFACTS_DIR,
    DEVICE,
    FEATURES_DATA_DIR,
    PROJECT_ROOT,
    set_seed,
)
from busan_metro_congestion.modeling.baseline import (
    OPERATIONAL_FEATURES,
    calculate_end_date,
    split_by_date_range,
    split_time_series,
    train_lightgbm,
    train_random_forest,
    validate_date_interval_yearly,
)
from busan_metro_congestion.modeling.ensemble import (
    blend_accurate_track,
    blend_fast_track,
    optimize_accurate_track_weights,
    optimize_fast_track_weights,
)
from busan_metro_congestion.modeling.lstm_net import MetroLSTMEstimator
from busan_metro_congestion.modeling.metrics import evaluate_predictions
from busan_metro_congestion.modeling.predict import (
    OPERATIONAL_FEATURE_SET,
    atomic_write_json,
    get_checkpoint_paths,
    run_batch_inference,
)
from busan_metro_congestion.preprocess.features import run_feature_pipeline
from busan_metro_congestion.preprocess.subway import StationTargetEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("pipeline_runner")


def load_config(config_path: Path) -> dict[str, Any]:
    """YAML 설정 로드 (파일이 없을 경우 기본 딕셔너리 반환)"""
    if not config_path.exists():
        logger.warning(
            f"설정 파일이 존재하지 않습니다: {config_path}. 기본 설정을 적용합니다."
        )
        return {"project": {"seed": 42}}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"project": {"seed": 42}}


def _make_temp_sibling(target_path: Path) -> Path:
    """원자적 교체를 위해 대상 파일과 같은 디렉터리에 임시 파일 경로를 생성합니다."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.stem}_",
        suffix=f".tmp{target_path.suffix}",
    )
    os.close(fd)
    return Path(tmp_name)


def _save_checkpoints_atomic(
    ckpt_dir: Path,
    lgb_model: Any,
    rf_model: Any,
    lstm_model: MetroLSTMEstimator,
    scaler_payload: dict[str, Any],
    target_encoder: StationTargetEncoder,
    manifest_payload: dict[str, Any],
) -> dict[str, Path]:
    """
    체크포인트·스케일러·타깃 인코더·매니페스트를 임시 파일에 먼저 기록한 뒤
    os.replace()로 원자적 완전 교체합니다 (매니페스트를 마지막에 커밋).
    """
    paths = get_checkpoint_paths(ckpt_dir)
    staged: list[tuple[Path, Path]] = []

    try:
        tmp_lgb = _make_temp_sibling(paths["lgbm"])
        staged.append((tmp_lgb, paths["lgbm"]))
        lgb_model.booster_.save_model(str(tmp_lgb))

        tmp_rf = _make_temp_sibling(paths["rf"])
        staged.append((tmp_rf, paths["rf"]))
        joblib.dump(rf_model, tmp_rf)

        if hasattr(lstm_model, "model_") and lstm_model.model_ is not None:
            tmp_lstm = _make_temp_sibling(paths["lstm"])
            staged.append((tmp_lstm, paths["lstm"]))
            torch.save(lstm_model.model_.state_dict(), tmp_lstm)

        tmp_scaler = _make_temp_sibling(paths["scaler"])
        staged.append((tmp_scaler, paths["scaler"]))
        joblib.dump(scaler_payload, tmp_scaler)

        tmp_enc = _make_temp_sibling(paths["target_encoder"])
        staged.append((tmp_enc, paths["target_encoder"]))
        target_encoder.save_mapping(tmp_enc)

        tmp_manifest = _make_temp_sibling(paths["manifest"])
        staged.append((tmp_manifest, paths["manifest"]))
        tmp_manifest.write_text(
            json.dumps(manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 매니페스트를 마지막에 커밋하여 부분 교체 상태 노출 차단
        for tmp_path, final_path in staged:
            os.replace(tmp_path, final_path)
    except Exception:
        for tmp_path, _ in staged:
            tmp_path.unlink(missing_ok=True)
        raise

    return paths


def _apply_split_target_encoding(
    train_df: pl.DataFrame,
    eval_df: pl.DataFrame | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame | None, StationTargetEncoder]:
    """학습 분할에만 StationTargetEncoder를 적합해 시계열 누수 없이 인코딩합니다."""
    encoder = StationTargetEncoder(m=10.0, n_splits=5, seed=seed)
    tr_base = (
        train_df.drop("station_target_encoded")
        if "station_target_encoded" in train_df.columns
        else train_df
    )
    tr_enc = encoder.fit_transform(
        tr_base,
        station_col="station_id",
        target_col="accumulated_passengers",
    ).sort(["station_id", "datetime"])

    ev_enc: pl.DataFrame | None = None
    if eval_df is not None:
        ev_base = (
            eval_df.drop("station_target_encoded")
            if "station_target_encoded" in eval_df.columns
            else eval_df
        )
        ev_enc = encoder.transform(ev_base, station_col="station_id").sort(
            ["station_id", "datetime"]
        )

    return tr_enc, ev_enc, encoder


def run_predict_pipeline(
    config_dict: dict[str, Any] | None = None,
    output_dir: Path | None = None,
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
) -> dict[str, Any]:
    """Set 1(13종) 학습 체크포인트 기반 운영 배치 추론 파이프라인 실행 함수."""
    cfg = config_dict or {}
    seed = int(cfg.get("project", {}).get("seed", 42))
    eff_output_path = output_path
    if eff_output_path is None and output_dir is not None:
        eff_output_path = Path(output_dir) / "predictions.parquet"

    return run_batch_inference(
        start_date=start_date,
        end_date=end_date,
        days=days,
        track=track,
        checkpoint_dir=checkpoint_dir,
        output_path=eff_output_path,
        manifest_out=manifest_out,
        station_master_path=station_master_path,
        calendar_master_path=calendar_master_path,
        reference_today=reference_today,
        seed=seed,
    )


def run_pipeline(
    mode: str = "train",
    step: str = "all",
    config_dict: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    features_path: Path | str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    **predict_kwargs: Any,
) -> dict[str, Any]:
    """
    부산 지하철 혼잡도 예측 Set 1(13종) 운영 E2E 파이프라인 실행 함수.

    모드:
    - 'train': Set 1 전처리, 앙상블 가중치 최적화, 평가 및 체크포인트 원자적 완전 교체
    - 'predict': Set 1 기학습 모델 기반 배치 추론 및 Parquet 산출물 원자적 교체
    """
    if mode == "predict":
        return run_predict_pipeline(
            config_dict=config_dict,
            output_dir=output_dir,
            **predict_kwargs,
        )
    elif mode != "train":
        raise ValueError(
            f"지원하지 않는 파이프라인 모드입니다: '{mode}'. ('train', 'predict' 중 선택)"
        )

    if step not in ("all", "preprocess", "train_eval"):
        raise ValueError(
            f"지원하지 않는 파이프라인 단계입니다: '{step}'. ('all', 'preprocess', 'train_eval' 중 선택)"
        )

    # 종료일 단독 지정 오류 검증
    if train_end and not train_start:
        raise ValueError("train_end를 지정하려면 train_start가 필요합니다.")
    if val_end and not val_start:
        raise ValueError("val_end를 지정하려면 val_start가 필요합니다.")
    if test_end and not test_start:
        raise ValueError("test_end를 지정하려면 test_start가 필요합니다.")

    # 시작일 기준 종료일 자동 계산 (윤년 고려 1년 뒤) 및 1년 단위 검증
    if train_start:
        if not train_end:
            train_end = calculate_end_date(train_start, years=1)
        else:
            validate_date_interval_yearly(train_start, train_end, label="train")

    if val_start:
        if not val_end:
            val_end = calculate_end_date(val_start, years=1)
        else:
            validate_date_interval_yearly(val_start, val_end, label="val")

    if test_start:
        if not test_end:
            test_end = calculate_end_date(test_start, years=1)
        else:
            validate_date_interval_yearly(test_start, test_end, label="test")

    # 커스텀 날짜 범위 지정 시 train_start와 val_start가 모두 필요
    if (train_start and not val_start) or (val_start and not train_start):
        raise ValueError(
            "사용자 지정 날짜 범위를 사용하려면 train_start와 val_start가 모두 지정되어야 합니다."
        )

    cfg = config_dict or {}
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    model_cfg = cfg.get("model", {})
    lgb_cfg = model_cfg.get("lightgbm", {})
    rf_cfg = model_cfg.get("random_forest", {})
    lstm_cfg = model_cfg.get("lstm", {})

    target_col = "accumulated_passengers"
    feature_cols = list(OPERATIONAL_FEATURES)

    logger.info("=" * 60)
    logger.info(f"부산 지하철 혼잡도 Set 1 운영 배치 파이프라인 시작 (Mode: {mode})")
    logger.info(f"   - 실행 디바이스: {DEVICE}")
    logger.info(
        f"   - 피처셋: {OPERATIONAL_FEATURE_SET} ({len(feature_cols)}종, 기상·축제 배제)"
    )
    logger.info(f"   - 실행 단계: {step}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # [1/4] Set 1 전처리 및 피처 엔지니어링 파이프라인 (기상·축제 무의존)
    # -----------------------------------------------------------------
    cached_feature_path = (
        Path(features_path)
        if features_path is not None
        else FEATURES_DATA_DIR / "metro_features_operational.parquet"
    )
    if step == "train_eval" and cached_feature_path.exists():
        logger.info(f"[1/4] 기존 Set 1 피처 데이터셋 로드: {cached_feature_path}")
        df_features = pl.read_parquet(cached_feature_path)
    else:
        logger.info("[1/4] Set 1 전처리 및 피처 엔지니어링 파이프라인 가동...")
        df_features = run_feature_pipeline(
            output_path=cached_feature_path,
            seed=seed,
            force_reload=(step != "train_eval"),
            include_external=False,
        )

    logger.info(
        f"피처 데이터셋 로드 완료: {len(df_features):,}행, {len(df_features.columns)}컬럼"
    )

    if step == "preprocess":
        logger.info("전처리 단계가 완료되었습니다. (--step preprocess)")
        return {"df_features": df_features}

    # -----------------------------------------------------------------
    # [2/4] 시계열 데이터 분할 및 누수 없는 타깃 인코딩
    # -----------------------------------------------------------------
    logger.info("[2/4] 시계열 데이터 분할 구성...")
    tr_df: pl.DataFrame
    val_df: pl.DataFrame
    ho_train: pl.DataFrame
    ho_test: pl.DataFrame
    custom_range_used = bool(train_start and train_end and val_start and val_end)

    if custom_range_used:
        assert train_start and train_end and val_start and val_end
        logger.info(
            f"사용자 지정 날짜 범위 분할 적용: Train({train_start}~{train_end}), Val({val_start}~{val_end})"
        )
        date_col = "date" if "date" in df_features.columns else "datetime"
        tr_df, val_df = split_by_date_range(
            df_features,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            date_col=date_col,
        )
        if len(tr_df) == 0 or len(val_df) == 0:
            raise ValueError(
                f"사용자 지정 날짜 범위에 데이터가 없습니다: "
                f"Train({train_start}~{train_end}: {len(tr_df)}행), "
                f"Val({val_start}~{val_end}: {len(val_df)}행)"
            )
        if test_start and test_end:
            logger.info(f"사용자 지정 Holdout Test 적용: ({test_start}~{test_end})")
            ho_train, ho_test = split_by_date_range(
                df_features,
                train_start=train_start,
                train_end=val_end,
                val_start=test_start,
                val_end=test_end,
                date_col=date_col,
            )
            if len(ho_test) == 0:
                raise ValueError(
                    f"사용자 지정 Test 날짜 범위({test_start}~{test_end})에 데이터가 없습니다."
                )
        else:
            ho_train = tr_df
            ho_test = val_df
    else:
        logger.info("시간 순서 보존 비율 분할(80:20)을 적용합니다.")
        train_full, ho_test = split_time_series(df_features, test_ratio=0.2)
        tr_df, val_df = split_time_series(train_full, test_ratio=0.2)
        ho_train = train_full

    # 분할별 학습셋에만 StationTargetEncoder를 적합해 시계열 누수 차단
    tr_df, val_enc, _ = _apply_split_target_encoding(tr_df, val_df, seed=seed)
    assert val_enc is not None
    val_df = val_enc
    ho_train, ho_test_enc, ho_encoder = _apply_split_target_encoding(
        ho_train, ho_test, seed=seed
    )
    assert ho_test_enc is not None
    ho_test = ho_test_enc

    logger.info(
        f"분할 완료 - 튜닝 학습: {len(tr_df):,}행, "
        f"검증: {len(val_df):,}행, 최종 테스트: {len(ho_test):,}행"
    )

    # -----------------------------------------------------------------
    # [3/4] 2-Track 모델 학습 및 앙상블 가중치 최적화
    # -----------------------------------------------------------------
    y_tr = np.ascontiguousarray(tr_df[target_col].to_numpy(), dtype=np.float32)
    y_val = np.ascontiguousarray(val_df[target_col].to_numpy(), dtype=np.float32)

    logger.info(f"[3/4] 2-Track 모델 학습 시작 ({len(feature_cols)}개 피처)")

    X_tr = np.ascontiguousarray(tr_df.select(feature_cols).to_numpy(), dtype=np.float32)
    X_val = np.ascontiguousarray(
        val_df.select(feature_cols).to_numpy(), dtype=np.float32
    )

    logger.info("  -> Fast Track: LightGBM 학습...")
    lgb_tune = train_lightgbm(
        X_tr,
        y_tr,
        n_estimators=lgb_cfg.get("n_estimators", 100),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        num_leaves=lgb_cfg.get("num_leaves", 31),
        n_jobs=lgb_cfg.get("n_jobs", -1),
        random_state=seed,
    )
    pred_lgb_val = np.clip(lgb_tune.predict(X_val), 0.0, None)

    logger.info("  -> Fast Track: Random Forest 학습...")
    rf_tune = train_random_forest(
        X_tr,
        y_tr,
        n_estimators=rf_cfg.get("n_estimators", 100),
        max_depth=rf_cfg.get("tune_max_depth", 8),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=seed,
    )
    pred_rf_val = np.clip(rf_tune.predict(X_val), 0.0, None)

    w_rf, w_lgb = optimize_fast_track_weights(
        y_val, pred_rf_val, pred_lgb_val, initial_weights=(0.3, 0.7)
    )

    logger.info("  -> Accurate Track: MetroLSTM 학습...")
    stn_tr = tr_df["station_id"].to_numpy() if "station_id" in tr_df.columns else None
    stn_val = (
        val_df["station_id"].to_numpy() if "station_id" in val_df.columns else None
    )

    mean_x_tr = X_tr.mean(axis=0, keepdims=True)
    std_x_tr = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_norm = (X_tr - mean_x_tr) / std_x_tr
    X_val_norm = (X_val - mean_x_tr) / std_x_tr

    mean_y_tr = float(y_tr.mean())
    std_y_tr = float(y_tr.std()) + 1e-6
    y_tr_norm = (y_tr - mean_y_tr) / std_y_tr

    lstm_tune = MetroLSTMEstimator(
        seq_len=lstm_cfg.get("seq_len", 3),
        hidden_dim1=lstm_cfg.get("hidden_dim1", 64),
        hidden_dim2=lstm_cfg.get("hidden_dim2", 32),
        dense_dim=lstm_cfg.get("dense_dim", 16),
        dropout_rate=lstm_cfg.get("dropout_rate", 0.2),
        lr=lstm_cfg.get("lr", 1e-3),
        epochs=lstm_cfg.get("tune_epochs", lstm_cfg.get("epochs", 10)),
        batch_size=lstm_cfg.get("batch_size", 64),
        random_state=seed,
        device=DEVICE,
    )
    lstm_tune.fit(X_tr_norm, y_tr_norm, station_ids=stn_tr)
    pred_lstm_val_norm = lstm_tune.predict(X_val_norm, station_ids=stn_val)
    pred_lstm_val = np.clip(pred_lstm_val_norm * std_y_tr + mean_y_tr, 0.0, None)

    w_rf_acc, w_lgb_acc, w_lstm_acc = optimize_accurate_track_weights(
        y_val, pred_rf_val, pred_lgb_val, pred_lstm_val, initial_weights=(0.2, 0.4, 0.4)
    )

    # -----------------------------------------------------------------
    # [4/4] Hold-out 평가 및 운영 배포 모델 재학습
    # -----------------------------------------------------------------
    logger.info("[4/4] Hold-out Test 성능 평가 및 운영 모델 학습...")
    X_ho_train = np.ascontiguousarray(
        ho_train.select(feature_cols).to_numpy(), dtype=np.float32
    )
    X_ho_test = np.ascontiguousarray(
        ho_test.select(feature_cols).to_numpy(), dtype=np.float32
    )

    y_ho_train = np.ascontiguousarray(ho_train[target_col].to_numpy(), dtype=np.float32)
    y_ho_test = np.ascontiguousarray(ho_test[target_col].to_numpy(), dtype=np.float32)
    stn_ho_tr = (
        ho_train["station_id"].to_numpy() if "station_id" in ho_train.columns else None
    )
    stn_ho_te = (
        ho_test["station_id"].to_numpy() if "station_id" in ho_test.columns else None
    )
    hours_test = ho_test["hour"].to_numpy() if "hour" in ho_test.columns else None

    final_lgb = train_lightgbm(
        X_ho_train,
        y_ho_train,
        n_estimators=lgb_cfg.get(
            "final_n_estimators", lgb_cfg.get("n_estimators", 120)
        ),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        num_leaves=lgb_cfg.get("num_leaves", 31),
        n_jobs=lgb_cfg.get("n_jobs", -1),
        random_state=seed,
    )
    pred_lgb_test = np.clip(final_lgb.predict(X_ho_test), 0.0, None)

    final_rf = train_random_forest(
        X_ho_train,
        y_ho_train,
        n_estimators=rf_cfg.get("final_n_estimators", rf_cfg.get("n_estimators", 100)),
        max_depth=rf_cfg.get("max_depth", 10),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=seed,
    )
    pred_rf_test = np.clip(final_rf.predict(X_ho_test), 0.0, None)

    mean_x_ho = X_ho_train.mean(axis=0, keepdims=True)
    std_x_ho = X_ho_train.std(axis=0, keepdims=True) + 1e-6
    X_ho_train_norm = (X_ho_train - mean_x_ho) / std_x_ho
    X_ho_test_norm = (X_ho_test - mean_x_ho) / std_x_ho

    mean_y_ho = float(y_ho_train.mean())
    std_y_ho = float(y_ho_train.std()) + 1e-6
    y_ho_train_norm = (y_ho_train - mean_y_ho) / std_y_ho

    final_lstm = MetroLSTMEstimator(
        seq_len=lstm_cfg.get("seq_len", 3),
        hidden_dim1=lstm_cfg.get("hidden_dim1", 64),
        hidden_dim2=lstm_cfg.get("hidden_dim2", 32),
        dense_dim=lstm_cfg.get("dense_dim", 16),
        dropout_rate=lstm_cfg.get("dropout_rate", 0.2),
        lr=lstm_cfg.get("lr", 1e-3),
        epochs=lstm_cfg.get("final_epochs", lstm_cfg.get("epochs", 12)),
        batch_size=lstm_cfg.get("batch_size", 64),
        random_state=seed,
        device=DEVICE,
    )
    final_lstm.fit(X_ho_train_norm, y_ho_train_norm, station_ids=stn_ho_tr)
    pred_lstm_test_norm = final_lstm.predict(X_ho_test_norm, station_ids=stn_ho_te)
    pred_lstm_test = np.clip(pred_lstm_test_norm * std_y_ho + mean_y_ho, 0.0, None)

    pred_fast_blend = np.clip(
        blend_fast_track(pred_rf_test, pred_lgb_test, w_rf=w_rf, w_lgb=w_lgb),
        0.0,
        None,
    )
    pred_accurate_blend = np.clip(
        blend_accurate_track(
            pred_rf_test,
            pred_lgb_test,
            pred_lstm_test,
            w_rf=w_rf_acc,
            w_lgb=w_lgb_acc,
            w_lstm=w_lstm_acc,
        ),
        0.0,
        None,
    )

    metrics_lgb = evaluate_predictions(y_ho_test, pred_lgb_test, hours=hours_test)
    metrics_rf = evaluate_predictions(y_ho_test, pred_rf_test, hours=hours_test)
    metrics_lstm = evaluate_predictions(y_ho_test, pred_lstm_test, hours=hours_test)
    metrics_fast = evaluate_predictions(y_ho_test, pred_fast_blend, hours=hours_test)
    metrics_accurate = evaluate_predictions(
        y_ho_test, pred_accurate_blend, hours=hours_test
    )

    logger.info("=" * 60)
    logger.info("최종 Hold-out Test 성능 평가 요약 (Set 1, 13종)")
    logger.info(
        f"  1) LightGBM      : MAE={metrics_lgb['mae']:.2f}, WAPE={metrics_lgb['wape']:.2f}%, Peak MAE={metrics_lgb['peak_mae']:.2f}, R2={metrics_lgb['r2']:.4f}"
    )
    logger.info(
        f"  2) Random Forest : MAE={metrics_rf['mae']:.2f}, WAPE={metrics_rf['wape']:.2f}%, Peak MAE={metrics_rf['peak_mae']:.2f}, R2={metrics_rf['r2']:.4f}"
    )
    logger.info(
        f"  3) MetroLSTM     : MAE={metrics_lstm['mae']:.2f}, WAPE={metrics_lstm['wape']:.2f}%, Peak MAE={metrics_lstm['peak_mae']:.2f}, R2={metrics_lstm['r2']:.4f}"
    )
    logger.info(
        f"  4) Fast-Track 앙상블    : MAE={metrics_fast['mae']:.2f}, WAPE={metrics_fast['wape']:.2f}%, Peak MAE={metrics_fast['peak_mae']:.2f}, R2={metrics_fast['r2']:.4f} (w_rf={w_rf:.3f}, w_lgb={w_lgb:.3f})"
    )
    logger.info(
        f"  5) Accurate-Track 앙상블: MAE={metrics_accurate['mae']:.2f}, WAPE={metrics_accurate['wape']:.2f}%, Peak MAE={metrics_accurate['peak_mae']:.2f}, R2={metrics_accurate['r2']:.4f} (w_rf={w_rf_acc:.3f}, w_lgb={w_lgb_acc:.3f}, w_lstm={w_lstm_acc:.3f})"
    )
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # 운영 배포 모델 최신 전 기간 재학습 (날짜 미지정 시)
    # -----------------------------------------------------------------
    deploy_df = ho_train
    deploy_lgb = final_lgb
    deploy_rf = final_rf
    deploy_lstm = final_lstm
    deploy_encoder = ho_encoder
    deploy_scaler = {
        "mean_x": mean_x_ho,
        "std_x": std_x_ho,
        "mean_y": mean_y_ho,
        "std_y": std_y_ho,
    }

    if not custom_range_used:
        logger.info(
            "운영 배포용 Set 1 최종 모델을 전체 수집 기간 데이터로 재학습합니다..."
        )
        deploy_df, _, deploy_encoder = _apply_split_target_encoding(
            df_features, None, seed=seed
        )
        X_deploy = np.ascontiguousarray(
            deploy_df.select(feature_cols).to_numpy(), dtype=np.float32
        )
        y_deploy = np.ascontiguousarray(
            deploy_df[target_col].to_numpy(), dtype=np.float32
        )
        stn_deploy = (
            deploy_df["station_id"].to_numpy()
            if "station_id" in deploy_df.columns
            else None
        )

        deploy_lgb = train_lightgbm(
            X_deploy,
            y_deploy,
            n_estimators=lgb_cfg.get(
                "final_n_estimators", lgb_cfg.get("n_estimators", 120)
            ),
            learning_rate=lgb_cfg.get("learning_rate", 0.05),
            num_leaves=lgb_cfg.get("num_leaves", 31),
            n_jobs=lgb_cfg.get("n_jobs", -1),
            random_state=seed,
        )
        deploy_rf = train_random_forest(
            X_deploy,
            y_deploy,
            n_estimators=rf_cfg.get(
                "final_n_estimators", rf_cfg.get("n_estimators", 100)
            ),
            max_depth=rf_cfg.get("max_depth", 10),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=seed,
        )

        mean_x_dep = X_deploy.mean(axis=0, keepdims=True)
        std_x_dep = X_deploy.std(axis=0, keepdims=True) + 1e-6
        X_deploy_norm = (X_deploy - mean_x_dep) / std_x_dep
        mean_y_dep = float(y_deploy.mean())
        std_y_dep = float(y_deploy.std()) + 1e-6
        y_deploy_norm = (y_deploy - mean_y_dep) / std_y_dep

        deploy_lstm = MetroLSTMEstimator(
            seq_len=lstm_cfg.get("seq_len", 3),
            hidden_dim1=lstm_cfg.get("hidden_dim1", 64),
            hidden_dim2=lstm_cfg.get("hidden_dim2", 32),
            dense_dim=lstm_cfg.get("dense_dim", 16),
            dropout_rate=lstm_cfg.get("dropout_rate", 0.2),
            lr=lstm_cfg.get("lr", 1e-3),
            epochs=lstm_cfg.get("final_epochs", lstm_cfg.get("epochs", 12)),
            batch_size=lstm_cfg.get("batch_size", 64),
            random_state=seed,
            device=DEVICE,
        )
        deploy_lstm.fit(X_deploy_norm, y_deploy_norm, station_ids=stn_deploy)
        deploy_scaler = {
            "mean_x": mean_x_dep,
            "std_x": std_x_dep,
            "mean_y": mean_y_dep,
            "std_y": std_y_dep,
        }

    # -----------------------------------------------------------------
    # 결과 영속화 (Checkpoints 원자적 완전 교체 & metrics.json)
    # -----------------------------------------------------------------
    base_out_dir = output_dir or (ARTIFACTS_DIR / "baseline")
    ckpt_dir = base_out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    project_version = str(cfg.get("project", {}).get("version", "0.1.0"))
    model_version = f"{project_version}-set1-{trained_at.strftime('%Y%m%dT%H%M%SZ')}"

    min_d = deploy_df["date"].min() if "date" in deploy_df.columns else None
    max_d = deploy_df["date"].max() if "date" in deploy_df.columns else None

    manifest_payload: dict[str, Any] = {
        "model_version": model_version,
        "feature_set": OPERATIONAL_FEATURE_SET,
        "selected_features": feature_cols,
        "trained_at": trained_at.isoformat(),
        "train_row_count": int(len(deploy_df)),
        "train_date_range": {
            "min_date": min_d.isoformat()
            if isinstance(min_d, date)
            else (str(min_d) if min_d is not None else None),
            "max_date": max_d.isoformat()
            if isinstance(max_d, date)
            else (str(max_d) if max_d is not None else None),
        },
        "weights": {
            "fast_track": {"w_rf": w_rf, "w_lgb": w_lgb},
            "accurate_track": {
                "w_rf": w_rf_acc,
                "w_lgb": w_lgb_acc,
                "w_lstm": w_lstm_acc,
            },
        },
        "lstm_config": {
            "input_dim": len(feature_cols),
            "seq_len": int(deploy_lstm.seq_len),
            "hidden_dim1": int(deploy_lstm.hidden_dim1),
            "hidden_dim2": int(deploy_lstm.hidden_dim2),
            "dense_dim": int(deploy_lstm.dense_dim),
            "dropout_rate": float(deploy_lstm.dropout_rate),
            "batch_size": int(deploy_lstm.batch_size),
        },
    }

    saved_ckpt_paths = _save_checkpoints_atomic(
        ckpt_dir=ckpt_dir,
        lgb_model=deploy_lgb,
        rf_model=deploy_rf,
        lstm_model=deploy_lstm,
        scaler_payload=deploy_scaler,
        target_encoder=deploy_encoder,
        manifest_payload=manifest_payload,
    )

    all_metrics = {
        "model_version": model_version,
        "feature_set": OPERATIONAL_FEATURE_SET,
        "selected_features": feature_cols,
        "weights": {
            "fast_track": {"w_rf": w_rf, "w_lgb": w_lgb},
            "accurate_track": {
                "w_rf": w_rf_acc,
                "w_lgb": w_lgb_acc,
                "w_lstm": w_lstm_acc,
            },
        },
        "models": {
            "lightgbm": metrics_lgb,
            "random_forest": metrics_rf,
            "metrolstm": metrics_lstm,
            "fast_track_ensemble": metrics_fast,
            "accurate_track_ensemble": metrics_accurate,
        },
    }

    metrics_file = base_out_dir / "metrics.json"
    atomic_write_json(all_metrics, metrics_file)

    logger.info(f"체크포인트 원자적 저장 완료: {ckpt_dir} (version={model_version})")
    logger.info(f"평가 메트릭 저장 완료: {metrics_file}")

    return {
        "model_version": model_version,
        "feature_set": OPERATIONAL_FEATURE_SET,
        "checkpoint_paths": {k: str(v) for k, v in saved_ckpt_paths.items()},
        "metrics": all_metrics,
        "fast_track_weights": (w_rf, w_lgb),
        "accurate_track_weights": (w_rf_acc, w_lgb_acc, w_lstm_acc),
        "selected_features": feature_cols,
        "models": {
            "lgbm": deploy_lgb,
            "rf": deploy_rf,
            "lstm": deploy_lstm,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서 정의 (Set 1 운영 train 및 predict 모드 분리)"""
    default_config_path = PROJECT_ROOT / "configs" / "config.yaml"
    parser = argparse.ArgumentParser(
        description="부산 지하철 혼잡도 예측 Set 1 운영 E2E 파이프라인 (모델 학습 및 배치 추론)"
    )
    subparsers = parser.add_subparsers(
        dest="mode",
        help="실행 모드 ('train': Set 1 모델 학습, 'predict': Set 1 배치 추론)",
    )

    # 1. train 모드 (Set 1 운영 모델 학습, 앙상블 블렌딩 및 체크포인트 원자적 교체)
    train_parser = subparsers.add_parser(
        "train", help="Set 1(13종) 운영 모델 학습 및 평가 파이프라인"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="설정 YAML 파일 경로",
    )
    train_parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["all", "preprocess", "train_eval"],
        help="실행할 파이프라인 단계 ('all', 'preprocess', 'train_eval', 미지정 시 config.yaml 적용)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="산출물 아티팩트 저장 디렉터리 경로",
    )
    train_parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="학습 시작일 (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="학습 종료일 (YYYY-MM-DD, 미지정 시 시작일 기준 윤년 고려 1년 뒤 자동 설정)",
    )
    train_parser.add_argument(
        "--val-start",
        type=str,
        default=None,
        help="검증 시작일 (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="검증 종료일 (YYYY-MM-DD, 미지정 시 시작일 기준 윤년 고려 1년 뒤 자동 설정)",
    )
    train_parser.add_argument(
        "--test-start",
        type=str,
        default=None,
        help="테스트 시작일 (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--test-end",
        type=str,
        default=None,
        help="테스트 종료일 (YYYY-MM-DD, 미지정 시 시작일 기준 윤년 고려 1년 뒤 자동 설정)",
    )

    # 2. predict 모드 (Set 1 운영 배치 추론)
    predict_parser = subparsers.add_parser(
        "predict",
        help="Set 1(13종) 학습 모델 기반 운영 배치 추론 및 Parquet 산출물 생성",
    )
    predict_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="예측 시작일 (YYYY-MM-DD, 미지정 시 실행 기준 KST 당일)",
    )
    predict_end_group = predict_parser.add_mutually_exclusive_group()
    predict_end_group.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="예측 종료일 (YYYY-MM-DD, 미지정 시 calendar_master.parquet 말일까지)",
    )
    predict_end_group.add_argument(
        "--days",
        type=int,
        default=None,
        help="시작일로부터 예측할 일수 (양의 정수, --end-date와 동시 지정 불가)",
    )
    predict_parser.add_argument(
        "--track",
        type=str,
        default="accurate",
        choices=["accurate", "fast"],
        help="추론 앙상블 트랙 ('accurate': RF+LGBM+LSTM 기본값, 'fast': RF+LGBM)",
    )
    predict_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Set 1 체크포인트 디렉터리 경로 (기본값: artifacts/baseline/checkpoints)",
    )
    predict_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="예측 산출물 Parquet 저장 경로 (기본값: artifacts/predictions/predictions.parquet)",
    )
    predict_parser.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="배치 추론 실행 요약 매니페스트 JSON 저장 경로 (선택)",
    )

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI 콘솔 스크립트 엔트리포인트 (uv run busan-metro)"""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()

    # 첫 인자가 서브커맨드('train', 'predict', '-h', '--help')가 아니면 기본 train 모드로 해석
    if argv and argv[0] not in ("train", "predict", "-h", "--help"):
        effective_argv = ["train"] + list(argv)
    elif not argv:
        effective_argv = ["train"]
    else:
        effective_argv = list(argv)

    args = parser.parse_args(effective_argv)

    if args.mode == "predict":
        return run_pipeline(
            mode="predict",
            start_date=args.start_date,
            end_date=args.end_date,
            days=args.days,
            track=args.track,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output_path,
            manifest_out=args.manifest_out,
        )

    # train 모드
    config = load_config(Path(args.config))
    pipeline_cfg = config.get("pipeline", {})
    effective_step = args.step or pipeline_cfg.get("step", "all")
    out_dir = Path(args.output_dir) if args.output_dir else None

    return run_pipeline(
        mode="train",
        step=effective_step,
        config_dict=config,
        output_dir=out_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


def cli_main() -> None:
    """CLI 콘솔 스크립트 실행 진입점 (정상 종료 0 보장)"""
    main()


if __name__ == "__main__":
    cli_main()
