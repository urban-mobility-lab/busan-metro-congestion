"""
전체 E2E 배치 학습, 2-Track 앙상블 블렌딩 및 논문 평가 파이프라인 통합 실행 진입점
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    calculate_end_date,
    get_feature_columns,
    split_by_date_range,
    split_rolling_time_series,
    split_time_series,
    train_lightgbm,
    train_random_forest,
    validate_date_interval_yearly,
    validate_short_horizon_lead_time,
)
from busan_metro_congestion.modeling.ensemble import (
    blend_accurate_track,
    blend_fast_track,
    optimize_accurate_track_weights,
    optimize_fast_track_weights,
)
from busan_metro_congestion.modeling.lstm_net import MetroLSTMEstimator
from busan_metro_congestion.modeling.metrics import evaluate_predictions
from busan_metro_congestion.preprocess.features import run_feature_pipeline

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


def run_predict_pipeline(
    horizon: str = "short",
    config_dict: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """학습된 모델 기반 예측 서빙 파이프라인 인터페이스."""
    if horizon not in ("short", "long"):
        raise ValueError(
            f"지원하지 않는 예측 호라이즌입니다: '{horizon}'. ('short', 'long' 중 선택)"
        )

    logger.info("=" * 60)
    logger.info("부산 지하철 혼잡도 예측 서빙 파이프라인 (Inference Mode)")
    logger.info(f"   - 예측 호라이즌: {horizon}")
    logger.info("   - 상태: 예측 서빙 인터페이스 준비 완료")
    logger.info("=" * 60)

    return {
        "mode": "predict",
        "horizon": horizon,
        "status": "template",
        "predictions": None,
    }


def run_pipeline(
    mode: str = "train",
    horizon: str = "short",
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
) -> dict[str, Any]:
    """
    부산 지하철 혼잡도 예측 E2E 파이프라인 실행 함수.

    모드:
    - 'train': 모델 학습 및 앙상블 가중치 최적화, 최종 평가
    - 'predict': 기학습 모델 기반 예측 서빙 인터페이스
    """
    if mode == "predict":
        return run_predict_pipeline(
            horizon=horizon,
            config_dict=config_dict,
            output_dir=output_dir,
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
    feature_cols = get_feature_columns(horizon)

    logger.info("=" * 60)
    logger.info(f"부산 지하철 혼잡도 예측 배치 파이프라인 시작 (Mode: {mode})")
    logger.info(f"   - 실행 디바이스: {DEVICE}")
    logger.info(f"   - 예측 호라이즌: {horizon} ({len(feature_cols)}개 피처)")
    logger.info(f"   - 실행 단계: {step}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # [1/4] 전처리 및 피처 엔지니어링 파이프라인
    # -----------------------------------------------------------------
    cached_feature_path = (
        Path(features_path)
        if features_path is not None
        else FEATURES_DATA_DIR / "metro_features.parquet"
    )
    if step == "train_eval" and cached_feature_path.exists():
        logger.info(f"[1/4] 기존 피처 데이터셋 로드: {cached_feature_path}")
        df_features = pl.read_parquet(cached_feature_path)
    else:
        logger.info("[1/4] 전처리 및 피처 엔지니어링 파이프라인 가동...")
        df_features = run_feature_pipeline(
            output_path=cached_feature_path,
            seed=seed,
            force_reload=(step != "train_eval"),
        )

    logger.info(
        f"피처 데이터셋 로드 완료: {len(df_features):,}행, {len(df_features.columns)}컬럼"
    )

    if step == "preprocess":
        logger.info("전처리 단계가 완료되었습니다. (--step preprocess)")
        return {"df_features": df_features}

    if horizon == "short":
        validate_short_horizon_lead_time(df_features)

    # -----------------------------------------------------------------
    # [2/4] 시계열 데이터 분할
    # -----------------------------------------------------------------
    logger.info("[2/4] 시계열 데이터 분할 구성...")
    split_ready = False
    tr_df: pl.DataFrame
    val_df: pl.DataFrame
    ho_train: pl.DataFrame
    ho_test: pl.DataFrame

    # 사용자 지정 날짜 범위가 주어진 경우 우선 적용
    if train_start and train_end and val_start and val_end:
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
        split_ready = True
    else:
        try:
            split_res = split_rolling_time_series(
                df_features,
                strategy="rolling",
                n_tracks=1,
            )
            valid_splits = [
                s
                for s in split_res["validation_splits"]
                if len(s["train_df"]) > 0 and len(s["val_df"]) > 0
            ]
            if len(valid_splits) > 0 and len(split_res["holdout_test"]["test_df"]) > 0:
                tr_df = valid_splits[0]["train_df"]
                val_df = valid_splits[0]["val_df"]
                ho_train = split_res["holdout_test"]["train_df"]
                ho_test = split_res["holdout_test"]["test_df"]
                split_ready = True
                logger.info(
                    f"표준 시계열 분할 적용: 검증 학습 {len(tr_df):,}행, 검증 {len(val_df):,}행, Holdout Test {len(ho_test):,}행"
                )
        except Exception as e:
            logger.warning(
                f"표준 시계열 분할 생성 중 예외 발생({e}). 시간 순서 비율 분할로 전환합니다."
            )
            split_ready = False

    if not split_ready:
        logger.info("시간 순서 보존 비율 분할(80:20)을 적용합니다.")
        train_full, ho_test = split_time_series(df_features, test_ratio=0.2)
        tr_df, val_df = split_time_series(train_full, test_ratio=0.2)
        ho_train = train_full

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

    # 입력 행렬 추출 (float32 연속 메모리 보장)
    X_tr = np.ascontiguousarray(tr_df.select(feature_cols).to_numpy(), dtype=np.float32)
    X_val = np.ascontiguousarray(
        val_df.select(feature_cols).to_numpy(), dtype=np.float32
    )

    # 1) Fast Track: LightGBM & Random Forest
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

    # Fast Track SLSQP 최적 가중치 탐색
    w_rf, w_lgb = optimize_fast_track_weights(
        y_val, pred_rf_val, pred_lgb_val, initial_weights=(0.3, 0.7)
    )

    # 2) Accurate Track: MetroLSTM (학습셋 기준 Z-score 정규화)
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

    # Accurate Track SLSQP 최적 가중치 탐색
    w_rf_acc, w_lgb_acc, w_lstm_acc = optimize_accurate_track_weights(
        y_val, pred_rf_val, pred_lgb_val, pred_lstm_val, initial_weights=(0.2, 0.4, 0.4)
    )

    # -----------------------------------------------------------------
    # [4/4] 최종 모델 학습, Hold-out Test 추론 및 평가
    # -----------------------------------------------------------------
    logger.info("[4/4] 최종 Hold-out Test 모델 학습 및 평가...")
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

    # 최종 모델 재학습
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

    # 앙상블 블렌딩
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

    # 평가 지표 산출
    metrics_lgb = evaluate_predictions(y_ho_test, pred_lgb_test, hours=hours_test)
    metrics_rf = evaluate_predictions(y_ho_test, pred_rf_test, hours=hours_test)
    metrics_lstm = evaluate_predictions(y_ho_test, pred_lstm_test, hours=hours_test)
    metrics_fast = evaluate_predictions(y_ho_test, pred_fast_blend, hours=hours_test)
    metrics_accurate = evaluate_predictions(
        y_ho_test, pred_accurate_blend, hours=hours_test
    )

    logger.info("=" * 60)
    logger.info("최종 Hold-out Test 성능 평가 요약")
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
    # 결과 영속화 (Checkpoints & metrics.json)
    # -----------------------------------------------------------------
    base_out_dir = output_dir or (ARTIFACTS_DIR / "baseline")
    ckpt_dir = base_out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 모델 가중치 저장
    lgb_path = ckpt_dir / f"lgbm_{horizon}.txt"
    final_lgb.booster_.save_model(str(lgb_path))

    rf_path = ckpt_dir / f"rf_{horizon}.joblib"
    joblib.dump(final_rf, rf_path)

    lstm_path = ckpt_dir / f"lstm_{horizon}.pt"
    if hasattr(final_lstm, "model_") and final_lstm.model_ is not None:
        torch.save(final_lstm.model_.state_dict(), lstm_path)

    scaler_path = ckpt_dir / f"scaler_{horizon}.joblib"
    joblib.dump(
        {
            "mean_x": mean_x_ho,
            "std_x": std_x_ho,
            "mean_y": mean_y_ho,
            "std_y": std_y_ho,
        },
        scaler_path,
    )

    # 메트릭 종합 딕셔너리
    all_metrics = {
        "horizon": horizon,
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

    metrics_file = base_out_dir / f"metrics_{horizon}.json"
    metrics_json = json.dumps(all_metrics, indent=2, ensure_ascii=False)
    metrics_file.write_text(metrics_json, encoding="utf-8")

    if output_dir is None:
        latest_metrics_file = base_out_dir / "metrics.json"
        latest_metrics_file.write_text(metrics_json, encoding="utf-8")
        logger.info(f"최신 평가 메트릭 저장 완료: {latest_metrics_file}")

    logger.info(f"체크포인트 저장 완료: {ckpt_dir}")
    logger.info(f"평가 메트릭 저장 완료: {metrics_file}")

    return {
        "metrics": all_metrics,
        "fast_track_weights": (w_rf, w_lgb),
        "accurate_track_weights": (w_rf_acc, w_lgb_acc, w_lstm_acc),
        "selected_features": feature_cols,
        "models": {
            "lgbm": final_lgb,
            "rf": final_rf,
            "lstm": final_lstm,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서 정의 (train 및 predict 모드 분리)"""
    default_config_path = PROJECT_ROOT / "configs" / "config.yaml"
    parser = argparse.ArgumentParser(
        description="부산 지하철 혼잡도 예측 E2E 파이프라인 (모델 학습 및 예측 서빙)"
    )
    subparsers = parser.add_subparsers(
        dest="mode", help="실행 모드 ('train': 모델 학습, 'predict': 예측 서빙)"
    )

    # 1. train 모드 (모델 학습, 앙상블 블렌딩 및 검증/평가)
    train_parser = subparsers.add_parser("train", help="모델 학습 및 평가 파이프라인")
    train_parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path),
        help="설정 YAML 파일 경로",
    )
    train_parser.add_argument(
        "--horizon",
        type=str,
        default=None,
        choices=["short", "long"],
        help="예측 호라이즌 ('short': 기상 포함 25종, 'long': Weather-Free 17종, 미지정 시 config.yaml 적용)",
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

    # 2. predict 모드 (학습된 모델 기반 예측 서빙 템플릿)
    predict_parser = subparsers.add_parser(
        "predict", help="학습된 모델 기반 예측 서빙 (템플릿)"
    )
    predict_parser.add_argument(
        "--horizon",
        type=str,
        default="short",
        choices=["short", "long"],
        help="예측 호라이즌 ('short': 기상 포함 25종, 'long': Weather-Free 17종)",
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
        return run_pipeline(mode="predict", horizon=args.horizon)

    # train 모드
    config = load_config(Path(args.config))
    pipeline_cfg = config.get("pipeline", {})
    effective_horizon = args.horizon or pipeline_cfg.get("horizon", "short")
    effective_step = args.step or pipeline_cfg.get("step", "all")
    out_dir = Path(args.output_dir) if args.output_dir else None

    return run_pipeline(
        mode="train",
        horizon=effective_horizon,
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
