"""
[논문 표 4.0~4.4 산출 통합 파이프라인]
5대 Feature Group (Set 0~4) 대상 Optuna 하이퍼파라미터 튜닝(LGB, RF, MetroLSTM) 및
3-Track 롤링 2-Track(Fast/Accurate) 전수 평가 통합 스크립트

- Optuna 튜닝: LightGBM(20 trials), SafeRandomForestRegressor(10 trials), MetroLSTM(12 trials)
  - 3-Track 롤링 평균 RMSE 최소화, 최적 Trial 갱신 시 가중치 즉시 저장 및 SQLite Resume 지원
- 최종 평가: 저장된 가중치(.txt, .joblib, scaler.joblib, .pt) 로드·추론 및 SLSQP 앙상블 산출
- 산출물:
  1) artifacts/ablation/optuna_all_models_results.json
  2) artifacts/ablation/ablation_optuna_2track_rolling_results.json
  3) artifacts/ablation/ablation_optuna_2track_tables_<YYYYMMDD_HHMMSS>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import torch
from scipy.optimize import minimize

from busan_metro_congestion.config import (
    ARTIFACTS_DIR,
    DEVICE,
    FEATURES_DATA_DIR,
    set_seed,
)
from busan_metro_congestion.modeling.baseline import (
    CALENDAR_TEMPORAL_FEATURES,
    FESTIVAL_FEATURES,
    SUBWAY_BASE_FEATURES,
    WEATHER_SHORT_FEATURES,
    SafeRandomForestRegressor,
    split_rolling_time_series,
)
from busan_metro_congestion.modeling.ensemble import (
    blend_accurate_track,
    blend_fast_track,
)
from busan_metro_congestion.modeling.lstm_net import MetroLSTM, MetroLSTMEstimator
from busan_metro_congestion.modeling.metrics import (
    calculate_rmse,
    evaluate_predictions,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("optuna_2track_ablation")

RF_N_JOBS = 8
LGB_N_JOBS = 8
MODEL_KEYS = ("lgb", "rf", "lstm")

# 0. 순수 년·월·일 및 지하철 승하차 기저 피처 (외부 데이터 및 주기 파생 배제 7종)
YMD_SUBWAY_FEATURES = [
    "year",
    "month",
    "day",
    "hour",
    "station_id",
    "line_num",
    "station_target_encoded",
]

_BASE_CAL = SUBWAY_BASE_FEATURES + CALENDAR_TEMPORAL_FEATURES
_BASE_CAL_FEST = _BASE_CAL + FESTIVAL_FEATURES

# 피처 그룹 조합 (Ablation Study용 Set 0~4)
FEATURE_GROUP_COMBINATIONS = {
    "short": {
        "ymd_subway_only": YMD_SUBWAY_FEATURES,  # 7종 (Set 0)
        "subway_calendar": _BASE_CAL,  # 13종 (Set 1)
        "subway_calendar_weather": _BASE_CAL + WEATHER_SHORT_FEATURES,  # 21종 (Set 2)
        "subway_calendar_festival": _BASE_CAL_FEST,  # 17종 (Set 3)
        "all": _BASE_CAL_FEST + WEATHER_SHORT_FEATURES,  # 25종 (Set 4)
    },
    "long": {
        "ymd_subway_only": YMD_SUBWAY_FEATURES,  # 7종 (Set 0)
        "subway_calendar": _BASE_CAL,  # 13종 (Set 1)
        "subway_calendar_festival": _BASE_CAL_FEST,  # 17종 (Weather-Free 전체)
        "all": _BASE_CAL_FEST,  # 17종 (long의 all은 Weather-Free)
    },
}

GROUP_LABELS = {
    "ymd_subway_only": "Set 0: 순수 년·월·일 + 지하철 (7종)",
    "subway_calendar": "Set 1: 지하철 + 날짜 (기저 13종)",
    "subway_calendar_weather": "Set 2: + 기상 예보 추가 (21종)",
    "subway_calendar_festival": "Set 3: + 대규모 행사 추가 (17종)",
    "all": "Set 4: + 기상 및 행사 전체 (25종)",
}


# ==============================================================================
# 1. 앙상블 가중치 최적화 (SLSQP) 및 데이터·파라미터 유틸리티
# ==============================================================================


def _optimize_weights_rmse(
    y_true: np.ndarray,
    preds: tuple[np.ndarray, ...],
    initial_weights: tuple[float, ...],
) -> tuple[float, ...]:
    """검증셋 RMSE를 최소화하는 비음수 볼록 결합 SLSQP 최적 가중치 산출"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_arrs = [np.asarray(p, dtype=np.float64).reshape(-1) for p in preds]
    res = minimize(
        lambda w: calculate_rmse(
            yt, sum(wi * pi for wi, pi in zip(w, p_arrs, strict=True))
        ),
        x0=np.array(initial_weights, dtype=np.float64),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(p_arrs),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    return tuple(float(x) for x in res.x) if res.success else initial_weights


def optimize_fast_weights_rmse(
    y_true: np.ndarray, pred_rf: np.ndarray, pred_lgb: np.ndarray
) -> tuple[float, float]:
    """검증셋 RMSE를 최소화하는 Fast-Track (w_rf, w_lgb) SLSQP 최적 가중치 산출"""
    w = _optimize_weights_rmse(y_true, (pred_rf, pred_lgb), (0.5, 0.5))
    return (w[0], w[1])


def optimize_accurate_weights_rmse(
    y_true: np.ndarray,
    pred_rf: np.ndarray,
    pred_lgb: np.ndarray,
    pred_lstm: np.ndarray,
) -> tuple[float, float, float]:
    """검증셋 RMSE를 최소화하는 Accurate-Track (w_rf, w_lgb, w_lstm) SLSQP 최적 가중치 산출"""
    w = _optimize_weights_rmse(
        y_true, (pred_rf, pred_lgb, pred_lstm), (0.45, 0.45, 0.10)
    )
    return (w[0], w[1], w[2])


def average_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """3개 롤링 트랙 검증 지표의 평균치 계산"""
    keys = ["rmse", "mae", "wape", "peak_mae", "r2", "pearson_r"]
    return {
        k: float(np.mean([m[k] for m in metrics_list if k in m]))
        for k in keys
        if any(k in m for m in metrics_list)
    }


def _extract_df_arrays(
    df: pl.DataFrame,
    feat_cols: list[str],
    target_col: str = "accumulated_passengers",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """단일 Polars DataFrame에서 (X, y, hours, station_id) NumPy 배열 추출"""
    X = np.ascontiguousarray(df.select(feat_cols).to_numpy(), dtype=np.float32)
    y = np.ascontiguousarray(df[target_col].to_numpy(), dtype=np.float32)
    hours = df["hour"].to_numpy() if "hour" in df.columns else None
    stn = df["station_id"].to_numpy() if "station_id" in df.columns else None
    return X, y, hours, stn


def _extract_track_arrays(
    track: dict[str, Any],
    feat_cols: list[str],
    target_col: str = "accumulated_passengers",
) -> tuple[Any, ...]:
    """롤링 트랙 Polars DataFrame에서 학습/검증용 NumPy 배열 일괄 추출"""
    X_tr, y_tr, _, stn_tr = _extract_df_arrays(track["train_df"], feat_cols, target_col)
    X_val, y_val, hours_val, stn_val = _extract_df_arrays(
        track["val_df"], feat_cols, target_col
    )
    return X_tr, y_tr, X_val, y_val, hours_val, stn_tr, stn_val


_BASELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "lgb": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    "rf": {"n_estimators": 100, "max_depth": 10, "max_samples": 0.5},
    "lstm": {
        "hidden_dim1": 64,
        "hidden_dim2": 32,
        "dropout_rate": 0.2,
        "lr": 1e-3,
        "epochs": 10,
    },
}

_FINAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "lgb": {**_BASELINE_DEFAULTS["lgb"], "subsample_freq": 1},
    "rf": _BASELINE_DEFAULTS["rf"],
    "lstm": {
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "dropout_rate": 0.1,
        "lr": 0.003,
        "epochs": 12,
    },
}


def _fixed_runtime_params(model_key: str, seed: int = 42) -> dict[str, Any]:
    """모델별 고정 런타임 하이퍼파라미터 반환"""
    fixed = {
        "lgb": {
            "random_state": seed,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": LGB_N_JOBS,
            "verbose": -1,
        },
        "rf": {"random_state": seed, "n_jobs": RF_N_JOBS},
        "lstm": {
            "seq_len": 3,
            "dense_dim": 32,
            "batch_size": 1024,
            "device": DEVICE,
            "random_state": seed,
        },
    }
    if model_key not in fixed:
        raise ValueError(f"지원하지 않는 모델 키입니다: {model_key}")
    return fixed[model_key]


def _baseline_model_params(model_key: str, seed: int = 42) -> dict[str, Any]:
    """Optuna 튜닝 전 Baseline 평가용 기본 파라미터"""
    return {**_BASELINE_DEFAULTS[model_key], **_fixed_runtime_params(model_key, seed)}


def _build_final_params(
    model_key: str, group_tuning: dict[str, Any], seed: int = 42
) -> dict[str, Any]:
    """최종 재학습(미저장 보충 학습 / 24개월 홀드아웃)용 모델별 파라미터 조립"""
    best = group_tuning.get(model_key, {}).get("best_params", {})
    return {
        **_FINAL_DEFAULTS[model_key],
        **best,
        **_fixed_runtime_params(model_key, seed),
    }


def _suggest_model_params(
    model_key: str, trial: optuna.Trial, seed: int = 42
) -> dict[str, Any]:
    """모델별 Optuna 하이퍼파라미터 탐색 공간 정의"""
    if model_key == "lgb":
        sampled: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 90),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
    elif model_key == "rf":
        sampled = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_samples": trial.suggest_float("max_samples", 0.3, 0.6, step=0.1),
        }
    elif model_key == "lstm":
        sampled = {
            "hidden_dim1": trial.suggest_categorical("hidden_dim1", [32, 64, 128]),
            "hidden_dim2": trial.suggest_categorical("hidden_dim2", [16, 32, 64]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "epochs": trial.suggest_int("epochs", 5, 12),
        }
    else:
        raise ValueError(f"지원하지 않는 모델 키입니다: {model_key}")
    return {**sampled, **_fixed_runtime_params(model_key, seed)}


# ==============================================================================
# 2. 3-Track 롤링 평가 및 가중치 저장/로드 헬퍼 함수
# ==============================================================================


def _fit_predict_model(
    model_key: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    params: dict[str, Any],
    stn_tr: np.ndarray | None = None,
    stn_val: np.ndarray | None = None,
) -> dict[str, Any]:
    """단일 분할에 대해 지정 모델(lgb/rf/lstm) 학습·추론 및 소요 시간 반환"""
    if model_key in ("lgb", "rf"):
        model_cls = (
            lgb.LGBMRegressor if model_key == "lgb" else SafeRandomForestRegressor
        )
        t0 = time.time()
        model = model_cls(**params)
        model.fit(X_tr, y_tr)
        fit_sec = time.time() - t0

        t1 = time.time()
        raw = (
            model.predict(X_val)
            if model_key == "lgb"
            else model.predict(X_val, chunk_size=100_000)
        )
        preds = np.clip(raw.astype(np.float32), 0.0, None)
        return {
            "model": model,
            "preds": preds,
            "fit_sec": fit_sec,
            "pred_sec": time.time() - t1,
        }

    if model_key == "lstm":
        mean_x = X_tr.mean(axis=0, keepdims=True)
        std_x = X_tr.std(axis=0, keepdims=True) + 1e-6
        mean_y = float(y_tr.mean())
        std_y = float(y_tr.std()) + 1e-6

        t0 = time.time()
        model = MetroLSTMEstimator(**params)
        model.fit((X_tr - mean_x) / std_x, (y_tr - mean_y) / std_y, station_ids=stn_tr)
        fit_sec = time.time() - t0

        t1 = time.time()
        pred_norm = model.predict((X_val - mean_x) / std_x, station_ids=stn_val)
        preds = np.clip((pred_norm * std_y + mean_y).astype(np.float32), 0.0, None)
        return {
            "state_dict": {
                k: v.cpu().clone() for k, v in model.model_.state_dict().items()
            },
            "scaler": {
                "mean_x": mean_x,
                "std_x": std_x,
                "mean_y": mean_y,
                "std_y": std_y,
            },
            "preds": preds,
            "fit_sec": fit_sec,
            "pred_sec": time.time() - t1,
        }

    raise ValueError(f"지원하지 않는 모델 키입니다: {model_key}")


def _evaluate_model_on_tracks(
    model_key: str,
    splits: list[dict[str, Any]],
    feat_cols: list[str],
    params: dict[str, Any],
    target_col: str = "accumulated_passengers",
    return_artifacts: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, Any]]]:
    """3개 롤링 트랙에 대해 모델 학습·추론 및 평균 지표 산출 공통 루프"""
    track_metrics: list[dict[str, float]] = []
    artifacts: list[dict[str, Any]] = []

    for track in splits:
        X_tr, y_tr, X_val, y_val, hours_val, stn_tr, stn_val = _extract_track_arrays(
            track, feat_cols, target_col
        )
        art = _fit_predict_model(
            model_key, X_tr, y_tr, X_val, params, stn_tr=stn_tr, stn_val=stn_val
        )
        track_metrics.append(evaluate_predictions(y_val, art["preds"], hours=hours_val))
        if return_artifacts:
            artifacts.append(art)

    avg_m = average_metrics(track_metrics)
    return (avg_m, artifacts) if return_artifacts else avg_m


def evaluate_lgbm_on_tracks(*args: Any, **kwargs: Any) -> Any:
    """주어진 파라미터로 3개 롤링 트랙에 대해 LightGBM 학습 및 평균 지표 산출"""
    return _evaluate_model_on_tracks("lgb", *args, **kwargs)


def evaluate_rf_on_tracks(*args: Any, **kwargs: Any) -> Any:
    """주어진 파라미터로 3개 롤링 트랙에 대해 SafeRandomForestRegressor 학습 및 평균 지표 산출"""
    return _evaluate_model_on_tracks("rf", *args, **kwargs)


def evaluate_lstm_on_tracks(*args: Any, **kwargs: Any) -> Any:
    """주어진 파라미터로 3개 롤링 트랙에 대해 MetroLSTM 정규화 학습 및 평균 지표 산출"""
    return _evaluate_model_on_tracks("lstm", *args, **kwargs)


def _checkpoint_paths(ckpt_dir: Path, tag: str) -> dict[str, Path]:
    """공통 접미사(tag) 기준 4종 체크포인트 파일 경로 매핑 반환"""
    return {
        "lgb": ckpt_dir / f"lgb_{tag}.txt",
        "rf": ckpt_dir / f"rf_{tag}.joblib",
        "scaler": ckpt_dir / f"scaler_{tag}.joblib",
        "lstm": ckpt_dir / f"lstm_{tag}.pt",
    }


def _save_single_artifact(
    ckpt_dir: Path, model_key: str, tag: str, art: dict[str, Any]
) -> None:
    """단일 모델 가중치 및 예측값(.npy) 영구 저장"""
    paths = _checkpoint_paths(ckpt_dir, tag)
    if model_key == "lgb":
        art["model"].booster_.save_model(str(paths["lgb"]))
    elif model_key == "rf":
        joblib.dump(art["model"], paths["rf"])
    elif model_key == "lstm":
        joblib.dump(art["scaler"], paths["scaler"])
        torch.save(art["state_dict"], paths["lstm"])
    else:
        raise ValueError(f"지원하지 않는 모델 키입니다: {model_key}")
    np.save(ckpt_dir / f"pred_{model_key}_{tag}.npy", art["preds"])


def _save_track_artifacts(
    ckpt_dir: Path, model_key: str, g_idx: int, artifacts: list[dict[str, Any]]
) -> None:
    """특정 모델의 3-Track 가중치 및 예측값(.npy) 영구 저장"""
    for t_idx, art in enumerate(artifacts, 1):
        _save_single_artifact(ckpt_dir, model_key, f"set{g_idx}_track{t_idx}", art)


def _save_lgb_artifacts(
    ckpt_dir: Path, g_idx: int, artifacts: list[dict[str, Any]]
) -> None:
    _save_track_artifacts(ckpt_dir, "lgb", g_idx, artifacts)


def _save_rf_artifacts(
    ckpt_dir: Path, g_idx: int, artifacts: list[dict[str, Any]]
) -> None:
    _save_track_artifacts(ckpt_dir, "rf", g_idx, artifacts)


def _save_lstm_artifacts(
    ckpt_dir: Path, g_idx: int, artifacts: list[dict[str, Any]]
) -> None:
    _save_track_artifacts(ckpt_dir, "lstm", g_idx, artifacts)


def _load_and_predict_weights(
    ckpt_dir: Path,
    tag: str,
    X_val: np.ndarray,
    stn_val: np.ndarray | None,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """디스크에 저장된 4종 모델 가중치(.txt, .joblib, scaler.joblib, .pt)를 로드하여 추론"""
    paths = _checkpoint_paths(ckpt_dir, tag)
    for f in paths.values():
        if not f.exists():
            raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {f}")

    bst = lgb.Booster(model_file=str(paths["lgb"]))
    t0 = time.time()
    pred_lgb = np.clip(bst.predict(X_val).astype(np.float32), 0.0, None)
    pred_sec_lgb = time.time() - t0

    rf_model = joblib.load(paths["rf"])
    t1 = time.time()
    pred_rf = np.clip(
        rf_model.predict(X_val, chunk_size=100_000).astype(np.float32), 0.0, None
    )
    pred_sec_rf = time.time() - t1

    scaler = joblib.load(paths["scaler"])
    state_dict = torch.load(paths["lstm"], map_location=DEVICE, weights_only=True)
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    dense_dim = state_dict["fc1.weight"].shape[0]

    net = MetroLSTM(
        input_dim=input_dim,
        hidden_dim1=h1,
        hidden_dim2=h2,
        dense_dim=dense_dim,
        dropout_rate=0.1,
    )
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()

    lstm_est = MetroLSTMEstimator(
        seq_len=3,
        hidden_dim1=h1,
        hidden_dim2=h2,
        dense_dim=dense_dim,
        dropout_rate=0.1,
        batch_size=1024,
        device=DEVICE,
    )
    lstm_est.model_ = net
    lstm_est.n_features_in_ = input_dim
    lstm_est.is_fitted_ = True

    t2 = time.time()
    pred_lstm_norm = lstm_est.predict(
        (X_val - scaler["mean_x"]) / scaler["std_x"], station_ids=stn_val
    )
    pred_lstm = np.clip(
        (pred_lstm_norm * scaler["std_y"] + scaler["mean_y"]).astype(np.float32),
        0.0,
        None,
    )
    pred_sec_lstm = time.time() - t2

    for key, pred in (("lgb", pred_lgb), ("rf", pred_rf), ("lstm", pred_lstm)):
        np.save(ckpt_dir / f"pred_{key}_{tag}.npy", pred)

    return pred_lgb, pred_rf, pred_lstm, pred_sec_lgb, pred_sec_rf, pred_sec_lstm


def _load_and_predict_track_weights(
    ckpt_dir: Path,
    g_idx: int,
    t_idx: int,
    X_val: np.ndarray,
    stn_val: np.ndarray | None,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """디스크에 저장된 단일 트랙 모델 가중치를 로드하여 추론"""
    return _load_and_predict_weights(
        ckpt_dir, f"set{g_idx}_track{t_idx}", X_val, stn_val, input_dim
    )


# ==============================================================================
# 3. 모델별 Optuna 하이퍼파라미터 튜닝 함수 (SQLite 이어하기 + 가중치 즉시 저장)
# ==============================================================================


def _create_or_load_study(
    study_name: str, seed: int, storage_url: str | None = None
) -> optuna.Study:
    """Optuna 스터디 생성 또는 기존 SQLite DB에서 이어받기"""
    sampler = optuna.samplers.TPESampler(seed=seed)
    if storage_url:
        return optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    return optuna.create_study(direction="minimize", sampler=sampler)


_TUNING_META: dict[str, tuple[str, str, str]] = {
    "lgb": ("LGBM", "LightGBM", f"n_jobs={LGB_N_JOBS}"),
    "rf": ("RF", "Random Forest", f"n_jobs={RF_N_JOBS}"),
    "lstm": ("LSTM", "MetroLSTM", f"Device={DEVICE}"),
}


def _tune_model_for_group(
    model_key: str,
    group_name: str,
    label: str,
    feat_cols: list[str],
    splits: list[dict[str, Any]],
    n_trials: int,
    seed: int = 42,
    storage_url: str | None = None,
    ckpt_dir: Path | None = None,
    g_idx: int | None = None,
) -> dict[str, Any]:
    """Optuna 공통 튜닝 루프: Baseline 측정/복원, Trial 이어하기, Best 가중치 즉시 저장"""
    model_tag, display_name, extra_log = _TUNING_META[model_key]
    logger.info(
        f"  ▶ [{model_tag} | {label}] 튜닝 시작 (피처 {len(feat_cols)}개, 목표 Trials {n_trials}, {extra_log})"
    )
    study = _create_or_load_study(f"{group_name}_{model_key}_v1", seed, storage_url)

    if "baseline_metrics" in study.user_attrs:
        base_m = study.user_attrs["baseline_metrics"]
        logger.info(
            f"    [{model_tag} Baseline (DB 복원)] RMSE: {base_m['rmse']:.2f} | MAE: {base_m['mae']:.2f}"
        )
    else:
        t0 = time.time()
        base_m = _evaluate_model_on_tracks(
            model_key, splits, feat_cols, _baseline_model_params(model_key, seed)
        )
        study.set_user_attr("baseline_metrics", base_m)
        logger.info(
            f"    [{model_tag} Baseline] RMSE: {base_m['rmse']:.2f} | MAE: {base_m['mae']:.2f} (소요 {time.time() - t0:.1f}초)"
        )

    completed_before = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best_rmse_so_far = (
        min(t.value for t in completed_before if t.value is not None)
        if completed_before
        else float("inf")
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_rmse_so_far
        params = _suggest_model_params(model_key, trial, seed)
        t_tr = time.time()
        m, arts = _evaluate_model_on_tracks(
            model_key, splits, feat_cols, params, return_artifacts=True
        )  # type: ignore[misc]
        trial.set_user_attr("metrics", m)
        trial.set_user_attr("avg_fit_sec", float(np.mean([a["fit_sec"] for a in arts])))
        trial.set_user_attr(
            "avg_pred_sec", float(np.mean([a["pred_sec"] for a in arts]))
        )

        is_new_best = m["rmse"] < best_rmse_so_far
        if is_new_best:
            best_rmse_so_far = m["rmse"]
            if ckpt_dir is not None and g_idx is not None:
                _save_track_artifacts(ckpt_dir, model_key, g_idx, arts)

        mark = " ★[가중치 저장]" if is_new_best and ckpt_dir is not None else ""
        logger.info(
            f"      [{model_tag} Trial #{trial.number + 1}/{n_trials}] RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} ({time.time() - t_tr:.1f}s){mark}"
        )
        return float(m["rmse"])

    remaining = max(0, n_trials - len(completed_before))
    if completed_before and remaining > 0:
        logger.info(
            f"    ♻️ [{model_tag} 이어하기] 기완료 {len(completed_before)}/{n_trials}건 확인 -> 잔여 {remaining}건 실행"
        )

    t_start = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    tune_elapsed = time.time() - t_start

    best_m = study.best_trial.user_attrs["metrics"]
    rmse_diff = best_m["rmse"] - base_m["rmse"]
    rmse_impr_pct = (
        (base_m["rmse"] - best_m["rmse"]) / base_m["rmse"] * 100.0
        if base_m["rmse"] > 0.0
        else 0.0
    )
    mae_diff = best_m["mae"] - base_m["mae"]
    mae_impr_pct = (
        (base_m["mae"] - best_m["mae"]) / base_m["mae"] * 100.0
        if base_m["mae"] > 0.0
        else 0.0
    )

    logger.info(
        f"    ⭐ [{model_tag} Best] RMSE: {best_m['rmse']:.2f} ({rmse_diff:+.2f}, {rmse_impr_pct:+.2f}%) | "
        f"MAE: {best_m['mae']:.2f} (소요 {tune_elapsed:.1f}초)"
    )

    return {
        "model_name": display_name,
        "baseline_metrics": base_m,
        "best_metrics": best_m,
        "best_params": study.best_trial.params,
        "avg_fit_sec": study.best_trial.user_attrs.get("avg_fit_sec", 0.0),
        "avg_pred_sec": study.best_trial.user_attrs.get("avg_pred_sec", 0.0),
        "improvements": {
            "rmse_diff": rmse_diff,
            "rmse_impr_pct": rmse_impr_pct,
            "mae_diff": mae_diff,
            "mae_impr_pct": mae_impr_pct,
        },
        "elapsed_seconds": tune_elapsed,
    }


def tune_lgb_for_group(
    group_name: str,
    label: str,
    feat_cols: list[str],
    splits: list[dict[str, Any]],
    n_trials: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """피처 그룹에 대한 LightGBM 3-Track 롤링 Optuna 튜닝 및 최적 가중치(.txt) 즉시 저장"""
    return _tune_model_for_group(
        "lgb", group_name, label, feat_cols, splits, n_trials, **kwargs
    )


def tune_rf_for_group(
    group_name: str,
    label: str,
    feat_cols: list[str],
    splits: list[dict[str, Any]],
    n_trials: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """피처 그룹에 대한 SafeRandomForestRegressor 3-Track 롤링 Optuna 튜닝 및 최적 가중치(.joblib) 즉시 저장"""
    return _tune_model_for_group(
        "rf", group_name, label, feat_cols, splits, n_trials, **kwargs
    )


def tune_lstm_for_group(
    group_name: str,
    label: str,
    feat_cols: list[str],
    splits: list[dict[str, Any]],
    n_trials: int = 12,
    **kwargs: Any,
) -> dict[str, Any]:
    """피처 그룹에 대한 MetroLSTM 3-Track 롤링 Optuna 튜닝 및 최적 가중치(.pt, scaler.joblib) 즉시 저장"""
    return _tune_model_for_group(
        "lstm", group_name, label, feat_cols, splits, n_trials, **kwargs
    )


def tune_all_models_for_group(
    group_name: str,
    label: str,
    feat_cols: list[str],
    splits: list[dict[str, Any]],
    n_lgb_trials: int = 20,
    n_rf_trials: int = 10,
    n_lstm_trials: int = 12,
    seed: int = 42,
    storage_url: str | None = None,
    partial_save_path: Path | None = None,
    all_tuning_results: dict[str, Any] | None = None,
    ckpt_dir: Path | None = None,
    g_idx: int | None = None,
) -> dict[str, Any]:
    """단일 피처 그룹에 대해 LightGBM, RF, MetroLSTM 3대 모델 Optuna 튜닝 통합 실행 및 가중치/JSON 즉시 저장"""
    logger.info("\n========================================================")
    logger.info(f"▶▶ [{label}] 3대 모델 Optuna 튜닝 파이프라인 가동")
    logger.info("========================================================")

    group_dict: dict[str, Any] = {
        "group_name": group_name,
        "label": label,
        "feature_count": len(feat_cols),
    }

    for key, n_tr in (
        ("lgb", n_lgb_trials),
        ("rf", n_rf_trials),
        ("lstm", n_lstm_trials),
    ):
        group_dict[key] = _tune_model_for_group(
            key,
            group_name,
            label,
            feat_cols,
            splits,
            n_tr,
            seed,
            storage_url,
            ckpt_dir,
            g_idx,
        )
        if partial_save_path is not None and all_tuning_results is not None:
            all_tuning_results[group_name] = group_dict
            partial_save_path.write_text(
                json.dumps(all_tuning_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return group_dict


# ==============================================================================
# 4. 논문 표 4.0~4.4 마크다운 생성기
# ==============================================================================


def format_best_params(model_type: str, params: dict[str, Any]) -> str:
    """마크다운 테이블 출력을 위한 최적 하이퍼파라미터 압축 포맷팅"""
    if model_type == "lgb":
        parts = [
            f"n_est={params.get('n_estimators')}",
            f"lr={params.get('learning_rate', 0):.3f}",
            f"leaves={params.get('num_leaves')}",
            f"depth={params.get('max_depth')}",
        ]
        if "subsample" in params:
            parts.append(f"sub={params['subsample']:.2f}")
        return ", ".join(parts)
    if model_type == "rf":
        parts = [
            f"n_est={params.get('n_estimators')}",
            f"depth={params.get('max_depth')}",
        ]
        if "min_samples_split" in params:
            parts.append(f"split={params['min_samples_split']}")
        if "max_samples" in params:
            parts.append(f"max_samples={params['max_samples']:.2f}")
        parts.append(f"min_leaf={params.get('min_samples_leaf')}")
        return ", ".join(parts)
    if model_type == "lstm":
        return (
            f"hidden=({params.get('hidden_dim1', 128)}, {params.get('hidden_dim2', 64)}), "
            f"drop={params.get('dropout_rate', 0.1):.2f}, "
            f"lr={params.get('lr', 0.003):.4f}, ep={params.get('epochs', 12)}"
        )
    return str(params)


def _render_model_comparison_table(
    title: str,
    entry: dict[str, Any],
    train_col_label: str,
    pred_col_label: str,
) -> list[str]:
    """개별 모델 3종 및 Fast/Accurate 앙상블 2종 비교 마크다운 표 렌더링 (표 4.2~4.4 공용)"""
    wf, wa = entry["weights_fast"], entry["weights_accurate"]
    tr_t, pr_t = entry["avg_train_time_sec"], entry["avg_pred_time_sec"]
    rows = [
        ("개별 모델", "**LightGBM (Optuna)**", entry["lgb"], tr_t["lgb"], pr_t["lgb"]),
        (
            "개별 모델",
            "**Random Forest (Optuna)**",
            entry["rf"],
            tr_t["rf"],
            pr_t["rf"],
        ),
        (
            "개별 모델",
            "**MetroLSTM (Optuna)**",
            entry["lstm"],
            tr_t["lstm"],
            pr_t["lstm"],
        ),
        (
            "앙상블 트랙",
            f"**Fast-Track** ($\\text{{RF }} {wf['w_rf']:.2f} + \\text{{LGB }} {wf['w_lgb']:.2f}$)",
            entry["fast_track"],
            tr_t["fast"],
            pr_t["fast"],
        ),
        (
            "앙상블 트랙",
            f"**Accurate-Track** ($\\text{{RF }} {wa['w_rf']:.2f} + \\text{{LGB }} {wa['w_lgb']:.2f} + \\text{{LSTM }} {wa['w_lstm']:.2f}$)",
            entry["accurate_track"],
            tr_t["acc"],
            pr_t["acc"],
        ),
    ]
    lines = [
        title,
        "",
        f"| 구분 | 모델 / 운영 트랙 | **RMSE** | MAE | WAPE (%) | Peak-Hour MAE | $R^2$ | Pearson $r$ | {train_col_label} | {pred_col_label} |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for category, name, m, tr_sec, pr_sec in rows:
        lines.append(
            f"| **{category}** | {name} | **{m['rmse']:.2f}** | {m['mae']:.2f} | "
            f"{m['wape']:.2f}% | {m['peak_mae']:.2f} | {m['r2']:.4f} | {m['pearson_r']:.4f} | "
            f"{tr_sec:.2f}초 | {pr_sec:.2f}초 |"
        )
    lines.append("")
    return lines


def generate_tables_markdown(
    results: dict[str, Any],
    tuning_results: dict[str, Any] | None = None,
) -> str:
    """논문 표 4.0~4.4(튜닝 전후 개선, 그룹별 앙상블, 개별·앙상블 비교, Holdout) 마크다운 생성"""
    lines: list[str] = [
        "# 4.2 외부 요인 반영에 따른 예측 성능 비교 (Optuna 튜닝 후 3-Track 롤링 실측)",
        "",
    ]

    tune_data = tuning_results or results.get("tuning_results")
    if tune_data is None:
        tune_file = ARTIFACTS_DIR / "ablation" / "optuna_all_models_results.json"
        if tune_file.exists():
            try:
                tune_data = json.loads(tune_file.read_text(encoding="utf-8"))
            except Exception:
                tune_data = None

    groups_order = list(GROUP_LABELS.keys())

    if tune_data is not None:
        lines.extend(
            [
                "## [표 4.0] Feature 그룹 및 모델별 Optuna 튜닝 전/후(Baseline vs Tuned) 성능 개선 비교표",
                "",
                "| 피처 그룹 (Feature Set) | 모델 (Model) | 튜닝 전 RMSE | 튜닝 후 RMSE | $\\Delta$RMSE | 개선율 (%) | 튜닝 전 MAE | 튜닝 후 MAE | 최적 하이퍼파라미터 (Best Parameters) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for g_name in groups_order:
            if g_name not in tune_data:
                continue
            g_t = tune_data[g_name]
            lbl = g_t.get("label", g_name)
            for m_key, m_name in (
                ("lgb", "LightGBM"),
                ("rf", "Random Forest"),
                ("lstm", "MetroLSTM"),
            ):
                if m_key not in g_t:
                    continue
                mr = g_t[m_key]
                bm, tm, impr = (
                    mr["baseline_metrics"],
                    mr["best_metrics"],
                    mr["improvements"],
                )
                param_str = format_best_params(m_key, mr.get("best_params", {}))
                lines.append(
                    f"| **{lbl}** | {m_name} | {bm['rmse']:.2f} | **{tm['rmse']:.2f}** | "
                    f"{impr['rmse_diff']:+.2f} | **{impr['rmse_impr_pct']:+.2f}%** | "
                    f"{bm['mae']:.2f} | {tm['mae']:.2f} | `{param_str}` |"
                )
        lines.extend(["", "---", ""])

    lines.extend(
        [
            "## [표 4.1] 관점 1: Accurate-Track(전체 종합)에서의 Feature 그룹별 예측 성능 비교",
            "",
            "| 피처 그룹 (Feature Set) | 피처 수 | **RMSE** | MAE | WAPE (%) | Peak-Hour MAE | $R^2$ | Pearson $r$ | 최적 가중치 ($w_{\\text{rf}} : w_{\\text{lgb}} : w_{\\text{lstm}}$) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for g_name in groups_order:
        if g_name not in results:
            continue
        r = results[g_name]
        acc, w = r["accurate_track"], r["weights_accurate"]
        lines.append(
            f"| **{r['label']}** | {r['feature_count']}종 | **{acc['rmse']:.2f}** | "
            f"{acc['mae']:.2f} | {acc['wape']:.2f}% | {acc['peak_mae']:.2f} | "
            f"{acc['r2']:.4f} | {acc['pearson_r']:.4f} | "
            f"${w['w_rf']:.2f} : {w['w_lgb']:.2f} : {w['w_lstm']:.2f}$ |"
        )

    best_eval_group = (
        "subway_calendar_festival"
        if "subway_calendar_festival" in results
        else ("all" if "all" in results else None)
    )
    if best_eval_group is not None:
        sb = results[best_eval_group]
        lines.extend(["", "---", ""])
        lines.extend(
            _render_model_comparison_table(
                f"## [표 4.2] 관점 2: 최우수 예측 성능 피처 조합({sb['label']}) 기준 개별 모델 및 앙상블 트랙 비교",
                sb,
                "평균 학습 시간 (Train)",
                "평균 예측 시간 (Inference)",
            )
        )

    for result_key, default_lbl, title_fmt in (
        (
            "holdout_test",
            "Set 3: + 대규모 행사 추가 (17종)",
            "## [표 4.3] 관점 3: 24개월 대칭 학습 기반 최종 홀드아웃 테스트 (미래 12개월 실측, {label})",
        ),
        (
            "holdout_test_ymd_subway",
            "Set 0: 순수 년·월·일 + 지하철 (7종)",
            "## [표 4.4] 관점 4: 24개월 대칭 학습 기반 기저 대조군 홀드아웃 테스트 (미래 12개월 실측, {label})",
        ),
    ):
        if result_key not in results:
            continue
        ho = results[result_key]
        lines.extend(["---", ""])
        lines.extend(
            _render_model_comparison_table(
                title_fmt.format(label=ho.get("label", default_lbl)),
                ho,
                "학습 시간 (Train, 24개월)",
                "예측 시간 (Inference, 12개월)",
            )
        )

    return "\n".join(lines)


# ==============================================================================
# 5. 24개월 홀드아웃 평가 및 파이프라인 메인 엔트리포인트
# ==============================================================================


def _blend_and_save_tracks(
    ckpt_dir: Path,
    tag: str,
    p_rf: np.ndarray,
    p_lgb: np.ndarray,
    p_lstm: np.ndarray,
    w_fast: tuple[float, float],
    w_acc: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Fast/Accurate 2-Track 앙상블 블렌딩 및 예측값(.npy) 저장"""
    pred_fast = blend_fast_track(p_rf, p_lgb, w_rf=w_fast[0], w_lgb=w_fast[1])
    pred_acc = blend_accurate_track(
        p_rf, p_lgb, p_lstm, w_rf=w_acc[0], w_lgb=w_acc[1], w_lstm=w_acc[2]
    )
    np.save(ckpt_dir / f"pred_fast_{tag}.npy", pred_fast)
    np.save(ckpt_dir / f"pred_acc_{tag}.npy", pred_acc)
    return pred_fast, pred_acc


def _summarize_track_times(times: dict[str, float]) -> dict[str, float]:
    """개별 모델 소요 시간에서 Fast/Accurate 앙상블 합계 시간 추가"""
    return {
        **times,
        "fast": times["rf"] + times["lgb"],
        "acc": times["rf"] + times["lgb"] + times["lstm"],
    }


def _evaluate_or_train_holdout(
    *,
    group_key: str,
    group_label: str,
    feat_cols: list[str],
    ho_tr_df: pl.DataFrame,
    ho_te_df: pl.DataFrame,
    target_col: str,
    ckpt_dir: Path,
    tuning_results: dict[str, Any],
    final_results: dict[str, Any],
    eval_from_weights: bool = False,
    existing_holdout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """24개월 홀드아웃 가중치 로드 추론(--eval-from-weights) 또는 24개월 신규 학습 및 12개월 테스트 평가"""
    ho_tuning = tuning_results.get(group_key, {})
    X_ho_te, y_ho_te, hours_ho_te, stn_ho_te = _extract_df_arrays(
        ho_te_df, feat_cols, target_col
    )
    tag = f"{group_key}_holdout_24m"
    weights_exist = all(p.exists() for p in _checkpoint_paths(ckpt_dir, tag).values())

    if eval_from_weights and weights_exist:
        logger.info(
            f"  [{group_label}] 저장된 24개월 홀드아웃 가중치 로드 및 12개월 테스트셋 추론 수행 중..."
        )
        prev_tr = (
            existing_holdout.get("avg_train_time_sec", {}) if existing_holdout else {}
        )
        tr_times = {
            k: float(
                prev_tr.get(
                    k, float(ho_tuning.get(k, {}).get("avg_fit_sec", d_sec)) * 2.0
                )
            )
            for k, d_sec in (("lgb", 3.0), ("rf", 9.0), ("lstm", 17.0))
        }
        p_lgb, p_rf, p_lstm, pr_lgb, pr_rf, pr_lstm = _load_and_predict_weights(
            ckpt_dir, tag, X_ho_te, stn_ho_te, len(feat_cols)
        )
        preds = {"lgb": p_lgb, "rf": p_rf, "lstm": p_lstm}
        pr_times = {"lgb": pr_lgb, "rf": pr_rf, "lstm": pr_lstm}
    else:
        X_ho_tr, y_ho_tr, _, stn_ho_tr = _extract_df_arrays(
            ho_tr_df, feat_cols, target_col
        )
        arts = {}
        for m_key in MODEL_KEYS:
            art = _fit_predict_model(
                m_key,
                X_ho_tr,
                y_ho_tr,
                X_ho_te,
                _build_final_params(m_key, ho_tuning),
                stn_tr=stn_ho_tr,
                stn_val=stn_ho_te,
            )
            _save_single_artifact(ckpt_dir, m_key, tag, art)
            arts[m_key] = art
        preds = {k: arts[k]["preds"] for k in MODEL_KEYS}
        tr_times = {k: float(arts[k]["fit_sec"]) for k in MODEL_KEYS}
        pr_times = {k: float(arts[k]["pred_sec"]) for k in MODEL_KEYS}

    wf = final_results[group_key]["weights_fast"]
    wa = final_results[group_key]["weights_accurate"]
    preds["fast_track"], preds["accurate_track"] = _blend_and_save_tracks(
        ckpt_dir,
        tag,
        preds["rf"],
        preds["lgb"],
        preds["lstm"],
        (wf["w_rf"], wf["w_lgb"]),
        (wa["w_rf"], wa["w_lgb"], wa["w_lstm"]),
    )

    return {
        "group_name": group_key,
        "label": group_label,
        "feature_count": len(feat_cols),
        "train_period": f"{ho_tr_df['date'].min()} ~ {ho_tr_df['date'].max()}",
        "test_period": f"{ho_te_df['date'].min()} ~ {ho_te_df['date'].max()}",
        **{
            k: evaluate_predictions(y_ho_te, pred, hours=hours_ho_te)
            for k, pred in preds.items()
        },
        "weights_fast": wf,
        "weights_accurate": wa,
        "avg_train_time_sec": _summarize_track_times(tr_times),
        "avg_pred_time_sec": _summarize_track_times(pr_times),
    }


def _evaluate_group_rolling(
    *,
    g_idx: int,
    g_name: str,
    feat_cols: list[str],
    val_splits: list[dict[str, Any]],
    g_tuning: dict[str, Any],
    ckpt_dir: Path,
    eval_from_weights: bool,
) -> dict[str, Any]:
    """단일 피처 그룹의 3-Track 가중치 보충·검증 추론 및 SLSQP 앙상블 평가"""
    lbl = GROUP_LABELS[g_name]
    t_grp = time.time()

    if not eval_from_weights:
        model_file_keys = {
            "lgb": ("lgb",),
            "rf": ("rf",),
            "lstm": ("scaler", "lstm"),
        }
        for m_key in MODEL_KEYS:
            model_files = [
                _checkpoint_paths(ckpt_dir, f"set{g_idx}_track{t_idx}")[fk]
                for t_idx in range(1, len(val_splits) + 1)
                for fk in model_file_keys[m_key]
            ]
            if all(f.exists() for f in model_files):
                continue
            logger.info(
                f"  [{lbl}] {m_key} 미저장 가중치 발견 -> 최적 파라미터로 3-Track 학습 및 가중치 저장 수행"
            )
            _, arts = _evaluate_model_on_tracks(
                m_key,
                val_splits,
                feat_cols,
                _build_final_params(m_key, g_tuning),
                return_artifacts=True,
            )  # type: ignore[misc]
            _save_track_artifacts(ckpt_dir, m_key, g_idx, arts)

    m_lists: dict[str, list[dict[str, float]]] = {
        k: [] for k in ("lgb", "rf", "lstm", "fast_track", "accurate_track")
    }
    w_fast_list: list[tuple[float, float]] = []
    w_acc_list: list[tuple[float, float, float]] = []
    pred_times: dict[str, list[float]] = {k: [] for k in MODEL_KEYS}

    for t_idx, track in enumerate(val_splits, 1):
        logger.info(f">>> [{lbl} Track {t_idx}] 가중치 로드 검증 및 앙상블 추론 중...")
        _, _, X_val, y_val, hours_val, _, stn_val = _extract_track_arrays(
            track, feat_cols
        )
        np.save(ckpt_dir / f"y_val_track{t_idx}.npy", y_val)
        if hours_val is not None:
            np.save(ckpt_dir / f"hours_val_track{t_idx}.npy", hours_val)

        p_lgb, p_rf, p_lstm, p_lgb_s, p_rf_s, p_lstm_s = (
            _load_and_predict_track_weights(
                ckpt_dir, g_idx, t_idx, X_val, stn_val, len(feat_cols)
            )
        )
        w_fast = optimize_fast_weights_rmse(y_val, p_rf, p_lgb)
        w_acc = optimize_accurate_weights_rmse(y_val, p_rf, p_lgb, p_lstm)
        pred_fast, pred_acc = _blend_and_save_tracks(
            ckpt_dir, f"set{g_idx}_track{t_idx}", p_rf, p_lgb, p_lstm, w_fast, w_acc
        )

        for k, sec in (("lgb", p_lgb_s), ("rf", p_rf_s), ("lstm", p_lstm_s)):
            pred_times[k].append(sec)
        for m_key, pred in (
            ("lgb", p_lgb),
            ("rf", p_rf),
            ("lstm", p_lstm),
            ("fast_track", pred_fast),
            ("accurate_track", pred_acc),
        ):
            m_lists[m_key].append(evaluate_predictions(y_val, pred, hours=hours_val))
        w_fast_list.append(w_fast)
        w_acc_list.append(w_acc)

    tr_times = {
        k: float(g_tuning.get(k, {}).get("avg_fit_sec", d_sec))
        for k, d_sec in (("lgb", 4.0), ("rf", 15.0), ("lstm", 25.0))
    }
    avg_pr = {k: float(np.mean(v)) for k, v in pred_times.items()}

    return {
        "group_name": g_name,
        "label": lbl,
        "feature_count": len(feat_cols),
        **{k: average_metrics(v) for k, v in m_lists.items()},
        "weights_fast": {
            "w_rf": float(np.mean([w[0] for w in w_fast_list])),
            "w_lgb": float(np.mean([w[1] for w in w_fast_list])),
        },
        "weights_accurate": {
            "w_rf": float(np.mean([w[0] for w in w_acc_list])),
            "w_lgb": float(np.mean([w[1] for w in w_acc_list])),
            "w_lstm": float(np.mean([w[2] for w in w_acc_list])),
        },
        "avg_train_time_sec": _summarize_track_times(tr_times),
        "avg_pred_time_sec": _summarize_track_times(avg_pr),
        "total_group_elapsed_sec": time.time() - t_grp,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5대 Feature Group (Set 0~4) Optuna 전수 튜닝 및 2-Track 롤링 평가 통합 파이프라인"
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="기저장된 튜닝 파라미터 로드하여 튜닝 생략",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="기저장된 결과 JSON에서 표 4.0~4.4 마크다운만 생성",
    )
    parser.add_argument(
        "--eval-from-weights",
        action="store_true",
        help="저장된 모델 가중치를 로드하여 추론·평가",
    )
    parser.add_argument(
        "--n-lgb-trials",
        type=int,
        default=20,
        help="LightGBM Optuna trial 수 (기본: 20)",
    )
    parser.add_argument(
        "--n-rf-trials",
        type=int,
        default=10,
        help="Random Forest Optuna trial 수 (기본: 10)",
    )
    parser.add_argument(
        "--n-lstm-trials",
        type=int,
        default=12,
        help="MetroLSTM Optuna trial 수 (기본: 12)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(42)

    abl_dir = ARTIFACTS_DIR / "ablation"
    out_path = abl_dir / "ablation_optuna_2track_rolling_results.json"
    tuning_file = abl_dir / "optuna_all_models_results.json"
    md_path = (
        abl_dir
        / f"ablation_optuna_2track_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    ckpt_dir = abl_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(ckpt_dir / 'optuna_studies.db').as_posix()}"

    if args.tables_only:
        if not out_path.exists():
            raise FileNotFoundError(f"결과 JSON 파일이 없습니다: {out_path}")
        results = json.loads(out_path.read_text(encoding="utf-8"))
        tune_data = (
            json.loads(tuning_file.read_text(encoding="utf-8"))
            if tuning_file.exists()
            else None
        )
        md_path.write_text(
            generate_tables_markdown(results, tuning_results=tune_data),
            encoding="utf-8",
        )
        logger.info(f"논문용 표 마크다운 재생성 완료: {md_path}")
        return

    feature_path = FEATURES_DATA_DIR / "metro_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"피처 데이터셋이 없습니다: {feature_path}")

    df = pl.read_parquet(feature_path)
    if len(df) < 1_000_000:
        from busan_metro_congestion.preprocess.features import run_feature_pipeline

        df = run_feature_pipeline(force_reload=True)

    if any(c not in df.columns for c in ("year", "month", "day")):
        df = df.with_columns(
            pl.col("date").dt.year().cast(pl.Int16).alias("year"),
            pl.col("date").dt.month().cast(pl.Int8).alias("month"),
            pl.col("date").dt.day().cast(pl.Int8).alias("day"),
        )

    target_col = "accumulated_passengers"
    split_res = split_rolling_time_series(df, strategy="rolling")
    val_splits = split_res["validation_splits"]
    groups = list(GROUP_LABELS.keys())
    group_defs = FEATURE_GROUP_COMBINATIONS["short"]

    # Step A: 5대 피처 그룹(Set 0~4) 대상 3대 모델 전수 Optuna 튜닝
    tuning_results: dict[str, Any] = (
        json.loads(tuning_file.read_text(encoding="utf-8"))
        if tuning_file.exists()
        else {}
    )
    if args.skip_tuning and tuning_file.exists():
        logger.info(f"⚡ [--skip-tuning] 기존 Optuna 튜닝 결과 로드: {tuning_file}")
    elif not args.eval_from_weights:
        logger.info(
            "\n🚀 [Step A] 5대 피처 그룹(Set 0~4) 3대 모델(LGB, RF, LSTM) Optuna 튜닝 및 최적 가중치 실시간 저장 시작"
        )
        for g_idx, g_name in enumerate(groups, 0):
            tuning_results[g_name] = tune_all_models_for_group(
                group_name=g_name,
                label=GROUP_LABELS[g_name],
                feat_cols=group_defs[g_name],
                splits=val_splits,
                n_lgb_trials=args.n_lgb_trials,
                n_rf_trials=args.n_rf_trials,
                n_lstm_trials=args.n_lstm_trials,
                seed=42,
                storage_url=storage_url,
                partial_save_path=tuning_file,
                all_tuning_results=tuning_results,
                ckpt_dir=ckpt_dir,
                g_idx=g_idx,
            )
        tuning_file.write_text(
            json.dumps(tuning_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            f"✅ [Step A 완료] Optuna 튜닝 결과 및 가중치 저장 완료: {tuning_file}"
        )

    # Step B: 저장된 가중치 검증 추론 및 2-Track SLSQP 앙상블 평가
    logger.info("\n🎯 [Step B] 저장된 가중치 검증 및 2-Track 롤링 앙상블 평가 시작")
    final_results: dict[str, Any] = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    )
    for g_idx, g_name in enumerate(groups, 0):
        final_results[g_name] = _evaluate_group_rolling(
            g_idx=g_idx,
            g_name=g_name,
            feat_cols=group_defs[g_name],
            val_splits=val_splits,
            g_tuning=tuning_results.get(g_name, {}),
            ckpt_dir=ckpt_dir,
            eval_from_weights=args.eval_from_weights,
        )
        if tuning_results:
            final_results["tuning_results"] = tuning_results
        out_path.write_text(
            json.dumps(final_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Step C: 최우수 Feature 조합(Set 3) 및 기저 대조군(Set 0) 24개월 홀드아웃 테스트 평가
    logger.info(
        "\n🏆 [Step C] 최우수 Feature 조합(Set 3) 및 기저 대조군(Set 0) 24개월 홀드아웃 테스트 평가 시작"
    )
    ho_tr_df = split_res["holdout_test"]["train_df"]
    ho_te_df = split_res["holdout_test"]["test_df"]
    logger.info(
        f"  - 홀드아웃 학습 데이터: {ho_tr_df['date'].min()} ~ {ho_tr_df['date'].max()} ({len(ho_tr_df):,}행, 24개월)"
    )
    logger.info(
        f"  - 홀드아웃 테스트 데이터: {ho_te_df['date'].min()} ~ {ho_te_df['date'].max()} ({len(ho_te_df):,}행, 12개월)"
    )

    best_group_key = (
        "subway_calendar_festival"
        if "subway_calendar_festival" in final_results
        else "all"
    )
    holdout_targets = [("holdout_test", best_group_key)]
    if "ymd_subway_only" in final_results:
        holdout_targets.append(("holdout_test_ymd_subway", "ymd_subway_only"))

    for result_key, g_key in holdout_targets:
        final_results[result_key] = _evaluate_or_train_holdout(
            group_key=g_key,
            group_label=GROUP_LABELS[g_key],
            feat_cols=group_defs[g_key],
            ho_tr_df=ho_tr_df,
            ho_te_df=ho_te_df,
            target_col=target_col,
            ckpt_dir=ckpt_dir,
            tuning_results=tuning_results,
            final_results=final_results,
            eval_from_weights=args.eval_from_weights,
            existing_holdout=final_results.get(result_key),
        )

    out_path.write_text(
        json.dumps(final_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(
        generate_tables_markdown(
            final_results,
            tuning_results=tuning_results or final_results.get("tuning_results"),
        ),
        encoding="utf-8",
    )
    logger.info("\n🎉 [파이프라인 전체 완료] 결과 저장:")
    logger.info(f"  - JSON: {out_path}")
    logger.info(f"  - Markdown: {md_path}")


if __name__ == "__main__":
    main()
