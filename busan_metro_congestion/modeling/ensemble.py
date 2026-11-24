"""
Two-Track 가중치 블렌딩 앙상블 모듈 (Fast Track 단독 & Accurate Track 종합)

운영 목적별 앙상블 체계:
1. Fast Track 단독 앙상블:
   - Random Forest + LightGBM (트리 기반 머신러닝 내부 앙상블)
   - 딥러닝 추론 지연 없이 밀리초(ms) 단위로 초고속 결과 산출 및 서빙 제출
2. Accurate Track 종합 앙상블:
   - Fast Track (RF, LightGBM) + Accurate Track (PyTorch LSTM)
   - 트리의 구조적 피처 해석력과 LSTM의 시계열 연속 모멘텀을 결합한 고정밀 결과 산출
3. 안전한 제약 최적화 (Non-negative Constrained Optimization / NNLS):
   - Scipy SLSQP 기반으로 검증셋 MAE를 최소화하되, 음수 방지(w_i >= 0) 및 볼록 결합(sum w_i = 1.0) 제약 적용
   - 모델 간 다중공선성으로 인한 오차 폭주 및 음수 통행량 외삽을 원천 차단
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import minimize

from busan_metro_congestion.modeling.metrics import calculate_mae

logger = logging.getLogger("modeling_ensemble")


# =====================================================================
# 1. Track 1: Fast Track 단독 앙상블 (Random Forest + LightGBM)
# =====================================================================


def blend_fast_track(
    pred_rf: np.ndarray,
    pred_lgb: np.ndarray,
    w_rf: float = 0.3,
    w_lgb: float = 0.7,
) -> np.ndarray:
    """
    Fast Track 내부 모델(Random Forest, LightGBM)의 예측값을 가중치 합산하여 초고속 예측 배열을 산출합니다.

    인자:
        pred_rf: Random Forest 예측 1D 배열
        pred_lgb: LightGBM 예측 1D 배열
        w_rf: Random Forest 가중치 (기본값: 0.3)
        w_lgb: LightGBM 가중치 (기본값: 0.7)

    반환값:
        Fast Track 블렌딩 1D NumPy 예측 배열
    """
    weight_sum = w_rf + w_lgb
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Fast Track 가중치의 합은 1.0이어야 합니다. (현재 합: {weight_sum:.4f})"
        )

    p_rf = np.asarray(pred_rf, dtype=np.float64).reshape(-1)
    p_lgb = np.asarray(pred_lgb, dtype=np.float64).reshape(-1)

    if len(p_rf) != len(p_lgb):
        raise ValueError(
            f"모델 예측 배열의 길이가 일치하지 않습니다: RF={len(p_rf)}, LGBM={len(p_lgb)}"
        )

    if len(p_rf) == 0:
        return np.array([], dtype=np.float64)

    blended = (w_rf * p_rf) + (w_lgb * p_lgb)
    return blended


def optimize_fast_track_weights(
    y_true: Any,
    pred_rf: np.ndarray,
    pred_lgb: np.ndarray,
    initial_weights: tuple[float, float] = (0.3, 0.7),
) -> tuple[float, float]:
    """
    검증 데이터셋의 MAE를 최소화하는 Fast Track 최적 가중치 (w_rf, w_lgb)를 탐색합니다.
    (w_i >= 0, w_rf + w_lgb = 1.0 볼록 제약 최적화)
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_rf = np.asarray(pred_rf, dtype=np.float64).reshape(-1)
    p_lgb = np.asarray(pred_lgb, dtype=np.float64).reshape(-1)

    def objective(weights: np.ndarray) -> float:
        w1, w2 = weights
        pred = (w1 * p_rf) + (w2 * p_lgb)
        return calculate_mae(yt, pred)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    res = minimize(
        objective,
        x0=np.array(initial_weights),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res.success:
        best_w = res.x
        logger.info(
            f"[Fast Track] 최적 가중치 탐색 성공: RF={best_w[0]:.4f}, LGBM={best_w[1]:.4f} "
            f"(최소 MAE: {res.fun:.2f})"
        )
        return (float(best_w[0]), float(best_w[1]))

    logger.warning("[Fast Track] 가중치 최적화 수렴 실패: 초기 가중치를 반환합니다.")
    return initial_weights


# =====================================================================
# 2. Track 2: Accurate Track 종합 앙상블 (RF + LightGBM + PyTorch LSTM)
# =====================================================================


def blend_accurate_track(
    pred_rf: np.ndarray,
    pred_lgb: np.ndarray,
    pred_lstm: np.ndarray,
    w_rf: float = 0.2,
    w_lgb: float = 0.4,
    w_lstm: float = 0.4,
) -> np.ndarray:
    """
    Fast Track(RF, LightGBM)과 Accurate Track(PyTorch LSTM) 3개 모델의 예측값을 가중치 합산하여
    최고 정밀도의 최종 통행량/혼잡도 예측 배열을 산출합니다.

    인자:
        pred_rf: Random Forest 예측 1D 배열
        pred_lgb: LightGBM 예측 1D 배열
        pred_lstm: PyTorch LSTM 예측 1D 배열
        w_rf, w_lgb, w_lstm: 각 모델 가중치 (합계 1.0)

    반환값:
        Accurate Track 블렌딩 1D NumPy 예측 배열
    """
    weight_sum = w_rf + w_lgb + w_lstm
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Accurate Track 가중치의 합은 1.0이어야 합니다. (현재 합: {weight_sum:.4f})"
        )

    p_rf = np.asarray(pred_rf, dtype=np.float64).reshape(-1)
    p_lgb = np.asarray(pred_lgb, dtype=np.float64).reshape(-1)
    p_lstm = np.asarray(pred_lstm, dtype=np.float64).reshape(-1)

    n = len(p_rf)
    if not (len(p_lgb) == n and len(p_lstm) == n):
        raise ValueError(
            f"각 모델 예측 배열의 길이가 일치하지 않습니다: "
            f"RF={len(p_rf)}, LGBM={len(p_lgb)}, LSTM={len(p_lstm)}"
        )

    if n == 0:
        return np.array([], dtype=np.float64)

    blended = (w_rf * p_rf) + (w_lgb * p_lgb) + (w_lstm * p_lstm)
    return blended


def optimize_accurate_track_weights(
    y_true: Any,
    pred_rf: np.ndarray,
    pred_lgb: np.ndarray,
    pred_lstm: np.ndarray,
    initial_weights: tuple[float, float, float] = (0.2, 0.4, 0.4),
) -> tuple[float, float, float]:
    """
    검증 데이터셋의 MAE를 최소화하는 Accurate Track 최적 가중치 (w_rf, w_lgb, w_lstm)를 탐색합니다.
    (w_i >= 0, sum w_i = 1.0 볼록 제약 최적화)
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p_rf = np.asarray(pred_rf, dtype=np.float64).reshape(-1)
    p_lgb = np.asarray(pred_lgb, dtype=np.float64).reshape(-1)
    p_lstm = np.asarray(pred_lstm, dtype=np.float64).reshape(-1)

    def objective(weights: np.ndarray) -> float:
        w1, w2, w3 = weights
        pred = (w1 * p_rf) + (w2 * p_lgb) + (w3 * p_lstm)
        return calculate_mae(yt, pred)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    res = minimize(
        objective,
        x0=np.array(initial_weights),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res.success:
        best_w = res.x
        logger.info(
            f"[Accurate Track] 최적 가중치 탐색 성공: RF={best_w[0]:.4f}, LGBM={best_w[1]:.4f}, LSTM={best_w[2]:.4f} "
            f"(최소 MAE: {res.fun:.2f})"
        )
        return (float(best_w[0]), float(best_w[1]), float(best_w[2]))

    logger.warning(
        "[Accurate Track] 가중치 최적화 수렴 실패: 초기 가중치를 반환합니다."
    )
    return initial_weights
