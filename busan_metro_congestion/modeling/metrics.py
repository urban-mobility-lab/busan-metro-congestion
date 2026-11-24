"""
논문 평가 지표 모듈 (MAE, RMSE, R², Pearson r, WAPE, Peak MAE)

혼잡도 및 승하차 통행량 예측 모델 성능 검증을 위한 지표 계산 모듈입니다.
NumPy 배열 및 Polars Series를 모두 지원하며 분모 0 및 결측에 대해 안전하게 방어합니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _to_numpy(arr: Any) -> np.ndarray:
    """Polars Series, 리스트 또는 배열을 1차원 float64 ndarray로 변환합니다."""
    if isinstance(arr, pl.Series):
        arr = arr.to_numpy()
    arr = np.asarray(arr, dtype=np.float64)
    return arr.reshape(-1)


def calculate_mae(y_true: Any, y_pred: Any) -> float:
    """평균 절대 오차 (Mean Absolute Error)를 계산합니다."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def calculate_rmse(y_true: Any, y_pred: Any) -> float:
    """평균 제곱근 오차 (Root Mean Squared Error)를 계산합니다."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def calculate_r2(y_true: Any, y_pred: Any) -> float:
    """결정 계수 (R-squared)를 계산합니다."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_pearson_r(y_true: Any, y_pred: Any) -> float:
    """피어슨 상관계수 (Pearson Correlation Coefficient)를 계산합니다."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) < 2:
        return 0.0
    std_yt = np.std(yt, ddof=1)
    std_yp = np.std(yp, ddof=1)
    if std_yt == 0.0 or std_yp == 0.0:
        return 0.0
    cov = np.cov(yt, yp)[0, 1]
    r = float(cov / (std_yt * std_yp))
    return float(np.clip(r, -1.0, 1.0))


DEFAULT_PEAK_HOURS = {7, 8, 9, 18, 19, 20}


def calculate_wape(y_true: Any, y_pred: Any) -> float:
    """
    가중 절대 백분율 오차 (Weighted Absolute Percentage Error, %)를 계산합니다.
    WAPE = (sum |y_true - y_pred| / sum |y_true|) * 100
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        return 0.0
    return float((np.sum(np.abs(yt - yp)) / denom) * 100.0)


def calculate_peak_hour_mae(
    y_true: Any,
    y_pred: Any,
    hours: Any | None = None,
    peak_hours: set[int] | list[int] | None = None,
) -> float:
    """
    출퇴근 첨두시간대(기본: 07~09시, 18~20시)에 한정한 MAE를 계산합니다.
    hours가 제공되지 않거나 첨두시간 데이터가 없을 경우 전체 MAE를 반환합니다.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) == 0:
        return 0.0
    if hours is None:
        return calculate_mae(yt, yp)

    h_arr = _to_numpy(hours).astype(int)
    if len(h_arr) != len(yt):
        return calculate_mae(yt, yp)

    target_peaks = set(peak_hours) if peak_hours is not None else DEFAULT_PEAK_HOURS
    mask = np.isin(h_arr, list(target_peaks))
    if not np.any(mask):
        return calculate_mae(yt, yp)

    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    hours: Any | None = None,
) -> dict[str, float]:
    """
    전체 평가 지표(MAE, RMSE, R², Pearson r, WAPE, Peak MAE)를 일괄 산출합니다.

    반환값:
        dict: {
            "mae": float,
            "rmse": float,
            "r2": float,
            "pearson_r": float,
            "wape": float,
            "peak_mae": float,
        }
    """
    return {
        "mae": round(calculate_mae(y_true, y_pred), 4),
        "rmse": round(calculate_rmse(y_true, y_pred), 4),
        "r2": round(calculate_r2(y_true, y_pred), 4),
        "pearson_r": round(calculate_pearson_r(y_true, y_pred), 4),
        "wape": round(calculate_wape(y_true, y_pred), 4),
        "peak_mae": round(calculate_peak_hour_mae(y_true, y_pred, hours=hours), 4),
    }
