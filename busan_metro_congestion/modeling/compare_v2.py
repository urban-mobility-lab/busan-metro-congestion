"""
운영 피처 v2 모델 비교 평가 (v1.0-baseline 대비)

- 운영 피처: 지하철 승하차 + 달력 특일·공휴일 + 주기성 (기상·행사 Feature 제외)
- 분할: v1.0-baseline 홀드아웃과 같은 학습 2023-08-01 ~ 2025-07-31, 평가 2025-08-01 ~ 2026-07-31
- 역 타깃 인코딩은 학습 구간으로만 다시 계산하고(학습 행 OOF), 평가 구간 실적은 쓰지 않습니다.
- 예측 시점은 D-1일 17시이며, 과거 실적은 D-2일까지만 입력에 씁니다.

비교 모델:
- v1_fast: v1.0-baseline Set 1 Fast-Track. 파라미터·가중치는 artifacts/ablation 결과에서 읽습니다.
- v2_fast: 같은 피처, v2 튜닝 파라미터·가중치
- v2_history_fast: + D-2일까지의 주간·최근 실적 Feature
- v2_seq2seq: 168시간(D-8 ~ D-2) 인코더 + 24시간 동시 출력 Seq2Seq LSTM, 역 임베딩
- v2_blend: v2_fast + v2_seq2seq (최종 v2)

v2 파라미터와 가중치는 홀드아웃 이전 검증 월(2024-07 ~ 2025-07 홀수 달)에서 정한 값입니다.

산출물: artifacts/ablation/v2/v2_comparison_results.json, v2_comparison_tables.md
실행:
    uv run python -m busan_metro_congestion.modeling.compare_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

from busan_metro_congestion.config import (
    ARTIFACTS_DIR,
    DEVICE,
    FEATURES_DATA_DIR,
    set_seed,
)
from busan_metro_congestion.modeling.baseline import (
    CALENDAR_TEMPORAL_FEATURES,
    SUBWAY_BASE_FEATURES,
    SafeRandomForestRegressor,
)
from busan_metro_congestion.modeling.day_ahead_lstm import (
    build_sequence_panel,
    fit_predict_seq2seq,
)
from busan_metro_congestion.modeling.ensemble import blend_fast_track
from busan_metro_congestion.modeling.metrics import (
    calculate_rmse,
    evaluate_predictions,
)
from busan_metro_congestion.preprocess.history_features import (
    HISTORY_FEATURES,
    add_history_features,
    drop_history_warmup,
)
from busan_metro_congestion.preprocess.subway import StationTargetEncoder

logger = logging.getLogger("compare_v2")

TARGET_COL = "accumulated_passengers"
KEYS = ["station_id", "date", "hour"]
SEED = 42
N_JOBS = 8

TRAIN_START = date(2023, 8, 1)
TRAIN_END = date(2025, 7, 31)
TEST_START = date(2025, 8, 1)
TEST_END = date(2026, 7, 31)

OPERATIONAL_FEATURES = SUBWAY_BASE_FEATURES + CALENDAR_TEMPORAL_FEATURES
HISTORY_OPERATIONAL_FEATURES = OPERATIONAL_FEATURES + HISTORY_FEATURES

V1_GROUP = "subway_calendar"
V1_DIR = ARTIFACTS_DIR / "ablation"
OUTPUT_DIR = ARTIFACTS_DIR / "ablation" / "v2"
FEATURE_PATH = FEATURES_DATA_DIR / "metro_features.parquet"

V2_TREE_PARAMS: dict[str, dict[str, Any]] = {
    "lgb": {
        "n_estimators": 120,
        "learning_rate": 0.09224717157613284,
        "num_leaves": 80,
        "max_depth": 10,
        "min_child_samples": 16,
        "subsample": 0.7196425388297226,
        "colsample_bytree": 0.7408242887997989,
        "reg_alpha": 0.976720358115074,
        "reg_lambda": 0.02617647983747478,
    },
    "rf": {
        "n_estimators": 75,
        "max_depth": 16,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "max_samples": 0.3,
    },
}
V2_SEQ2SEQ_PARAMS: dict[str, Any] = {
    "hidden_dim": 64,
    "dropout": 0.28184968246925673,
    "lr": 0.002354770601447807,
    "weight_decay": 0.0002463768595899747,
    "epochs": 30,
}
V2_SEQ2SEQ_SEEDS = (42, 43, 44)
V2_WEIGHTS: dict[str, dict[str, float]] = {
    "v2_fast": {"w_rf": 0.9032114849882324, "w_lgb": 0.09678851501176755},
    "v2_history_fast": {"w_rf": 0.3286743807545822, "w_lgb": 0.6713256192454178},
    "v2_blend": {"w_seq2seq": 0.4481601443170572, "w_fast": 0.5518398556829428},
}

MODEL_LABELS = {
    "v1_fast": "v1.0-baseline Fast-Track (Set 1)",
    "v2_fast": "v2 Fast-Track (재튜닝)",
    "v2_history_fast": "v2 Fast-Track + 과거 실적",
    "v2_seq2seq": "v2 Seq2Seq LSTM",
    "v2_blend": "v2 Fast-Track + Seq2Seq (최종)",
}
REFERENCE_MODEL = "v1_fast"
METRIC_COLUMNS = [
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("r2", "R²"),
    ("pearson_r", "Pearson r"),
    ("wape", "WAPE (%)"),
    ("peak_mae", "Peak MAE"),
]


def load_v1_baseline(ablation_dir: Path = V1_DIR) -> dict[str, Any]:
    """v1.0-baseline Set 1의 튜닝 파라미터와 Fast-Track 가중치를 읽습니다(파일은 수정하지 않음)."""
    tuning = json.loads(
        (ablation_dir / "optuna_all_models_results.json").read_text(encoding="utf-8")
    )[V1_GROUP]
    rolling = json.loads(
        (ablation_dir / "ablation_optuna_2track_rolling_results.json").read_text(
            encoding="utf-8"
        )
    )[V1_GROUP]
    return {
        "params": {
            "lgb": {"num_leaves": 31, **tuning["lgb"]["best_params"]},
            "rf": {"max_depth": 10, **tuning["rf"]["best_params"]},
        },
        "weights": {
            "w_rf": float(rolling["weights_fast"]["w_rf"]),
            "w_lgb": float(rolling["weights_fast"]["w_lgb"]),
        },
    }


def lgb_params(best: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    return {
        "subsample_freq": 1,
        **best,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }


def rf_params(best: dict[str, Any], seed: int = SEED) -> dict[str, Any]:
    return {**best, "random_state": seed, "n_jobs": N_JOBS}


def prepare_frame(df: pl.DataFrame) -> pl.DataFrame:
    """운영 피처만 남기고 D-2일까지의 과거 실적 Feature를 붙인 뒤 시작 구간을 제외합니다."""
    columns = list(
        dict.fromkeys([*KEYS, TARGET_COL, *OPERATIONAL_FEATURES, "datetime"])
    )
    frame = df.select([c for c in columns if c in df.columns])
    return drop_history_warmup(add_history_features(frame)).sort(KEYS)


def split_period(
    frame: pl.DataFrame,
    train_start: date = TRAIN_START,
    train_end: date = TRAIN_END,
    test_start: date = TEST_START,
    test_end: date = TEST_END,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """학습·평가 행을 나누고 역 타깃 인코딩을 학습 구간으로만 다시 계산합니다."""
    if train_end >= test_start:
        raise ValueError("학습 구간은 평가 구간보다 앞서야 합니다.")
    train = frame.filter(pl.col("date").is_between(train_start, train_end))
    test = frame.filter(pl.col("date").is_between(test_start, test_end)).sort(KEYS)
    if train.is_empty() or test.is_empty():
        raise ValueError("학습 또는 평가 구간 데이터가 없습니다.")
    encoder = StationTargetEncoder(m=10.0, n_splits=5, seed=SEED)
    encoded_train = encoder.fit_transform(
        train.drop("station_target_encoded", strict=False),
        station_col="station_id",
        target_col=TARGET_COL,
    )
    encoded_test = encoder.transform(
        test.drop("station_target_encoded", strict=False), station_col="station_id"
    )
    return encoded_train, encoded_test


def _matrix(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df.select(columns).to_numpy(), dtype=np.float32)


def fit_predict_fast(
    train: pl.DataFrame,
    test: pl.DataFrame,
    columns: list[str],
    params: dict[str, dict[str, Any]],
    weights: dict[str, float],
    seed: int = SEED,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """RF·LightGBM을 학습해 개별 예측과 Fast-Track 결합 예측, 학습·예측 시간을 반환합니다."""
    X_train, X_test = _matrix(train, columns), _matrix(test, columns)
    y_train = np.ascontiguousarray(train[TARGET_COL].to_numpy(), dtype=np.float32)
    preds: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for key, model in (
        ("rf", SafeRandomForestRegressor(**rf_params(params["rf"], seed))),
        ("lgb", lgb.LGBMRegressor(**lgb_params(params["lgb"], seed))),
    ):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        raw = (
            model.predict(X_test, chunk_size=100_000)
            if key == "rf"
            else model.predict(X_test)
        )
        preds[key] = np.clip(np.asarray(raw, dtype=np.float64), 0.0, None)
        seconds[key] = time.perf_counter() - start
    preds["fast"] = blend_fast_track(
        preds["rf"], preds["lgb"], w_rf=weights["w_rf"], w_lgb=weights["w_lgb"]
    )
    return preds, seconds


def predict_seq2seq(
    frame: pl.DataFrame,
    test_keys: pl.DataFrame,
    params: dict[str, Any] = V2_SEQ2SEQ_PARAMS,
    seeds: tuple[int, ...] = V2_SEQ2SEQ_SEEDS,
    train_start: date = TRAIN_START,
    train_end: date = TRAIN_END,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """Seq2Seq LSTM을 시드별로 학습해 평균 예측을 평가 행 순서(test_keys)에 맞춰 반환합니다."""
    panel = build_sequence_panel(frame)
    result = fit_predict_seq2seq(
        panel,
        train_start,
        train_end,
        test_keys["date"].min(),
        test_keys["date"].max(),
        params,
        seeds=seeds,
        device=device,
    )
    aligned = test_keys.select(KEYS).join(
        result.with_columns(pl.col("station_id").cast(test_keys["station_id"].dtype)),
        on=KEYS,
        how="left",
        maintain_order="left",
    )
    if aligned["prediction"].null_count():
        raise ValueError("Seq2Seq 예측이 평가 행 일부를 채우지 못했습니다.")
    return aligned["prediction"].to_numpy().astype(np.float64)


def blend_with_seq2seq(
    fast: np.ndarray, seq2seq: np.ndarray, w_seq2seq: float, w_fast: float
) -> np.ndarray:
    if abs(w_seq2seq + w_fast - 1.0) > 1e-4 or min(w_seq2seq, w_fast) < 0.0:
        raise ValueError("Seq2Seq 결합 가중치는 0 이상이고 합이 1이어야 합니다.")
    return w_seq2seq * np.asarray(seq2seq) + w_fast * np.asarray(fast)


def bootstrap_rmse_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    dates: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """RMSE(a) - RMSE(b)와 날짜 단위 부트스트랩 95% 신뢰구간을 계산합니다."""
    _, day_index = np.unique(dates, return_inverse=True)
    n_days = int(day_index.max()) + 1
    sse_a = np.bincount(day_index, (y_true - pred_a) ** 2, n_days)
    sse_b = np.bincount(day_index, (y_true - pred_b) ** 2, n_days)
    counts = np.bincount(day_index, minlength=n_days)

    sampled = np.random.default_rng(seed).integers(0, n_days, (n_boot, n_days))
    n_rows = counts[sampled].sum(axis=1)
    diffs = np.sqrt(sse_a[sampled].sum(axis=1) / n_rows) - np.sqrt(
        sse_b[sampled].sum(axis=1) / n_rows
    )
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff": calculate_rmse(y_true, pred_a) - calculate_rmse(y_true, pred_b),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def summarize(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    hours: np.ndarray,
    dates: np.ndarray,
    reference: str = REFERENCE_MODEL,
) -> dict[str, Any]:
    """모델별 6개 지표와 기준선 대비 RMSE 차이·개선율·신뢰구간을 계산합니다."""
    metrics = {
        name: evaluate_predictions(y_true, pred, hours=hours)
        for name, pred in preds.items()
    }
    base = preds[reference]
    base_rmse = calculate_rmse(y_true, base)
    versus = {}
    for name, pred in preds.items():
        if name == reference:
            continue
        diff = bootstrap_rmse_difference(y_true, pred, base, dates)
        versus[name] = {
            **diff,
            "improvement_pct": -diff["diff"] / base_rmse * 100.0 if base_rmse else 0.0,
        }
    return {"metrics": metrics, "versus_reference": versus}


def render_markdown(results: dict[str, Any]) -> str:
    """비교 결과 JSON을 Markdown 표로 바꿉니다."""
    period = results["period"]
    lines = [
        "# 운영 피처 v2 모델 비교 평가 (v1.0-baseline 대비)",
        "",
        f"- 학습 {period['train']}, 평가(홀드아웃) {period['test']}, "
        f"평가 {results['n_test_rows']:,}행",
        "- 운영 피처: 지하철 승하차 + 달력 특일·공휴일 + 주기성 (기상·행사 제외)",
        "- 예측 시점 D-1일 17시, 과거 실적은 D-2일까지만 사용",
        "- 모든 모델의 역 타깃 인코딩을 학습 구간으로만 다시 계산",
        "",
        "## 6개 지표",
        "",
        "| 모델 | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |",
        "| :--- | " + " | ".join("---:" for _ in METRIC_COLUMNS) + " |",
    ]
    for name, metric in results["metrics"].items():
        values = []
        for key, _ in METRIC_COLUMNS:
            digits = 4 if key in ("r2", "pearson_r") else 2
            values.append(f"{metric[key]:.{digits}f}")
        lines.append(f"| {MODEL_LABELS.get(name, name)} | " + " | ".join(values) + " |")

    lines += [
        "",
        f"## {MODEL_LABELS[results['reference']]} 대비 RMSE",
        "",
        "| 모델 | RMSE 차이 | 개선율 | 날짜 부트스트랩 95% 신뢰구간 |",
        "| :--- | ---: | ---: | :---: |",
    ]
    for name, diff in results["versus_reference"].items():
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {diff['diff']:+.2f} | "
            f"{diff['improvement_pct']:+.2f}% | "
            f"[{diff['ci_low']:+.2f}, {diff['ci_high']:+.2f}] |"
        )
    lines += [
        "",
        "음수 RMSE 차이(양수 개선율)는 기준선보다 정확하다는 뜻입니다. "
        "신뢰구간이 0을 포함하지 않으면 날짜 단위로 봐도 차이가 유지됩니다.",
        "",
        "## 학습·예측 시간 (초)",
        "",
        "| 모델 | 시간 |",
        "| :--- | ---: |",
    ]
    for name, seconds in results["seconds"].items():
        lines.append(f"| {name} | {seconds:.1f} |")
    return "\n".join(lines) + "\n"


def _cached(cache_dir: Path, name: str, compute: Any) -> np.ndarray:
    path = cache_dir / f"{name}.npy"
    if path.exists():
        return np.load(path)
    values = np.asarray(compute(), dtype=np.float64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return values


def run_comparison(
    df: pl.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    v1: dict[str, Any] | None = None,
    seq2seq_params: dict[str, Any] = V2_SEQ2SEQ_PARAMS,
    seq2seq_seeds: tuple[int, ...] = V2_SEQ2SEQ_SEEDS,
    period: tuple[date, date, date, date] = (
        TRAIN_START,
        TRAIN_END,
        TEST_START,
        TEST_END,
    ),
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """기준선과 v2 모델을 같은 분할로 학습·평가하고 JSON·Markdown을 저장합니다."""
    set_seed(SEED)
    v1 = v1 or load_v1_baseline()
    train_start, train_end, test_start, test_end = period
    frame = prepare_frame(df)
    train, test = split_period(frame, *period)
    cache_dir = output_dir / "checkpoints"
    preds: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}

    for name, columns, params, weights in (
        ("v1_fast", OPERATIONAL_FEATURES, v1["params"], v1["weights"]),
        ("v2_fast", OPERATIONAL_FEATURES, V2_TREE_PARAMS, V2_WEIGHTS["v2_fast"]),
        (
            "v2_history_fast",
            HISTORY_OPERATIONAL_FEATURES,
            V2_TREE_PARAMS,
            V2_WEIGHTS["v2_history_fast"],
        ),
    ):
        start = time.perf_counter()
        preds[name] = _cached(
            cache_dir,
            name,
            lambda c=columns, p=params, w=weights: fit_predict_fast(
                train, test, c, p, w
            )[0]["fast"],
        )
        seconds[name] = time.perf_counter() - start
        logger.info(
            f"[{name}] RMSE {calculate_rmse(test[TARGET_COL], preds[name]):.2f}"
        )

    start = time.perf_counter()
    preds["v2_seq2seq"] = _cached(
        cache_dir,
        "v2_seq2seq",
        lambda: predict_seq2seq(
            frame,
            test.select(KEYS),
            seq2seq_params,
            seq2seq_seeds,
            train_start,
            train_end,
            device,
        ),
    )
    seconds["v2_seq2seq"] = time.perf_counter() - start
    weights = V2_WEIGHTS["v2_blend"]
    preds["v2_blend"] = blend_with_seq2seq(
        preds["v2_fast"], preds["v2_seq2seq"], weights["w_seq2seq"], weights["w_fast"]
    )

    y_true = test[TARGET_COL].to_numpy().astype(np.float64)
    summary = summarize(y_true, preds, test["hour"].to_numpy(), test["date"].to_numpy())
    results = {
        "period": {
            "train": f"{train_start} ~ {train_end}",
            "test": f"{test_start} ~ {test_end}",
        },
        "n_test_rows": len(test),
        "reference": REFERENCE_MODEL,
        "features": {
            "operational": OPERATIONAL_FEATURES,
            "history": HISTORY_FEATURES,
        },
        "params": {
            "v1_fast": v1,
            "v2_trees": V2_TREE_PARAMS,
            "v2_seq2seq": {**seq2seq_params, "seeds": list(seq2seq_seeds)},
            "v2_weights": V2_WEIGHTS,
        },
        **summary,
        "seconds": seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v2_comparison_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "v2_comparison_tables.md").write_text(
        render_markdown(results), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="운영 피처 v2 모델 비교 평가")
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if not args.feature_path.exists():
        raise FileNotFoundError(f"피처 데이터셋이 없습니다: {args.feature_path}")
    results = run_comparison(pl.read_parquet(args.feature_path), args.output_dir)
    for name, metric in results["metrics"].items():
        logger.info(f"{MODEL_LABELS[name]}: RMSE {metric['rmse']:.2f}")


if __name__ == "__main__":
    main()
