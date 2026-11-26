"""
운영 피처 v2 모델(과거 실적 Feature, Seq2Seq LSTM, 비교 평가) 단위 테스트

합성 데이터만 사용합니다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from busan_metro_congestion.config import ARTIFACTS_DIR
from busan_metro_congestion.modeling import compare_v2
from busan_metro_congestion.modeling.compare_v2 import (
    HISTORY_OPERATIONAL_FEATURES,
    OPERATIONAL_FEATURES,
    blend_with_seq2seq,
    bootstrap_rmse_difference,
    load_v1_baseline,
    prepare_frame,
    render_markdown,
    run_comparison,
    split_period,
    summarize,
)
from busan_metro_congestion.modeling.day_ahead_lstm import (
    SequenceBatcher,
    build_sequence_panel,
    fit_predict_seq2seq,
)
from busan_metro_congestion.modeling.metrics import calculate_rmse
from busan_metro_congestion.preprocess.history_features import (
    HISTORY_FEATURES,
    add_history_features,
)

TARGET = "accumulated_passengers"
CPU = torch.device("cpu")
LSTM_PARAMS = {"hidden_dim": 16, "epochs": 2, "lr": 1e-3, "batch_size": 64}
CALENDAR = {
    "season",
    "is_weekend",
    "is_holiday_eve",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_cos",
}


def make_panel(
    start: date, days: int, stations: tuple[int, ...] = (1, 2)
) -> pl.DataFrame:
    """역×일×시간 완전 격자의 합성 운영 피처 테이블을 생성합니다."""
    rows = []
    rng = np.random.default_rng(0)
    for station in stations:
        for offset in range(days):
            current = start + timedelta(days=offset)
            for hour in range(1, 25):
                level = 100.0 * station + 20.0 * hour + 30.0 * current.weekday()
                rows.append(
                    {
                        "station_id": station,
                        "line_num": 1,
                        "datetime": datetime(current.year, current.month, current.day)
                        + timedelta(hours=hour - 1),
                        "date": current,
                        "hour": hour,
                        TARGET: level + float(rng.normal(0, 5)),
                        "station_target_encoded": 0.0,
                        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                        "weekday_sin": float(np.sin(current.weekday())),
                        "is_holiday": current.day == 1,
                    }
                )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("hour").cast(pl.Int8))
        .with_columns(*[pl.lit(0.0, dtype=pl.Float32).alias(c) for c in CALENDAR])
    )


def _actual(panel: pl.DataFrame, day: date, hour: int = 9) -> float:
    return panel.filter((pl.col("date") == day) & (pl.col("hour") == hour))[TARGET][0]


def test_history_values_match_definition():
    panel = make_panel(date(2024, 1, 1), 60, stations=(1,))
    target = date(2024, 2, 20)
    row = add_history_features(panel).filter(
        (pl.col("date") == target) & (pl.col("hour") == 9)
    )
    assert row["lag_2d"][0] == pytest.approx(_actual(panel, target - timedelta(days=2)))
    weekly = [_actual(panel, target - timedelta(days=7 * w)) for w in (1, 2, 3, 4)]
    assert row["lag_7d"][0] == pytest.approx(weekly[0])
    assert row["mean_4w_same_hour"][0] == pytest.approx(np.mean(weekly))
    last_week = [_actual(panel, target - timedelta(days=d)) for d in range(2, 9)]
    assert row["mean_7d_same_hour"][0] == pytest.approx(np.mean(last_week))


def test_history_features_ignore_issue_day_and_later_actuals():
    panel = make_panel(date(2024, 1, 1), 60)
    target = date(2024, 2, 20)
    changed = panel.with_columns(
        pl.when(pl.col("date") >= target - timedelta(days=1))
        .then(pl.col(TARGET) * 1000.0)
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    )
    select = pl.col("date") == target
    before = add_history_features(panel).filter(select).select(HISTORY_FEATURES)
    after = add_history_features(changed).filter(select).select(HISTORY_FEATURES)
    assert before.equals(after)


def test_prepare_frame_keeps_only_operational_features_without_missing_history():
    raw = make_panel(date(2024, 1, 1), 40).with_columns(
        pl.lit(1.0).alias("temperature"), pl.lit(1.0).alias("event_impact_score")
    )
    frame = prepare_frame(raw)
    assert "temperature" not in frame.columns
    assert "event_impact_score" not in frame.columns
    assert set(HISTORY_OPERATIONAL_FEATURES) <= set(frame.columns)
    assert frame.select(HISTORY_FEATURES).null_count().sum_horizontal().item() == 0
    assert frame["date"].min() == date(2024, 1, 30)


def test_split_period_encodes_with_train_targets_only():
    frame = prepare_frame(make_panel(date(2024, 1, 1), 70))
    period = (date(2024, 1, 30), date(2024, 2, 29), date(2024, 3, 1), date(2024, 3, 10))
    _, test = split_period(frame, *period)
    shifted = frame.with_columns(
        pl.when(pl.col("date") >= period[2])
        .then(pl.col(TARGET) * 100.0)
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    )
    _, shifted_test = split_period(shifted, *period)
    assert test["station_target_encoded"].to_list() == (
        shifted_test["station_target_encoded"].to_list()
    )
    assert test["date"].min() == period[2] and test["date"].max() == period[3]
    with pytest.raises(ValueError):
        split_period(frame, period[0], period[2], period[2], period[3])


def test_sequence_panel_rejects_incomplete_grid():
    frame = prepare_frame(make_panel(date(2024, 1, 1), 40))
    with pytest.raises(ValueError):
        build_sequence_panel(frame.filter(pl.col("date") != date(2024, 2, 3)))
    with pytest.raises(ValueError):
        build_sequence_panel(frame.filter(pl.col("hour") != 5))


def test_seq2seq_inputs_exclude_issue_day_and_target_day():
    frame = prepare_frame(make_panel(date(2024, 1, 1), 90))
    target = date(2024, 3, 20)
    changed = frame.with_columns(
        pl.when(pl.col("date") >= target - timedelta(days=1))
        .then(pl.col(TARGET) * 1000.0)
        .otherwise(pl.col(TARGET))
        .alias(TARGET)
    )
    outputs = []
    for data in (frame, changed):
        panel = build_sequence_panel(data)
        batcher = SequenceBatcher(panel, date(2024, 1, 30), date(2024, 3, 1), CPU)
        day = torch.tensor([panel.dates.index(target)])
        outputs.append(batcher.assemble(torch.tensor([0]), day))
    for original, modified in zip(*outputs):
        assert torch.equal(original, modified)
    assert outputs[0][0].shape[1] == 168
    assert outputs[0][1].shape[1] == 24


def test_seq2seq_returns_full_grid_and_beats_constant():
    frame = prepare_frame(make_panel(date(2024, 1, 1), 120))
    panel = build_sequence_panel(frame)
    result = fit_predict_seq2seq(
        panel,
        date(2024, 1, 30),
        date(2024, 3, 31),
        date(2024, 4, 1),
        date(2024, 4, 15),
        {**LSTM_PARAMS, "epochs": 5},
        seeds=(1, 2),
        device=CPU,
    )
    assert len(result) == 2 * 15 * 24
    assert result["prediction"].min() >= 0.0
    assert result["hour"].min() == 1 and result["hour"].max() == 24
    joined = result.join(
        frame.select(["station_id", "date", "hour", TARGET]),
        on=["station_id", "date", "hour"],
    )
    y_true = joined[TARGET].to_numpy()
    assert len(joined) == len(result)
    assert calculate_rmse(y_true, joined["prediction"].to_numpy()) < np.std(y_true)


def test_blend_with_seq2seq_rejects_invalid_weights():
    fast, seq = np.array([10.0, 20.0]), np.array([30.0, 40.0])
    assert blend_with_seq2seq(fast, seq, 0.25, 0.75).tolist() == [15.0, 25.0]
    with pytest.raises(ValueError):
        blend_with_seq2seq(fast, seq, 0.5, 0.6)
    with pytest.raises(ValueError):
        blend_with_seq2seq(fast, seq, -0.1, 1.1)


def test_bootstrap_detects_clear_improvement():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 2400)
    dates = np.repeat(np.arange(100), 24)
    good = y + rng.normal(0, 1, len(y))
    bad = y + rng.normal(0, 10, len(y))
    diff = bootstrap_rmse_difference(y, good, bad, dates, n_boot=300)
    assert diff["diff"] < 0 and diff["ci_high"] < 0


def test_summarize_reports_six_metrics_and_improvement_against_reference():
    y = np.array([100.0, 200.0, 300.0, 400.0] * 6)
    hours = np.tile(np.array([7, 8, 12, 18]), 6)
    dates = np.repeat(np.arange(6), 4)
    preds = {"v1_fast": y + 20.0, "v2_blend": y + 10.0}
    summary = summarize(y, preds, hours, dates)
    assert set(summary["metrics"]["v2_blend"]) == {
        "mae",
        "rmse",
        "r2",
        "pearson_r",
        "wape",
        "peak_mae",
    }
    versus = summary["versus_reference"]
    assert list(versus) == ["v2_blend"]
    assert versus["v2_blend"]["diff"] == pytest.approx(-10.0)
    assert versus["v2_blend"]["improvement_pct"] == pytest.approx(50.0)


def test_load_v1_baseline_reads_set1_operational_results():
    v1 = load_v1_baseline()
    assert v1["weights"]["w_rf"] + v1["weights"]["w_lgb"] == pytest.approx(1.0)
    assert {"n_estimators", "learning_rate"} <= set(v1["params"]["lgb"])
    assert {"n_estimators", "max_depth"} <= set(v1["params"]["rf"])
    assert set(OPERATIONAL_FEATURES) == {
        "station_target_encoded",
        "line_num",
        "station_id",
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
    }


def _digest(paths: list) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_run_comparison_end_to_end_keeps_v1_artifacts(tmp_path, monkeypatch):
    v1_files = sorted((ARTIFACTS_DIR / "ablation").glob("*.json")) + sorted(
        (ARTIFACTS_DIR / "ablation").glob("*.md")
    )
    before = _digest(v1_files)

    small = {
        "lgb": {"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15},
        "rf": {"n_estimators": 10, "max_depth": 6, "max_samples": 0.5},
    }
    monkeypatch.setattr(compare_v2, "V2_TREE_PARAMS", small)
    output = tmp_path / "v2"
    results = run_comparison(
        make_panel(date(2024, 1, 1), 110),
        output_dir=output,
        v1={"params": small, "weights": {"w_rf": 0.5, "w_lgb": 0.5}},
        seq2seq_params=LSTM_PARAMS,
        seq2seq_seeds=(1,),
        period=(
            date(2024, 1, 30),
            date(2024, 3, 31),
            date(2024, 4, 1),
            date(2024, 4, 19),
        ),
        device=CPU,
    )

    assert _digest(v1_files) == before
    assert set(results["metrics"]) == {
        "v1_fast",
        "v2_fast",
        "v2_history_fast",
        "v2_seq2seq",
        "v2_blend",
    }
    assert results["n_test_rows"] == 2 * 19 * 24
    saved = json.loads((output / "v2_comparison_results.json").read_text("utf-8"))
    assert saved["metrics"] == results["metrics"]
    markdown = (output / "v2_comparison_tables.md").read_text("utf-8")
    assert markdown == render_markdown(results)
    assert "Peak MAE" in markdown and "v1.0-baseline" in markdown
    assert sorted(p.name for p in (output / "checkpoints").glob("*.npy")) == [
        "v1_fast.npy",
        "v2_fast.npy",
        "v2_history_fast.npy",
        "v2_seq2seq.npy",
    ]
