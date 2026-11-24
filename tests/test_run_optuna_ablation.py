"""
scripts/run_optuna_2track_ablation.py 및 관련 유틸리티 단위 테스트
"""

from __future__ import annotations

import numpy as np
import polars as pl

from scripts.run_optuna_2track_ablation import (
    average_metrics,
    evaluate_lgbm_on_tracks,
    evaluate_lstm_on_tracks,
    evaluate_rf_on_tracks,
    format_best_params,
    generate_tables_markdown,
    optimize_accurate_weights_rmse,
    optimize_fast_weights_rmse,
)


def test_optuna_2track_ablation_helpers_and_table_generation(monkeypatch, tmp_path):
    """가중치 최적화(SLSQP 실제 수렴) 및 표 4.1/4.2 마크다운 생성 정합성 검증"""
    import scripts.run_optuna_2track_ablation as ablation
    from busan_metro_congestion.modeling.metrics import calculate_rmse

    y_true = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
    # 초기값 x0=(0.5, 0.5) 및 (0.45, 0.45, 0.10)에서 큰 오차가 발생하도록 비대칭 예측 구성
    p_rf = y_true + 40.0
    p_lgb_perfect = y_true.copy()
    p_lgb_biased = y_true + 20.0
    p_lstm_perfect = y_true.copy()

    # 1. Fast-Track SLSQP 최적화: 초기값(0.5, 0.5, RMSE=20.0)에서 최적 모델(LGB=1.0)로 수렴 검증
    w_rf, w_lgb = optimize_fast_weights_rmse(y_true, p_rf, p_lgb_perfect)
    assert np.isclose(w_rf + w_lgb, 1.0)
    assert w_rf >= 0.0 and w_lgb >= 0.0
    assert w_lgb > 0.95 and w_rf < 0.05
    assert calculate_rmse(y_true, w_rf * p_rf + w_lgb * p_lgb_perfect) < 1.0

    # 2. Accurate-Track SLSQP 최적화: 초기값(0.45, 0.45, 0.10, RMSE=27.0)에서 LSTM=1.0으로 수렴 검증
    w_rf_a, w_lgb_a, w_lstm_a = optimize_accurate_weights_rmse(
        y_true, p_rf, p_lgb_biased, p_lstm_perfect
    )
    assert np.isclose(w_rf_a + w_lgb_a + w_lstm_a, 1.0)
    assert all(w >= 0.0 for w in (w_rf_a, w_lgb_a, w_lstm_a))
    assert w_lstm_a > 0.95
    assert (
        calculate_rmse(
            y_true, w_rf_a * p_rf + w_lgb_a * p_lgb_biased + w_lstm_a * p_lstm_perfect
        )
        < 1.0
    )

    # 3. 평균 지표 계산 검증
    avg_m = average_metrics(
        [
            {
                "rmse": 100.0,
                "mae": 50.0,
                "wape": 10.0,
                "peak_mae": 70.0,
                "r2": 0.95,
                "pearson_r": 0.97,
            },
            {
                "rmse": 120.0,
                "mae": 60.0,
                "wape": 12.0,
                "peak_mae": 80.0,
                "r2": 0.97,
                "pearson_r": 0.99,
            },
        ]
    )
    assert np.isclose(avg_m["rmse"], 110.0)
    assert np.isclose(avg_m["mae"], 55.0)
    assert np.isclose(avg_m["wape"], 11.0)
    assert np.isclose(avg_m["peak_mae"], 75.0)
    assert np.isclose(avg_m["r2"], 0.96)
    assert np.isclose(avg_m["pearson_r"], 0.98)

    # 4. 표 4.1 및 표 4.2 마크다운 생성 구조 및 수치 반영 검증 (디스크 아티팩트 격리)
    monkeypatch.setattr(ablation, "ARTIFACTS_DIR", tmp_path)
    dummy_group = {
        "feature_count": 13,
        "lgb": avg_m,
        "rf": avg_m,
        "lstm": avg_m,
        "fast_track": avg_m,
        "accurate_track": avg_m,
        "weights_fast": {"w_rf": 0.45, "w_lgb": 0.55},
        "weights_accurate": {"w_rf": 0.40, "w_lgb": 0.40, "w_lstm": 0.20},
        "avg_train_time_sec": {
            "lgb": 4.0,
            "rf": 15.0,
            "lstm": 75.0,
            "fast": 19.0,
            "acc": 94.0,
        },
        "avg_pred_time_sec": {
            "lgb": 0.5,
            "rf": 0.5,
            "lstm": 5.0,
            "fast": 1.0,
            "acc": 6.0,
        },
    }
    mock_results = {
        "subway_calendar": {**dummy_group, "label": "Set 1: 지하철 + 날짜 (기저 13종)"},
        "subway_calendar_weather": {
            **dummy_group,
            "label": "Set 2: + 기상 예보 추가 (21종)",
        },
        "subway_calendar_festival": {
            **dummy_group,
            "label": "Set 3: + 대규모 행사 추가 (17종)",
        },
        "all": {
            **dummy_group,
            "label": "Set 4: + 기상 및 행사 전체 (25종)",
            "feature_count": 25,
        },
    }
    md_out = generate_tables_markdown(mock_results)
    assert "## [표 4.0]" not in md_out
    assert "## [표 4.1]" in md_out
    assert "## [표 4.2]" in md_out
    assert "Set 4: + 기상 및 행사 전체 (25종)" in md_out
    assert "**110.00**" in md_out
    assert "55.00" in md_out
    assert "0.40 : 0.40 : 0.20" in md_out


def test_optuna_table_4_0_generation():
    """논문 [표 4.0] Optuna 튜닝 전/후 개선 비교표 마크다운 생성 검증"""
    dummy_m_base = {
        "rmse": 180.0,
        "mae": 90.0,
        "wape": 14.0,
        "peak_mae": 130.0,
        "r2": 0.95,
        "pearson_r": 0.96,
    }
    dummy_m_tuned = {
        "rmse": 140.0,
        "mae": 70.0,
        "wape": 11.0,
        "peak_mae": 95.0,
        "r2": 0.97,
        "pearson_r": 0.98,
    }
    mock_model_res = {
        "baseline_metrics": dummy_m_base,
        "best_metrics": dummy_m_tuned,
        "improvements": {
            "rmse_diff": -40.0,
            "rmse_impr_pct": 22.22,
            "mae_diff": -20.0,
            "mae_impr_pct": 22.22,
        },
        "best_params": {
            "n_estimators": 120,
            "learning_rate": 0.08,
            "num_leaves": 45,
            "max_depth": 8,
        },
    }
    mock_tuning = {
        "subway_calendar": {
            "label": "Set 1: 지하철 + 날짜 (기저 13종)",
            "lgb": mock_model_res,
            "rf": {
                **mock_model_res,
                "best_params": {
                    "n_estimators": 60,
                    "max_depth": 12,
                    "max_samples": 0.35,
                    "min_samples_leaf": 4,
                },
            },
            "lstm": {
                **mock_model_res,
                "best_params": {
                    "hidden_dim1": 128,
                    "hidden_dim2": 64,
                    "dropout_rate": 0.1,
                    "lr": 0.003,
                },
            },
        }
    }
    mock_results = {
        "subway_calendar": {
            "label": "Set 1: 지하철 + 날짜 (기저 13종)",
            "feature_count": 13,
            "accurate_track": dummy_m_tuned,
            "weights_accurate": {"w_rf": 0.4, "w_lgb": 0.4, "w_lstm": 0.2},
        }
    }
    md_out = generate_tables_markdown(mock_results, tuning_results=mock_tuning)
    assert "## [표 4.0]" in md_out
    assert "LightGBM" in md_out
    assert "Random Forest" in md_out
    assert "MetroLSTM" in md_out
    assert "-40.00" in md_out
    assert "+22.22%" in md_out


def test_format_best_params():
    """최적 파라미터 포맷팅 헬퍼 함수 정상 동작 검증"""
    lgb_p = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
    }
    rf_p = {
        "n_estimators": 80,
        "max_depth": 14,
        "max_samples": 0.35,
        "min_samples_leaf": 5,
    }
    lstm_p = {"hidden_dim1": 128, "hidden_dim2": 64, "dropout_rate": 0.1, "lr": 0.003}

    assert "n_est=100" in format_best_params("lgb", lgb_p)
    assert "lr=0.050" in format_best_params("lgb", lgb_p)
    assert "max_samples=0.35" in format_best_params("rf", rf_p)
    assert "hidden=(128, 64)" in format_best_params("lstm", lstm_p)


def _make_learnable_splits(
    seed: int = 42, n: int = 80
) -> list[dict[str, pl.DataFrame]]:
    """선형 신호(y = 100 + 25*f1 - 15*f2)를 갖는 2-Track 테스트 분할 생성"""
    rng = np.random.default_rng(seed)

    def _build() -> pl.DataFrame:
        f1 = rng.standard_normal(n).astype(np.float32)
        f2 = rng.standard_normal(n).astype(np.float32)
        y = np.clip(100.0 + 25.0 * f1 - 15.0 * f2, 0.0, None).astype(np.float32)
        return pl.DataFrame(
            {
                "f1": f1,
                "f2": f2,
                "accumulated_passengers": y,
                "station_id": np.repeat(np.arange(4), n // 4),
                "hour": rng.integers(5, 24, size=n),
            }
        )

    d1, d2 = _build(), _build()
    return [{"train_df": d1, "val_df": d2}, {"train_df": d2, "val_df": d1}]


def test_evaluate_models_on_dummy_tracks():
    """3-Track 롤링 평가 헬퍼 함수(LGB, RF, LSTM)의 지표 산출·비음수 예측·아티팩트 반환 검증"""
    feat_cols = ["f1", "f2"]
    splits = _make_learnable_splits(seed=42)
    expected_keys = {"rmse", "mae", "wape", "peak_mae", "r2", "pearson_r"}

    # LGBM
    m_lgb, lgb_arts = evaluate_lgbm_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 20, "num_leaves": 15, "verbose": -1, "random_state": 42},
        return_artifacts=True,
    )
    assert expected_keys.issubset(m_lgb.keys())
    assert all(np.isfinite(m_lgb[k]) for k in expected_keys)
    assert m_lgb["r2"] > 0.5
    assert len(lgb_arts) == 2 and all((a["preds"] >= 0).all() for a in lgb_arts)

    # RF
    m_rf, rf_arts = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {
            "n_estimators": 20,
            "max_depth": 6,
            "max_samples": 0.8,
            "n_jobs": 1,
            "random_state": 42,
        },
        return_artifacts=True,
    )
    assert expected_keys.issubset(m_rf.keys())
    assert all(np.isfinite(m_rf[k]) for k in expected_keys)
    assert m_rf["r2"] > 0.5
    assert len(rf_arts) == 2 and all((a["preds"] >= 0).all() for a in rf_arts)

    # LSTM
    m_lstm, lstm_arts = evaluate_lstm_on_tracks(
        splits,
        feat_cols,
        {
            "seq_len": 2,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dense_dim": 8,
            "epochs": 2,
            "batch_size": 32,
            "device": "cpu",
            "random_state": 42,
        },
        return_artifacts=True,
    )
    assert expected_keys.issubset(m_lstm.keys())
    assert all(np.isfinite(m_lstm[k]) for k in expected_keys)
    assert len(lstm_arts) == 2 and all((a["preds"] >= 0).all() for a in lstm_arts)


def test_tune_all_models_for_group_dummy(tmp_path):
    """tune_all_models_for_group 실행 정합성(개선율 수식, 부분 JSON 저장, 3대 모델 스키마) 검증"""
    import json

    from scripts.run_optuna_2track_ablation import tune_all_models_for_group

    feat_cols = ["f1", "f2"]
    splits = _make_learnable_splits(seed=42)
    partial_json = tmp_path / "partial_tuning.json"
    all_tuning: dict = {}

    res = tune_all_models_for_group(
        group_name="test_group",
        label="Test Group (2 features)",
        feat_cols=feat_cols,
        splits=splits,
        n_lgb_trials=1,
        n_rf_trials=1,
        n_lstm_trials=1,
        seed=42,
        partial_save_path=partial_json,
        all_tuning_results=all_tuning,
    )
    assert partial_json.exists()
    saved = json.loads(partial_json.read_text(encoding="utf-8"))
    assert "test_group" in saved

    for m_key in ("lgb", "rf", "lstm"):
        m_res = res[m_key]
        base_rmse = m_res["baseline_metrics"]["rmse"]
        best_rmse = m_res["best_metrics"]["rmse"]
        assert np.isfinite(base_rmse) and np.isfinite(best_rmse)
        assert len(m_res["best_params"]) > 0
        assert np.isclose(
            m_res["improvements"]["rmse_diff"], best_rmse - base_rmse, atol=1e-6
        )
        assert np.isclose(
            m_res["improvements"]["rmse_impr_pct"],
            (base_rmse - best_rmse) / base_rmse * 100.0,
            atol=1e-6,
        )


def test_idempotency_and_sqlite_resume(monkeypatch, tmp_path):
    """동일 데이터 재실행 멱등성, SQLite 이어하기(재평가 0회 보장) 및 3대 모델 가중치 로드 추론 일치 검증"""
    import scripts.run_optuna_2track_ablation as ablation
    from scripts.run_optuna_2track_ablation import tune_lgb_for_group

    np.random.seed(42)
    n = 60
    feat_cols = ["f1", "f2"]
    d1 = pl.DataFrame(
        {
            "f1": np.random.randn(n).astype(np.float32),
            "f2": np.random.randn(n).astype(np.float32),
            "accumulated_passengers": (np.random.rand(n) * 100.0).astype(np.float32),
            "station_id": np.random.randint(0, 5, size=n),
            "hour": np.random.randint(5, 24, size=n),
        }
    )
    d2 = pl.DataFrame(
        {
            "f1": np.random.randn(n).astype(np.float32),
            "f2": np.random.randn(n).astype(np.float32),
            "accumulated_passengers": (np.random.rand(n) * 100.0).astype(np.float32),
            "station_id": np.random.randint(0, 5, size=n),
            "hour": np.random.randint(5, 24, size=n),
        }
    )
    splits = [{"train_df": d1, "val_df": d2}, {"train_df": d2, "val_df": d1}]

    # 1. LGBM, RF, LSTM 2회 반복 실행 시 100% 동일한 지표 산출(멱등성) 검증
    m_lgb_1 = evaluate_lgbm_on_tracks(
        splits,
        feat_cols,
        {
            "n_estimators": 5,
            "num_leaves": 7,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
            "random_state": 42,
        },
    )
    m_lgb_2 = evaluate_lgbm_on_tracks(
        splits,
        feat_cols,
        {
            "n_estimators": 5,
            "num_leaves": 7,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
            "random_state": 42,
        },
    )
    assert np.isclose(m_lgb_1["rmse"], m_lgb_2["rmse"], atol=1e-6)

    m_rf_1 = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "max_depth": 3, "max_samples": 0.5, "random_state": 42},
    )
    m_rf_2 = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "max_depth": 3, "max_samples": 0.5, "random_state": 42},
    )
    assert np.isclose(m_rf_1["rmse"], m_rf_2["rmse"], atol=1e-6)

    m_lstm_1 = evaluate_lstm_on_tracks(
        splits,
        feat_cols,
        {
            "seq_len": 2,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dense_dim": 8,
            "epochs": 2,
            "batch_size": 32,
            "random_state": 42,
        },
    )
    m_lstm_2 = evaluate_lstm_on_tracks(
        splits,
        feat_cols,
        {
            "seq_len": 2,
            "hidden_dim1": 16,
            "hidden_dim2": 8,
            "dense_dim": 8,
            "epochs": 2,
            "batch_size": 32,
            "random_state": 42,
        },
    )
    assert np.isclose(m_lstm_1["rmse"], m_lstm_2["rmse"], atol=1e-6)

    # 2. SQLite 스토리지 기반 이어하기(Resume) 및 실제 가중치 파일 즉시 저장 검증
    db_url = f"sqlite:///{(tmp_path / 'test_optuna.db').as_posix()}"
    res_step1 = tune_lgb_for_group(
        "g_test",
        "Label",
        feat_cols,
        splits,
        n_trials=1,
        seed=42,
        storage_url=db_url,
        ckpt_dir=tmp_path,
        g_idx=1,
    )
    assert (tmp_path / "lgb_set1_track1.txt").exists()
    assert (tmp_path / "pred_lgb_set1_track1.npy").exists()
    saved_lgb_preds_t1 = np.load(tmp_path / "pred_lgb_set1_track1.npy")

    # 재실행 시 SQLite DB에서 Baseline·Trial을 복원하며 모델 재평가가 단 1회도 호출되지 않아야 함
    def _fail_if_reevaluated(*args, **kwargs):
        raise AssertionError("SQLite Resume 시 모델 재평가가 호출되면 안 됩니다.")

    with monkeypatch.context() as mp:
        mp.setattr(ablation, "_evaluate_model_on_tracks", _fail_if_reevaluated)
        res_step2 = tune_lgb_for_group(
            "g_test",
            "Label",
            feat_cols,
            splits,
            n_trials=1,
            seed=42,
            storage_url=db_url,
            ckpt_dir=tmp_path,
            g_idx=1,
        )
    assert np.isclose(
        res_step1["best_metrics"]["rmse"], res_step2["best_metrics"]["rmse"], atol=1e-6
    )

    # 3. 3대 모델(LGB, RF, LSTM) 가중치 저장 후 _load_and_predict_track_weights 로드 추론 일치 검증
    from scripts.run_optuna_2track_ablation import (
        _load_and_predict_track_weights,
        _save_lstm_artifacts,
        _save_rf_artifacts,
    )

    _, rf_arts = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "max_depth": 3, "max_samples": 0.5, "random_state": 42},
        return_artifacts=True,
    )
    _save_rf_artifacts(tmp_path, 1, rf_arts)

    _, lstm_arts = evaluate_lstm_on_tracks(
        splits,
        feat_cols,
        {
            "seq_len": 3,
            "hidden_dim1": 32,
            "hidden_dim2": 16,
            "dense_dim": 32,
            "dropout_rate": 0.1,
            "epochs": 1,
            "batch_size": 32,
            "random_state": 42,
        },
        return_artifacts=True,
    )
    _save_lstm_artifacts(tmp_path, 1, lstm_arts)

    X_val_t1 = np.ascontiguousarray(d2.select(feat_cols).to_numpy(), dtype=np.float32)
    stn_val_t1 = d2["station_id"].to_numpy()
    p_lgb_l, p_rf_l, p_lstm_l, _, _, _ = _load_and_predict_track_weights(
        tmp_path, 1, 1, X_val_t1, stn_val_t1, len(feat_cols)
    )
    assert np.allclose(p_lgb_l, saved_lgb_preds_t1, atol=1e-5)
    assert np.allclose(p_rf_l, rf_arts[0]["preds"], atol=1e-5)
    assert np.allclose(p_lstm_l, lstm_arts[0]["preds"], atol=1e-5)


def test_optuna_2track_ablation_tables_only_cli(monkeypatch, tmp_path):
    """--tables-only 실행 시 기존 표를 덮어쓰지 않고 실행 시각 파일명으로 마크다운 생성 검증"""
    import re
    import shutil
    import sys

    import scripts.run_optuna_2track_ablation as ablation
    from busan_metro_congestion.config import ARTIFACTS_DIR

    abl_dir = tmp_path / "ablation"
    abl_dir.mkdir()
    for name in (
        "ablation_optuna_2track_rolling_results.json",
        "optuna_all_models_results.json",
    ):
        shutil.copy(ARTIFACTS_DIR / "ablation" / name, abl_dir / name)
    legacy_md = abl_dir / "ablation_optuna_2track_tables.md"
    legacy_md.write_text("기존 표", encoding="utf-8")

    monkeypatch.setattr(ablation, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_optuna_2track_ablation.py", "--tables-only"])
    ablation.main()

    assert legacy_md.read_text(encoding="utf-8") == "기존 표"
    generated = list(abl_dir.glob("ablation_optuna_2track_tables_*.md"))
    assert len(generated) == 1
    assert re.fullmatch(
        r"ablation_optuna_2track_tables_\d{8}_\d{6}\.md", generated[0].name
    )
    content = generated[0].read_text(encoding="utf-8")
    assert "4.1" in content
    assert "4.4" in content


def test_evaluate_or_train_holdout_train_and_reload(monkeypatch, tmp_path):
    """24개월 홀드아웃 신규 학습·저장 후 eval_from_weights=True 재학습 없이 가중치 로드 추론 일치 검증"""
    from datetime import date, timedelta

    import scripts.run_optuna_2track_ablation as ablation
    from scripts.run_optuna_2track_ablation import _evaluate_or_train_holdout

    rng = np.random.default_rng(42)
    n = 60
    feat_cols = ["f1", "f2"]
    base_date = date(2024, 1, 1)
    tr_df = pl.DataFrame(
        {
            "date": [base_date + timedelta(days=i // 5) for i in range(n)],
            "f1": rng.standard_normal(n).astype(np.float32),
            "f2": rng.standard_normal(n).astype(np.float32),
            "accumulated_passengers": (rng.random(n) * 100.0).astype(np.float32),
            "station_id": np.repeat(np.arange(5), n // 5),
            "hour": rng.integers(5, 24, size=n),
        }
    )
    te_df = pl.DataFrame(
        {
            "date": [base_date + timedelta(days=30 + i // 5) for i in range(n)],
            "f1": rng.standard_normal(n).astype(np.float32),
            "f2": rng.standard_normal(n).astype(np.float32),
            "accumulated_passengers": (rng.random(n) * 100.0).astype(np.float32),
            "station_id": np.repeat(np.arange(5), n // 5),
            "hour": rng.integers(5, 24, size=n),
        }
    )
    tuning_results = {
        "dummy_group": {
            "lgb": {"best_params": {"n_estimators": 5, "num_leaves": 7}},
            "rf": {"best_params": {"n_estimators": 5, "max_depth": 3}},
            "lstm": {
                "best_params": {
                    "hidden_dim1": 16,
                    "hidden_dim2": 8,
                    "epochs": 1,
                }
            },
        }
    }
    final_results = {
        "dummy_group": {
            "weights_fast": {"w_rf": 0.6, "w_lgb": 0.4},
            "weights_accurate": {"w_rf": 0.5, "w_lgb": 0.3, "w_lstm": 0.2},
        }
    }

    trained = _evaluate_or_train_holdout(
        group_key="dummy_group",
        group_label="Dummy Holdout",
        feat_cols=feat_cols,
        ho_tr_df=tr_df,
        ho_te_df=te_df,
        target_col="accumulated_passengers",
        ckpt_dir=tmp_path,
        tuning_results=tuning_results,
        final_results=final_results,
        eval_from_weights=False,
    )
    for fname in (
        "lgb_dummy_group_holdout_24m.txt",
        "rf_dummy_group_holdout_24m.joblib",
        "scaler_dummy_group_holdout_24m.joblib",
        "lstm_dummy_group_holdout_24m.pt",
        "pred_fast_dummy_group_holdout_24m.npy",
        "pred_acc_dummy_group_holdout_24m.npy",
    ):
        assert (tmp_path / fname).exists()

    def _fail_if_retrained(*args, **kwargs):
        raise AssertionError("eval_from_weights=True 시 모델을 재학습하면 안 됩니다.")

    monkeypatch.setattr(ablation, "_fit_predict_model", _fail_if_retrained)
    reloaded = _evaluate_or_train_holdout(
        group_key="dummy_group",
        group_label="Dummy Holdout",
        feat_cols=feat_cols,
        ho_tr_df=tr_df,
        ho_te_df=te_df,
        target_col="accumulated_passengers",
        ckpt_dir=tmp_path,
        tuning_results=tuning_results,
        final_results=final_results,
        eval_from_weights=True,
        existing_holdout=trained,
    )
    for key in ("lgb", "rf", "lstm", "fast_track", "accurate_track"):
        assert np.isclose(trained[key]["rmse"], reloaded[key]["rmse"], atol=1e-4)
        assert np.isclose(trained[key]["mae"], reloaded[key]["mae"], atol=1e-4)


def test_load_weights_infers_custom_lstm_dense_dim(tmp_path):
    """_load_and_predict_track_weights가 dense_dim!=32인 LSTM 체크포인트도 state_dict에서 동적 추론하는지 검증"""
    from scripts.run_optuna_2track_ablation import (
        _load_and_predict_track_weights,
        _save_lgb_artifacts,
        _save_lstm_artifacts,
        _save_rf_artifacts,
    )

    feat_cols = ["f1", "f2"]
    splits = _make_learnable_splits(seed=42, n=60)
    _, lgb_arts = evaluate_lgbm_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "num_leaves": 7, "verbose": -1, "random_state": 42},
        return_artifacts=True,
    )
    _, rf_arts = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "max_depth": 3, "max_samples": 0.5, "random_state": 42},
        return_artifacts=True,
    )
    _, lstm_arts = evaluate_lstm_on_tracks(
        splits,
        feat_cols,
        {
            "seq_len": 3,
            "hidden_dim1": 24,
            "hidden_dim2": 12,
            "dense_dim": 16,
            "dropout_rate": 0.1,
            "epochs": 1,
            "batch_size": 32,
            "random_state": 42,
        },
        return_artifacts=True,
    )
    _save_lgb_artifacts(tmp_path, 2, lgb_arts)
    _save_rf_artifacts(tmp_path, 2, rf_arts)
    _save_lstm_artifacts(tmp_path, 2, lstm_arts)

    val_df = splits[0]["val_df"]
    X_val = np.ascontiguousarray(val_df.select(feat_cols).to_numpy(), dtype=np.float32)
    stn_val = val_df["station_id"].to_numpy()

    _, _, p_lstm, _, _, _ = _load_and_predict_track_weights(
        tmp_path, 2, 1, X_val, stn_val, len(feat_cols)
    )
    assert np.allclose(p_lstm, lstm_arts[0]["preds"], atol=1e-5)


def test_tune_model_zero_baseline_no_zero_division(monkeypatch):
    """Baseline RMSE/MAE가 0.0일 때 ZeroDivisionError 없이 개선율 0.0을 반환하는지 검증"""
    import scripts.run_optuna_2track_ablation as ablation
    from scripts.run_optuna_2track_ablation import tune_lgb_for_group

    zero_metrics = {
        "rmse": 0.0,
        "mae": 0.0,
        "wape": 0.0,
        "peak_mae": 0.0,
        "r2": 1.0,
        "pearson_r": 1.0,
    }

    def fake_eval(
        model_key,
        splits,
        feat_cols,
        params,
        target_col="accumulated_passengers",
        return_artifacts=False,
    ):
        if return_artifacts:
            return zero_metrics, [{"fit_sec": 0.01, "pred_sec": 0.01}]
        return zero_metrics

    monkeypatch.setattr(ablation, "_evaluate_model_on_tracks", fake_eval)
    res = tune_lgb_for_group(
        "zero_group", "Zero Label", ["f1", "f2"], [], n_trials=1, seed=42
    )
    assert res["improvements"]["rmse_diff"] == 0.0
    assert res["improvements"]["rmse_impr_pct"] == 0.0
    assert res["improvements"]["mae_diff"] == 0.0
    assert res["improvements"]["mae_impr_pct"] == 0.0


def test_evaluate_group_rolling_retrains_only_missing_model(monkeypatch, tmp_path):
    """_evaluate_group_rolling에서 일부 모델 가중치만 누락 시 기존 모델을 덮어쓰지 않고 누락 모델만 보충 학습 검증"""
    import scripts.run_optuna_2track_ablation as ablation
    from scripts.run_optuna_2track_ablation import (
        _evaluate_group_rolling,
        _save_lgb_artifacts,
        _save_rf_artifacts,
    )

    feat_cols = ["f1", "f2"]
    splits = _make_learnable_splits(seed=42, n=60)
    g_tuning = {
        "lgb": {
            "best_params": {"n_estimators": 5, "num_leaves": 7},
            "avg_fit_sec": 0.1,
        },
        "rf": {"best_params": {"n_estimators": 5, "max_depth": 3}, "avg_fit_sec": 0.2},
        "lstm": {
            "best_params": {"hidden_dim1": 16, "hidden_dim2": 8, "epochs": 1},
            "avg_fit_sec": 0.3,
        },
    }

    # lgb와 rf만 미리 저장하고 lstm은 누락시킴
    _, lgb_arts = evaluate_lgbm_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "num_leaves": 7, "verbose": -1, "random_state": 42},
        return_artifacts=True,
    )
    _, rf_arts = evaluate_rf_on_tracks(
        splits,
        feat_cols,
        {"n_estimators": 5, "max_depth": 3, "max_samples": 0.5, "random_state": 42},
        return_artifacts=True,
    )
    _save_lgb_artifacts(tmp_path, 1, lgb_arts)
    _save_rf_artifacts(tmp_path, 1, rf_arts)

    retrained_models: list[str] = []
    real_eval = ablation._evaluate_model_on_tracks

    def spy_eval(model_key, *args, **kwargs):
        retrained_models.append(model_key)
        return real_eval(model_key, *args, **kwargs)

    monkeypatch.setattr(ablation, "_evaluate_model_on_tracks", spy_eval)
    res = _evaluate_group_rolling(
        g_idx=1,
        g_name="subway_calendar",
        feat_cols=feat_cols,
        val_splits=splits,
        g_tuning=g_tuning,
        ckpt_dir=tmp_path,
        eval_from_weights=False,
    )

    assert retrained_models == ["lstm"]
    assert (tmp_path / "lstm_set1_track1.pt").exists()
    assert (tmp_path / "scaler_set1_track2.joblib").exists()
    for key in ("lgb", "rf", "lstm", "fast_track", "accurate_track"):
        assert np.isfinite(res[key]["rmse"])
