"""
E2E 파이프라인(run_pipeline.py) 통합 실행 및 아티팩트 영속화 단위 테스트
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

import busan_metro_congestion.run_pipeline as pipeline_mod
from busan_metro_congestion.config import SAMPLES_DIR
from busan_metro_congestion.modeling.baseline import WEATHER_SHORT_FEATURES
from busan_metro_congestion.preprocess.features import (
    run_feature_pipeline as real_run_feature_pipeline,
)
from busan_metro_congestion.preprocess.festival import (
    build_festival_interim,
    build_festival_processed,
)
from busan_metro_congestion.preprocess.weather import process_weather_data
from busan_metro_congestion.run_pipeline import load_config, main, run_pipeline

FAST_TEST_CONFIG: dict[str, Any] = {
    "project": {"seed": 42},
    "model": {
        "lightgbm": {"n_estimators": 20, "final_n_estimators": 20},
        "random_forest": {
            "n_estimators": 15,
            "final_n_estimators": 15,
            "tune_max_depth": 6,
            "max_depth": 6,
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


@pytest.fixture(scope="module")
def sample_feature_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """data/samples/ 기반의 격리된 전처리 입력 경로 픽스처."""
    base_dir = tmp_path_factory.mktemp("run_pipeline_samples")

    weather_df = process_weather_data(
        file_path=SAMPLES_DIR / "weather" / "weather_forecast_sample.parquet",
        track="forecast",
    ).filter(pl.col("lead_time_hours") <= 12)
    weather_path = base_dir / "weather_processed_sample.parquet"
    weather_df.write_parquet(weather_path)

    fest_interim = build_festival_interim(
        raw_festival_dir=SAMPLES_DIR / "festival",
        output_path=base_dir / "festival_interim_sample.parquet",
    )
    festival_path = base_dir / "festival_processed_sample.parquet"
    build_festival_processed(interim_df=fest_interim, output_path=festival_path)

    return {
        "subway_path": SAMPLES_DIR / "subway" / "subway_2023_sample.csv",
        "weather_path": weather_path,
        "festival_path": festival_path,
    }


@pytest.fixture
def isolated_sample_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_feature_inputs: dict[str, Path],
) -> Path:
    """run_pipeline이 data/raw/ 및 추적 파일을 건드리지 않고 샘플 입력과 tmp_path만 사용하도록 격리."""
    features_out = tmp_path / "metro_features_test.parquet"

    def _run_sample_features(**kwargs: Any):
        kwargs.setdefault("subway_path", sample_feature_inputs["subway_path"])
        kwargs.setdefault("weather_path", sample_feature_inputs["weather_path"])
        kwargs.setdefault("festival_path", sample_feature_inputs["festival_path"])
        kwargs["output_path"] = features_out
        return real_run_feature_pipeline(**kwargs)

    monkeypatch.setattr(pipeline_mod, "run_feature_pipeline", _run_sample_features)
    return features_out


def test_load_config(tmp_path: Path):
    """설정 파일 로드 정상 동작 및 미존재 시 폴백 기본값 검증"""
    # 1. 미존재 파일 폴백 검증
    non_existent = tmp_path / "not_found.yaml"
    cfg_fallback = load_config(non_existent)
    assert isinstance(cfg_fallback, dict)
    assert cfg_fallback.get("project", {}).get("seed") == 42

    # 2. 유효한 YAML 파일 로드 검증
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("project:\n  seed: 2026\n", encoding="utf-8")
    cfg_custom = load_config(custom_yaml)
    assert cfg_custom.get("project", {}).get("seed") == 2026


def test_run_pipeline_preprocess_step(isolated_sample_pipeline: Path):
    """--step preprocess 단계만 실행 시 마스터 피처 반환 및 스키마 검증"""
    res = run_pipeline(step="preprocess", features_path=isolated_sample_pipeline)
    assert "df_features" in res
    df = res["df_features"]
    assert len(df) > 0
    assert "accumulated_passengers" in df.columns
    assert "station_target_encoded" in df.columns


def test_run_pipeline_all_short(tmp_path: Path, isolated_sample_pipeline: Path):
    """horizon='short' 2-Track 학습, SLSQP 가중치 제약식 및 체크포인트 영속화 검증"""
    out_dir = tmp_path / "artifacts_test_short"
    res = run_pipeline(
        horizon="short",
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=out_dir,
        features_path=isolated_sample_pipeline,
    )

    # 1. 반환 객체 구조 및 5종 모델 등록 검증
    assert "metrics" in res
    assert "models" in res
    assert "fast_track_weights" in res
    assert "accurate_track_weights" in res

    metrics = res["metrics"]
    assert metrics["horizon"] == "short"
    expected_models = [
        "lightgbm",
        "random_forest",
        "metrolstm",
        "fast_track_ensemble",
        "accurate_track_ensemble",
    ]
    for m_name in expected_models:
        assert m_name in metrics["models"]
        m_eval = metrics["models"][m_name]
        for metric_key in ("mae", "wape", "peak_mae", "r2"):
            assert metric_key in m_eval
            assert np.isfinite(m_eval[metric_key])
        assert m_eval["mae"] > 0.0

    # 2. SLSQP 가중치 합 및 비음수 제약 불변식 검증 (sum(w) == 1.0, w >= 0.0)
    w_rf, w_lgb = res["fast_track_weights"]
    assert pytest.approx(w_rf + w_lgb, 1e-4) == 1.0
    assert w_rf >= 0.0 and w_lgb >= 0.0

    w_rf_acc, w_lgb_acc, w_lstm_acc = res["accurate_track_weights"]
    assert pytest.approx(w_rf_acc + w_lgb_acc + w_lstm_acc, 1e-4) == 1.0
    assert w_rf_acc >= 0.0 and w_lgb_acc >= 0.0 and w_lstm_acc >= 0.0

    # 3. 모델 체크포인트 3종(lgb, rf, lstm), LSTM Scaler 및 메트릭 파일 영속화 검증
    ckpt_dir = out_dir / "checkpoints"
    lgb_file = ckpt_dir / "lgbm_short.txt"
    rf_file = ckpt_dir / "rf_short.joblib"
    lstm_file = ckpt_dir / "lstm_short.pt"
    scaler_file = ckpt_dir / "scaler_short.joblib"
    metrics_file = out_dir / "metrics_short.json"

    assert lgb_file.exists() and lgb_file.stat().st_size > 0
    assert rf_file.exists() and rf_file.stat().st_size > 0
    assert lstm_file.exists() and lstm_file.stat().st_size > 0
    assert scaler_file.exists() and scaler_file.stat().st_size > 0
    assert metrics_file.exists() and metrics_file.stat().st_size > 0

    scaler_data = joblib.load(scaler_file)
    assert set(scaler_data.keys()) == {"mean_x", "std_x", "mean_y", "std_y"}
    assert np.isfinite(scaler_data["mean_y"]) and scaler_data["std_y"] > 0.0

    # 4. 저장된 메트릭 JSON 역직렬화 내용 정합성 검증
    loaded_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert loaded_metrics["horizon"] == "short"
    assert "fast_track_ensemble" in loaded_metrics["models"]


def test_run_pipeline_all_long(tmp_path: Path, isolated_sample_pipeline: Path):
    """horizon='long'(Weather-Free 17종) 파이프라인 무결성 검증"""
    out_dir = tmp_path / "artifacts_test_long"
    res = run_pipeline(
        horizon="long",
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=out_dir,
        features_path=isolated_sample_pipeline,
    )

    metrics = res["metrics"]
    assert metrics["horizon"] == "long"
    assert len(res["selected_features"]) == 17
    assert set(WEATHER_SHORT_FEATURES).isdisjoint(res["selected_features"])
    assert res["models"]["lgbm"].n_features_in_ == 17
    assert res["models"]["rf"].n_features_in_ == 17
    assert res["models"]["lstm"].n_features_in_ == 17

    ckpt_dir = out_dir / "checkpoints"
    assert (ckpt_dir / "lgbm_long.txt").exists()
    assert (ckpt_dir / "rf_long.joblib").exists()
    assert (ckpt_dir / "lstm_long.pt").exists()
    assert (ckpt_dir / "scaler_long.joblib").exists()
    assert (out_dir / "metrics_long.json").exists()

    acc_metrics = metrics["models"]["accurate_track_ensemble"]
    assert np.isfinite(acc_metrics["mae"])
    assert acc_metrics["mae"] > 0.0


def test_run_pipeline_short_rejects_excessive_lead_time(
    tmp_path: Path, isolated_sample_pipeline: Path
):
    """단기 모델(short)은 lead_time_hours > 12 입력을 차단하고, 장기 모델(long)은 허용하는지 검증합니다."""
    res_pre = run_pipeline(step="preprocess", features_path=isolated_sample_pipeline)
    df_bad_lead = res_pre["df_features"].with_columns(
        pl.lit(24, dtype=pl.Int16).alias("lead_time_hours")
    )
    bad_features_path = tmp_path / "metro_features_lead24.parquet"
    df_bad_lead.write_parquet(bad_features_path)

    with pytest.raises(ValueError, match="최대 12시간 이내의 기상 예보만 허용"):
        run_pipeline(
            horizon="short",
            step="train_eval",
            config_dict=FAST_TEST_CONFIG,
            output_dir=tmp_path / "out_short_fail",
            features_path=bad_features_path,
        )

    res_long = run_pipeline(
        horizon="long",
        step="train_eval",
        config_dict=FAST_TEST_CONFIG,
        output_dir=tmp_path / "out_long_ok",
        features_path=bad_features_path,
    )
    assert res_long["metrics"]["horizon"] == "long"


def test_run_pipeline_default_metrics_path_unified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_sample_pipeline: Path
):
    """기본 출력 실행 시 메트릭이 artifacts/baseline/metrics*.json에만 저장되고 최상위 중복 복사가 없는지 검증합니다."""
    fake_artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(pipeline_mod, "ARTIFACTS_DIR", fake_artifacts)

    run_pipeline(
        horizon="short",
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=None,
        features_path=isolated_sample_pipeline,
    )

    assert (fake_artifacts / "baseline" / "metrics_short.json").exists()
    assert (fake_artifacts / "baseline" / "metrics.json").exists()
    assert not (fake_artifacts / "metrics.json").exists()


def test_run_pipeline_custom_date_range(
    monkeypatch: pytest.MonkeyPatch, isolated_sample_pipeline: Path
):
    """사용자 지정 날짜 범위의 1년 종료일 자동 계산(윤년 포함) 및 split_by_date_range 전달을 검증합니다."""
    real_split = pipeline_mod.split_by_date_range
    captured_calls: list[dict[str, str]] = []

    def spy_split_by_date_range(
        df: pl.DataFrame,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        date_col: str = "date",
    ):
        captured_calls.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
            }
        )
        return real_split(
            df,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            date_col=date_col,
        )

    monkeypatch.setattr(pipeline_mod, "split_by_date_range", spy_split_by_date_range)

    # 1. 종료일 생략 시 시작일 기준 윤년 고려 1년 뒤(2024-05-31, 2025-05-31) 자동 계산 후 분할에 전달됨
    with pytest.raises(ValueError, match="사용자 지정 날짜 범위에 데이터가 없습니다"):
        run_pipeline(
            horizon="short",
            step="train_eval",
            features_path=isolated_sample_pipeline,
            train_start="2023-06-01",
            val_start="2024-06-01",
        )
    assert len(captured_calls) == 1
    assert captured_calls[0] == {
        "train_start": "2023-06-01",
        "train_end": "2024-05-31",
        "val_start": "2024-06-01",
        "val_end": "2025-05-31",
    }

    # 2. 명시적 2년 학습 구간 지정 시 정확히 전달되고 빈 검증 구간에서 예외 발생
    captured_calls.clear()
    with pytest.raises(ValueError, match="사용자 지정 날짜 범위에 데이터가 없습니다"):
        run_pipeline(
            horizon="short",
            step="train_eval",
            features_path=isolated_sample_pipeline,
            train_start="2023-06-01",
            train_end="2025-05-31",
            val_start="2025-06-01",
            val_end="2026-05-31",
        )
    assert len(captured_calls) == 1
    assert captured_calls[0] == {
        "train_start": "2023-06-01",
        "train_end": "2025-05-31",
        "val_start": "2025-06-01",
        "val_end": "2026-05-31",
    }


def test_run_pipeline_invalid_arguments():
    """지원하지 않는 모드, step, horizon, 롤링 파라미터 또는 1년 단위 미충족 시 ValueError 검증"""
    # 1. 지원하지 않는 모드
    with pytest.raises(ValueError, match="지원하지 않는 파이프라인 모드"):
        run_pipeline(mode="invalid_mode")

    # 2. 지원하지 않는 step
    with pytest.raises(ValueError, match="지원하지 않는 파이프라인 단계"):
        run_pipeline(step="invalid_step")

    # 3. 지원하지 않는 horizon (train / predict 모드 공통)
    with pytest.raises(ValueError, match="지원하지 않는 예측 호라이즌"):
        run_pipeline(horizon="invalid_horizon")
    with pytest.raises(ValueError, match="지원하지 않는 예측 호라이즌"):
        run_pipeline(mode="predict", horizon="invalid_horizon")

    # 4. 시작일 없이 종료일만 단독 지정
    with pytest.raises(ValueError, match="train_end를 지정하려면 train_start가 필요"):
        run_pipeline(train_end="2024-05-31")

    # 5. 1년 단위 미충족 종료일 지정 (예: 5개월 뒤)
    with pytest.raises(ValueError, match="1년 단위.*윤년 고려"):
        run_pipeline(
            train_start="2023-06-01",
            train_end="2023-11-30",
            val_start="2023-12-01",
        )

    # 6. 종료일이 시작일보다 앞섬
    with pytest.raises(ValueError, match="앞설 수 없습니다"):
        run_pipeline(
            train_start="2023-06-01",
            train_end="2023-05-31",
            val_start="2023-06-01",
        )


def test_main_cli_train_and_predict(tmp_path: Path, isolated_sample_pipeline: Path):
    """CLI main 엔트리포인트의 train 및 predict 서브커맨드 디스패치 검증"""
    # 1. predict 서브커맨드 CLI 호출
    res_pred = main(["predict", "--horizon", "short"])
    assert res_pred["mode"] == "predict"

    # 2. train 서브커맨드 CLI 호출
    res_train = main(
        [
            "train",
            "--step",
            "preprocess",
            "--output-dir",
            str(tmp_path / "cli_artifacts"),
        ]
    )
    assert "df_features" in res_train

    # 3. 서브커맨드 생략 시 기본 train 모드 디스패치 검증
    res_default = main(
        [
            "--step",
            "preprocess",
        ]
    )
    assert "df_features" in res_default


def test_config_yaml_parameters_propagate_to_models_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_sample_pipeline: Path
):
    """config.yaml의 모델 하이퍼파라미터 및 pipeline 기본값이 실제 모델과 CLI에 반영되는지 검증합니다."""
    custom_cfg: dict[str, Any] = {
        "project": {"seed": 7},
        "pipeline": {"horizon": "long", "step": "preprocess"},
        "model": {
            "lightgbm": {
                "n_estimators": 12,
                "learning_rate": 0.08,
                "num_leaves": 15,
                "n_jobs": 2,
            },
            "random_forest": {
                "n_estimators": 11,
                "max_depth": 5,
                "n_jobs": 2,
            },
            "lstm": {
                "seq_len": 2,
                "hidden_dim1": 16,
                "hidden_dim2": 8,
                "dense_dim": 8,
                "dropout_rate": 0.15,
                "lr": 0.002,
                "epochs": 2,
                "batch_size": 32,
            },
        },
    }

    res = run_pipeline(
        horizon="short",
        step="all",
        config_dict=custom_cfg,
        output_dir=tmp_path / "cfg_prop_out",
        features_path=isolated_sample_pipeline,
    )
    lgb_model = res["models"]["lgbm"]
    rf_model = res["models"]["rf"]
    lstm_model = res["models"]["lstm"]

    assert lgb_model.get_params()["learning_rate"] == pytest.approx(0.08)
    assert lgb_model.get_params()["num_leaves"] == 15
    assert lgb_model.get_params()["n_jobs"] == 2
    assert rf_model.get_params()["n_jobs"] == 2
    assert lstm_model.dropout_rate == pytest.approx(0.15)
    assert lstm_model.lr == pytest.approx(0.002)

    # CLI에서 --step/--horizon 미지정 시 config.yaml의 pipeline 섹션(horizon, step) 모두 전달 검증
    cfg_yaml = tmp_path / "custom_pipeline.yaml"
    cfg_yaml.write_text(
        "project:\n  seed: 7\npipeline:\n  horizon: long\n  step: preprocess\n",
        encoding="utf-8",
    )
    captured_cli_kwargs: dict[str, Any] = {}
    real_run_pipeline = pipeline_mod.run_pipeline

    def spy_run_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured_cli_kwargs.update(kwargs)
        return real_run_pipeline(**kwargs)

    monkeypatch.setattr(pipeline_mod, "run_pipeline", spy_run_pipeline)
    res_cli = main(["train", "--config", str(cfg_yaml)])
    assert "df_features" in res_cli
    assert captured_cli_kwargs["horizon"] == "long"
    assert captured_cli_kwargs["step"] == "preprocess"
    assert captured_cli_kwargs["config_dict"]["project"]["seed"] == 7
