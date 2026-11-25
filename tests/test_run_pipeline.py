"""
Set 1 운영 E2E 파이프라인(run_pipeline.py) 통합 실행 및 아티팩트 영속화 단위 테스트
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import pytest

import busan_metro_congestion.run_pipeline as pipeline_mod
from busan_metro_congestion.config import SAMPLES_DIR
from busan_metro_congestion.modeling.baseline import (
    FESTIVAL_FEATURES,
    OPERATIONAL_FEATURES,
    WEATHER_SHORT_FEATURES,
)
from busan_metro_congestion.modeling.predict import (
    get_checkpoint_paths,
    validate_predictions_df,
)
from busan_metro_congestion.preprocess.features import (
    OPERATIONAL_MASTER_COLUMNS,
)
from busan_metro_congestion.preprocess.features import (
    run_feature_pipeline as real_run_feature_pipeline,
)
from busan_metro_congestion.run_pipeline import (
    _apply_split_target_encoding,
    _save_checkpoints_atomic,
    load_config,
    main,
    run_pipeline,
)

FAST_TEST_CONFIG: dict[str, Any] = {
    "project": {"seed": 42, "version": "0.1.0"},
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


@pytest.fixture
def isolated_sample_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """run_pipeline이 data/raw/·기상·축제 파일 없이 지하철 샘플과 tmp_path만 사용하도록 격리."""
    features_out = tmp_path / "metro_features_operational_test.parquet"

    def _run_sample_features(**kwargs: Any):
        kwargs.setdefault(
            "subway_path", SAMPLES_DIR / "subway" / "subway_2023_sample.csv"
        )
        kwargs["output_path"] = features_out
        return real_run_feature_pipeline(**kwargs)

    monkeypatch.setattr(pipeline_mod, "run_feature_pipeline", _run_sample_features)
    return features_out


def test_load_config(tmp_path: Path):
    """설정 파일 로드 정상 동작 및 미존재 시 폴백 기본값 검증"""
    non_existent = tmp_path / "not_found.yaml"
    cfg_fallback = load_config(non_existent)
    assert isinstance(cfg_fallback, dict)
    assert cfg_fallback.get("project", {}).get("seed") == 42

    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("project:\n  seed: 2026\n", encoding="utf-8")
    cfg_custom = load_config(custom_yaml)
    assert cfg_custom.get("project", {}).get("seed") == 2026


def test_run_pipeline_preprocess_step(isolated_sample_pipeline: Path):
    """--step preprocess 실행 시 Set 1(기상·축제 배제 17컬럼) 피처 데이터 반환 검증"""
    res = run_pipeline(step="preprocess", features_path=isolated_sample_pipeline)
    assert "df_features" in res
    df = res["df_features"]
    assert len(df) > 0
    assert df.columns == OPERATIONAL_MASTER_COLUMNS
    assert set(WEATHER_SHORT_FEATURES).isdisjoint(df.columns)
    assert set(FESTIVAL_FEATURES).isdisjoint(df.columns)


def test_run_pipeline_all_operational_set1(
    tmp_path: Path, isolated_sample_pipeline: Path
):
    """Set 1(13종) 운영 2-Track 학습, SLSQP 가중치 제약식 및 체크포인트 6종 원자적 영속화 검증"""
    out_dir = tmp_path / "artifacts_test_operational"
    res = run_pipeline(
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=out_dir,
        features_path=isolated_sample_pipeline,
    )

    # 1. Set 1(13종) 피처 및 5종 평가 모델 검증
    assert res["feature_set"] == "subway_calendar"
    assert res["selected_features"] == OPERATIONAL_FEATURES
    assert len(res["selected_features"]) == 13
    assert set(WEATHER_SHORT_FEATURES).isdisjoint(res["selected_features"])
    assert set(FESTIVAL_FEATURES).isdisjoint(res["selected_features"])
    assert res["models"]["lgbm"].n_features_in_ == 13
    assert res["models"]["rf"].n_features_in_ == 13
    assert res["models"]["lstm"].n_features_in_ == 13

    metrics = res["metrics"]
    assert metrics["feature_set"] == "subway_calendar"
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

    # 2. SLSQP 가중치 합 및 비음수 제약 불변식 검증
    w_rf, w_lgb = res["fast_track_weights"]
    assert pytest.approx(w_rf + w_lgb, 1e-4) == 1.0
    assert w_rf >= 0.0 and w_lgb >= 0.0

    w_rf_acc, w_lgb_acc, w_lstm_acc = res["accurate_track_weights"]
    assert pytest.approx(w_rf_acc + w_lgb_acc + w_lstm_acc, 1e-4) == 1.0
    assert w_rf_acc >= 0.0 and w_lgb_acc >= 0.0 and w_lstm_acc >= 0.0

    # 3. 운영 체크포인트 6종(lgbm, rf, lstm, scaler, station_target_encoder, model_manifest) 및 metrics.json 검증
    ckpt_dir = out_dir / "checkpoints"
    expected_files = [
        ckpt_dir / "lgbm.txt",
        ckpt_dir / "rf.joblib",
        ckpt_dir / "lstm.pt",
        ckpt_dir / "scaler.joblib",
        ckpt_dir / "station_target_encoder.parquet",
        ckpt_dir / "model_manifest.json",
        out_dir / "metrics.json",
    ]
    for f in expected_files:
        assert f.exists() and f.stat().st_size > 0, f"산출물 누락: {f}"

    scaler_data = joblib.load(ckpt_dir / "scaler.joblib")
    assert set(scaler_data.keys()) == {"mean_x", "std_x", "mean_y", "std_y"}
    assert np.isfinite(scaler_data["mean_y"]) and scaler_data["std_y"] > 0.0

    # 기본(날짜 미지정) 실행 시 운영 배포 모델이 전 기간 데이터로 재학습되어 매니페스트에 기록되는지 검증
    manifest_data = json.loads(
        (ckpt_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    full_features_rows = len(pl.read_parquet(isolated_sample_pipeline))
    assert manifest_data["model_version"] == res["model_version"]
    assert manifest_data["feature_set"] == "subway_calendar"
    assert manifest_data["selected_features"] == OPERATIONAL_FEATURES
    assert manifest_data["train_row_count"] == full_features_rows
    assert manifest_data["lstm_config"]["input_dim"] == 13
    assert manifest_data["lstm_config"]["seq_len"] == 2
    assert manifest_data["weights"]["fast_track"]["w_rf"] == pytest.approx(w_rf)
    assert manifest_data["weights"]["accurate_track"]["w_lstm"] == pytest.approx(
        w_lstm_acc
    )

    loaded_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert loaded_metrics["feature_set"] == "subway_calendar"
    assert "fast_track_ensemble" in loaded_metrics["models"]


def test_apply_split_target_encoding_prevents_eval_leakage():
    """학습/평가 분할 시 기존 station_target_encoded를 폐기하고 학습셋만으로 재적합하여 평가셋 누출을 차단하는지 검증합니다."""
    train_rows = [
        {
            "station_id": 119,
            "date": date(2023, 6, 1 + d),
            "datetime": datetime(2023, 6, 1 + d, h),
            "accumulated_passengers": 100.0,
            "station_target_encoded": 99999.0,  # 오염된 사전 컬럼 주입
        }
        for d in range(6)
        for h in (8, 18)
    ]
    eval_rows = [
        {
            "station_id": 119,
            "date": date(2023, 6, 10),
            "datetime": datetime(2023, 6, 10, h),
            "accumulated_passengers": 500_000.0,  # 극단적 미래 이상치
            "station_target_encoded": 99999.0,
        }
        for h in (8, 18)
    ]
    train_df = pl.DataFrame(train_rows)
    eval_df = pl.DataFrame(eval_rows)

    tr_enc, ev_enc, encoder = _apply_split_target_encoding(train_df, eval_df, seed=42)
    assert ev_enc is not None
    # 평가셋의 극단값(500,000)과 기존 오염 컬럼(99,999)이 전혀 반영되지 않고 학습셋 평균(100.0)이어야 함
    assert encoder.global_mean == pytest.approx(100.0, rel=1e-4)
    assert encoder.station_map[119] == pytest.approx(100.0, rel=1e-4)
    assert (tr_enc["station_target_encoded"] < 200.0).all()
    assert ev_enc["station_target_encoded"].to_list() == pytest.approx(
        [100.0, 100.0], rel=1e-4
    )


def test_save_checkpoints_atomic_commits_manifest_last_and_cleans_up_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    isolated_sample_pipeline: Path,
):
    """체크포인트 원자적 교체 시 manifest가 최후 커밋되고, 스테이징 실패 시 기존 파일 보존 및 임시 파일이 정리되는지 검증합니다."""
    out_dir = tmp_path / "atomic_ckpt_out"
    res = run_pipeline(
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=out_dir,
        features_path=isolated_sample_pipeline,
    )
    ckpt_dir = out_dir / "checkpoints"
    paths = get_checkpoint_paths(ckpt_dir)
    manifest_payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    scaler_payload = joblib.load(paths["scaler"])

    from busan_metro_congestion.preprocess.subway import StationTargetEncoder

    encoder = StationTargetEncoder.load_mapping(paths["target_encoder"])

    # 1. os.replace 커밋 순서 검증: model_manifest.json이 반드시 마지막에 커밋되어야 함
    replaced_targets: list[str] = []
    real_replace = pipeline_mod.os.replace

    def _spy_replace(src: Any, dst: Any) -> None:
        replaced_targets.append(Path(dst).name)
        real_replace(src, dst)

    monkeypatch.setattr(pipeline_mod.os, "replace", _spy_replace)
    _save_checkpoints_atomic(
        ckpt_dir=ckpt_dir,
        lgb_model=res["models"]["lgbm"],
        rf_model=res["models"]["rf"],
        lstm_model=res["models"]["lstm"],
        scaler_payload=scaler_payload,
        target_encoder=encoder,
        manifest_payload=manifest_payload,
    )
    assert len(replaced_targets) == 6
    assert replaced_targets[-1] == "model_manifest.json"

    # 2. 스테이징 도중 예외 발생 시 기존 체크포인트 불변 및 임시 파일(.tmp.*) 0건 정리 검증
    monkeypatch.undo()
    original_snapshots = {k: p.read_bytes() for k, p in paths.items()}

    class _FailingEncoder:
        def save_mapping(self, path: Path | str) -> None:
            raise OSError("simulated staging failure")

    with pytest.raises(OSError, match="simulated staging failure"):
        _save_checkpoints_atomic(
            ckpt_dir=ckpt_dir,
            lgb_model=res["models"]["lgbm"],
            rf_model=res["models"]["rf"],
            lstm_model=res["models"]["lstm"],
            scaler_payload=scaler_payload,
            target_encoder=_FailingEncoder(),  # type: ignore[arg-type]
            manifest_payload=manifest_payload,
        )

    for key, orig_bytes in original_snapshots.items():
        assert paths[key].read_bytes() == orig_bytes
    assert list(ckpt_dir.glob("*.tmp*")) == []


def test_run_pipeline_default_metrics_path_unified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_sample_pipeline: Path
):
    """기본 출력 실행 시 메트릭이 artifacts/baseline/metrics.json에만 저장되고 최상위 중복 복사가 없는지 검증합니다."""
    fake_artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(pipeline_mod, "ARTIFACTS_DIR", fake_artifacts)

    run_pipeline(
        step="all",
        config_dict=FAST_TEST_CONFIG,
        output_dir=None,
        features_path=isolated_sample_pipeline,
    )

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

    with pytest.raises(ValueError, match="사용자 지정 날짜 범위에 데이터가 없습니다"):
        run_pipeline(
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

    captured_calls.clear()
    with pytest.raises(ValueError, match="사용자 지정 날짜 범위에 데이터가 없습니다"):
        run_pipeline(
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


def test_run_pipeline_custom_date_range_success_and_holdout_isolation(
    tmp_path: Path,
):
    """사용자 지정 train/val/test 기간 정상 학습 시 holdout 테스트 데이터가 배포 체크포인트에 혼입되지 않음을 검증합니다."""
    records: list[dict[str, Any]] = []
    # Train: 2023-06-01~2024-05-31, Val: 2024-06-01~2025-05-31, Test: 2025-06-01~2026-05-31
    sample_dates = [
        # Train (6일)
        date(2023, 6, 1),
        date(2023, 6, 2),
        date(2023, 8, 1),
        date(2023, 10, 1),
        date(2024, 1, 1),
        date(2024, 5, 31),
        # Val (4일)
        date(2024, 6, 1),
        date(2024, 9, 1),
        date(2025, 1, 1),
        date(2025, 5, 31),
        # Holdout Test (2일)
        date(2025, 6, 1),
        date(2026, 5, 31),
    ]
    for stn in (119, 203):
        for d in sample_dates:
            for h in (7, 8, 9, 18):
                rad = 2.0 * np.pi * (h % 24) / 24.0
                records.append(
                    {
                        "station_id": stn,
                        "line_num": 1 if stn == 119 else 2,
                        "datetime": datetime(d.year, d.month, d.day, h),
                        "date": d,
                        "hour": h,
                        "accumulated_passengers": 200.0
                        + float(stn % 10) * 15.0
                        + h * 5.0,
                        "station_target_encoded": 100.0,
                        "is_weekend": d.weekday() >= 5,
                        "is_holiday": False,
                        "is_holiday_eve": False,
                        "season": 2,
                        "hour_sin": float(np.sin(rad)),
                        "hour_cos": float(np.cos(rad)),
                        "month_sin": 0.0,
                        "month_cos": 1.0,
                        "weekday_sin": 0.0,
                        "weekday_cos": 1.0,
                    }
                )
    feat_path = tmp_path / "synthetic_3yr_operational.parquet"
    pl.DataFrame(records).with_columns(
        pl.col("station_id").cast(pl.Int64),
        pl.col("line_num").cast(pl.Int8),
        pl.col("hour").cast(pl.Int8),
        pl.col("season").cast(pl.Int8),
        pl.col("accumulated_passengers").cast(pl.Float32),
        pl.col("station_target_encoded").cast(pl.Float32),
    ).select(OPERATIONAL_MASTER_COLUMNS).sort(["station_id", "datetime"]).write_parquet(
        feat_path
    )

    out_dir = tmp_path / "custom_holdout_out"
    res = run_pipeline(
        step="train_eval",
        config_dict=FAST_TEST_CONFIG,
        output_dir=out_dir,
        features_path=feat_path,
        train_start="2023-06-01",
        val_start="2024-06-01",
        test_start="2025-06-01",
    )
    assert res["feature_set"] == "subway_calendar"
    manifest = json.loads(
        (out_dir / "checkpoints" / "model_manifest.json").read_text(encoding="utf-8")
    )
    # custom_range_used=True이므로 ho_train(Train+Val=10일×2역×4시간=80행)만 배포 모델에 사용, Test(2025-06-01~) 배제
    assert manifest["train_row_count"] == 10 * 2 * 4
    assert manifest["train_date_range"]["min_date"] == "2023-06-01"
    assert manifest["train_date_range"]["max_date"] == "2025-05-31"

    # test 기간에 데이터가 없는 경우 명시 실패 검증
    with pytest.raises(ValueError, match="사용자 지정 Test 날짜 범위"):
        run_pipeline(
            step="train_eval",
            config_dict=FAST_TEST_CONFIG,
            output_dir=tmp_path / "empty_test_out",
            features_path=feat_path,
            train_start="2023-06-01",
            val_start="2024-06-01",
            test_start="2027-06-01",
        )


def test_run_pipeline_invalid_arguments():
    """지원하지 않는 모드, step 또는 날짜 파라미터 오류 시 ValueError 검증"""
    with pytest.raises(ValueError, match="지원하지 않는 파이프라인 모드"):
        run_pipeline(mode="invalid_mode")

    with pytest.raises(ValueError, match="지원하지 않는 파이프라인 단계"):
        run_pipeline(step="invalid_step")

    with pytest.raises(ValueError, match="train_end를 지정하려면 train_start가 필요"):
        run_pipeline(train_end="2024-05-31")

    with pytest.raises(ValueError, match="val_end를 지정하려면 val_start가 필요"):
        run_pipeline(val_end="2025-05-31")

    with pytest.raises(ValueError, match="test_end를 지정하려면 test_start가 필요"):
        run_pipeline(test_end="2026-05-31")

    with pytest.raises(ValueError, match="train_start와 val_start가 모두 지정"):
        run_pipeline(train_start="2023-06-01")

    with pytest.raises(ValueError, match="train_start와 val_start가 모두 지정"):
        run_pipeline(val_start="2024-06-01")

    with pytest.raises(ValueError, match="1년 단위.*윤년 고려"):
        run_pipeline(
            train_start="2023-06-01",
            train_end="2023-11-30",
            val_start="2023-12-01",
        )

    with pytest.raises(ValueError, match="앞설 수 없습니다"):
        run_pipeline(
            train_start="2023-06-01",
            train_end="2023-05-31",
            val_start="2023-06-01",
        )


def test_main_cli_train_and_predict(tmp_path: Path, isolated_sample_pipeline: Path):
    """CLI main 엔트리포인트의 Set 1 train 및 predict 서브커맨드 디스패치 검증"""
    out_dir = tmp_path / "cli_artifacts"
    cfg_yaml = tmp_path / "fast_operational.yaml"
    cfg_yaml.write_text(
        (
            "project:\n"
            "  seed: 42\n"
            "  version: '0.1.0'\n"
            "pipeline:\n"
            "  step: all\n"
            "model:\n"
            "  lightgbm:\n"
            "    n_estimators: 10\n"
            "    final_n_estimators: 10\n"
            "  random_forest:\n"
            "    n_estimators: 10\n"
            "    final_n_estimators: 10\n"
            "    tune_max_depth: 5\n"
            "    max_depth: 5\n"
            "  lstm:\n"
            "    seq_len: 2\n"
            "    hidden_dim1: 16\n"
            "    hidden_dim2: 8\n"
            "    dense_dim: 8\n"
            "    tune_epochs: 2\n"
            "    final_epochs: 2\n"
            "    batch_size: 32\n"
        ),
        encoding="utf-8",
    )

    # 1. train 서브커맨드 CLI 호출 (Set 1 체크포인트 생성)
    res_train = main(
        [
            "train",
            "--config",
            str(cfg_yaml),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert res_train["feature_set"] == "subway_calendar"
    assert len(res_train["selected_features"]) == 13

    # 2. predict 서브커맨드 CLI 호출 (생성된 체크포인트 기반 배치 추론)
    pred_out = tmp_path / "predictions.parquet"
    manifest_out = tmp_path / "predict_run_manifest.json"
    res_pred = main(
        [
            "predict",
            "--start-date",
            "2026-12-01",
            "--days",
            "2",
            "--checkpoint-dir",
            str(out_dir / "checkpoints"),
            "--output-path",
            str(pred_out),
            "--manifest-out",
            str(manifest_out),
        ]
    )
    assert res_pred["mode"] == "predict"
    assert res_pred["status"] == "ok"
    assert res_pred["feature_set"] == "subway_calendar"
    assert res_pred["row_count"] == 114 * 2 * 24
    assert pred_out.exists()
    assert manifest_out.exists()
    validate_predictions_df(pl.read_parquet(pred_out))
    manifest_json = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest_json["track"] == "accurate"
    assert manifest_json["start_date"] == "2026-12-01"
    assert manifest_json["end_date"] == "2026-12-02"
    assert manifest_json["row_count"] == 114 * 2 * 24

    # 2-b. predict --track fast 및 output_dir 폴백 경로 검증
    fast_out_dir = tmp_path / "fast_dir_out"
    res_fast_dir = run_pipeline(
        mode="predict",
        output_dir=fast_out_dir,
        start_date="2026-12-01",
        days=1,
        track="fast",
        checkpoint_dir=out_dir / "checkpoints",
    )
    assert res_fast_dir["track"] == "fast"
    assert (fast_out_dir / "predictions.parquet").exists()
    validate_predictions_df(pl.read_parquet(fast_out_dir / "predictions.parquet"))

    # 2-c. CLI 상호 배타 옵션(--end-date + --days) 및 제거된 --horizon 옵션 거부 검증
    with pytest.raises(SystemExit) as exc_mutex:
        main(
            [
                "predict",
                "--start-date",
                "2026-12-01",
                "--end-date",
                "2026-12-02",
                "--days",
                "2",
            ]
        )
    assert exc_mutex.value.code == 2

    with pytest.raises(SystemExit) as exc_horizon:
        main(["train", "--horizon", "short"])
    assert exc_horizon.value.code == 2

    # 3. 서브커맨드 생략 시 기본 train 모드 디스패치 검증
    res_default = main(["--step", "preprocess"])
    assert "df_features" in res_default


def test_config_yaml_parameters_propagate_to_models_and_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_sample_pipeline: Path
):
    """config.yaml의 모델 하이퍼파라미터 및 pipeline.step 기본값이 실제 모델과 CLI에 반영되는지 검증합니다."""
    custom_cfg: dict[str, Any] = {
        "project": {"seed": 7},
        "pipeline": {"step": "preprocess"},
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
        step="all",
        config_dict=custom_cfg,
        output_dir=tmp_path / "cfg_prop_out",
        features_path=isolated_sample_pipeline,
    )
    lgb_model = res["models"]["lgbm"]
    rf_model = res["models"]["rf"]
    lstm_model = res["models"]["lstm"]

    assert lgb_model.get_params()["n_estimators"] == 12
    assert lgb_model.get_params()["learning_rate"] == pytest.approx(0.08)
    assert lgb_model.get_params()["num_leaves"] == 15
    assert lgb_model.get_params()["n_jobs"] == 2
    assert rf_model.get_params()["n_estimators"] == 11
    assert rf_model.get_params()["max_depth"] == 5
    assert rf_model.get_params()["n_jobs"] == 2
    assert lstm_model.seq_len == 2
    assert lstm_model.hidden_dim1 == 16
    assert lstm_model.hidden_dim2 == 8
    assert lstm_model.dense_dim == 8
    assert lstm_model.dropout_rate == pytest.approx(0.15)
    assert lstm_model.lr == pytest.approx(0.002)
    assert lstm_model.batch_size == 32

    cfg_yaml = tmp_path / "custom_pipeline.yaml"
    cfg_yaml.write_text(
        "project:\n  seed: 7\npipeline:\n  step: preprocess\n",
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
    assert captured_cli_kwargs["step"] == "preprocess"
    assert captured_cli_kwargs["config_dict"]["project"]["seed"] == 7
