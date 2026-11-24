"""
2-Track 베이스라인 모델 및 평가 지표 단위 테스트
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from busan_metro_congestion.modeling.baseline import (
    WEATHER_SHORT_FEATURES,
    SafeRandomForestRegressor,
    calculate_end_date,
    generate_rolling_tracks,
    get_feature_columns,
    split_by_date_range,
    split_rolling_time_series,
    split_time_series,
    train_lightgbm,
    train_random_forest,
    validate_date_interval_yearly,
)
from busan_metro_congestion.modeling.metrics import (
    calculate_mae,
    calculate_peak_hour_mae,
    calculate_pearson_r,
    calculate_r2,
    calculate_rmse,
    calculate_wape,
    evaluate_predictions,
)


def create_mock_features_df(
    n_days: int = 2, stations: list[int] | None = None
) -> pl.DataFrame:
    """
    Data Contract(MASTER_FEATURE_COLUMNS 25종 및 메타 컬럼)를 완벽히 준수하는
    인메모리 Mock Polars DataFrame을 생성합니다.
    (station_id, datetime) 기준 정렬을 보장합니다.
    """
    if stations is None:
        stations = [101, 102, 103, 104, 105]

    n_hours = n_days * 24
    base_dt = datetime(2023, 6, 1, 0, 0)
    records = []

    for stn in stations:
        for i in range(n_hours):
            dt = base_dt + timedelta(hours=i)
            records.append(
                {
                    "station_id": stn,
                    "datetime": dt,
                    "date": dt.date(),
                    "hour": dt.hour,
                    "line_num": 1,
                    "station_target_encoded": 100.0 + float(stn % 10) * 10.0,
                    "nearest_event_rank": 1.0,
                    "max_dist_weight": 0.5,
                    "max_scale_weight": 0.5,
                    "event_impact_score": 0.25,
                    "season": 2,
                    "is_weekend": 0,
                    "is_holiday": 0,
                    "is_holiday_eve": 0,
                    "hour_sin": float(np.sin(2 * np.pi * dt.hour / 24)),
                    "hour_cos": float(np.cos(2 * np.pi * dt.hour / 24)),
                    "month_sin": float(np.sin(2 * np.pi * dt.month / 12)),
                    "month_cos": float(np.cos(2 * np.pi * dt.month / 12)),
                    "weekday_sin": float(np.sin(2 * np.pi * dt.weekday() / 7)),
                    "weekday_cos": float(np.cos(2 * np.pi * dt.weekday() / 7)),
                    "lead_time_hours": 1.0,
                    "precipitation": 0.0,
                    "pop": 10.0,
                    "temperature": 20.0,
                    "rainfall_type": 0,
                    "humidity": 60.0,
                    "wind_speed": 2.5,
                    "sky_condition": 1,
                    "accumulated_passengers": (
                        50.0
                        + 30.0 * float(np.sin(dt.hour / 4.0))
                        + float(stn % 10) * 5.0
                    ),
                }
            )

    df = pl.DataFrame(records)
    return df.sort(["station_id", "datetime"])


def test_metrics_calculation():
    """평가 지표(MAE, RMSE, R², Pearson r, WAPE, Peak MAE)의 정확성과 안정성을 검증합니다."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0, 52.0])

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    pearson = calculate_pearson_r(y_true, y_pred)

    assert pytest.approx(mae, abs=1e-2) == 2.0
    assert pytest.approx(rmse, abs=1e-2) == 2.0976
    assert pytest.approx(r2, abs=1e-2) == 0.9780
    assert pytest.approx(pearson, abs=1e-2) == 0.9910
    assert -1.0 <= pearson <= 1.0

    # 완전 일치 시 지표 불변식 검증 (MAE=0, RMSE=0, R2=1, r=1, WAPE=0)
    assert calculate_mae(y_true, y_true) == 0.0
    assert calculate_rmse(y_true, y_true) == 0.0
    assert pytest.approx(calculate_r2(y_true, y_true), abs=1e-5) == 1.0
    assert pytest.approx(calculate_pearson_r(y_true, y_true), abs=1e-5) == 1.0
    assert calculate_wape(y_true, y_true) == 0.0

    # 역방향 상관관계(-1.0) 검증
    assert (
        pytest.approx(
            calculate_pearson_r(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])),
            abs=1e-5,
        )
        == -1.0
    )

    # edge case: 분모 0 방어
    const_arr = np.array([10.0, 10.0, 10.0])
    assert calculate_pearson_r(const_arr, const_arr) == 0.0
    assert calculate_wape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0

    # Polars Series 입력 호환성 검증
    s_true = pl.Series("y_true", y_true)
    s_pred = pl.Series("y_pred", y_pred)
    assert pytest.approx(calculate_mae(s_true, s_pred), abs=1e-2) == 2.0
    assert pytest.approx(calculate_pearson_r(s_true, s_pred), abs=1e-2) == 0.9910

    eval_dict = evaluate_predictions(y_true, y_pred, hours=np.array([7, 8, 9, 12, 18]))
    assert {"mae", "rmse", "r2", "pearson_r", "wape", "peak_mae"}.issubset(
        set(eval_dict.keys())
    )
    # WAPE = (10 / 150) * 100 = 6.6667%
    assert pytest.approx(eval_dict["wape"], abs=1e-2) == 6.67
    # peak hours: 7, 8, 9, 18 -> y_true: [10, 20, 30, 50], y_pred: [12, 18, 33, 52] -> abs diff: [2, 2, 3, 2] -> mean: 2.25
    assert pytest.approx(eval_dict["peak_mae"], abs=1e-2) == 2.25

    # hours=None 또는 길이 불일치 시 fallback 검증
    assert calculate_peak_hour_mae(y_true, y_pred, hours=None) == mae
    assert calculate_peak_hour_mae(y_true, y_pred, hours=np.array([1, 2])) == mae


def test_get_feature_columns_tracks():
    """2-Track 호라이즌 피처 컬럼 분기 및 Weather-Free 여부를 검증합니다."""
    short_cols = get_feature_columns("short")
    long_cols = get_feature_columns("long")

    # Short Track에는 기상 8종 포함
    for wf in WEATHER_SHORT_FEATURES:
        assert wf in short_cols

    # Long Track에는 기상 피처 전면 제외 (Weather-Free)
    for wf in WEATHER_SHORT_FEATURES:
        assert wf not in long_cols

    assert len(short_cols) == 25
    assert len(long_cols) == 17
    assert set(short_cols) - set(long_cols) == set(WEATHER_SHORT_FEATURES)

    with pytest.raises(ValueError, match="지원하지 않는 예측 호라이즌입니다"):
        get_feature_columns("ultra_short")


def test_split_time_series_leakage_free():
    """시간 순서 보존 분할을 통한 시계열 데이터 누출 차단을 검증합니다."""
    base_time = datetime(2023, 6, 1, 0, 0)
    times = [base_time + timedelta(hours=i) for i in range(100)]
    df = pl.DataFrame(
        {
            "datetime": times,
            "accumulated_passengers": [float(i) for i in range(100)],
        }
    )

    train_df, test_df = split_time_series(df, test_ratio=0.2)

    assert len(train_df) == 80
    assert len(test_df) == 20

    max_train_time = train_df["datetime"].max()
    min_test_time = test_df["datetime"].min()
    assert max_train_time < min_test_time


def test_split_time_series_multi_station_preservation():
    """(station_id, datetime) 정렬 테이블에서 모든 역사의 기간이 공평하게 분할되고 정렬 순서가 보존되는지 검증합니다."""
    base_time = datetime(2023, 6, 1, 0, 0)
    times = [base_time + timedelta(hours=i) for i in range(10)]
    # 역 101번 10시간, 역 102번 10시간 (총 20행, (station_id, datetime) 정렬 상태)
    stn_ids = [101] * 10 + [102] * 10
    all_times = times + times
    df = pl.DataFrame(
        {
            "station_id": stn_ids,
            "datetime": all_times,
            "accumulated_passengers": [float(i) for i in range(20)],
        }
    )

    train_df, test_df = split_time_series(df, test_ratio=0.2)

    # 10시간 중 8시간(80%) Train, 2시간(20%) Test -> 각 역마다 8개, 2개 (총 16개, 4개)
    assert len(train_df) == 16
    assert len(test_df) == 4
    assert train_df["station_id"].to_list() == [101] * 8 + [102] * 8
    assert test_df["station_id"].to_list() == [101] * 2 + [102] * 2


def test_split_by_date_range_leakage_free():
    """시계열 롤링 검증을 위한 날짜 범위 엄밀 분할을 검증합니다."""
    from datetime import date

    dates = [date(2023, 6, 1) + timedelta(days=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "date": dates,
            "accumulated_passengers": [float(i * 10) for i in range(10)],
        }
    )

    train_df, val_df = split_by_date_range(
        df,
        train_start="2023-06-01",
        train_end="2023-06-07",
        val_start="2023-06-08",
        val_end="2023-06-10",
        date_col="date",
    )

    assert len(train_df) == 7
    assert len(val_df) == 3
    assert train_df["date"].max() < val_df["date"].min()

    # 문자열 타입 날짜 컬럼 자동 변환 검증
    df_str = df.with_columns(pl.col("date").cast(pl.Utf8))
    train_str, val_str = split_by_date_range(
        df_str,
        train_start="2023-06-01",
        train_end="2023-06-07",
        val_start="2023-06-08",
        val_end="2023-06-10",
        date_col="date",
    )
    assert len(train_str) == 7
    assert len(val_str) == 3


def test_lstm_dataset_cross_station_isolation():
    """역사 경계 침범 방지(Cross-Station Mixing 방지)를 검증합니다."""
    from busan_metro_congestion.modeling.lstm_net import (
        MetroLSTM,
        MetroSlidingWindowDataset,
    )

    # 역 101번 5개 시간대, 역 102번 5개 시간대 (총 10개)
    station_ids = np.array([101, 101, 101, 101, 101, 102, 102, 102, 102, 102])
    X = np.ones((10, 4), dtype=np.float32)
    y = np.arange(10, dtype=np.float32)
    seq_len = 3

    dataset = MetroSlidingWindowDataset(X, y, seq_len=seq_len, station_ids=station_ids)

    # 역 101에서 유효 인덱스: 0, 1 (idx 0 -> 0..2(x), 3(y), idx 1 -> 1..3(x), 4(y))
    # idx 2 -> 2..4(x), 5(y: 102번 역!) -> 경계 침범이므로 제외되어야 함!
    # 역 102에서 유효 인덱스: 5, 6
    # 총 유효 윈도우 수는 4개여야 함
    assert len(dataset) == 4

    seq_x, target_y = dataset[0]
    assert seq_x.shape == (3, 4)
    assert float(target_y) == 3.0

    # station_ids 미제공 시 기본 순차 슬라이딩 윈도우 검증 (10 - 3 = 7개)
    dataset_no_stn = MetroSlidingWindowDataset(X, y, seq_len=seq_len)
    assert len(dataset_no_stn) == 7

    # 시퀀스 길이 이하의 짧은 데이터 입력 시 빈 데이터셋(len=0) 안전 반환 검증
    dataset_short = MetroSlidingWindowDataset(X[:2], y[:2], seq_len=seq_len)
    assert len(dataset_short) == 0

    # MetroLSTM 순전파 검증
    model = MetroLSTM(input_dim=4, hidden_dim1=16, hidden_dim2=8, dense_dim=4)
    out = model(seq_x.unsqueeze(0))
    assert out.shape == (1,)


def test_lstm_dataset_sample_features_integration():
    """Data Contract 규격 Mock 피처(240행, 5개역)로 LSTM 슬라이딩 윈도우 생성 및 역사 격리를 검증합니다."""
    from busan_metro_congestion.modeling.lstm_net import (
        MetroLSTM,
        MetroSlidingWindowDataset,
    )

    df_features = create_mock_features_df()

    cols = get_feature_columns("short")
    X = df_features.select(cols).to_numpy()
    y = df_features["accumulated_passengers"].to_numpy()
    station_ids = df_features["station_id"].to_numpy()
    seq_len = 3

    dataset = MetroSlidingWindowDataset(X, y, seq_len=seq_len, station_ids=station_ids)

    # 5개 역 * (48시간 - 3) = 225개 윈도우 정확히 생성 검증
    assert len(dataset) == 225

    # 정렬 순서가 ["datetime", "station_id"]로 깨졌을 때 교차 역사 윈도우가 전부 차단되는지 검증
    df_broken = df_features.sort(["datetime", "station_id"])
    ds_broken = MetroSlidingWindowDataset(
        df_broken.select(cols).to_numpy(),
        df_broken["accumulated_passengers"].to_numpy(),
        seq_len=seq_len,
        station_ids=df_broken["station_id"].to_numpy(),
    )
    assert len(ds_broken) == 0

    # 첫 번째 윈도우 형상 및 결측 부재 검증
    seq_x, target_y = dataset[0]
    assert seq_x.shape == (3, len(cols))
    assert np.isnan(seq_x.numpy()).sum() == 0
    assert not np.isnan(float(target_y))

    # 모든 유효 인덱스에 대해 시작 역과 타깃 역이 동일한지(역사 경계 침범 차단) 전수 검증
    for real_idx in dataset.valid_indices:
        assert station_ids[real_idx] == station_ids[real_idx + seq_len]

    # 실제 피처 차원으로 MetroLSTM 순전파 연산 성공 검증
    model = MetroLSTM(input_dim=len(cols), hidden_dim1=16, hidden_dim2=8, dense_dim=4)
    out = model(seq_x.unsqueeze(0))
    assert out.shape == (1,)
    assert not np.isnan(float(out.detach().squeeze()))


def test_lstm_training_and_prediction():
    """MetroLSTM 학습 루프의 손실 감소, Early Stopping 조기 종료 및 배치 추론을 검증합니다."""
    import torch
    from torch.utils.data import DataLoader

    from busan_metro_congestion.modeling.lstm_net import (
        MetroLSTM,
        MetroSlidingWindowDataset,
        predict_lstm,
        train_lstm_model,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    X = np.random.randn(40, 4).astype(np.float32)
    y = (X[:, 0] * 3.0 - X[:, 1] * 2.0).astype(np.float32)

    dataset = MetroSlidingWindowDataset(X, y, seq_len=3)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 1. Early Stopping 실제 발동 검증: lr=0.0으로 검증 손실 개선이 멈출 때 patience=2 후 3에폭에서 조기 종료
    model_es = MetroLSTM(
        input_dim=4, hidden_dim1=8, hidden_dim2=4, dense_dim=4, dropout_rate=0.0
    )
    es_res = train_lstm_model(
        model_es, loader, val_loader=loader, epochs=15, lr=0.0, patience=2
    )
    assert "model" in es_res
    assert len(es_res["val_losses"]) == 3
    assert len(es_res["train_losses"]) == 3
    assert es_res["best_val_loss"] == pytest.approx(es_res["val_losses"][0], rel=1e-6)

    # 2. 정상 학습 시 손실 감소 및 배치 추론 무결성 검증
    model_train = MetroLSTM(
        input_dim=4, hidden_dim1=16, hidden_dim2=8, dense_dim=8, dropout_rate=0.0
    )
    train_res = train_lstm_model(
        model_train, loader, val_loader=loader, epochs=15, lr=2e-2, patience=15
    )
    assert train_res["train_losses"][-1] < train_res["train_losses"][0]

    preds = predict_lstm(train_res["model"], loader)
    assert len(preds) == len(dataset)
    assert np.isnan(preds).sum() == 0
    assert np.std(preds) > 0.0


def test_ensemble_blending():
    """Two-Track 앙상블 (Fast Track 및 Accurate Track) 가중치 블렌딩 및 최적화 동작을 검증합니다."""
    from busan_metro_congestion.modeling.ensemble import (
        blend_accurate_track,
        blend_fast_track,
        optimize_accurate_track_weights,
        optimize_fast_track_weights,
    )

    pred_rf = np.array([10.0, 20.0, 30.0])
    pred_lgb = np.array([12.0, 18.0, 32.0])
    pred_lstm = np.array([11.0, 19.0, 31.0])
    y_true = np.array([11.0, 19.0, 31.0])

    # 1. Fast Track 단독 앙상블 (RF + LightGBM)
    fast_blended = blend_fast_track(pred_rf, pred_lgb, w_rf=0.3, w_lgb=0.7)
    assert len(fast_blended) == 3
    # 수치 계산 정확성 검증: 0.3*10 + 0.7*12 = 11.4, 0.3*20 + 0.7*18 = 18.6, 0.3*30 + 0.7*32 = 31.4
    np.testing.assert_allclose(fast_blended, [11.4, 18.6, 31.4])

    # Fast Track 가중치 불일치 예외
    with pytest.raises(ValueError, match="Fast Track 가중치의 합은 1.0"):
        blend_fast_track(pred_rf, pred_lgb, w_rf=0.5, w_lgb=0.6)

    # 배열 길이 불일치 예외
    with pytest.raises(ValueError, match="길이가 일치하지 않습니다"):
        blend_fast_track(np.array([10.0, 20.0]), np.array([10.0]))

    # 빈 배열 입력 안전 처리
    assert len(blend_fast_track(np.array([]), np.array([]))) == 0

    # Fast Track 가중치 최적화: 초기값(0.9, 0.1)에서 멀리 떨어진 최적해(0.0, 1.0)로 실제 수렴 검증
    bad_rf = y_true + 50.0
    perfect_lgb = y_true.copy()
    best_fast_w = optimize_fast_track_weights(
        y_true, bad_rf, perfect_lgb, initial_weights=(0.9, 0.1)
    )
    assert pytest.approx(sum(best_fast_w), abs=1e-3) == 1.0
    assert best_fast_w[0] >= 0.0 and best_fast_w[1] >= 0.0
    assert best_fast_w[1] > 0.95
    opt_fast_mae = calculate_mae(
        y_true, blend_fast_track(bad_rf, perfect_lgb, *best_fast_w)
    )
    init_fast_mae = calculate_mae(
        y_true, blend_fast_track(bad_rf, perfect_lgb, 0.9, 0.1)
    )
    assert opt_fast_mae < 1.0
    assert opt_fast_mae < init_fast_mae

    # 2. Accurate Track 종합 앙상블 (RF + LightGBM + LSTM)
    acc_blended = blend_accurate_track(pred_rf, pred_lgb, pred_lstm, 0.2, 0.4, 0.4)
    assert len(acc_blended) == 3
    # 수치 계산 정확성 검증: 0.2*10 + 0.4*12 + 0.4*11 = 11.2, 0.2*20 + 0.4*18 + 0.4*19 = 18.8, 0.2*30 + 0.4*32 + 0.4*31 = 31.2
    np.testing.assert_allclose(acc_blended, [11.2, 18.8, 31.2])

    # Accurate Track 가중치 불일치 예외
    with pytest.raises(ValueError, match="Accurate Track 가중치의 합은 1.0"):
        blend_accurate_track(pred_rf, pred_lgb, pred_lstm, 0.5, 0.5, 0.5)

    # Accurate Track 배열 길이 불일치 예외
    with pytest.raises(ValueError, match="길이가 일치하지 않습니다"):
        blend_accurate_track(np.array([10.0, 20.0]), np.array([10.0]), np.array([10.0]))

    # Accurate Track 빈 배열 입력 안전 처리
    assert len(blend_accurate_track(np.array([]), np.array([]), np.array([]))) == 0

    # Accurate Track 가중치 최적화: 초기값(0.8, 0.1, 0.1)에서 최적 모델(LSTM) 가중치로 수렴 검증
    bad_lgb = y_true + 25.0
    perfect_lstm = y_true.copy()
    best_acc_w = optimize_accurate_track_weights(
        y_true, bad_rf, bad_lgb, perfect_lstm, initial_weights=(0.8, 0.1, 0.1)
    )
    assert pytest.approx(sum(best_acc_w), abs=1e-3) == 1.0
    assert all(w >= 0.0 for w in best_acc_w)
    assert best_acc_w[2] > 0.95
    opt_acc_mae = calculate_mae(
        y_true, blend_accurate_track(bad_rf, bad_lgb, perfect_lstm, *best_acc_w)
    )
    init_acc_mae = calculate_mae(
        y_true, blend_accurate_track(bad_rf, bad_lgb, perfect_lstm, 0.8, 0.1, 0.1)
    )
    assert opt_acc_mae < 1.0
    assert opt_acc_mae < init_acc_mae


def test_ensemble_pipeline_sample_integration():
    """Data Contract 기반 전체 파이프라인에서 RF, LightGBM, LSTM의 예측 길이가 50개로 일치하고 앙상블 블렌딩되는지 검증합니다."""
    from torch.utils.data import DataLoader

    from busan_metro_congestion.modeling.ensemble import (
        blend_accurate_track,
        blend_fast_track,
        optimize_accurate_track_weights,
    )
    from busan_metro_congestion.modeling.lstm_net import (
        MetroLSTM,
        MetroSlidingWindowDataset,
        predict_lstm,
        prepare_lstm_inference_dataframe,
        train_lstm_model,
    )

    df_features = create_mock_features_df()
    train_df, test_df = split_time_series(df_features, test_ratio=0.2)
    cols = get_feature_columns("short")
    X_train = np.ascontiguousarray(train_df.select(cols).to_numpy(), dtype=np.float32)
    y_train = np.ascontiguousarray(
        train_df["accumulated_passengers"].to_numpy(), dtype=np.float32
    )
    X_test = np.ascontiguousarray(test_df.select(cols).to_numpy(), dtype=np.float32)
    y_test = np.ascontiguousarray(
        test_df["accumulated_passengers"].to_numpy(), dtype=np.float32
    )

    # 1. Fast Track (RF, LightGBM) 학습 및 50개 테스트 예측 산출
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=10,
        max_depth=5,
        random_state=42,
    )
    lgb_model = train_lightgbm(
        X_train,
        y_train,
        n_estimators=10,
        num_leaves=15,
        random_state=42,
    )
    pred_rf = np.clip(rf_model.predict(X_test), 0.0, None)
    pred_lgb = np.clip(lgb_model.predict(X_test), 0.0, None)
    assert len(pred_rf) == 50 and len(pred_lgb) == 50

    # 2. Fast Track 단독 앙상블 블렌딩 (길이 50개 정상 산출)
    blended_fast = blend_fast_track(pred_rf, pred_lgb, w_rf=0.3, w_lgb=0.7)
    assert len(blended_fast) == 50
    assert np.isnan(blended_fast).sum() == 0

    # 3. Accurate Track (LSTM): 테스트 구간 50개를 모두 예측하기 위한 Lookback 버퍼(마지막 3시간) 연결
    seq_len = 3
    eval_df = prepare_lstm_inference_dataframe(
        test_df, lookback_df=train_df, seq_len=seq_len
    )

    ds_train = MetroSlidingWindowDataset(
        train_df.select(cols).to_numpy(),
        train_df["accumulated_passengers"].to_numpy(),
        seq_len=seq_len,
        station_ids=train_df["station_id"].to_numpy(),
    )
    ds_eval = MetroSlidingWindowDataset(
        eval_df.select(cols).to_numpy(),
        eval_df["accumulated_passengers"].to_numpy(),
        seq_len=seq_len,
        station_ids=eval_df["station_id"].to_numpy(),
    )
    assert len(ds_eval) == 50  # Lookback 연결로 50개 타깃과 1:1 완벽 정렬

    train_loader = DataLoader(ds_train, batch_size=16, shuffle=False)
    eval_loader = DataLoader(ds_eval, batch_size=16, shuffle=False)
    lstm_model = MetroLSTM(
        input_dim=len(cols), hidden_dim1=16, hidden_dim2=8, dense_dim=4
    )
    train_res = train_lstm_model(lstm_model, train_loader, epochs=2)
    pred_lstm = predict_lstm(train_res["model"], eval_loader)
    assert len(pred_lstm) == 50

    # 4. Accurate Track 종합 앙상블 (RF + LightGBM + LSTM) 크래시 없이 50개 블렌딩 성공 검증
    blended_acc = blend_accurate_track(
        pred_rf, pred_lgb, pred_lstm, w_rf=0.2, w_lgb=0.4, w_lstm=0.4
    )
    assert len(blended_acc) == 50
    assert np.isnan(blended_acc).sum() == 0
    assert (blended_acc >= 0).all()

    # 5. 가중치 최적화(NNLS) 정상 수렴 검증
    best_weights = optimize_accurate_track_weights(y_test, pred_rf, pred_lgb, pred_lstm)
    assert pytest.approx(sum(best_weights), abs=1e-3) == 1.0


def test_prepare_lstm_inference_dataframe():
    """prepare_lstm_inference_dataframe 헬퍼 함수의 역사별 Lookback 버퍼 정합 동작을 검증합니다."""
    from busan_metro_congestion.modeling.lstm_net import (
        prepare_lstm_inference_dataframe,
    )

    df_features = create_mock_features_df(n_days=2, stations=[101, 102])
    train_df, test_df = split_time_series(df_features, test_ratio=0.2)

    # 1. lookback_df가 제공되었을 때: 각 역사별 마지막 3개 행(tail)이 test_df 앞에 정확히 부착됨
    res_with_lb = prepare_lstm_inference_dataframe(
        test_df, lookback_df=train_df, seq_len=3
    )
    assert len(res_with_lb) == len(test_df) + (2 * 3)
    assert res_with_lb.columns == test_df.columns
    for stn in (101, 102):
        expected_lb_dts = (
            train_df.filter(pl.col("station_id") == stn)
            .sort("datetime")
            .tail(3)["datetime"]
            .to_list()
        )
        expected_test_dts = (
            test_df.filter(pl.col("station_id") == stn)
            .sort("datetime")["datetime"]
            .to_list()
        )
        actual_dts = res_with_lb.filter(pl.col("station_id") == stn)[
            "datetime"
        ].to_list()
        assert actual_dts == expected_lb_dts + expected_test_dts

    # 2. lookback_df가 None이거나 비어있을 때: 원본 test_df 정렬 반환
    res_none = prepare_lstm_inference_dataframe(test_df, lookback_df=None)
    assert len(res_none) == len(test_df)

    empty_df = test_df.filter(pl.col("station_id") == 9999)
    res_empty = prepare_lstm_inference_dataframe(test_df, lookback_df=empty_df)
    assert len(res_empty) == len(test_df)


def test_safe_random_forest_regressor():
    """SafeRandomForestRegressor의 float32 연속 메모리 보장 및 청크 분할 추론 검증."""
    rng = np.random.default_rng(42)
    # 1. Fortran-order (non-contiguous) float64 배열 생성 (SIMD 충돌 유발 조건 모의)
    X_train_f = np.asfortranarray(rng.standard_normal((100, 5), dtype=np.float64))
    y_train_f = np.asfortranarray(rng.standard_normal((100,), dtype=np.float64))

    X_test_f = np.asfortranarray(rng.standard_normal((50, 5), dtype=np.float64))

    # train_random_forest가 SafeRandomForestRegressor를 반환하는지 검증
    model = train_random_forest(X_train_f, y_train_f, n_estimators=5, max_depth=3)
    assert isinstance(model, SafeRandomForestRegressor)

    # 청크 분할(chunk_size=10) 추론과 일반 추론 결과의 일치 여부 및 float32 반환 검증
    preds_chunked = model.predict(X_test_f, chunk_size=10)
    preds_direct = model.predict(X_test_f, chunk_size=100)

    assert preds_chunked.dtype == np.float32
    assert preds_direct.dtype == np.float32
    assert np.allclose(preds_chunked, preds_direct, atol=1e-5)


def test_date_utilities_leap_year():
    """윤년을 고려한 1년 종료일 계산 및 1년 단위 검증 함수 테스트."""
    # 1. 윤년 포함 1년 계산 (2023-06-01 -> 2024-05-31: 2024년 2월 29일 윤일 포함 366일)
    assert calculate_end_date("2023-06-01", 1) == "2024-05-31"

    # 2. 평년 1년 계산 (2024-03-01 -> 2025-02-28: 365일)
    assert calculate_end_date("2024-03-01", 1) == "2025-02-28"

    # 3. 윤일 시작 1년 계산 (2024-02-29 -> 2025-02-28: 366일)
    assert calculate_end_date("2024-02-29", 1) == "2025-02-28"

    # 4. 2년 기간 계산
    assert calculate_end_date("2023-06-01", 2) == "2025-05-31"

    # 5. validate_date_interval_yearly 정상 판정 검증
    assert validate_date_interval_yearly("2023-06-01", "2024-05-31") == 1
    assert validate_date_interval_yearly("2023-06-01", "2025-05-31") == 2

    # 6. 1년 단위 미충족 예외 검증
    with pytest.raises(ValueError, match="1년 단위.*윤년 고려"):
        validate_date_interval_yearly("2023-06-01", "2023-12-31")

    with pytest.raises(ValueError, match="1년 단위.*윤년 고려"):
        validate_date_interval_yearly("2023-06-01", "2024-06-01")  # 하루 초과

    # 7. 종료일이 시작일보다 앞선 경우 예외 검증
    with pytest.raises(ValueError, match="앞설 수 없습니다"):
        validate_date_interval_yearly("2023-06-01", "2022-06-01")


def test_generate_rolling_tracks_and_splits():
    """동적 롤링 트랙 생성 및 split_rolling_time_series 파라미터 제어 검증."""
    # 1. 기본 3-Track, 1M shift 생성 검증
    tracks, holdout = generate_rolling_tracks(
        n_tracks=3, shift_months=1, base_train_start="2023-06-01"
    )
    assert len(tracks) == 3
    assert tracks[0]["train_start"] == "2023-06-01"
    assert tracks[0]["train_end"] == "2024-05-31"
    assert tracks[0]["val_start"] == "2024-06-01"
    assert tracks[0]["val_end"] == "2025-05-31"

    assert tracks[1]["train_start"] == "2023-07-01"
    assert tracks[1]["val_end"] == "2025-06-30"

    assert tracks[2]["train_start"] == "2023-08-01"
    assert tracks[2]["val_end"] == "2025-07-31"

    # 마지막 val_end(2025-07-31) 바로 다음 날 test_start 시작
    assert holdout["test_start"] == "2025-08-01"
    assert holdout["test_end"] == "2026-07-31"

    # 2. 2-Track, 2M shift 생성 검증
    tracks_2m, holdout_2m = generate_rolling_tracks(
        n_tracks=2, shift_months=2, base_train_start="2023-06-01"
    )
    assert len(tracks_2m) == 2
    assert holdout_2m["test_start"] == "2025-08-01"
    assert holdout_2m["test_end"] == "2026-07-31"

    # 3. Track 수 범위 초과 (최대 3) 예외 검증
    with pytest.raises(ValueError, match="Track 수는 1 이상 3 이하"):
        generate_rolling_tracks(n_tracks=4)

    with pytest.raises(ValueError, match="Track 수는 1 이상 3 이하"):
        generate_rolling_tracks(n_tracks=0)


def test_split_rolling_time_series():
    """3-Track 롤링 슬라이딩 윈도우 + Holdout Test 분할 및 데이터 누출 방지 검증."""
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2026, 8, 31)
    n_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    rng = np.random.default_rng(42)

    mock_multi_year_df = pl.DataFrame(
        {
            "date": [d.date() for d in dates],
            "datetime": dates,
            "station_id": [101] * n_days,
            "accumulated_passengers": rng.integers(50, 200, size=n_days),
        }
    )

    res = split_rolling_time_series(mock_multi_year_df, strategy="rolling")

    assert res["strategy"] == "rolling"
    assert len(res["validation_splits"]) == 3

    expected_names = ["Track 1", "Track 2", "Track 3"]
    for i, track in enumerate(res["validation_splits"]):
        assert track["name"] == expected_names[i]
        tr_df = track["train_df"]
        val_df = track["val_df"]
        assert len(tr_df) > 0
        assert len(val_df) > 0
        assert tr_df["date"].max() < val_df["date"].min()

    ho = res["holdout_test"]
    assert len(ho["train_df"]) > 0
    assert len(ho["test_df"]) > 0
    assert ho["train_df"]["date"].max() < ho["test_df"]["date"].min()
    assert ho["train_df"]["date"].min() == datetime(2023, 8, 1).date()
    assert ho["train_df"]["date"].max() == datetime(2025, 7, 31).date()
    assert ho["test_df"]["date"].min() == datetime(2025, 8, 1).date()
