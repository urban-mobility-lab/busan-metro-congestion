"""
MetroLSTMEstimator scikit-learn 호환성 단위 테스트
"""

import numpy as np
import pytest
from sklearn.base import clone

from busan_metro_congestion.modeling.lstm_net import MetroLSTMEstimator


@pytest.fixture
def mock_tabular_data():
    """테스트용 합성 시계열 (N=60, F=4) 데이터"""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.normal(0, 0.1, 60)).astype(np.float32)
    return X, y


def test_lstm_estimator_sklearn_contract(mock_tabular_data):
    """scikit-learn BaseEstimator 규격 (get_params, clone, fit, predict) 검증"""
    X, y = mock_tabular_data

    est = MetroLSTMEstimator(
        seq_len=2,
        hidden_dim1=16,
        hidden_dim2=8,
        dense_dim=8,
        epochs=3,
        batch_size=16,
        random_state=42,
    )

    # 1. get_params / clone 검증
    cloned = clone(est)
    assert cloned.seq_len == 2
    assert cloned.hidden_dim1 == 16

    # 2. fit & predict 검증
    cloned.fit(X, y)
    assert getattr(cloned, "is_fitted_", False) is True

    preds = cloned.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),), (
        f"예측값 차원이 입력 차원({len(X)})과 일치해야 합니다."
    )
    assert not np.isnan(preds).any()


def test_lstm_estimator_with_station_ids():
    """역사 ID 경계가 있는 데이터셋에서의 학습 및 추론 검증"""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3)).astype(np.float32)
    y = rng.uniform(50, 150, 50).astype(np.float32)
    station_ids = np.array([101] * 25 + [102] * 25)

    est = MetroLSTMEstimator(
        seq_len=3,
        hidden_dim1=16,
        hidden_dim2=8,
        dense_dim=8,
        epochs=2,
        batch_size=16,
        random_state=42,
    )
    est.fit(X, y, station_ids=station_ids)
    preds = est.predict(X, station_ids=station_ids)

    assert preds.shape == (50,)
    assert not np.isnan(preds).any()
    assert len(np.unique(preds)) > 2
    assert (preds[0:3] == preds[3]).all()
    assert (preds[25:28] == preds[28]).all()


def test_lstm_estimator_unfitted_raises():
    """fit되지 않은 상태에서 predict 호출 시 예외 발생 검증"""
    est = MetroLSTMEstimator()
    with pytest.raises(RuntimeError, match="fit"):
        est.predict(np.zeros((10, 4)))
