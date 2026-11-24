"""
Accurate Track: PyTorch 기반 시계열 LSTM 네트워크 및 Dataset

역사별 독립 시계열 원칙을 준수하여 역사 경계 침범(Cross-Station Mixing)을 차단하고,
2계층 LSTM 네트워크 및 조기 종료(Early Stopping) 학습 루프를 제공합니다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, Dataset

from busan_metro_congestion.config import DEVICE

logger = logging.getLogger("modeling_lstm")


class MetroLSTM(nn.Module):
    """
    부산 지하철 시계열 혼잡도 및 통행량 예측을 위한 2계층 LSTM 네트워크
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dense_dim: int = 16,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 연산:
            x: (batch_size, seq_len, input_dim)
        반환:
            (batch_size,) 스칼라 예측값 텐서
        """
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # 마지막 타임스텝의 은닉 상태 추출
        last_hidden = out[:, -1, :]
        dense = self.relu(self.fc1(last_hidden))
        pred = self.fc2(dense).squeeze(-1)
        return pred


class MetroSlidingWindowDataset(Dataset):
    """
    (N, Features) 2차원 데이터를 받아 (seq_len, Features) 3차원 윈도우 시퀀스로 변환하는 Dataset

    역사 경계 격리 원칙 (Cross-Station Boundary Isolation):
    station_ids 배열이 주어질 경우, 시작 시점과 끝 시점의 역사 ID가 동일한지 벡터화 검사
    (is_valid = station_ids[:-seq_len] == station_ids[seq_len:])를 수행하여,
    서로 다른 역사의 데이터가 단일 윈도우에 혼합되지 않는 안전한 인덱스(valid_indices)만 추출합니다.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 3,
        station_ids: np.ndarray | None = None,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

        n = len(X)
        if n <= seq_len:
            self.valid_indices = np.array([], dtype=np.int64)
        elif station_ids is not None:
            stn = np.asarray(station_ids)
            # 시작 시점과 끝 시점의 station_id가 동일한 윈도우만 유효 판정
            is_valid = stn[:-seq_len] == stn[seq_len:]
            self.valid_indices = np.flatnonzero(is_valid)
        else:
            # station_ids 미제공 시 기본 순차 슬라이딩 인덱스
            self.valid_indices = np.arange(n - seq_len, dtype=np.int64)

        if len(self.valid_indices) > 0:
            v_idx = torch.as_tensor(self.valid_indices, dtype=torch.long)
            self.seq_X = torch.stack([self.X[v_idx + k] for k in range(seq_len)], dim=1)
            self.target_y = self.y[v_idx + seq_len]
        else:
            feat_dim = self.X.shape[1] if self.X.ndim > 1 else 0
            self.seq_X = torch.empty((0, seq_len, feat_dim), dtype=torch.float32)
            self.target_y = torch.empty((0,), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seq_X[idx], self.target_y[idx]


def train_lstm_model(
    model: MetroLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """
    PyTorch 시계열 LSTM 모델 학습 및 검증 루프 (Early Stopping 및 벡터화 배치 슬라이싱 포함)
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    train_ds = getattr(train_loader, "dataset", None)
    use_fast_train = (
        train_ds is not None
        and hasattr(train_ds, "seq_X")
        and hasattr(train_ds, "target_y")
        and len(train_ds) > 0
    )
    if use_fast_train:
        train_all_x = train_ds.seq_X.to(device)
        train_all_y = train_ds.target_y.to(device)
        train_bs = train_loader.batch_size or 1024
        n_train = len(train_ds)

    val_ds = getattr(val_loader, "dataset", None) if val_loader is not None else None
    use_fast_val = (
        val_ds is not None
        and hasattr(val_ds, "seq_X")
        and hasattr(val_ds, "target_y")
        and len(val_ds) > 0
    )
    if use_fast_val and val_loader is not None:
        val_all_x = val_ds.seq_X.to(device)
        val_all_y = val_ds.target_y.to(device)
        val_bs = val_loader.batch_size or 1024
        n_val = len(val_ds)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        if use_fast_train:
            is_shuffle = isinstance(
                getattr(train_loader, "sampler", None),
                torch.utils.data.RandomSampler,
            )
            perm = (
                torch.randperm(n_train, device=device)
                if is_shuffle
                else torch.arange(n_train, device=device)
            )
            for i in range(0, n_train, train_bs):
                idx = perm[i : i + train_bs]
                batch_x = train_all_x[idx]
                batch_y = train_all_y[idx]

                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
        else:
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                if use_fast_val:
                    for i in range(0, n_val, val_bs):
                        vx = val_all_x[i : i + val_bs]
                        vy = val_all_y[i : i + val_bs]
                        v_pred = model(vx)
                        val_loss += criterion(v_pred, vy).item()
                        val_batches += 1
                else:
                    for vx, vy in val_loader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        v_pred = model(vx)
                        val_loss += criterion(v_pred, vy).item()
                        val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early Stopping 적용 (Epoch {epoch}/{epochs}, Best Val Loss: {best_val_loss:.4f})"
                    )
                    break
        else:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 최적 가중치 복원
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(device)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss if val_loader else avg_train_loss,
    }


def predict_lstm(
    model: MetroLSTM,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> np.ndarray:
    """학습된 MetroLSTM 모델로 배치 추론을 수행하고 1D NumPy 배열을 반환합니다."""
    model.eval()
    model.to(device)
    preds: list[np.ndarray] = []

    ds = getattr(loader, "dataset", None)
    use_fast = ds is not None and hasattr(ds, "seq_X") and len(ds) > 0

    with torch.no_grad():
        if use_fast:
            bs = loader.batch_size or 1024
            n_samples = len(ds)
            for i in range(0, n_samples, bs):
                batch_x = ds.seq_X[i : i + bs].to(device)
                p = model(batch_x).cpu().numpy()
                preds.append(p)
        else:
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                p = model(batch_x).cpu().numpy()
                preds.append(p)

    if not preds:
        return np.array([], dtype=np.float32)
    return np.concatenate(preds, axis=0)


def prepare_lstm_inference_dataframe(
    test_df: pl.DataFrame,
    lookback_df: pl.DataFrame | None = None,
    seq_len: int = 3,
    station_col: str = "station_id",
    time_col: str = "datetime",
) -> pl.DataFrame:
    """
    LSTM 추론 시 슬라이딩 윈도우(seq_len)로 인한 앞부분 예측 결손을 방지하기 위해,
    lookback_df의 역사별 직전 seq_len개 시점을 test_df 앞에 연결하고
    (station_id, datetime) 순으로 정렬된 평가용 DataFrame을 반환합니다.

    lookback_df가 제공되면 반환된 DataFrame으로 생성한 MetroSlidingWindowDataset의
    길이는 test_df의 행 수와 1:1로 정확히 일치하여, Fast Track(트리 모델)과의
    앙상블 블렌딩 시 차원 불일치 없이 즉시 결합할 수 있습니다.
    """
    if lookback_df is None or len(lookback_df) == 0:
        return test_df.sort([station_col, time_col])

    # 각 역사별 마지막 seq_len개 시점만 슬라이싱
    lb_tail = lookback_df.group_by(station_col, maintain_order=True).tail(seq_len)
    combined = pl.concat([lb_tail, test_df]).sort([station_col, time_col])
    return combined


class MetroLSTMEstimator(BaseEstimator, RegressorMixin):
    """
    scikit-learn BaseEstimator 호환용 MetroLSTM 회귀 추정기(Estimator).

    BaseEstimator와 RegressorMixin을 상속하여 get_params, set_params, fit, predict 규격을 준수합니다.
    (N, p) 2차원 입력을 받아 내부적으로 MetroSlidingWindowDataset을 통해 시계열 윈도우로 변환 후 학습/추론하며,
    predict 시 앞단 결손(seq_len) 구간을 첫 유효 예측값으로 자동 패딩하여 (N,) 일치 차원을 보장합니다.
    """

    def __init__(
        self,
        seq_len: int = 3,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dense_dim: int = 16,
        dropout_rate: float = 0.2,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        patience: int = 3,
        random_state: int = 42,
        device: str | torch.device = DEVICE,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dense_dim = dense_dim
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        station_ids: np.ndarray | None = None,
    ) -> MetroLSTMEstimator:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X_arr.shape[1]

        dataset = MetroSlidingWindowDataset(
            X_arr, y_arr, seq_len=self.seq_len, station_ids=station_ids
        )
        if len(dataset) == 0:
            raise ValueError(
                f"학습 데이터 샘플 수({len(X_arr)})가 seq_len({self.seq_len})보다 작거나 유효 윈도우가 없습니다."
            )

        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )

        model = MetroLSTM(
            input_dim=self.n_features_in_,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            dense_dim=self.dense_dim,
            dropout_rate=self.dropout_rate,
        )

        dev = torch.device(self.device) if isinstance(self.device, str) else self.device
        train_res = train_lstm_model(
            model=model,
            train_loader=loader,
            val_loader=None,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            device=dev,
        )
        self.model_ = train_res["model"]
        self.is_fitted_ = True
        return self

    def predict(
        self,
        X: np.ndarray,
        station_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("모델이 아직 fit되지 않았습니다.")

        X_arr = np.asarray(X, dtype=np.float32)
        dummy_y = np.zeros(len(X_arr), dtype=np.float32)

        dataset = MetroSlidingWindowDataset(
            X_arr, dummy_y, seq_len=self.seq_len, station_ids=station_ids
        )
        preds = np.zeros(len(X_arr), dtype=np.float32)

        if len(dataset) == 0:
            return preds

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        dev = torch.device(self.device) if isinstance(self.device, str) else self.device
        raw_preds = predict_lstm(self.model_, loader, device=dev)

        if len(raw_preds) > 0:
            target_indices = dataset.valid_indices + self.seq_len
            preds[target_indices] = raw_preds
            if station_ids is not None:
                stn_arr = np.asarray(station_ids)
                is_target = np.zeros(len(X_arr), dtype=bool)
                is_target[target_indices] = True
                for s in np.unique(stn_arr):
                    s_indices = np.flatnonzero(stn_arr == s)
                    s_targets = s_indices[is_target[s_indices]]
                    if len(s_targets) > 0:
                        first_s_idx = s_targets[0]
                        fill_val = preds[first_s_idx]
                        missing_prefix = s_indices[s_indices < first_s_idx]
                        preds[missing_prefix] = fill_val
                    else:
                        preds[s_indices] = raw_preds[0]
            else:
                first_idx = target_indices[0]
                preds[:first_idx] = raw_preds[0]

        return preds
