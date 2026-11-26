"""
전날 17시 시나리오용 Seq2Seq LSTM

역×예측일 하나를 샘플로 삼아 D일 24개 시간대를 한 번에 예측합니다.
- 인코더: D-8일~D-2일의 시간별 승객 수(168시간)와 해당 시점의 달력
- 디코더: D일 시간대별 달력, 같은 요일·시간의 과거 4주 실적, 선택적 행사 Feature
- 역 ID는 임베딩으로 처리하고, 승객 수는 학습 구간 역 평균으로 나눠 비율로 다룹니다.
- 출력은 과거 4주 평균 비율에 대한 보정값이며, 손실은 RMSE 평가와 맞추기 위해
  역 평균을 곱한 원 단위 오차로 계산합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from busan_metro_congestion.config import DEVICE

TARGET_COL = "accumulated_passengers"
ENCODER_CALENDAR = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "is_holiday",
    "is_holiday_eve",
]
DECODER_CALENDAR = [*ENCODER_CALENDAR, "month_sin", "month_cos"]
DECODER_LAGS = ["lag_7d", "lag_14d", "lag_21d", "lag_28d"]
ENCODER_DAY_OFFSETS = np.arange(-8, -1)
HOURS_PER_DAY = 24


@dataclass
class SequencePanel:
    """역×일×시간 격자로 정렬한 입력 배열"""

    stations: np.ndarray
    dates: list[date]
    y: np.ndarray
    calendar_enc: np.ndarray
    calendar_dec: np.ndarray
    lags: np.ndarray
    extras: np.ndarray


def build_sequence_panel(
    df: pl.DataFrame, extra_cols: list[str] | None = None
) -> SequencePanel:
    """피처 테이블을 (역, 일, 24시간) 배열로 변환합니다. 격자에 빈칸이 있으면 오류를 냅니다."""
    extra_cols = extra_cols or []
    ordered = df.sort(["station_id", "date", "hour"])
    stations = ordered["station_id"].unique(maintain_order=True).to_numpy()
    dates = ordered["date"].unique().sort().to_list()
    n_stations, n_days = len(stations), len(dates)

    contiguous = (dates[-1] - dates[0]).days + 1 == n_days
    if not contiguous or len(ordered) != n_stations * n_days * HOURS_PER_DAY:
        raise ValueError("역×일×시간 격자가 완전하지 않아 시퀀스를 만들 수 없습니다.")
    if ordered.select(DECODER_LAGS).null_count().sum_horizontal().item() > 0:
        raise ValueError("과거 실적 Feature에 결측이 있습니다.")

    def to_array(columns: list[str]) -> np.ndarray:
        shape = (n_stations, n_days, HOURS_PER_DAY, len(columns))
        if not columns:
            return np.zeros(shape, dtype=np.float32)
        values = ordered.select(pl.col(columns).cast(pl.Float32)).to_numpy()
        return np.array(values, dtype=np.float32).reshape(shape)

    return SequencePanel(
        stations=stations,
        dates=dates,
        y=to_array([TARGET_COL])[..., 0],
        calendar_enc=to_array(ENCODER_CALENDAR)[0],
        calendar_dec=to_array(DECODER_CALENDAR)[0],
        lags=to_array(DECODER_LAGS),
        extras=to_array(extra_cols),
    )


class DayAheadSeq2Seq(nn.Module):
    """인코더 상태에서 시작해 예측일 24개 시간대의 보정값을 출력하는 Seq2Seq LSTM"""

    def __init__(
        self,
        n_stations: int,
        encoder_dim: int,
        decoder_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_stations, embedding_dim)
        layer_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            encoder_dim + embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout,
        )
        self.decoder = nn.LSTM(
            decoder_dim + embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, encoder_x: torch.Tensor, decoder_x: torch.Tensor, station: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.embedding(station).unsqueeze(1)
        enc_static = embedded.expand(-1, encoder_x.shape[1], -1)
        dec_static = embedded.expand(-1, decoder_x.shape[1], -1)
        _, state = self.encoder(torch.cat([encoder_x, enc_static], dim=-1))
        out, _ = self.decoder(torch.cat([decoder_x, dec_static], dim=-1), state)
        return self.head(out).squeeze(-1)


class SequenceBatcher:
    """한 폴드의 정규화 기준을 고정하고 (역, 일) 샘플을 배치 텐서로 조립합니다."""

    def __init__(
        self,
        panel: SequencePanel,
        train_start: date,
        train_end: date,
        device: torch.device,
    ) -> None:
        self.panel = panel
        self.device = device
        train_days = self.day_indices(panel, train_start, train_end)
        train_y = panel.y[:, train_days]

        scale = np.maximum(train_y.mean(axis=(1, 2)), 1.0)
        extra_scale = np.abs(panel.extras[:, train_days]).max(axis=(0, 1, 2))
        extra_scale = np.where(extra_scale > 0, extra_scale, 1.0)

        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.loss_scale = float(scale.mean())
        self.y = torch.as_tensor(panel.y, device=device)
        self.calendar_enc = torch.as_tensor(panel.calendar_enc, device=device)
        self.calendar_dec = torch.as_tensor(panel.calendar_dec, device=device)
        self.lags = torch.as_tensor(panel.lags, device=device)
        self.extras = torch.as_tensor(panel.extras / extra_scale, device=device)
        self.offsets = torch.as_tensor(ENCODER_DAY_OFFSETS, device=device)

    @staticmethod
    def day_indices(panel: SequencePanel, start: date, end: date) -> np.ndarray:
        """구간 안에서 인코더 입력 7일을 모두 확보할 수 있는 날짜 인덱스를 반환합니다."""
        first = -int(ENCODER_DAY_OFFSETS.min())
        return np.array(
            [i for i, d in enumerate(panel.dates) if start <= d <= end and i >= first],
            dtype=np.int64,
        )

    @property
    def encoder_dim(self) -> int:
        return 1 + self.calendar_enc.shape[-1]

    @property
    def decoder_dim(self) -> int:
        return self.calendar_dec.shape[-1] + self.lags.shape[-1] + self.extras.shape[-1]

    def assemble(
        self, station: torch.Tensor, day: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """인코더 입력, 디코더 입력, 과거 4주 평균 비율을 반환합니다."""
        scale = self.scale[station]
        enc_days = day.unsqueeze(1) + self.offsets
        history = self.y[station.unsqueeze(1), enc_days].flatten(1)
        encoder_x = torch.cat(
            [
                (history / scale.unsqueeze(1)).unsqueeze(-1),
                self.calendar_enc[enc_days].flatten(1, 2),
            ],
            dim=-1,
        )
        lag_ratio = self.lags[station, day] / scale.view(-1, 1, 1)
        decoder_x = torch.cat(
            [self.calendar_dec[day], lag_ratio, self.extras[station, day]], dim=-1
        )
        return encoder_x, decoder_x, lag_ratio.mean(dim=-1)


def _make_samples(
    n_stations: int, days: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    station, day = np.meshgrid(np.arange(n_stations), days, indexing="ij")
    return (
        torch.as_tensor(station.ravel(), device=device),
        torch.as_tensor(day.ravel(), device=device),
    )


def fit_seq2seq(
    panel: SequencePanel,
    train_start: date,
    train_end: date,
    params: dict[str, Any],
    seed: int = 42,
    device: torch.device = DEVICE,
) -> tuple[DayAheadSeq2Seq, SequenceBatcher]:
    """학습 구간의 (역, 일) 샘플로 고정된 에폭 수만큼 학습합니다."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batcher = SequenceBatcher(panel, train_start, train_end, device)
    model = DayAheadSeq2Seq(
        n_stations=len(panel.stations),
        encoder_dim=batcher.encoder_dim,
        decoder_dim=batcher.decoder_dim,
        hidden_dim=params.get("hidden_dim", 64),
        dropout=params.get("dropout", 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
    )
    epochs = params.get("epochs", 20)
    batch_size = params.get("batch_size", 512)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    days = SequenceBatcher.day_indices(panel, train_start, train_end)
    station, day = _make_samples(len(panel.stations), days, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(station), device=device, generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_station, batch_day = station[index], day[index]
            encoder_x, decoder_x, base = batcher.assemble(batch_station, batch_day)
            ratio = base + model(encoder_x, decoder_x, batch_station)
            scale = batcher.scale[batch_station].unsqueeze(1)
            residual = (ratio * scale - batcher.y[batch_station, batch_day]) / (
                batcher.loss_scale
            )
            loss = residual.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model, batcher


def predict_seq2seq(
    model: DayAheadSeq2Seq,
    batcher: SequenceBatcher,
    start: date,
    end: date,
    batch_size: int = 2048,
) -> pl.DataFrame:
    """구간의 모든 역·시간대 예측값을 키 컬럼과 함께 반환합니다."""
    panel = batcher.panel
    days = SequenceBatcher.day_indices(panel, start, end)
    station, day = _make_samples(len(panel.stations), days, batcher.device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(station), batch_size):
            batch_station = station[first : first + batch_size]
            batch_day = day[first : first + batch_size]
            encoder_x, decoder_x, base = batcher.assemble(batch_station, batch_day)
            ratio = base + model(encoder_x, decoder_x, batch_station)
            outputs.append(ratio * batcher.scale[batch_station].unsqueeze(1))
    prediction = torch.cat(outputs).clamp(min=0.0).cpu().numpy().ravel()

    station_idx = np.repeat(station.cpu().numpy(), HOURS_PER_DAY)
    day_idx = np.repeat(day.cpu().numpy(), HOURS_PER_DAY)
    hours = np.tile(np.arange(1, HOURS_PER_DAY + 1), len(station))
    first_date = panel.dates[0]
    return pl.DataFrame(
        {
            "station_id": panel.stations[station_idx],
            "date": [first_date + timedelta(days=int(i)) for i in day_idx],
            "hour": hours,
            "prediction": prediction,
        }
    ).with_columns(pl.col("hour").cast(pl.Int8), pl.col("date").cast(pl.Date))


def fit_predict_seq2seq(
    panel: SequencePanel,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    params: dict[str, Any],
    seeds: tuple[int, ...] = (42,),
    device: torch.device = DEVICE,
) -> pl.DataFrame:
    """시드별로 학습해 예측값을 평균합니다."""
    averaged: pl.DataFrame | None = None
    for seed in seeds:
        model, batcher = fit_seq2seq(
            panel, train_start, train_end, params, seed=seed, device=device
        )
        current = predict_seq2seq(model, batcher, test_start, test_end)
        averaged = (
            current
            if averaged is None
            else averaged.with_columns(
                pl.col("prediction") + current["prediction"],
            )
        )
    return averaged.with_columns(pl.col("prediction") / len(seeds))
