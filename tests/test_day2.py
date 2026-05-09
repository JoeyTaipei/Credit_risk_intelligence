from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.preprocess import create_synthetic_time_series
from src.models.lstm_encoder import LSTMEncoder


def test_time_series_shape(sample_credit_df: pd.DataFrame) -> None:
    sequences = create_synthetic_time_series(sample_credit_df)

    assert sequences.shape == (100, 12, 4)


def test_time_series_range(sample_credit_df: pd.DataFrame) -> None:
    sequences = create_synthetic_time_series(sample_credit_df)

    assert np.all(sequences >= 0)
    assert np.all(sequences <= 1)


def test_lstm_forward_shape() -> None:
    model = LSTMEncoder(input_size=4, embedding_dim=32)
    batch = torch.rand(8, 12, 4)

    output = model(batch)

    assert output.shape == (8, 32)


def test_lstm_trains_one_step() -> None:
    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    head = nn.Linear(32, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3,
    )
    criterion = nn.BCEWithLogitsLoss()
    batch = torch.rand(8, 12, 4)
    target = torch.randint(0, 2, (8, 1)).float()

    optimizer.zero_grad()
    logits = head(encoder(batch))
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
