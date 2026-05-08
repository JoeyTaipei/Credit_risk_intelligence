from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.predict import predict_single_borrower, risk_level_from_score
from src.models.fusion import LateFusionClassifier


class MockXGB:
    def apply(self, X):
        import numpy as np

        return np.ones((len(X), 3), dtype=int)


class MockLSTM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], 32)


class MockTextEncoder:
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        return torch.ones(len(texts), 32)


class MockFusion(nn.Module):
    def forward(
        self,
        tabular_emb: torch.Tensor,
        lstm_emb: torch.Tensor,
        gnn_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor([[0.0]])


def test_fusion_forward_shape() -> None:
    model = LateFusionClassifier()
    embeddings = [torch.rand(8, 32) for _ in range(4)]

    output = model(*embeddings)

    assert output.shape == (8, 1)


def test_fusion_all_embeddings_required() -> None:
    model = LateFusionClassifier()
    embeddings = [torch.rand(8, 32) for _ in range(3)]

    with pytest.raises(TypeError):
        model(*embeddings)


def test_predict_returns_required_keys(sample_credit_df) -> None:
    borrower = sample_credit_df.iloc[0]
    models = {
        "xgb": MockXGB(),
        "lstm_encoder": MockLSTM(),
        "gnn_encoder": object(),
        "text_encoder": MockTextEncoder(),
        "fusion_model": MockFusion(),
        "node_embeddings": torch.ones(32),
    }

    result = predict_single_borrower(borrower, "Loan for debt consolidation.", models)

    assert {"risk_score", "risk_level", "top_shap_features", "embeddings"}.issubset(result)


def test_risk_level_thresholds() -> None:
    assert risk_level_from_score(0.1) == "低風險"
    assert risk_level_from_score(0.45) == "中風險"
    assert risk_level_from_score(0.8) == "高風險"
