"""Single-borrower inference helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.preprocess import create_synthetic_time_series
from src.models.fusion import get_xgb_leaf_embeddings


def risk_level_from_score(score: float) -> str:
    """Map a probability score to a Traditional Chinese risk level."""
    if score < 0.3:
        return "低風險"
    if score <= 0.6:
        return "中風險"
    return "高風險"


def _ensure_2d_embedding(value: Any) -> torch.Tensor:
    """Convert an embedding-like value to a single-row FloatTensor."""
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    tensor = tensor.float()
    return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor


def _top_shap_features(models: dict[str, Any], row_df: pd.DataFrame) -> list[tuple[str, float, Any]]:
    """Return top SHAP features when an explainer is supplied; otherwise fallback rows."""
    explainer = models.get("shap_explainer")
    if explainer is not None:
        shap_values = explainer(row_df)
        values = getattr(shap_values, "values", shap_values)
        values = np.asarray(values).reshape(-1)
        top_idx = np.argsort(np.abs(values))[::-1][:3]
        return [(row_df.columns[i], float(values[i]), row_df.iloc[0, i]) for i in top_idx]

    fallback_features = row_df.columns[:3]
    return [(name, 0.0, row_df.iloc[0][name]) for name in fallback_features]


def predict_single_borrower(
    borrower_row: pd.Series,
    loan_description: str,
    models: dict,
) -> dict:
    """
    Full inference pipeline for one borrower.
    models dict keys: xgb, lstm_encoder, gnn_encoder,
                      text_encoder, fusion_model, node_embeddings
    Returns: {
        risk_score: float (0-1),
        risk_level: str (低/中/高風險),
        top_shap_features: list of (name, value, raw_value),
        embeddings: dict of 4 embedding vectors
    }
    """
    row_df = borrower_row.to_frame().T
    xgb_model = models["xgb"]
    lstm_encoder = models["lstm_encoder"]
    text_encoder = models["text_encoder"]
    fusion_model = models["fusion_model"]

    targetless_df = row_df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    with torch.no_grad():
        tabular_emb = get_xgb_leaf_embeddings(xgb_model, targetless_df)

        sequence = torch.FloatTensor(create_synthetic_time_series(row_df))
        lstm_emb = _ensure_2d_embedding(lstm_encoder(sequence))

        if "node_embedding" in models:
            gnn_emb = _ensure_2d_embedding(models["node_embedding"])
        else:
            node_embeddings = models["node_embeddings"]
            if isinstance(node_embeddings, torch.Tensor):
                gnn_emb = _ensure_2d_embedding(node_embeddings[0])
            else:
                gnn_emb = _ensure_2d_embedding(node_embeddings)

        text_emb = _ensure_2d_embedding(text_encoder.encode_texts([loan_description]))

        logit = fusion_model(tabular_emb, lstm_emb, gnn_emb, text_emb)
        risk_score = float(torch.sigmoid(logit).item())

    return {
        "risk_score": risk_score,
        "risk_level": risk_level_from_score(risk_score),
        "top_shap_features": _top_shap_features(models, targetless_df),
        "embeddings": {
            "tabular": tabular_emb.squeeze(0).detach().cpu().tolist(),
            "lstm": lstm_emb.squeeze(0).detach().cpu().tolist(),
            "gnn": gnn_emb.squeeze(0).detach().cpu().tolist(),
            "text": text_emb.squeeze(0).detach().cpu().tolist(),
        },
    }
