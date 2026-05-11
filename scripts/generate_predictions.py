"""Generate predictions.csv and shap_values.pkl for the Streamlit demo.

Run once from the project root:
    python scripts/generate_predictions.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
N_DEMO = 10   # number of borrowers to show in the dashboard


def _risk_level(score: float) -> str:
    if score < 0.3:
        return "低風險"
    if score <= 0.6:
        return "中風險"
    return "高風險"


def _action(score: float) -> str:
    if score < 0.3:
        return "可核准放款"
    if score <= 0.6:
        return "需補件審查"
    return "建議拒絕或擔保"


def main() -> None:
    from src.data.graph_builder import build_borrower_graph
    from src.data.preprocess import create_synthetic_time_series
    from src.models.fusion import LateFusionClassifier, get_xgb_leaf_embeddings
    from src.models.gnn_encoder import GraphSAGEEncoder
    from src.models.lstm_encoder import LSTMEncoder

    # ── Load data ──────────────────────────────────────────────────────────────
    val_df = pd.read_parquet(PROCESSED / "val.parquet").head(N_DEMO).reset_index(drop=True)
    X = val_df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    # ── Load models ────────────────────────────────────────────────────────────
    with open(PROCESSED / "xgb_baseline.pkl", "rb") as f:
        xgb = pickle.load(f)

    lstm_enc = LSTMEncoder(input_size=4, hidden_size=64, num_layers=2, embedding_dim=32)
    lstm_enc.load_state_dict(torch.load(PROCESSED / "lstm_encoder.pt", map_location="cpu", weights_only=True))
    lstm_enc.eval()

    gnn_enc = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    gnn_enc.load_state_dict(torch.load(PROCESSED / "gnn_encoder.pt", map_location="cpu", weights_only=True))
    gnn_enc.eval()

    fusion = LateFusionClassifier()
    fusion.load_state_dict(torch.load(PROCESSED / "fusion_model.pt", map_location="cpu", weights_only=True))
    fusion.eval()

    # ── Build embeddings ───────────────────────────────────────────────────────
    tabular_emb = get_xgb_leaf_embeddings(xgb, X)          # (N, 32)

    seq = torch.FloatTensor(create_synthetic_time_series(val_df))  # (N, 12, 4)
    with torch.no_grad():
        lstm_emb = lstm_enc(seq)                            # (N, 32)

    edge_index, node_features = build_borrower_graph(val_df)
    with torch.no_grad():
        gnn_emb = gnn_enc(node_features, edge_index)        # (N, 32)

    # Reuse pre-computed text embeddings (first N rows of val set)
    text_all = torch.load(PROCESSED / "text_embeddings_val.pt", map_location="cpu").float()
    text_emb = text_all[:N_DEMO]                            # (N, 32)

    # ── Run fusion ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = fusion(tabular_emb, lstm_emb, gnn_emb, text_emb)  # (N, 1)
        risk_scores = torch.sigmoid(logits).squeeze(1).numpy()

    # ── Compute SHAP values (XGBoost TreeExplainer) ────────────────────────────
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_explanation = explainer(X)   # shap.Explanation shape (N, n_features)

    # Save SHAP as a shap.Explanation for the waterfall plot in the app
    with open(PROCESSED / "shap_values.pkl", "wb") as f:
        pickle.dump(shap_explanation, f)
    print(f"Saved shap_values.pkl  ({N_DEMO} rows, {shap_explanation.shape[1]} features)")

    # ── Build predictions DataFrame ────────────────────────────────────────────
    shap_vals = shap_explanation.values   # (N, n_features)
    top1_idx = np.argmax(np.abs(shap_vals), axis=1)
    feature_names = list(X.columns)

    records = []
    for i in range(N_DEMO):
        score = float(risk_scores[i])
        top_feat = feature_names[top1_idx[i]]
        shap1 = float(shap_vals[i, top1_idx[i]])
        records.append({
            "borrower_id": i,
            "risk_score": round(score, 4),
            "risk_level": _risk_level(score),
            "recommended_action": _action(score),
            "top_shap_feature": top_feat,
            "shap_value_top1": round(shap1, 4),
            "MonthlyIncome": int(val_df.loc[i, "MonthlyIncome"]),
            "DebtRatio": round(float(val_df.loc[i, "DebtRatio"]), 3),
            "NumberOfTimes90DaysLate": int(val_df.loc[i, "NumberOfTimes90DaysLate"]),
            "age_bucket": str(val_df.loc[i, "age_bucket"]),
        })

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(PROCESSED / "predictions.csv", index=False)
    print(f"Saved predictions.csv  ({N_DEMO} borrowers)")
    print("\nRisk scores:")
    for _, r in pred_df.iterrows():
        print(f"  borrower #{int(r.borrower_id):2d}  {r.risk_score:.1%}  {r.risk_level}  ({r.recommended_action})")


if __name__ == "__main__":
    main()
