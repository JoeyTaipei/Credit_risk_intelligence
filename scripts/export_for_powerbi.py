"""Export val-set fusion predictions to CSV for Power BI dashboard.

Loads data/processed/val.parquet and all trained checkpoints,
runs batch inference through the full 4-encoder → fusion pipeline,
computes per-row top-3 SHAP features, and writes
data/processed/powerbi_dashboard.csv with 18 business-ready columns.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph_builder import build_borrower_graph
from src.data.preprocess import create_synthetic_time_series
from src.inference.predict import risk_level_from_score
from src.models.fusion import LateFusionClassifier, get_xgb_leaf_embeddings
from src.models.gnn_encoder import GraphSAGEEncoder
from src.models.lstm_encoder import LSTMEncoder
from src.utils.openai_report import generate_credit_report

PROCESSED = ROOT / "data" / "processed"
TARGET_COL = "SeriousDlqin2yrs"


def _load_models():
    with (PROCESSED / "xgb_baseline.pkl").open("rb") as fh:
        xgb = pickle.load(fh)

    lstm = LSTMEncoder(input_size=4, embedding_dim=32)
    lstm.load_state_dict(torch.load(PROCESSED / "lstm_encoder.pt", map_location="cpu"))
    lstm.eval()

    gnn = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    gnn.load_state_dict(torch.load(PROCESSED / "gnn_encoder.pt", map_location="cpu"))
    gnn.eval()

    fusion = LateFusionClassifier()
    fusion.load_state_dict(torch.load(PROCESSED / "fusion_model.pt", map_location="cpu"))
    fusion.eval()

    # Pre-computed 32-dim sentence-BERT projections from Day 4 text stage
    text_emb = torch.load(PROCESSED / "text_embeddings_val.pt", map_location="cpu").float()

    return xgb, lstm, gnn, fusion, text_emb


def _batch_embeddings(val_df, xgb, lstm, gnn, text_emb):
    X_val = val_df.drop(columns=[TARGET_COL], errors="ignore")
    with torch.no_grad():
        tab = get_xgb_leaf_embeddings(xgb, X_val)
        sequences = torch.FloatTensor(create_synthetic_time_series(val_df))
        lstm_out = lstm(sequences)
        # Build val-split graph for inductive GNN inference
        edge_index, node_features = build_borrower_graph(val_df)
        gnn_out = gnn(node_features, edge_index)
    return tab, lstm_out, gnn_out, text_emb


def _run_fusion(fusion, tab, lstm, gnn, text):
    with torch.no_grad():
        logits = fusion(tab, lstm, gnn, text)       # (n, 1)
        scores = torch.sigmoid(logits).squeeze(1)   # (n,)
    return scores.numpy()


def _top3_shap(xgb, val_df):
    """Return top-3 SHAP features per row as parallel name lists and a (n, 3) value array."""
    X = val_df.drop(columns=[TARGET_COL], errors="ignore")
    explainer = shap.TreeExplainer(xgb)
    raw = explainer.shap_values(X)

    # Normalise across XGBoost's possible output shapes
    if isinstance(raw, list):
        sv = raw[1]           # binary: [neg_class, pos_class]
    elif raw.ndim == 3:
        sv = raw[:, :, 1]
    else:
        sv = raw              # already (n, features)

    feature_names = list(X.columns)
    n = len(val_df)
    # Sort descending by |SHAP| within each row; keep top 3
    top3_idx = np.argsort(-np.abs(sv), axis=1)[:, :3]   # (n, 3)

    top_names = [[feature_names[top3_idx[i, j]] for j in range(3)] for i in range(n)]
    top_vals = sv[np.arange(n)[:, None], top3_idx]       # (n, 3), signed values

    return top_names, top_vals


def _age_bucket_series(age: pd.Series) -> pd.Series:
    return pd.cut(
        age.clip(upper=100),
        bins=[0, 30, 45, 60, 100],
        labels=["青年", "壯年", "中年", "資深"],
    ).astype(str)


def _recommended_action(score: float) -> str:
    if score > 0.6:
        return "拒絕或要求擔保"
    if score > 0.3:
        return "人工複審"
    return "核准"


def _build_ai_summaries(
    val_df: pd.DataFrame,
    risk_scores: np.ndarray,
    risk_levels: list[str],
    top_names: list[list[str]],
    top_vals: np.ndarray,
    recommended_actions: list[str],
    n_top: int = 10,
) -> list[str]:
    """AI summaries for top-N borrowers via Claude; template string for the rest.

    Calling Claude for only the 10 highest-risk rows bounds API cost while
    ensuring the most consequential decisions have a full-length narrative report.
    """
    X_val = val_df.drop(columns=[TARGET_COL], errors="ignore")
    top_indices = set(np.argsort(risk_scores)[-n_top:].tolist())

    summaries: list[str] = []
    for i in range(len(val_df)):
        score = float(risk_scores[i])
        action = recommended_actions[i]
        top_feat = top_names[i][0]

        if i in top_indices:
            borrower_data = X_val.iloc[i].to_dict()
            prediction = {"risk_score": score, "risk_level": risk_levels[i]}
            # Pass all three top features with their actual SHAP values
            shap_top = [
                (top_names[i][j], float(top_vals[i, j]), borrower_data.get(top_names[i][j]))
                for j in range(3)
            ]
            report = generate_credit_report(borrower_data, prediction, shap_top)
            summaries.append(report)
        else:
            summaries.append(
                f"風險評分 {score:.0%}，主要風險因子：{top_feat}，建議：{action}"
            )

    return summaries


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("[INFO] Loading val.parquet ...")
    val_df = pd.read_parquet(PROCESSED / "val.parquet").reset_index(drop=True)
    print(f"[INFO] val.parquet shape: {val_df.shape}")

    # Capture NaN status before any downstream filling; val.parquet is already
    # preprocessed so this will be all-False, but the condition is preserved
    # faithfully per the business definition of a thin-file borrower.
    income_is_nan = val_df["MonthlyIncome"].isna()

    print("[INFO] Loading model checkpoints ...")
    xgb, lstm, gnn, fusion, text_emb = _load_models()

    n_val = len(val_df)
    if text_emb.shape[0] != n_val:
        raise ValueError(
            f"text_embeddings_val.pt has {text_emb.shape[0]} rows "
            f"but val.parquet has {n_val} rows — re-run the text stage."
        )

    print("[INFO] Computing modality embeddings ...")
    tab, lstm_out, gnn_out, text_out = _batch_embeddings(val_df, xgb, lstm, gnn, text_emb)

    print("[INFO] Running fusion model ...")
    risk_scores = _run_fusion(fusion, tab, lstm_out, gnn_out, text_out)

    print("[INFO] Computing per-row top-3 SHAP features ...")
    top_names, top_vals = _top3_shap(xgb, val_df)

    # --- Derived business columns ---
    risk_levels = [risk_level_from_score(s) for s in risk_scores]
    recommended_actions = [_recommended_action(s) for s in risk_scores]

    total_past_due = val_df["total_past_due"].to_numpy(dtype=int)
    is_thin_file = (income_is_nan.to_numpy()) | (total_past_due == 0)

    estimated_loss = np.where(
        np.array(risk_levels) == "高風險",
        500_000.0 * risk_scores,
        0.0,
    )

    print("[INFO] Generating AI summaries (top 10 high-risk via Claude Opus) ...")
    ai_summaries = _build_ai_summaries(
        val_df, risk_scores, risk_levels, top_names, top_vals, recommended_actions, n_top=10
    )

    out = pd.DataFrame({
        "borrower_id": range(n_val),
        "age": val_df["age"].to_numpy(dtype=int),
        "MonthlyIncome": val_df["MonthlyIncome"].to_numpy(),
        "DebtRatio": val_df["DebtRatio"].to_numpy(),
        "RevolvingUtilizationOfUnsecuredLines": val_df["RevolvingUtilizationOfUnsecuredLines"].to_numpy(),
        "total_past_due": total_past_due,
        "risk_score": risk_scores.round(4),
        "risk_level": risk_levels,
        "top_shap_feature": [row[0] for row in top_names],
        "shap_value_1": top_vals[:, 0].round(4),
        "top_shap_feature_2": [row[1] for row in top_names],
        "top_shap_feature_3": [row[2] for row in top_names],
        "age_bucket": _age_bucket_series(val_df["age"]),
        "is_thin_file": is_thin_file,
        "estimated_loss": np.round(estimated_loss, 2),
        "recommended_action": recommended_actions,
        "ai_summary": ai_summaries,
    })

    out_path = PROCESSED / "powerbi_dashboard.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] Exported {out.shape[0]} rows × {out.shape[1]} cols → {out_path}")
    print("\nFirst 3 rows:")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
