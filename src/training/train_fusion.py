"""Standalone late-fusion trainer with swappable LSTM encoder source.

Usage:
    # Default: uses lstm_encoder.pt trained on synthetic data
    python -m src.training.train_fusion --lstm_source synthetic

    # Use the LC-trained encoder; saves fusion_model_lc.pt
    python -m src.training.train_fusion --lstm_source lending_club

All other encoders (XGBoost leaf embeddings, GraphSAGE, frozen sentence-BERT)
are identical in both modes — only the LSTM checkpoint path changes.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COL = "SeriousDlqin2yrs"

# Checkpoint name written by each mode — never overwrites the other.
_OUTPUT_NAME = {
    "synthetic":     "fusion_model.pt",
    "lending_club":  "fusion_model_lc.pt",
}
_LSTM_CHECKPOINT = {
    "synthetic":     "lstm_encoder.pt",
    "lending_club":  "lstm_encoder_lc.pt",
}


# ---------------------------------------------------------------------------
# Embedding helpers  (mirrors train_loop.py but accepts explicit state_path)
# ---------------------------------------------------------------------------

def _make_lstm_embeddings(
    df: Any,
    state_path: Path,
) -> torch.Tensor:
    """Pass synthetic sequences through the specified LSTM encoder checkpoint.

    The encoder checkpoint is swapped via state_path; the input sequences are
    always the synthetic 12-month series so that all fusion splits use the same
    tabular feature space regardless of which LSTM was trained.
    """
    import pandas as pd
    from src.data.preprocess import create_synthetic_time_series
    from src.models.lstm_encoder import LSTMEncoder

    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    encoder.load_state_dict(torch.load(state_path, map_location="cpu"))
    encoder.eval()
    with torch.no_grad():
        seqs = torch.FloatTensor(create_synthetic_time_series(df))
        return encoder(seqs).detach().cpu()


def _make_gnn_embeddings(df: Any, state_path: Path) -> torch.Tensor:
    from src.data.graph_builder import build_borrower_graph
    from src.models.gnn_encoder import GraphSAGEEncoder

    edge_index, node_features = build_borrower_graph(df)
    encoder = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    encoder.load_state_dict(torch.load(state_path, map_location="cpu"))
    encoder.eval()
    with torch.no_grad():
        return encoder(node_features, edge_index).detach().cpu()


def _get_xgb_leaf_embeddings(model: Any, X: Any) -> torch.Tensor:
    from src.models.fusion import get_xgb_leaf_embeddings
    return get_xgb_leaf_embeddings(model, X)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _fusion_metrics(labels: torch.Tensor, probs: torch.Tensor) -> dict[str, float]:
    y_true = labels.cpu().numpy().ravel()
    y_prob = probs.cpu().numpy().ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc":      float(roc_auc_score(y_true, y_prob)),
        "pr_auc":       float(average_precision_score(y_true, y_prob)),
        "f1@0.5":       float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_fusion(lstm_source: str) -> dict[str, float]:
    """Train the late-fusion classifier and return validation metrics.

    Args:
        lstm_source: "synthetic" uses lstm_encoder.pt;
                     "lending_club" uses lstm_encoder_lc.pt.
    """
    import pandas as pd
    from src.models.fusion import LateFusionClassifier

    lstm_ckpt  = PROCESSED_DIR / _LSTM_CHECKPOINT[lstm_source]
    output_path = PROCESSED_DIR / _OUTPUT_NAME[lstm_source]

    required = {
        "train":      PROCESSED_DIR / "train.parquet",
        "val":        PROCESSED_DIR / "val.parquet",
        "xgb":        PROCESSED_DIR / "xgb_baseline.pkl",
        "lstm":       lstm_ckpt,
        "gnn":        PROCESSED_DIR / "gnn_encoder.pt",
        "text_train": PROCESSED_DIR / "text_embeddings_train.pt",
        "text_val":   PROCESSED_DIR / "text_embeddings_val.pt",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts:\n  " + "\n  ".join(missing))

    print(f"[INFO] lstm_source={lstm_source}  checkpoint={lstm_ckpt.name}")

    torch.manual_seed(42)
    train_df = pd.read_parquet(required["train"])
    val_df   = pd.read_parquet(required["val"])
    xgb_model = _load_pickle(required["xgb"])

    X_train = train_df.drop(columns=[TARGET_COL])
    X_val   = val_df.drop(columns=[TARGET_COL])
    train_labels = torch.FloatTensor(train_df[TARGET_COL].to_numpy()).view(-1, 1)
    val_labels   = torch.FloatTensor(val_df[TARGET_COL].to_numpy()).view(-1, 1)

    print("[INFO] Building embeddings ...")
    train_tabular = _get_xgb_leaf_embeddings(xgb_model, X_train)
    val_tabular   = _get_xgb_leaf_embeddings(xgb_model, X_val)

    train_lstm = _make_lstm_embeddings(train_df, required["lstm"])
    val_lstm   = _make_lstm_embeddings(val_df,   required["lstm"])
    print(f"[INFO]   LSTM embeddings  train={tuple(train_lstm.shape)}  val={tuple(val_lstm.shape)}")

    train_gnn = _make_gnn_embeddings(train_df, required["gnn"])
    val_gnn   = _make_gnn_embeddings(val_df,   required["gnn"])

    train_text = torch.load(required["text_train"], map_location="cpu").float()
    val_text   = torch.load(required["text_val"],   map_location="cpu").float()

    train_dataset = TensorDataset(train_tabular, train_lstm, train_gnn, train_text, train_labels)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = LateFusionClassifier()
    pos_count  = train_labels.sum()
    neg_count  = train_labels.numel() - pos_count
    pos_weight = neg_count / pos_count.clamp(min=1.0)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state    = model.state_dict()
    patience      = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(1, 31):
        model.train()
        train_loss_total, n = 0.0, 0
        for tab, lstm, gnn, text, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(tab, lstm, gnn, text), labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = train_loss_total / n

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tabular, val_lstm, val_gnn, val_text)
            val_loss   = criterion(val_logits, val_labels).item()
            val_prob   = torch.sigmoid(val_logits)
            final_metrics = _fusion_metrics(val_labels, val_prob)

        print(
            f"[INFO] epoch={epoch:2d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={final_metrics['roc_auc']:.4f}  "
            f"val_pr_auc={final_metrics['pr_auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience      = 0
        else:
            patience += 1
            if patience >= 7:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Saved fusion model -> {output_path}")
    print(
        f"[INFO] Final val  AUC={final_metrics['roc_auc']:.4f}  "
        f"PR-AUC={final_metrics['pr_auc']:.4f}  "
        f"F1@0.5={final_metrics['f1@0.5']:.4f}"
    )
    return final_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Late-fusion trainer with swappable LSTM source.")
    parser.add_argument(
        "--lstm_source",
        choices=["synthetic", "lending_club"],
        default="synthetic",
        help=(
            "synthetic (default): use lstm_encoder.pt, save fusion_model.pt. "
            "lending_club: use lstm_encoder_lc.pt, save fusion_model_lc.pt."
        ),
    )
    args = parser.parse_args()
    run_fusion(args.lstm_source)


if __name__ == "__main__":
    main()
