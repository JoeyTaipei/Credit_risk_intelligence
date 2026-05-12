"""Standalone LSTM encoder trainer with swappable data source.

Usage:
    # Original synthetic data (default — identical to run_lstm_stage in train_loop.py)
    python -m src.training.train_lstm --stage lstm --data_source synthetic

    # Real Lending Club sequences (requires lc_sequences.pt to exist)
    python -m src.training.train_lstm --stage lstm --data_source lending_club

The --data_source flag lets us hot-swap between the synthetic sequences built
from cs-training.csv and the real monthly trajectories built from loan.csv,
without touching any model code.  Default is synthetic so existing CI / demo
flows are unaffected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_synthetic() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load synthetic sequences derived from cs-training.csv splits."""
    import pandas as pd
    from src.data.preprocess import create_synthetic_time_series

    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing processed splits: run --stage xgb first.\n"
            f"  Expected: {train_path}\n           {val_path}"
        )

    TARGET = "SeriousDlqin2yrs"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = torch.FloatTensor(create_synthetic_time_series(train_df))
    y_train = torch.FloatTensor(train_df[TARGET].to_numpy()).view(-1, 1)
    X_val = torch.FloatTensor(create_synthetic_time_series(val_df))
    y_val = torch.FloatTensor(val_df[TARGET].to_numpy()).view(-1, 1)
    return X_train, y_train, X_val, y_val


def _load_lending_club() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load real Lending Club sequences with a temporal train/val split.

    Temporal split (leak-free):
      Input  → sequences[:, :10, :]  — first 10 months of borrower history
      Label  → lc_labels.pt          — whether is_late appears in months 10–11

    The LSTM must learn to predict FUTURE delinquency from PAST payment behavior.
    loan_status is not used anywhere in either the features or the labels.
    """
    from sklearn.model_selection import train_test_split

    seq_path = PROCESSED_DIR / "lc_sequences.pt"
    lbl_path = PROCESSED_DIR / "lc_labels.pt"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Lending Club sequences not found: {seq_path}\n"
            f"  Run: python -m src.data.lending_club_timeseries"
        )
    if not lbl_path.exists():
        raise FileNotFoundError(f"Lending Club labels not found: {lbl_path}")

    sequences = torch.load(seq_path, map_location="cpu")  # (N, 12, 4)
    labels = torch.load(lbl_path, map_location="cpu")     # (N,)

    # Only the first 10 months are input — months 10–11 are the prediction target
    # and must not be seen by the model during training or inference.
    input_sequences = sequences[:, :10, :]  # (N, 10, 4)

    idx = np.arange(len(input_sequences))
    y_np = labels.numpy()
    train_idx, val_idx = train_test_split(idx, test_size=0.20, random_state=42, stratify=y_np)

    X_train = input_sequences[train_idx]
    y_train = labels[train_idx].view(-1, 1)
    X_val = input_sequences[val_idx]
    y_val = labels[val_idx].view(-1, 1)

    pos_rate = y_np.mean()
    print(f"[INFO] LC dataset: {len(sequences):,} borrowers, {pos_rate:.3f} positive rate")
    print(f"[INFO] Input shape: {tuple(X_train.shape[1:])}  Train: {len(train_idx):,}  Val: {len(val_idx):,}")
    return X_train, y_train, X_val, y_val


def run_lstm(data_source: str) -> float:
    """Train LSTM encoder and return final validation AUC.

    Args:
        data_source: "synthetic" or "lending_club"
    """
    from src.models.lstm_encoder import LSTMEncoder

    torch.manual_seed(42)

    if data_source == "synthetic":
        X_train, y_train, X_val, y_val = _load_synthetic()
        output_name = "lstm_encoder.pt"
    elif data_source == "lending_club":
        X_train, y_train, X_val, y_val = _load_lending_club()
        output_name = "lstm_encoder_lc.pt"  # keep separate — don't overwrite synthetic model
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    print(f"[INFO] data_source={data_source}  X_train={tuple(X_train.shape)}  X_val={tuple(X_val.shape)}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=256, shuffle=False
    )

    encoder = LSTMEncoder(input_size=4, embedding_dim=32)
    head = nn.Linear(32, 1)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=1e-3
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = encoder.state_dict()
    patience_counter = 0
    final_val_auc = 0.0

    for epoch in range(1, 21):
        encoder.train()
        head.train()
        train_loss_total, n = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(head(encoder(bx)), by)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * bx.size(0)
            n += bx.size(0)
        train_loss = train_loss_total / n

        encoder.eval()
        head.eval()
        val_loss_total, n_val = 0.0, 0
        val_logits: list[torch.Tensor] = []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = head(encoder(bx))
                val_loss_total += criterion(logits, by).item() * bx.size(0)
                n_val += bx.size(0)
                val_logits.append(logits)

        val_loss = val_loss_total / n_val
        val_prob = torch.sigmoid(torch.cat(val_logits)).cpu().numpy().ravel()
        final_val_auc = float(roc_auc_score(y_val.numpy().ravel(), val_prob))

        print(
            f"[INFO] epoch={epoch:2d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_auc={final_val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    encoder.load_state_dict(best_state)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / output_name
    torch.save(encoder.state_dict(), out_path)
    print(f"[INFO] Saved encoder → {out_path}")
    print(f"[INFO] Final val AUC ({data_source}): {final_val_auc:.4f}")
    return final_val_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM encoder trainer with swappable data source.")
    parser.add_argument(
        "--stage",
        choices=["lstm"],
        default="lstm",
        help="Training stage (currently only 'lstm' is supported here).",
    )
    parser.add_argument(
        "--data_source",
        choices=["synthetic", "lending_club"],
        default="synthetic",
        help=(
            "synthetic: use create_synthetic_time_series on cs-training.csv splits (default). "
            "lending_club: use lc_sequences.pt built from loan.csv."
        ),
    )
    args = parser.parse_args()
    run_lstm(args.data_source)


if __name__ == "__main__":
    main()
