"""Train Aligned Fusion v0 on four cs-training2-derived modality embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RANDOM_STATE = 42


class AlignedFusionMLP(nn.Module):
    """Three-layer MLP over concatenated aligned modality embeddings."""

    def __init__(self, input_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned artifact: {path}")
    return torch.load(path, map_location="cpu")


def _safe_torch_save(payload, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    torch.save(payload, path)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1@0.5": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _graph_fallback_used() -> bool:
    metadata_path = PROCESSED_DIR / "graph_emb_aligned_metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(metadata.get("used_zero_fallback", False))


def _print_alignment_report(n: int, graph_zero_fallback: bool) -> None:
    graph_note = "zero fallback" if graph_zero_fallback else "GraphSAGE embeddings"
    print("\nModality     | Source       | N      | Real/Synthetic | Notes")
    print("-------------|--------------|--------|----------------|-------")
    print(f"Tabular      | cs-training2 | {n:<6} | Real           | 5-fold ensemble")
    print(f"Time Series  | cs-training2 | {n:<6} | Synthetic      | LC encoder weights")
    print(f"Graph        | cs-training2 | {n:<6} | Synthetic      | {graph_note}")
    print(f"Text         | cs-training2 | {n:<6} | Synthetic      | template generated")
    print(f"Labels       | cs-training2 | {n:<6} | Real           | SeriousDlqin2yrs")


def load_aligned_inputs() -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Load and validate all aligned embeddings before training."""
    tabular = _load_tensor(PROCESSED_DIR / "tabular_emb_aligned.pt").float()
    lstm = _load_tensor(PROCESSED_DIR / "lstm_emb_aligned.pt").float()
    graph = _load_tensor(PROCESSED_DIR / "graph_emb_aligned.pt").float()
    text = _load_tensor(PROCESSED_DIR / "text_emb_aligned.pt").float()
    labels = _load_tensor(PROCESSED_DIR / "aligned_labels.pt").float().view(-1)

    tensors = {
        "tabular": tabular,
        "lstm": lstm,
        "graph": graph,
        "text": text,
    }
    n = len(labels)
    for name, tensor in tensors.items():
        if tensor.ndim != 2 or tensor.shape[1] != 32:
            raise ValueError(f"{name} embedding must have shape (N, 32), got {tuple(tensor.shape)}")
        if tensor.shape[0] != n:
            raise ValueError(f"{name} N={tensor.shape[0]} does not match labels N={n}")

    fused = torch.cat([tabular, lstm, graph, text], dim=1)
    if fused.shape != (n, 128):
        raise ValueError(f"Concatenated embedding shape mismatch: {tuple(fused.shape)}")

    graph_zero_fallback = _graph_fallback_used()
    _print_alignment_report(n, graph_zero_fallback)
    return fused, labels, graph_zero_fallback


def train_fusion_aligned(
    batch_size: int = 2048,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, float | bool | int]:
    """Train aligned fusion MLP and save the best state."""
    output_path = PROCESSED_DIR / "fusion_model_aligned.pt"
    metadata_path = PROCESSED_DIR / "fusion_model_aligned_metadata.json"
    for path in [output_path, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    X, labels, graph_zero_fallback = load_aligned_inputs()
    y_np = labels.numpy().astype(int)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_np,
    )

    train_dataset = TensorDataset(X[train_idx], labels[train_idx].view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_val = X[val_idx]
    y_val = labels[val_idx].view(-1, 1)

    torch.manual_seed(RANDOM_STATE)
    model = AlignedFusionMLP(input_dim=128, dropout=0.3)
    pos_count = labels[train_idx].sum()
    neg_count = len(train_idx) - pos_count
    criterion = nn.BCEWithLogitsLoss(pos_weight=(neg_count / pos_count.clamp(min=1.0)).view(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_auc = -np.inf
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    final_metrics: dict[str, float] = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            seen += len(batch_labels)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_prob = torch.sigmoid(val_logits).view(-1).numpy()
        final_metrics = _binary_metrics(y_np[val_idx], val_prob)
        train_loss = total_loss / max(seen, 1)
        print(
            f"[INFO] epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_auc={final_metrics['roc_auc']:.6f} "
            f"val_pr_auc={final_metrics['pr_auc']:.6f}"
        )

        if final_metrics["roc_auc"] > best_auc:
            best_auc = final_metrics["roc_auc"]
            best_epoch = epoch
            best_metrics = dict(final_metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    payload = {
        "state_dict": best_state,
        "input_dim": 128,
        "dropout": 0.3,
        "random_state": RANDOM_STATE,
    }
    _safe_torch_save(payload, output_path)

    result = {
        "best_epoch": int(best_epoch),
        "val_auc": float(best_metrics["roc_auc"]),
        "val_pr_auc": float(best_metrics["pr_auc"]),
        "val_f1_at_0_5": float(best_metrics["f1@0.5"]),
        "last_epoch_val_auc": float(final_metrics["roc_auc"]),
        "last_epoch_val_pr_auc": float(final_metrics["pr_auc"]),
        "graph_used_zero_fallback": graph_zero_fallback,
        "n_borrowers": int(len(labels)),
    }
    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {output_path}")
    print(
        f"[INFO] Final aligned fusion Val AUC={result['val_auc']:.6f} "
        f"PR-AUC={result['val_pr_auc']:.6f} "
        f"graph_zero_fallback={graph_zero_fallback} "
        f"N={len(labels):,}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_fusion_aligned(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
