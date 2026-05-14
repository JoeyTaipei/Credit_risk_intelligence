"""Generate aligned 32-dim tabular embeddings for cs-training2 borrowers."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.features_v2 import TARGET_COL, load_credit_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training2.csv"
BATCH_SIZE = 10_000
EMBEDDING_DIM = 32


def _safe_torch_save(tensor: torch.Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    torch.save(tensor, path)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _fold_model_paths() -> list[Path]:
    zero_based = [PROCESSED_DIR / f"xgb_fold_{i}.pkl" for i in range(5)]
    one_based = [PROCESSED_DIR / f"xgb_fold_{i}.pkl" for i in range(1, 6)]
    if all(path.exists() for path in zero_based):
        return zero_based
    if all(path.exists() for path in one_based):
        return one_based
    missing = [str(path) for path in zero_based + one_based if not path.exists()]
    raise FileNotFoundError(f"Could not find a complete 5-fold XGBoost model set: {missing}")


def _xgb_leaf_table_size(model: Any) -> int:
    try:
        tree_df = model.get_booster().trees_to_dataframe()
        return int(tree_df["Node"].max()) + 1
    except Exception:
        return 4096


def _make_borrower_ids(data_path: Path) -> torch.Tensor:
    raw_df = pd.read_csv(data_path)
    unnamed_cols = [column for column in raw_df.columns if column.startswith("Unnamed")]
    if unnamed_cols:
        ids = raw_df.loc[raw_df["age"] != 0, unnamed_cols[0]].to_numpy()
    else:
        ids = np.arange(len(raw_df), dtype=np.int64)[raw_df["age"].to_numpy() != 0]
    return torch.as_tensor(ids, dtype=torch.long)


def generate_tabular_embeddings(
    data_path: Path = DEFAULT_DATA_PATH,
    batch_size: int = BATCH_SIZE,
) -> dict[str, tuple[int, ...]]:
    """Create aligned tabular embeddings, labels, and borrower ids."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tabular_path = PROCESSED_DIR / "tabular_emb_aligned.pt"
    labels_path = PROCESSED_DIR / "aligned_labels.pt"
    ids_path = PROCESSED_DIR / "aligned_borrower_ids.pt"
    metadata_path = PROCESSED_DIR / "tabular_emb_aligned_metadata.json"

    for path in [tabular_path, labels_path, ids_path, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    feature_df = load_credit_features_v2(data_path)
    X = feature_df.drop(columns=[TARGET_COL])
    y = torch.as_tensor(feature_df[TARGET_COL].astype(int).to_numpy(), dtype=torch.float32)
    borrower_ids = _make_borrower_ids(data_path)
    if len(borrower_ids) != len(X):
        raise ValueError(f"Borrower id count {len(borrower_ids)} does not match feature rows {len(X)}")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    embeddings = torch.zeros((len(X), EMBEDDING_DIM), dtype=torch.float32)
    model_paths = _fold_model_paths()

    for fold_idx, model_path in enumerate(model_paths, start=1):
        print(f"[INFO] Loading XGBoost fold model {fold_idx}: {model_path.name}")
        model = _load_pickle(model_path)
        table_size = _xgb_leaf_table_size(model)
        torch.manual_seed(42 + fold_idx)
        embedding_table = nn.Embedding(table_size, EMBEDDING_DIM)
        fold_embeddings = torch.zeros_like(embeddings)

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                leaves = model.apply(X.iloc[start:end]).astype(np.int64)
                max_leaf = int(leaves.max())
                if max_leaf >= table_size:
                    raise ValueError(
                        f"Fold {fold_idx} leaf index {max_leaf} exceeds embedding table size {table_size}"
                    )
                leaf_tensor = torch.as_tensor(leaves, dtype=torch.long)
                fold_embeddings[start:end] = embedding_table(leaf_tensor).mean(dim=1)

        embeddings += fold_embeddings / len(model_paths)

    _safe_torch_save(embeddings, tabular_path)
    _safe_torch_save(y, labels_path)
    _safe_torch_save(borrower_ids, ids_path)
    metadata = {
        "data_path": str(data_path),
        "n_borrowers": int(len(X)),
        "embedding_dim": EMBEDDING_DIM,
        "fold_models": [path.name for path in model_paths],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[INFO] Saved {tabular_path} shape={tuple(embeddings.shape)}")
    print(f"[INFO] Saved {labels_path} shape={tuple(y.shape)}")
    print(f"[INFO] Saved {ids_path} shape={tuple(borrower_ids.shape)}")
    return {
        "tabular": tuple(embeddings.shape),
        "labels": tuple(y.shape),
        "borrower_ids": tuple(borrower_ids.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tabular_embeddings(data_path=args.data_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
