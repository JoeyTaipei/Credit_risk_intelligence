"""Generate aligned GraphSAGE embeddings for cs-training2 borrowers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from src.data.features_v2 import TARGET_COL, load_credit_features_v2
from src.models.gnn_encoder import GraphSAGEEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training2.csv"
DEFAULT_GNN_PATH = PROCESSED_DIR / "gnn_encoder.pt"
BATCH_SIZE = 10_000
GRAPH_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "age",
    "total_past_due",
]


def _safe_torch_save(tensor: torch.Tensor, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    torch.save(tensor, path)


def _build_topk_cosine_graph(
    feature_df,
    threshold: float = 0.85,
    max_neighbors: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match graph_builder logic with a scalable top-k nearest-neighbor pass."""
    X = feature_df[GRAPH_FEATURES].fillna(feature_df[GRAPH_FEATURES].median()).to_numpy()
    X_norm = MinMaxScaler().fit_transform(X).astype(np.float32)
    n = X_norm.shape[0]

    nn_model = NearestNeighbors(
        n_neighbors=max_neighbors + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn_model.fit(X_norm)
    distances, indices = nn_model.kneighbors(X_norm, return_distance=True)
    similarities = 1.0 - distances

    edge_set: set[tuple[int, int]] = set()
    for src in range(n):
        kept = 0
        for dst, sim in zip(indices[src], similarities[src]):
            if dst == src:
                continue
            if sim <= threshold:
                continue
            edge_set.add((src, int(dst)))
            edge_set.add((int(dst), src))
            kept += 1
            if kept >= max_neighbors:
                break

    if edge_set:
        edges = np.array(sorted(edge_set), dtype=np.int64)
        edge_index = torch.as_tensor(edges.T, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    node_features = torch.as_tensor(X_norm, dtype=torch.float32)
    return edge_index, node_features


def generate_graph_embeddings(
    data_path: Path = DEFAULT_DATA_PATH,
    gnn_path: Path = DEFAULT_GNN_PATH,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, ...]:
    """Create aligned GraphSAGE embeddings or a logged zero-vector fallback."""
    output_path = PROCESSED_DIR / "graph_emb_aligned.pt"
    metadata_path = PROCESSED_DIR / "graph_emb_aligned_metadata.json"
    for path in [output_path, metadata_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    feature_df = load_credit_features_v2(data_path)
    n = len(feature_df)
    used_zero_fallback = False
    fallback_reason = ""
    num_edges = 0

    try:
        if not gnn_path.exists():
            raise FileNotFoundError(f"Missing GraphSAGE checkpoint: {gnn_path}")
        edge_index, node_features = _build_topk_cosine_graph(feature_df)
        num_edges = int(edge_index.shape[1])
        encoder = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
        encoder.load_state_dict(torch.load(gnn_path, map_location="cpu"))
        encoder.eval()
        with torch.no_grad():
            output = encoder(node_features, edge_index).detach().cpu().float()
    except Exception as exc:
        used_zero_fallback = True
        fallback_reason = str(exc)
        output = torch.zeros((n, 32), dtype=torch.float32)
        print(f"[WARN] GraphSAGE zero-vector fallback: {n} borrowers")
        print(f"[WARN] GraphSAGE fallback reason: {fallback_reason}")

    if output.shape != (n, 32):
        used_zero_fallback = True
        fallback_reason = f"Unexpected GraphSAGE shape {tuple(output.shape)}"
        output = torch.zeros((n, 32), dtype=torch.float32)
        print(f"[WARN] GraphSAGE zero-vector fallback: {n} borrowers")

    _safe_torch_save(output, output_path)
    metadata = {
        "data_path": str(data_path),
        "n_borrowers": int(n),
        "embedding_dim": 32,
        "threshold": 0.85,
        "max_neighbors": 10,
        "num_edges": num_edges,
        "used_zero_fallback": used_zero_fallback,
        "fallback_reason": fallback_reason,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {output_path} shape={tuple(output.shape)}")
    print(f"[INFO] Graph metadata: used_zero_fallback={used_zero_fallback}, num_edges={num_edges}")
    return tuple(output.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--gnn-path", type=Path, default=DEFAULT_GNN_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_graph_embeddings(
        data_path=args.data_path,
        gnn_path=args.gnn_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
