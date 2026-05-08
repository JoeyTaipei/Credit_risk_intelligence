"""Builds a borrower similarity graph from tabular features for PyTorch Geometric."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# The five features chosen for similarity capture distinct credit dimensions:
# utilization (behaviour), debt (leverage), income (capacity),
# age (credit history length), total_past_due (delinquency severity).
_GRAPH_FEATURES: list[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "age",
    "total_past_due",
]


def build_borrower_graph(
    df: pd.DataFrame,
    threshold: float = 0.85,
    max_neighbors: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a borrower similarity graph from tabular features.

    Each node is a borrower.  An edge (i, j) exists when borrower j is among
    borrower i's top-max_neighbors most similar borrowers AND their cosine
    similarity exceeds threshold.  Edges are added in both directions so the
    graph is undirected — GraphSAGE aggregates symmetrically.

    NOTE: similarity computation is O(n²) in memory and time.  This is fine
    for the 840-row demo split; for 150 000 rows use approximate nearest-
    neighbour search (e.g. faiss or annoy) before production deployment.

    Args:
        df:            Cleaned borrower DataFrame containing _GRAPH_FEATURES.
        threshold:     Minimum cosine similarity to draw an edge.  0.85 means
                       two borrowers share ≥85% of their directional credit
                       profile — a tight cluster that approximates "financially
                       similar cohort" without over-connecting the graph.
        max_neighbors: Maximum edges per node (out-degree cap).  Prevents hub
                       collapse where one typical borrower connects to everyone.
        seed:          Unused here (graph construction is deterministic) but
                       kept for API consistency across the pipeline.

    Returns:
        edge_index:    LongTensor of shape (2, num_edges) in PyG COO format.
                       Row 0 = source node indices, row 1 = destination node
                       indices.  COO (coordinate) format stores only the
                       non-zero positions of the adjacency matrix, which is
                       memory-efficient for sparse graphs and directly consumed
                       by all PyG conv layers.
        node_features: FloatTensor of shape (n_nodes, 5) — the MinMax-scaled
                       feature matrix used both as node input to GraphSAGE and
                       as the normalised representation for cosine similarity.
    """
    df = df.reset_index(drop=True)

    # Step 1 — Extract the five credit-dimension features.
    # Fill any residual NaNs defensively (should not occur after clean_tabular).
    X = df[_GRAPH_FEATURES].fillna(df[_GRAPH_FEATURES].median()).values.astype(np.float64)
    n = X.shape[0]

    # Step 2 — MinMax-scale each feature to [0, 1].
    # Scaling is mandatory before cosine similarity: without it, MonthlyIncome
    # (range ~$1 500–$23 000) would dominate the dot product and the similarity
    # would effectively collapse to income similarity only.
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X).astype(np.float32)

    # Step 3 — Compute full pairwise cosine similarity matrix (n × n).
    # WHY cosine over euclidean: cosine measures the *angle* between feature
    # vectors, not their magnitude.  Two borrowers with the same financial
    # behaviour pattern at different income levels will have high cosine
    # similarity but high euclidean distance.  Angle-based similarity captures
    # "same type of borrower" better than distance-based similarity when features
    # span very different scales — even after MinMax scaling, the distributions
    # differ in shape, making cosine the more robust choice.
    sim_matrix = cosine_similarity(X_norm)  # (n, n), values in [-1, 1]

    # Step 4 — Build edge lists: for each borrower i, find neighbors j where
    # sim(i, j) > threshold, then keep only the top max_neighbors by similarity.
    # WHY threshold=0.85: in a 5-dimensional normalised space, 0.85 cosine
    # similarity means the two borrowers' credit profiles point in nearly the
    # same direction.  In business terms this approximates "same financial cohort"
    # — useful for identifying risk clusters without connecting the entire graph.
    # WHY max_neighbors=10: caps hub formation.  Without a degree cap, a very
    # typical borrower (near the centroid) would connect to thousands of others,
    # dominating gradient flow during GraphSAGE training.
    np.fill_diagonal(sim_matrix, -1.0)  # exclude self-loops

    src_list: list[int] = []
    dst_list: list[int] = []

    for i in range(n):
        row = sim_matrix[i]
        # Indices of all borrowers above the similarity threshold
        candidates = np.where(row > threshold)[0]
        if len(candidates) == 0:
            continue
        # Sort candidates by descending similarity, keep top max_neighbors
        top_k = candidates[np.argsort(row[candidates])[::-1][:max_neighbors]]
        src_list.extend([i] * len(top_k))
        dst_list.extend(top_k.tolist())

    # Step 5 — Deduplicate and add reverse edges to make the graph undirected.
    # GraphSAGE aggregates information from neighbours; symmetric edges ensure
    # each borrower receives its neighbours' features regardless of who
    # "initiated" the edge during construction.
    edge_set: set[tuple[int, int]] = set()
    for s, d in zip(src_list, dst_list):
        edge_set.add((s, d))
        edge_set.add((d, s))  # reverse direction for undirected graph

    if edge_set:
        edges = np.array(sorted(edge_set), dtype=np.int64)  # sort for reproducibility
        # Step 5 cont. — COO format: row 0 = sources, row 1 = destinations.
        # PyG's message-passing layers expect exactly this layout; internally
        # they scatter-gather along this coordinate list.
        edge_index = torch.tensor(edges.T, dtype=torch.long)  # (2, num_edges)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Step 6 — Package the scaled feature matrix as the node feature tensor.
    # Same matrix used for similarity is reused as node input to GraphSAGE —
    # no information leakage since the features are not label-derived.
    node_features = torch.tensor(X_norm, dtype=torch.float)  # (n_nodes, 5)

    return edge_index, node_features


def get_graph_stats(edge_index: torch.Tensor, n_nodes: int) -> dict:
    """Compute summary statistics for a graph — used on the PPT slide.

    Args:
        edge_index: LongTensor of shape (2, num_edges) in COO format.
        n_nodes:    Total number of nodes (borrowers).

    Returns:
        Dictionary with keys: num_nodes, num_edges, avg_degree, density,
        is_connected.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    num_edges = edge_index.shape[1]
    avg_degree = round(num_edges / n_nodes, 2) if n_nodes > 0 else 0.0

    # Density = fraction of all possible directed edges that exist.
    possible = n_nodes * (n_nodes - 1)
    density = round(num_edges / possible, 6) if possible > 0 else 0.0

    # Weak connectivity: treat directed edges as undirected and check
    # whether all nodes belong to a single component.
    if num_edges > 0:
        rows = edge_index[0].numpy()
        cols = edge_index[1].numpy()
        adj = csr_matrix(
            (np.ones(num_edges), (rows, cols)), shape=(n_nodes, n_nodes)
        )
        n_components, _ = connected_components(adj, directed=False, connection="weak")
        is_connected = bool(n_components == 1)
    else:
        is_connected = False

    return {
        "num_nodes": n_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "density": density,
        "is_connected": is_connected,
    }
