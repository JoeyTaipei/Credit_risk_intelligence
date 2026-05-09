from __future__ import annotations

import pandas as pd
import torch

from src.data.graph_builder import build_borrower_graph, get_graph_stats
from src.data.preprocess import clean_tabular, engineer_tabular_features
from src.models.gnn_encoder import GraphSAGEEncoder


def _graph_input(sample_credit_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_tabular(sample_credit_df)
    return engineer_tabular_features(cleaned)


def test_graph_builder_output_types(sample_credit_df: pd.DataFrame) -> None:
    edge_index, node_features = build_borrower_graph(_graph_input(sample_credit_df))

    assert edge_index.dtype == torch.long
    assert node_features.dtype == torch.float


def test_graph_builder_edge_index_shape(sample_credit_df: pd.DataFrame) -> None:
    edge_index, _ = build_borrower_graph(_graph_input(sample_credit_df))

    assert edge_index.shape[0] == 2


def test_gnn_forward_shape() -> None:
    model = GraphSAGEEncoder(input_dim=5, embedding_dim=32)
    node_features = torch.rand(100, 5)
    edge_index = torch.randint(0, 100, (2, 200), dtype=torch.long)

    output = model(node_features, edge_index)

    assert output.shape == (100, 32)


def test_graph_stats_keys(sample_credit_df: pd.DataFrame) -> None:
    edge_index, node_features = build_borrower_graph(_graph_input(sample_credit_df))

    stats = get_graph_stats(edge_index, n_nodes=node_features.shape[0])

    expected = {"num_nodes", "num_edges", "avg_degree", "density", "is_connected"}
    assert expected.issubset(stats)
