"""Preprocessing stubs for upcoming Day deliverables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw GiveMeSomeCredit tabular data.

    Args:
        df: Raw borrower credit DataFrame.

    Returns:
        Cleaned borrower credit DataFrame.
    """
    raise NotImplementedError("Day 1 deliverable")


def create_synthetic_time_series(df: pd.DataFrame, window: int = 12) -> np.ndarray:
    """Create synthetic borrower time-series windows from tabular data.

    Args:
        df: Borrower credit DataFrame.
        window: Number of synthetic time steps to create per borrower.

    Returns:
        Synthetic time-series feature array.
    """
    raise NotImplementedError("Day 2 deliverable")


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer model-ready tabular features.

    Args:
        df: Cleaned borrower credit DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    raise NotImplementedError("Day 3 deliverable")


def build_borrower_graph(df: pd.DataFrame, threshold: float = 0.85) -> tuple:
    """Build borrower relationship graph inputs.

    Args:
        df: Borrower feature DataFrame.
        threshold: Similarity threshold for connecting borrower nodes.

    Returns:
        Tuple containing graph artifacts.
    """
    raise NotImplementedError("Day 4 deliverable")
