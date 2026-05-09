"""Late fusion classifier: concatenates four 32-dim modality embeddings → risk logit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_xgb_leaf_embeddings(
    xgb_model,
    X: pd.DataFrame,
    dim: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Convert XGBoost tree leaf assignments to dense borrower embeddings.

    XGBoost's model.apply() returns, for each sample, the integer index of the
    leaf node it landed in for every tree.  Two samples that land in the same
    leaf have identical risk profiles according to that tree's splits.  By
    embedding leaf indices into a dense vector space we convert this categorical
    structure into a representation an MLP can operate on.

    WHY leaf embeddings over raw features fed directly into the MLP:
        (1) Leaf assignment is a non-linear, interaction-aware summary of all
            tabular features — XGBoost has already applied optimal splits and
            captured interactions (e.g. "age < 35 AND utilization > 0.8").
            Passing raw features to the MLP would require it to re-learn those
            interactions from scratch with far less data.
        (2) Samples in the same leaf receive the SAME embedding vector, which
            acts as a learned cluster prototype.  This imposes a useful inductive
            bias: borrowers the tree model considers equivalent get the same
            fusion-layer starting point.
        (3) It decouples the tabular signal from the MLP's architectural choices,
            making the tabular modality a clean "plug" like the other three.

    NOTE: XGBoost operates outside PyTorch's autograd graph — gradients cannot
    flow through model.apply().  The embedding table is therefore initialised
    with a fixed seed and NOT trained.  The credit signal lives in WHICH leaf
    a sample falls into, not in the embedding table's exact values.

    Args:
        xgb_model: Trained XGBClassifier (must be fitted, supports .apply()).
        X:         Feature DataFrame with the same column names used at training.
        dim:       Output embedding dimension per borrower.
        seed:      Seed for the random embedding initialisation.

    Returns:
        FloatTensor of shape (n_samples, dim).
    """
    # model.apply() requires a DataFrame to preserve feature names;
    # passing a raw ndarray raises a feature-name validation error.
    # Returns float32 node indices — cast to int64 for nn.Embedding lookup.
    leaves = xgb_model.apply(X).astype(np.int64)  # (n_samples, n_trees)
    n_samples, n_trees = leaves.shape

    # XGBoost leaf indices are positional node IDs in the full binary tree, not
    # contiguous integers starting at 0.  E.g. tree 0 uses {10, 12, 35, …, 80}.
    # Setting num_embeddings = max_index + 1 allocates one slot per possible
    # node position; unused slots consume a few KB but avoid any re-indexing.
    max_leaf = int(leaves.max()) + 1  # observed max = 90 → table size = 91

    torch.manual_seed(seed)
    embedding_table = nn.Embedding(num_embeddings=max_leaf, embedding_dim=dim)
    # embedding_table is intentionally not added to self — it lives here only
    # to convert the discrete leaf space to a dense vector space.

    leaf_tensor = torch.tensor(leaves, dtype=torch.long)  # (n_samples, n_trees)

    with torch.no_grad():
        # Look up each tree's leaf embedding: (n_samples, n_trees, dim)
        tree_embeddings = embedding_table(leaf_tensor)

        # Mean pool across trees: each tree in the XGBoost ensemble has roughly
        # equal predictive authority (all are trained on the full dataset with
        # the same learning rate), so uniform averaging is appropriate.
        # This also keeps output dimensionality fixed regardless of n_trees.
        tabular_emb = tree_embeddings.mean(dim=1)  # (n_samples, dim)

    return tabular_emb.float()


class LateFusionClassifier(nn.Module):
    """MLP that fuses four modality embeddings into a default-risk logit.

    Takes four pre-computed 32-dim embeddings (one per modality) and produces
    a single raw logit.  Apply torch.sigmoid() to convert to P(default).

    Input:  four FloatTensors of shape (batch, 32)
    Output: FloatTensor of shape (batch, 1) — raw logit
    """

    def __init__(
        self,
        tabular_dim: int = 32,
        lstm_dim: int = 32,
        gnn_dim: int = 32,
        text_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            tabular_dim: XGBoost leaf embedding dimension.
            lstm_dim:    LSTM time-series embedding dimension.
            gnn_dim:     GraphSAGE node embedding dimension.
            text_dim:    sentence-BERT projection dimension.
            hidden_dim:  Width of the first hidden layer.  64 is large enough
                         to learn cross-modal interactions while staying
                         regularisable on 840 training samples.
            dropout:     Dropout rate between hidden layers.  Set to 0.3 (higher
                         than each individual encoder's 0.2) because the fusion
                         input is 128-dim and the risk of co-adaptation across
                         modality dimensions is higher than within a single encoder.
        """
        super().__init__()

        # The concatenated embedding is physically:
        #   [tabular | lstm | gnn | text]
        #   dim 0-31:   XGBoost leaf cluster prototype (what kind of risk profile)
        #   dim 32-63:  LSTM temporal behaviour (how payment behaviour evolved)
        #   dim 64-95:  GraphSAGE social cohort (who the borrower is similar to)
        #   dim 96-127: sentence-BERT loan narrative (what the borrower claimed)
        concat_dim = tabular_dim + lstm_dim + gnn_dim + text_dim  # 128

        # WHY late fusion (concatenate after individual encoders):
        #   Each encoder learned representations in its own feature space using
        #   its own optimal inductive bias (trees for tabular, recurrence for
        #   sequences, message passing for graphs, transformers for text).
        #   Concatenating at the decision layer preserves that structure.  Early
        #   fusion would force a single model to handle all four modalities
        #   simultaneously and destroy modality-specific preprocessing pipelines.
        #
        # WHY concat over learned attention weights across modalities:
        #   Attention-based fusion (e.g. learned softmax weights per modality)
        #   would allow "trust text less, trust tabular more for this borrower",
        #   which is more expressive.  But with 840 training samples and four
        #   modalities of synthetic data, the extra parameters risk overfitting.
        #   Concat is the right call for this portfolio demo; attention fusion
        #   would be the upgrade path with real data.
        self.classifier = nn.Sequential(
            # Layer 1: 128 → hidden_dim
            # Learns to combine cross-modal signals: e.g. "high LSTM delinquency
            # AND low GNN risk (isolated borrower) = moderate overall risk".
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: hidden_dim → 32
            # Second compression step forces the network to distil the most
            # credit-predictive combination of cross-modal features before the
            # final linear classifier.
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output logit: 32 → 1
            # No sigmoid here — BCEWithLogitsLoss is numerically more stable than
            # BCE(sigmoid(x)) and is used during training.  At inference time,
            # apply torch.sigmoid(logit) to get P(default ∈ [0, 1]).
            nn.Linear(32, 1),
        )

    def forward(
        self,
        tabular_emb: torch.Tensor,
        lstm_emb: torch.Tensor,
        gnn_emb: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse modality embeddings and return a default-risk logit.

        Args:
            tabular_emb: XGBoost leaf embeddings, shape (batch, tabular_dim).
            lstm_emb:    LSTM time-series embeddings, shape (batch, lstm_dim).
            gnn_emb:     GraphSAGE node embeddings, shape (batch, gnn_dim).
            text_emb:    sentence-BERT projected embeddings, shape (batch, text_dim).

        Returns:
            Raw logit of shape (batch, 1).
        """
        # Concatenate all four modality embeddings along the feature axis.
        # Each embedding is already (batch, 32); result is (batch, 128).
        fused = torch.cat([tabular_emb, lstm_emb, gnn_emb, text_emb], dim=1)

        # Feed through the three-layer MLP to produce the risk logit.
        logit = self.classifier(fused)  # (batch, 1)

        return logit
