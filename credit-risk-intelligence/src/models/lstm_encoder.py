"""Bi-LSTM encoder that maps a borrower's monthly payment sequence to a fixed embedding."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """Encode a variable-length payment sequence into a compact borrower embedding.

    Input:  (batch, seq_len, input_size)  — one row per timestep per borrower
    Output: (batch, embedding_dim)        — one fixed-size vector per borrower

    The embedding is consumed by the late-fusion MLP on Day 5; keeping it at
    embedding_dim=32 prevents the LSTM from dominating the fusion input by
    sheer dimensionality relative to the other three encoder outputs.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        embedding_dim: int = 32,
    ) -> None:
        """
        Args:
            input_size:    Number of features per timestep (utilization, payment_ratio,
                           is_late, balance — matches create_synthetic_time_series output).
            hidden_size:   LSTM hidden units per layer.  64 is large enough to capture
                           sequential delinquency patterns while staying lightweight.
            num_layers:    Stacked LSTM depth.  2 layers lets the upper layer learn
                           abstractions over patterns found by the lower layer.
            dropout:       Applied between LSTM layers (not after the last layer).
                           Regularises the recurrent weights to prevent memorising
                           the synthetic noise in the training sequences.
            embedding_dim: Output embedding size.  32 dimensions — one quarter of
                           hidden_size — acts as an information bottleneck, forcing
                           the projection to distil only the most predictive
                           temporal signal for the fusion classifier.
        """
        super().__init__()

        # batch_first=True: expects input shape (batch, seq_len, features).
        # PyTorch's default is (seq_len, batch, features), which requires an
        # awkward transpose before every forward call.  batch_first matches the
        # shape produced by a standard DataLoader and is easier to reason about.
        #
        # dropout is applied to the output of each LSTM layer *except* the last,
        # so it has no effect when num_layers == 1.  Guard against a misleading
        # PyTorch warning by zeroing it in that case.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projects the LSTM's last hidden state from hidden_size → embedding_dim.
        # This is a learned compression: the linear layer can up-weight the hidden
        # dimensions most correlated with default risk and suppress noise dimensions.
        self.projection = nn.Linear(hidden_size, embedding_dim)

        # Expose embedding_dim so the fusion layer can query it without
        # instantiating the encoder to find out.
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of payment sequences through the LSTM and return embeddings.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch, embedding_dim).
        """
        # lstm() returns (output, (h_n, c_n)).
        #   output: (batch, seq_len, hidden_size)  — all hidden states, every step
        #   h_n:    (num_layers, batch, hidden_size) — final hidden state per layer
        #   c_n:    (num_layers, batch, hidden_size) — final cell state per layer
        # We discard output and c_n; only h_n carries the learned sequence summary.
        _, (h_n, _) = self.lstm(x)

        # h_n[-1]: select the top (last) LSTM layer's final hidden state.
        # Shape: (batch, hidden_size).
        #
        # Why last hidden state instead of mean pooling over all timesteps?
        # The LSTM's forget and input gates have already decided — during the
        # forward pass — which timestep information to accumulate and which to
        # discard.  h_n[-1] is the result of that learned compression.
        # Mean pooling treats all timesteps equally, throwing away that sequential
        # selectivity; the last hidden state preserves it.
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Project to the compact embedding space consumed by the fusion layer.
        embedding = self.projection(last_hidden)  # (batch, embedding_dim)

        return embedding
