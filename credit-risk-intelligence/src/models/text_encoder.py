"""Frozen sentence-BERT encoder with a learnable projection head for loan text."""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TextEncoder(nn.Module):
    """Encode loan application text into compact 32-dim borrower embeddings.

    Architecture:
        SentenceTransformer (frozen) → 384-dim sentence embedding
        nn.Linear(384, 32)           → 32-dim projection for fusion

    The SentenceTransformer weights are frozen by default.  Only the projection
    head is trained, which takes a few seconds per epoch and is robust even on
    small or synthetic corpora.

    Input:  list of strings (one per borrower)
    Output: FloatTensor of shape (n_texts, embedding_dim)
    """

    # all-MiniLM-L6-v2 produces 384-dim sentence embeddings.  Changing the
    # model_name requires updating this constant accordingly.
    _SBERT_OUTPUT_DIM: int = 384

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 32,
        freeze: bool = True,
    ) -> None:
        """
        Args:
            model_name:    HuggingFace / sentence-transformers model identifier.
                           all-MiniLM-L6-v2 is chosen over bert-base-uncased or
                           larger BERT variants for three reasons:
                           (1) It was fine-tuned specifically for semantic sentence
                               similarity tasks using contrastive learning on 1B+
                               sentence pairs — it already "speaks" the semantic
                               language of short texts like loan descriptions.
                           (2) It is 5× smaller than bert-base (22M vs 110M params)
                               and runs ~5× faster on CPU, critical for a 7-day
                               demo without a dedicated GPU.
                           (3) Its 384-dim output is already a high-quality
                               semantic summary; projecting to 32 from 384 is a
                               smaller information bottleneck than projecting from
                               768 (bert-base), which makes the projection head's
                               job easier.
            embedding_dim: Output size of the projection head.  Fixed at 32 to
                           match LSTMEncoder and GraphSAGEEncoder so no modality
                           dominates the fusion layer by dimensionality alone.
            freeze:        If True, disable gradient computation for all
                           SentenceTransformer parameters.  This is the correct
                           call when training on synthetic data:
                           (a) Synthetic loan descriptions lack genuine credit
                               signal — fine-tuning would overfit noise.
                           (b) The projection head has only 384×32 + 32 = 12,320
                               parameters, small enough to train cleanly on the
                               few hundred training samples available.
                           (c) Frozen pre-trained weights already encode rich
                               semantic knowledge; the projection head learns
                               only which 32 semantic dimensions are most
                               relevant to credit default prediction.
        """
        super().__init__()

        # Load the pre-trained sentence encoder.  On first run this downloads
        # ~90MB from HuggingFace; subsequent runs use the local cache.
        self.sentence_transformer = SentenceTransformer(model_name)

        if freeze:
            # Disable gradient tracking for all transformer parameters so that
            # back-propagation never updates the pre-trained weights.
            # This makes training faster, memory-lighter, and more stable on
            # small or synthetic datasets where the transformer would otherwise
            # overfit to spurious lexical patterns in the template-generated text.
            for param in self.sentence_transformer.parameters():
                param.requires_grad = False

        self._freeze = freeze

        # Projection head: compress 384-dim semantic embedding to 32-dim.
        # The learned linear mapping selects the semantic directions in the
        # 384-dim space that are most predictive of credit default given the
        # downstream supervised signal from the fusion classifier.
        # Risk of this compression: at 32/384 ≈ 8% retention we may lose
        # semantic nuance — mitigated by the fact that all-MiniLM is already
        # a compressed representation of full BERT.
        self.projection = nn.Linear(self._SBERT_OUTPUT_DIM, embedding_dim)

        self.embedding_dim = embedding_dim

    def encode_texts(self, texts: list[str]) -> torch.FloatTensor:
        """Encode a list of loan description strings into embeddings.

        Args:
            texts: Raw loan description strings, one per borrower.

        Returns:
            FloatTensor of shape (n_texts, embedding_dim).
        """
        # encode() returns a tensor on the same device as the model.
        # show_progress_bar=False keeps the training loop output clean.
        # batch_size=64 balances throughput with CPU memory on a laptop.
        sentence_embeddings = self.sentence_transformer.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=64,
        )  # (n_texts, 384)

        if self._freeze:
            # When the transformer is frozen its outputs have no grad_fn;
            # detach() makes that contract explicit and avoids a confusing
            # autograd error if the tensor happens to retain a computation graph
            # from a prior non-frozen call.
            sentence_embeddings = sentence_embeddings.detach()

        # Project from 384-dim semantic space to 32-dim fusion-ready embedding.
        # This is the only trainable step; the rest is deterministic inference.
        projected = self.projection(sentence_embeddings)  # (n_texts, embedding_dim)

        return projected

    def forward(self, texts: list[str]) -> torch.FloatTensor:
        """nn.Module interface; delegates to encode_texts.

        Keeping forward() consistent with the other encoders (LSTMEncoder,
        GraphSAGEEncoder) means the fusion layer can call any encoder with a
        single forward() pattern without special-casing the text modality.

        Args:
            texts: Raw loan description strings, one per borrower.

        Returns:
            FloatTensor of shape (n_texts, embedding_dim).
        """
        return self.encode_texts(texts)
