"""GraphSAGE encoder that maps borrower nodes to fixed-size embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# SAGEConv is PyG's GraphSAGE convolution layer.  Imported here; callers must
# have torch_geometric installed (pip install torch-geometric).
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Encode each borrower node in a similarity graph into a 32-dim embedding.

    Input:  node features (n_nodes, input_dim)  +  edge_index (2, num_edges)
    Output: node embeddings (n_nodes, embedding_dim)

    The embedding for each node aggregates information from borrowers up to
    num_layers hops away — with 2 layers, each borrower sees its direct
    neighbours AND its neighbours' neighbours (2-hop receptive field).
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            input_dim:     Number of node features per borrower (matches the
                           5-column matrix returned by build_borrower_graph).
            hidden_dim:    Width of the intermediate GraphSAGE layer.  64 gives
                           enough capacity to learn non-linear combinations of
                           the 5 credit features across the neighbourhood.
            embedding_dim: Final node embedding size.  32 dimensions — the same
                           target as LSTMEncoder — so no single modality
                           dominates the fusion input by dimensionality.
            num_layers:    Number of SAGEConv message-passing rounds.
                           2 layers = 2-hop receptive field: a borrower aggregates
                           from its direct neighbours, then those neighbours
                           aggregate from *their* neighbours.  In credit terms:
                           "borrower → borrower's cohort → cohort's cohort".
                           A 3rd layer rarely helps on small graphs and risks
                           over-smoothing (all embeddings converge to the same
                           vector as hops increase).
            dropout:       Applied between the two SAGEConv layers to regularise
                           neighbourhood aggregation.  Without dropout, nodes
                           with many edges can over-fit to their specific
                           neighbourhood composition.
        """
        super().__init__()

        # WHY SAGEConv over GCNConv:
        # GCNConv normalises by the symmetric degree matrix of the *full* graph,
        # which requires materialising the entire adjacency at every forward pass
        # and is transductive (cannot generalise to unseen nodes).
        # SAGEConv uses neighbourhood *sampling* (mini-batch friendly) and
        # concatenates the node's own representation with its aggregated
        # neighbours, making it inductive — it can produce embeddings for new
        # borrowers who were not in the training graph, which is essential when
        # new applicants arrive daily in production.
        self.conv1 = SAGEConv(input_dim, hidden_dim)

        # Second layer aggregates 2-hop information: each node's updated hidden
        # state already encodes 1-hop context; passing through conv2 folds in
        # the 1-hop-updated representations of its neighbours, giving effective
        # 2-hop coverage without an explicit 3rd layer.
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Propagate node features through the graph and return node embeddings.

        Args:
            x:          Node feature matrix, shape (n_nodes, input_dim).
            edge_index: Graph connectivity in COO format, shape (2, num_edges).

        Returns:
            Node embedding matrix, shape (n_nodes, embedding_dim).
        """
        # Layer 1: each node aggregates its 1-hop neighbours' raw features.
        # SAGEConv concatenates [self_features || mean(neighbour_features)] and
        # applies a learned linear transformation — this "mean aggregator" is
        # computationally cheap and empirically competitive on dense tabular node
        # features like ours.
        x = self.conv1(x, edge_index)   # (n_nodes, hidden_dim)
        x = F.relu(x)                   # non-linearity allows learning of
                                        # non-linear cohort risk interactions

        # Dropout: randomly zeros node representations between layers.
        # This prevents the model from memorising specific neighbour identities
        # and forces it to learn robust, neighbourhood-agnostic credit signals.
        x = self.dropout(x)

        # Layer 2: each node now aggregates the *updated* 1-hop representations
        # of its neighbours, giving an effective 2-hop receptive field.
        # No activation after the final layer — the fusion MLP downstream will
        # apply its own non-linearity.  Keeping raw logits also makes the
        # embedding space more flexible for the linear projection in fusion.py.
        x = self.conv2(x, edge_index)   # (n_nodes, embedding_dim)

        return x
