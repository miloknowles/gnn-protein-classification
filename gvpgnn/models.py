from typing import Optional

import torch
import torch.nn as nn

from pydantic import BaseModel

from .gvp_core import GVP, GVPConvLayer, LayerNorm
from torch_scatter import scatter_mean


class ClassifierGNNParams(BaseModel):
  node_in_dim: tuple[int, int]
  node_h_dim: tuple[int, int]
  edge_in_dim: tuple[int, int]
  edge_h_dim: tuple[int, int]
  n_categories: int = 10,
  num_layers: int = 4,
  drop_rate: float = 0.1


class ClassifierGNN(nn.Module):
  """
  An adapted GVP-GNN for the protein structure classification task.
  
  Usage
  -----
  This model takes in protein structure graphs of type `torch_geometric.data.Data` 
  or `torch_geometric.data.Batch` and returns a categorical distribution over
  possible labels for each graph in the batch in a `torch.Tensor` of shape [n_nodes]
  
  In this specific problem formulation, the output labels are 10 different
  architecture level classifications of protein domains, based on CATH.
  
  Should be used with `gvpgnn.datasets.ProteinGraphDataset`, or with generators
  of `torch_geometric.data.Batch` objects with the same attributes.

  References
  ----------
  * Code: https://github.com/drorlab/gvp-pytorch/blob/main/run_cpd.py
  * Paper: https://arxiv.org/abs/2009.01411
  
  Parameters
  ----------
  `node_in_dim`: Node dimensions in input graph. Should be (6, 3) if using original features.
  `node_h_dim`: Node dimensions to use in internal GVP-GNN layers. Authors use (100, 16).
  `node_in_dim`: Edge dimensions in input graph. Should be (32, 1) if using original features.
  `edge_h_dim`: Edge dimensions to embed to before use in GVP-GNN layers. Authors use (32, 1).
  `n_categories`: The number of output categories for classification
  `num_layers`: The number of internal GVP-GNN layers.
  `drop_rate`: The rate to use in all dropout layers.
  """
  def __init__(
    self,
    node_in_dim: tuple[int, int],
    node_h_dim: tuple[int, int],
    edge_in_dim: tuple[int, int],
    edge_h_dim: tuple[int, int],
    n_categories: int = 10,
    num_layers: int = 3,
    drop_rate: float = 0.1
  ):
    super(ClassifierGNN, self).__init__()
    self.n_categories = n_categories

    # Map the NODE embeddings to their hidden dimension.
    self.W_v = nn.Sequential(
      GVP(node_in_dim, node_h_dim, activations=(None, None)),
      LayerNorm(node_in_dim),
    )

    # Map the EDGE embeddings to their hidden dimension.
    self.W_e = nn.Sequential(
      GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
      LayerNorm(edge_in_dim),
    )

    # Apply a variable number of messaging passing updates (with GVPs used internally).
    self.layers = nn.ModuleList(
      GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(num_layers)
    )

    # Collapse the vector dimension, leaving only the scalar part of the embeddings
    # at each node. Note that the GVP modules allow information to flow from vector
    # embeddings to scalar embeddings, so the information in vector embeddings is
    # not lost during this operation.
    ns, _ = node_h_dim
    self.W_out = nn.Sequential(
      GVP(node_h_dim, (ns, 0), activations=(None, None)),
      LayerNorm(node_h_dim)
    )

    # Final dense block, which receives the average node embedding and outputs logits.
    self.dense = nn.Sequential(
      nn.Linear(ns, 2*ns),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop_rate),
      nn.Linear(2*ns, 2*ns),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop_rate),
      nn.Linear(2*ns, n_categories)
    )

  def forward(
    self,
    h_V: tuple[torch.Tensor, torch.Tensor],
    edge_index,
    h_E: tuple[torch.Tensor, torch.Tensor],
    graph_indices: Optional[torch.Tensor] = None
  ):
    """
    Outputs the categorical distribution over labels.

    Parameters
    ----------
    * `h_V`: tuple (s, V) of node embeddings
    * `edge_index`: `torch.Tensor` of shape [2, num_edges]
    * `h_E`: tuple (s, V) of edge embeddings
    * `graph_indices`: the graph index that each node belongs to
    """
    h_V = self.W_v(h_V)
    h_E = self.W_e(h_E)

    for layer in self.layers:
      h_V = layer(h_V, edge_index, h_E)

    out = self.W_out(h_V)

    # If we're NOT doing inference over batches, the 0th dimension is the node
    # dimension. If the input data is batched, then the `graph_indices` input
    # maps each node index to a graph in the batch. For example, if batch[i] = k,
    # that means that node i belongs to graph k.
    if graph_indices is None:
      out = out.mean(dim=0, keepdims=True)
    else:
      out = scatter_mean(out, graph_indices, dim=0)

    logits = self.dense(out).squeeze(-1)
    return logits