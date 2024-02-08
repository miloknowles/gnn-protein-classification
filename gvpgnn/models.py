from typing import Optional

import torch
import torch.nn as nn

from pydantic import BaseModel

from .gvp_core import GVP, GVPConvLayer, LayerNorm

from torch_geometric.nn import (
  TransformerConv,
  global_mean_pool as gap,
  global_add_pool as gsp,
  global_max_pool as gmp,
  TopKPooling
)


class ClassifierGNNParams(BaseModel):
  """Parameters for the model.
  
  This Pydantic class mainly helps us strip the relevant parameters out of a
  larger configuration object.
  """
  node_in_dim: tuple[int, int]
  node_h_dim: tuple[int, int]
  edge_in_dim: tuple[int, int]
  edge_h_dim: tuple[int, int]
  n_categories: int = 10
  n_gvp_layers: int = 3
  n_pool_layers: int = 3
  n_conv_heads: int = 1
  drop_rate: float = 0.1
  pooling_op: str = "conv"


class NaiveGlobalPooling(nn.Module):
  """
  Applies simple global pooling operations to the graph.
  """
  def __init__(self, node_scalar_dim: int):
    super(NaiveGlobalPooling, self).__init__()
    self.node_scalar_dim = node_scalar_dim

  def get_output_dim(self) -> int:
    """Returns the output dimensionality of this block."""
    return 3 * self.node_scalar_dim

  def forward(self, x, edge_index, graph_indices):
    return torch.concat([
      gmp(x, graph_indices),
      gap(x, graph_indices),
      gsp(x, graph_indices)
    ], dim=-1)


class NoPoolingBlock(nn.Module):
  """
  An identity module that provides the same interface as the other blocks.
  """
  def __init__(self, node_scalar_dim: int):
    super(NoPoolingBlock, self).__init__()
    self.node_scalar_dim = node_scalar_dim

  def get_output_dim(self) -> int:
    """Returns the output dimensionality of this block."""
    return self.node_scalar_dim

  def forward(self, x, edge_index, graph_indices):
    return x


class TransformerConvPoolingBlock(nn.Module):
  """
  Applies one or more graph convolutions with attention.
  """
  def __init__(self, node_scalar_dim: int, n_conv_layers: int = 2, n_conv_heads: int = 2, drop_rate: float = 0.1):
    super(TransformerConvPoolingBlock, self).__init__()

    self.node_scalar_dim = node_scalar_dim
    self.n_conv_heads = n_conv_heads

    # Apply convolution(s) with attention to help propagate more information around the graph.
    self.conv_layers = nn.ModuleList(
      TransformerConv(
        node_scalar_dim,
        node_scalar_dim,
        concat=True,
        beta=False,
        dropout=drop_rate,
        edge_dim=None,
        heads=n_conv_heads,
      ) for _ in range(n_conv_layers)
    )

    # Transform multi-head output back into a single head.
    self.transf_layers = nn.ModuleList(
      nn.Linear(node_scalar_dim*n_conv_heads, node_scalar_dim) for _ in range(n_conv_layers)
    )

  def get_output_dim(self) -> int:
    """Returns the output dimensionality of this block."""
    return 3 * self.node_scalar_dim

  def forward(self, x, edge_index, graph_indices):
    for i in range(len(self.conv_layers)):
      x = torch.relu(
        self.transf_layers[i](
          self.conv_layers[i](x, edge_index)
        )
      )
    return torch.concat([
      gmp(x, graph_indices),
      gap(x, graph_indices),
      gsp(x, graph_indices)
    ], dim=-1)


class TopKPoolingBlock(nn.Module):
  """
  Applies graph convolutions and Top-K Pooling to steadily reduce the size of the graph.

  Based on: https://github.com/deepfindr/gnn-project/blob/main/model.py
  """
  def __init__(self, node_scalar_dim: int, n_pool_layers: int = 3, n_conv_heads: int = 1, drop_rate: float = 0.1):
    super(TopKPoolingBlock, self).__init__()
    self.conv_layers = nn.ModuleList([])
    self.transf_layers = nn.ModuleList([])
    self.pooling_layers = nn.ModuleList([])
    self.bn_layers = nn.ModuleList([])
    self.node_scalar_dim = node_scalar_dim
    
    for _ in range(n_pool_layers):
      self.conv_layers.append(
        TransformerConv(
          node_scalar_dim,
          node_scalar_dim,
          concat=True,
          beta=False,
          dropout=drop_rate,
          edge_dim=None,
          heads=n_conv_heads,
        )
      )
      self.transf_layers.append(
        nn.Linear(node_scalar_dim*n_conv_heads, node_scalar_dim)
      )
      self.bn_layers.append(
        nn.BatchNorm1d(node_scalar_dim)
      )
      self.pooling_layers.append(
        TopKPooling(node_scalar_dim, ratio=0.5) 
      )

  def get_output_dim(self) -> int:
    """Returns the output dimensionality of this block."""
    return len(self.pooling_layers) * 3 * self.node_scalar_dim

  def forward(self, x, edge_index, graph_indices):
    _edge_index = edge_index # modified
    _graph_indices = graph_indices # modified

    global_features = []
  
    for i in range(len(self.pooling_layers)):
      x = self.conv_layers[i](x, _edge_index)
      x = torch.relu(self.transf_layers[i](x))
      x = self.bn_layers[i](x)
      x, _edge_index, _, _graph_indices, _, _ = \
        self.pooling_layers[i](x, _edge_index, None, _graph_indices)
      global_features.append(torch.cat([
        gmp(x, _graph_indices),
        gap(x, _graph_indices),
        gsp(x, _graph_indices)
      ], dim=-1))

    return torch.concat(global_features, dim=-1)


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
  `n_gvp_layers`: The number of internal GVP-GNN layers.
  `drop_rate`: The rate to use in all dropout layers.
  """
  def __init__(
    self,
    node_in_dim: tuple[int, int],
    node_h_dim: tuple[int, int],
    edge_in_dim: tuple[int, int],
    edge_h_dim: tuple[int, int],
    n_categories: int = 10,
    n_gvp_layers: int = 3,
    n_pool_layers: int = 3,
    drop_rate: float = 0.1,
    n_conv_heads: int = 1,
    pooling_op: str = "topk"
  ):
    super(ClassifierGNN, self).__init__()
    self.n_categories = n_categories
    ns, nv = node_h_dim

    # First, embed the (potential high-dimensional) scalar features for each node
    # to a lower dimension. Since the features come from a static, pre-trained network,
    # this gives the model a chance to modify them before message passing.
    self.W_features = nn.Sequential(
      nn.Linear(node_in_dim[0], ns*2),
      nn.ReLU(inplace=True),
      nn.Linear(ns*2, ns),
      nn.ReLU(inplace=True),
    )

    # Map the NODE embeddings to their hidden dimension.
    self.W_v = nn.Sequential(
      LayerNorm((ns, node_in_dim[1])),
      GVP((ns, node_in_dim[1]), node_h_dim, activations=(None, None)),
    )

    # Map the EDGE embeddings to their hidden dimension.
    self.W_e = nn.Sequential(
      LayerNorm(edge_in_dim),
      GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
    )

    # Apply a variable number of GVP-style messaging passing updates.
    self.gvp_conv_layers = nn.ModuleList(
      GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) for _ in range(n_gvp_layers)
    )

    # Apply one last GVP, and get rid of the vector (R^3) features in the process.
    # This leaves only scalar features for the final blocks leading to classification.
    self.W_out = nn.Sequential(
      LayerNorm(node_h_dim),
      GVP(node_h_dim, (ns, 0)),
    )

    if pooling_op == "naive":
      self.pooling_op = NaiveGlobalPooling(node_h_dim[0])
    elif pooling_op == "conv":
      self.pooling_op = TransformerConvPoolingBlock(
          node_h_dim[0], n_conv_layers=n_pool_layers, n_conv_heads=n_conv_heads, drop_rate=drop_rate
      )
    elif pooling_op == "topk":
      self.pooling_op = TopKPoolingBlock(
        node_h_dim[0], n_pool_layers=n_pool_layers, n_conv_heads=n_conv_heads, drop_rate=drop_rate
      )
    else:
      raise NotImplementedError(f"Unrecognized pooling operation {pooling_op}.")

    # Final dense block, which receives the pooled features as inputs, and outputs logits.
    self.dense = nn.Sequential(
      nn.Linear(self.pooling_op.get_output_dim(), 2*ns),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop_rate),
      nn.Linear(2*ns, ns),
      nn.ReLU(inplace=True),
      nn.Dropout(p=drop_rate),
      nn.Linear(ns, n_categories)
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
    h_V = self.W_v((self.W_features(h_V[0]), h_V[1]))
    h_E = self.W_e(h_E)

    for layer in self.gvp_conv_layers:
      h_V = layer(h_V, edge_index, h_E)

    x = self.W_out(h_V)
    x = self.pooling_op(x, edge_index, graph_indices)

    return self.dense(x).squeeze(-1)