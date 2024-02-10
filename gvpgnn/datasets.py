from typing import Optional
import random
import glob
import json
import numpy as np
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import gvpgnn.embeddings as embeddings
import gvpgnn.voxels as voxels

from scipy.spatial.transform import Rotation


class RandomRotation3d(object):
  """Randomly apply a 3D rotation to a set of points.

  We sample a random rotation matrix and apply it to all of the points..
  """
  def __init__(self):
    pass

  def __call__(self, points: torch.Tensor):
    R = Rotation.random().as_matrix()
    return torch.matmul(points, R)


class ProteinVoxelDataset(data.Dataset):
  """
  A dataset for voxelizing proteins.
  """
  def __init__(
    self,
    json_folder: str,
    plm: Optional[str] = None,
    device: str = "cpu",
    voxel_grid_dim: int = 512,
    apply_random_rotation: bool = False,
  ):
    super(ProteinVoxelDataset, self).__init__()

    # This defines the ordering of examples in the dataset.
    self.json_folder = json_folder
    self.filenames = glob.glob(f"{self.json_folder}/*.json")
    self.voxel_grid_dim = voxel_grid_dim
    self.device = device
    self.plm = plm
    self.transform = RandomRotation3d() if apply_random_rotation else None

    if len(self.filenames) == 0:
      raise FileNotFoundError("Couldn't find any files in the dataset folder you passed. Does it exist and have JSON files in it?")

    # Count up the number of nodes in each graph (used by the sampler).
    self.node_counts = [0 for _ in range(len(self.filenames))]

    # Determine the frequence of labels in the dataset (used by the sampler).
    n_classes = 10
    self.label_counts = [0 for _ in range(n_classes)]
    self.node_labels = [0 for _ in range(len(self.filenames))]

    # Open up each file to see how many nodes it contains:
    for i, filename in enumerate(self.filenames):
      with open(filename, "r") as f:
        m = json.load(f)
        self.node_counts[i] = len(m['seq'])
        self.label_counts[m['task_label']] += 1
        self.node_labels[i] = m['task_label']

    # Calculate weights for each label.
    self.class_weights = (1 / n_classes) * (len(self.filenames) / np.array(self.label_counts)) # (10)
    # Calculate weights for each example in the dataset (does not need to sum to 1).
    self.sampler_weights = [self.class_weights[self.node_labels[i]] for i in range(len(self.node_labels))]

    self.letter_to_num = {
      'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
      'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
      'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
      'N': 2, 'Y': 18, 'M': 12, '_': -1, 
    }
    self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

  def __len__(self) -> int:
    return len(self.filenames)
  
  def __getitem__(self, i) -> dict:
    """Load the filename with index `i` and featurize it."""
    with open(self.filenames[i], "r") as f:
      data = json.load(f)
      return self._featurize(data, i)

  def _featurize(self, data: dict, i: int) -> dict:
    name = data['name']
    with torch.no_grad():
      coords = torch.as_tensor(
        data['coords'],
        device=self.device,
        dtype=torch.float32
      )

      # Optionally apply a random rotation to the data.
      if self.transform is not None:
        coords = self.transform(coords)

      # NOTE(milo): Find neighbors based on the position of C-alpha.
      O_N = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 0]), G=self.voxel_grid_dim).unsqueeze(-1)
      O_Ca = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 1]), G=self.voxel_grid_dim).unsqueeze(-1)
      O_C = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 2]), G=self.voxel_grid_dim).unsqueeze(-1)
      O_O = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 3]), G=self.voxel_grid_dim).unsqueeze(-1)
      O = torch.concat([O_N, O_Ca, O_C, O_O], dim=-1)

      # The node scalar features can optionally include embeddings (by concatenating).
      if self.plm is not None:
        embedding = torch.load(self.filenames[i].replace(".json", ".pt"))
      else:
        embedding = None

    return dict(
      name=name,
      task_label=data['task_label'],
      coords=coords,
      occupancy_grid=O,
      embedding=embedding,
    )

#===============================================================================

def _normalize(tensor, dim=-1):
  '''
  Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
  '''
  return torch.nan_to_num(
    torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
  '''
  From https://github.com/jingraham/neurips19-graph-protein-design
  
  Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
  That is, if `D` has shape [...dims], then the returned tensor will have
  shape [...dims, D_count].
  '''
  D_mu = torch.linspace(D_min, D_max, D_count, device=device)
  D_mu = D_mu.view([1, -1])
  D_sigma = (D_max - D_min) / D_count
  D_expand = torch.unsqueeze(D, -1)

  RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
  return RBF


class ProteinGraphDataset(data.Dataset):
  """
  A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
  protein structures into featurized protein graphs as described in the manuscript.

  * Source: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
  * README: https://github.com/drorlab/gvp-pytorch/blob/main/README.md
  
  Returned graphs are of type `torch_geometric.data.Data` with attributes:
  -x          alpha carbon coordinates, shape [n_nodes, 3]
  -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
  -name       name of the protein structure, string
  -node_s     node scalar features, shape [n_nodes, 6] 
  -node_v     node vector features, shape [n_nodes, 3, 3]
  -edge_s     edge scalar features, shape [n_edges, 32]
  -edge_v     edge scalar features, shape [n_edges, 1, 3]
  -edge_index edge indices, shape [2, n_edges]
  -mask       node mask, `False` for nodes with missing data that are excluded from message passing
  
  Portions from https://github.com/jingraham/neurips19-graph-protein-design.

  Parameters
  ----------
  * `json_folder`: A folder with JSON-style representations of protein backbones.
  * `num_positional_embeddings`: number of positional embeddings
  * `top_k`: number of edges to draw per node (as destination node)
  * `device`: if `cuda`, will do preprocessing on the GPU
  """
  def __init__(
    self,
    json_folder: str,
    num_positional_embeddings: int = 16,
    top_k: int = 30,
    num_rbf: int = 16,
    edge_algorithm: str = "knn_graph", # or radius_graph
    r_ball_radius: float = 8, # angstroms
    virtual_node: bool = True,
    plm: Optional[str] = None,
    device: str = "cpu",
    precomputed_embeddings: bool = True
  ):
    super(ProteinGraphDataset, self).__init__()
    
    self.json_folder = json_folder
    self.top_k = top_k
    self.num_rbf = num_rbf
    self.edge_algorithm = edge_algorithm
    self.r_ball_radius = r_ball_radius
    self.num_positional_embeddings = num_positional_embeddings
    self.virtual_node = virtual_node
    self.plm = plm
    self.device = device
    self.precomputed_embeddings = precomputed_embeddings

    # If a protein language model is given, initialize it here.
    if not self.precomputed_embeddings:
      self.plm_model, self.plm_alphabet = embeddings.esm2_model_dictionary[self.plm]()
      self.plm_embedding_dim = embeddings.esm2_embedding_dims[self.plm]
      self.plm_layer = embeddings.esm2_embedding_layer[self.plm]

    # This defines the ordering of examples in the dataset.
    self.filenames = glob.glob(f"{self.json_folder}/*.json")

    if len(self.filenames) == 0:
      raise FileNotFoundError("Couldn't find any files in the dataset folder you passed. Does it exist and have JSON files in it?")

    # Count up the number of nodes in each graph (used by the sampler).
    self.node_counts = [0 for _ in range(len(self.filenames))]

    # Determine the frequence of labels in the dataset (used by the sampler).
    # TODO(milo): The number of labels should be a parameter.
    n_classes = 10
    self.label_counts = [0 for _ in range(n_classes)]
    self.node_labels = [0 for _ in range(len(self.filenames))]

    # Open up each file to see how many nodes it contains:
    for i, filename in enumerate(self.filenames):
      with open(filename, "r") as f:
        m = json.load(f)
        self.node_counts[i] = len(m['seq'])
        self.label_counts[m['task_label']] += 1
        self.node_labels[i] = m['task_label']

    # Calculate weightsx for each label.
    self.class_weights = (1 / n_classes) * (len(self.filenames) / np.array(self.label_counts)) # (10)
    # Calculate weights for each example in the dataset (does not need to sum to 1).
    self.sampler_weights = [self.class_weights[self.node_labels[i]] for i in range(len(self.node_labels))]

    self.letter_to_num = {
      'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
      'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
      'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
      'N': 2, 'Y': 18, 'M': 12, '_': -1, 
    }
    self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

  def __len__(self) -> int:
    return len(self.filenames)
  
  def __getitem__(self, i) -> dict:
    """Load the filename with index `i` and featurize it."""
    with open(self.filenames[i], "r") as f:
      data = json.load(f)
      return self._featurize_as_graph(data, i)
  
  def _featurize_as_graph(self, protein, i: int) -> dict:
    name = protein['name']
    with torch.no_grad():
      coords = torch.as_tensor(
        protein['coords'], device=self.device, dtype=torch.float32
      )
      seq = torch.as_tensor(
        [self.letter_to_num[a] for a in protein['seq']], device=self.device, dtype=torch.long
      )

      mask = torch.isfinite(coords.sum(dim=(1,2)))
      coords[~mask] = np.inf
      
      # NOTE(milo): Find neighbors based on the position of C-alpha.
      X_ca = coords[:, 1]

      if self.edge_algorithm == "knn_graph":
        edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
      elif self.edge_algorithm == "radius_graph":
        edge_index = torch_cluster.radius_graph(X_ca, r=self.r_ball_radius)
      else:
        raise NotImplementedError()

      pos_embeddings = self._positional_embeddings(edge_index)
      E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
      rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

      dihedrals = self._dihedrals(coords)                     
      orientations = self._orientations(X_ca)
      sidechains = self._sidechains(coords)

      # The node scalar features can optionally include embeddings (by concatenating).
      if self.plm is not None:
        if self.precomputed_embeddings:
          # TODO(milo): Hacky but works.
          node_embedding = torch.load(self.filenames[i].replace(".json", ".pt"))
        else:
          node_embedding = embeddings.extract_embedding_single(
            self.plm_model,
            self.plm_alphabet,
            self.plm_layer,
            # Can pass unknown/missing residues to the language model:
            # TODO(milo): What are the '.' '-' and '<mask>' tokens used for?
            protein['seq'].replace('_', '<unk>')
          )
        node_s = torch.concat([dihedrals, node_embedding], dim=-1)
      else:
        node_s = dihedrals

      node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
      edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
      edge_v = _normalize(E_vectors).unsqueeze(-2)

      node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
      
    data = torch_geometric.data.Data(
      x=X_ca,
      seq=seq,
      # distance_matrix=distance_matrix,
      name=name,
      task_label=protein['task_label'],
      node_s=node_s,
      node_v=node_v,
      edge_s=edge_s,
      edge_v=edge_v,
      edge_index=edge_index,
      mask=mask
    )

    # Optionally add a virtual node that is receives/sends messages to all other nodes.
    if self.virtual_node:
      tf = torch_geometric.transforms.VirtualNode()
      data = tf(data)

    return data

  def _dihedrals(self, X, eps=1e-7):
    """https://github.com/jingraham/neurips19-graph-protein-design"""
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features
  
  def _positional_embeddings(
    self,
    edge_index, 
    num_embeddings=None,
    period_range=[2, 1000]
  ):
    """https://github.com/jingraham/neurips19-graph-protein-design"""
    num_embeddings = num_embeddings or self.num_positional_embeddings
    d = edge_index[0] - edge_index[1]
   
    frequency = torch.exp(
      torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
      * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

  def _orientations(self, X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

  def _sidechains(self, X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n, dim=-1), dim=-1)
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec
  

class BatchSampler(data.Sampler):
  '''
  From https://github.com/jingraham/neurips19-graph-protein-design.
  
  A `torch.utils.data.Sampler` which samples batches according to a
  maximum number of graph nodes.
  
  :param node_counts: array of node counts in the dataset to sample from
  :param max_nodes: the maximum number of nodes in any batch, including batches of a single element
  :param shuffle: if `True`, batches in shuffled order
  '''
  def __init__(
    self,
    node_counts: list[int],
    max_nodes: int = 3000,
    shuffle: bool = True,
    sampler_weights: Optional[np.ndarray] = None
  ):
    self.node_counts = node_counts
    self.shuffle = shuffle
    self.max_nodes = max_nodes
    self.sampler_weights = sampler_weights
    self._form_batches()
  
  def _form_batches(self):
    """Form batches with random sampling.
    
    Each batch consists of one or more graphs. Graphs are added to the batch
    until the maximum number of nodes have been met.
    """
    self.batches = []

    # Optionally apply weighted sampling to deal with class imbalances.
    if self.sampler_weights is not None:
      assert(len(self.sampler_weights) == len(self.node_counts))
      idx = list(data.WeightedRandomSampler(self.sampler_weights, len(self.node_counts), replacement=True))
      # Skip any graphs in the dataset that are larger than the maximum size.
      idx = [idx[j] for j in range(len(idx)) if (self.node_counts[idx[j]] <= self.max_nodes)]
    else:
      # Skip any graphs in the dataset that are larger than the maximum size.
      idx = [i for i in range(len(self.node_counts)) if self.node_counts[i] <= self.max_nodes]
      
    if self.shuffle:
      random.shuffle(idx)

    while idx:
      batch = []
      n_nodes = 0
      while idx and (n_nodes + self.node_counts[idx[0]]) <= self.max_nodes:
        next_idx, idx = idx[0], idx[1:]
        n_nodes += self.node_counts[next_idx]
        batch.append(next_idx)
      self.batches.append(batch)

  def __len__(self) -> int:
    """Returns the number of batches.""" 
    if not self.batches:
      self._form_batches()
    return len(self.batches)
  
  def __iter__(self) -> list[int]:
    """Generator function for batches.
    
    Each batch is a list of indices in the dataset to sample.
    """
    if not self.batches:
      self._form_batches()
    for batch in self.batches:
      yield batch