import glob
import json
import random
import numpy as np
import torch
import torch.utils.data as data
import gvpgnn.voxels as voxels

from scipy.spatial.transform import Rotation


class RandomRotation3d(object):
  """Randomly apply a 3D rotation to a set of points.

  We sample a random rotation matrix and apply it to all of the points..
  """
  def __init__(self, device):
    self.device = device

  def __call__(self, points: torch.Tensor):
    R = torch.Tensor(Rotation.random().as_matrix()).to(self.device)
    return torch.matmul(points, R)


class ProteinVoxelDataset(data.Dataset):
  """A torch `Dataset` that converts protein structures into 3D voxel grids."""
  def __init__(
    self,
    json_folder: str,
    voxel_grid_dim: int = 512,
    apply_random_rotation: bool = False,
    device: str = "cpu",
  ):
    super(ProteinVoxelDataset, self).__init__()

    # This defines the ordering of examples in the dataset.
    self.json_folder = json_folder
    self.filenames = glob.glob(f"{self.json_folder}/*.json")
    self.voxel_grid_dim = voxel_grid_dim
    self.device = device
    self.transform = RandomRotation3d(device=device) if apply_random_rotation else None

    if len(self.filenames) == 0:
      raise FileNotFoundError("Couldn't find any files in the dataset folder you passed. Does it exist and have JSON files in it?")

    # Count up the number of nodes in each graph (used by the sampler).
    self.node_counts = torch.zeros(len(self.filenames))

    # Determine the frequence of labels in the dataset (used by the sampler).
    num_categories = 10
    self.label_counts = torch.zeros(num_categories, dtype=torch.int)
    self.node_labels = torch.zeros(len(self.filenames), dtype=torch.int)

    # Open up each file to see how many nodes it contains:
    for i, filename in enumerate(self.filenames):
      with open(filename, "r") as f:
        m = json.load(f)
        self.node_counts[i] = len(m['seq'])
        self.label_counts[m['task_label']] += 1
        self.node_labels[i] = m['task_label']

    # Calculate weights for each label.
    self.class_weights = (1 / num_categories) * (len(self.filenames) / np.array(self.label_counts)) # (10)

    # Calculate weights for each example in the dataset (does not need to sum to 1).
    self.sampler_weights = [self.class_weights[self.node_labels[i]] for i in range(len(self.node_labels))]

  def __len__(self) -> int:
    """Returns the length of the dataset."""
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
      O_N = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 0]), G=self.voxel_grid_dim).unsqueeze(0)
      O_Ca = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 1]), G=self.voxel_grid_dim).unsqueeze(0)
      O_C = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 2]), G=self.voxel_grid_dim).unsqueeze(0)
      O_O = voxels.create_occupancy_grid(voxels.center_and_scale_unit_box(coords[:, 3]), G=self.voxel_grid_dim).unsqueeze(0)

      # TODO(milo): Experiment with keeping the four channels separate.
      # O = torch.concat([O_N, O_Ca, O_C, O_O], dim=0) # channel dim first
      O = 0.25 * (O_N + O_Ca + O_C + O_O)

    return dict(
      name=name,
      task_label=data['task_label'],
      occupancy_grid=O,
    )


class VoxelBatchSampler(data.Sampler):
  """A `torch.utils.data.Sampler` for sampling minibatches of 3D voxel grids.
  
  Parameters
  ----------
  * `batch_size` : The number of examples to include in the batch
  * `sampler_weights` : How much to weight each example in the dataset (does not have to sum to one)
  * `shuffle`: Whether to randomly shuffle the dataset before sampling batches
  """
  def __init__(
    self,
    batch_size: int,
    sampler_weights: np.ndarray,
    shuffle: bool = True,
  ):
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.sampler_weights = sampler_weights
    self._form_batches()
  
  def _form_batches(self):
    """Form batches with random sampling.
    """
    self.batches = []
    idx = list(data.WeightedRandomSampler(self.sampler_weights, len(self.sampler_weights), replacement=True))

    if self.shuffle:
      random.shuffle(idx)

    N = len(idx)
    B = self.batch_size

    for b in range(N // B - 1):
      self.batches.append(idx[b*B : b*B + B])

  def __len__(self) -> int:
    """Returns the number of batches.""" 
    return len(self.batches)
  
  def __iter__(self):
    """Generator function for batches.
    
    Each batch is a list of indices in the dataset to sample.
    """
    for batch in self.batches:
      yield batch