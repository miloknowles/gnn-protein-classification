import numpy as np

import torch
import torch_geometric

from .datasets import BatchSampler


def print_confusion(mat: np.ndarray, lookup: dict, n_categories: int = 10):
  """Prints out a confusion matrix during training."""
  counts = mat.astype(np.int32)
  mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
  mat = np.round(mat * 1000).astype(np.int32)
  res = '\n'
  for i in range(n_categories):
    res += '\t{}'.format(lookup[i])
  res += '\tCount\n'
  for i in range(n_categories):
    res += '{}\t'.format(lookup[i])
    res += '\t'.join('{}'.format(n) for n in mat[i])
    res += '\t{}\n'.format(sum(counts[i]))
  print(res)


def get_best_system_device() -> str:
  """Detect the best available device on this machine."""
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  return device


def dataloader_factory(dataset: torch.utils.data.Dataset, args: any) -> torch_geometric.data.DataLoader:
  """Convenience function for intializing a DataLoader using a dataset and some arguments."""
  return torch_geometric.data.DataLoader(
    dataset,
    num_workers=args.num_workers,
    batch_sampler=BatchSampler(
      dataset.node_counts,
      max_nodes=args.max_nodes
    )
  )
