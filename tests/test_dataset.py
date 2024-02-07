import numpy as np
import torch_geometric

from gvpgnn.datasets import ProteinGraphDataset, BatchSampler
from gvpgnn.paths import data_folder


def test_weighted_sampling():
  """Make sure that the weighted sampling produces a uniform distribution over class labels."""
  dataset_version = "cleaned_skip_missing"
  split_name = "train"

  dataset = ProteinGraphDataset(
    data_folder(f"{dataset_version}/{split_name}"),
    num_positional_embeddings=1,
    top_k=1,
    num_rbf=1,
    device="cpu"
  )

  sampler = BatchSampler(
    dataset.node_counts,
    max_nodes=3000,
    sampler_weights=dataset.sampler_weights
  )

  dataloader = torch_geometric.data.DataLoader(
    dataset,
    num_workers=1,
    batch_sampler=sampler,
  )

  # Gather statistics on sampled labels:
  N = 100
  C = 10
  label_freq = np.zeros(C)

  for i, batch in enumerate(dataloader):
    if i > N:
      break
    batch_labels = batch.task_label.numpy()
    for label in batch_labels:
      label_freq[label] += 1

  dist = label_freq / label_freq.sum()

  print("Label distribution:")
  print(dist)

  diff_from_uniform = np.abs(dist - (1/C))
  tol = 0.02 # 2%

  print("Diff. from a uniform distribution:")
  print(diff_from_uniform)
  np.testing.assert_array_less(diff_from_uniform, tol)