import numpy as np
import torch_geometric

from gvpgnn.graph_dataset import ProteinGraphDataset, BatchSampler
from gvpgnn.paths import data_folder


def test_load_with_precomputed_embeddings():
  """Make sure that embeddings aren't computed if they've already been precomputed."""
  dataset_version = "cleaned_with_esm2_t6_8M_UR50D"
  split_name = "train"

  dataset = ProteinGraphDataset(
    data_folder(f"{dataset_version}/{split_name}"),
    num_positional_embeddings=1,
    top_k=1,
    num_rbf=1,
    device="cpu",
    precomputed_embeddings=True
  )
  # Ensure that the embeddings are included in the node scalar features.
  print(dataset[0])
  assert(dataset[0].node_s.shape[1] == 326) # 320 + 6


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

  dataloader = torch_geometric.loader.DataLoader(
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