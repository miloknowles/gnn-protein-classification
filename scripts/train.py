import sys; sys.path.append('..')

import argparse
from datetime import datetime
from functools import partial
import tqdm, os, json

import torch
import torch.nn as nn
import gvpgnn.datasets as datasets
import gvpgnn.models as models
import gvpgnn.paths as paths
import numpy as np
import torch_geometric

print = partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default=paths.models_folder(),
                    help='Directory to save trained models')
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                    help='Number of threads for loading data')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='Max number of nodes per batch')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='Training epochs')
parser.add_argument('--train', action="store_true", help="Train a model from scratch or from a checkpoint.")
parser.add_argument('--test', action="store_true", help="Test a trained model")
parser.add_argument('--test-set-path', type=str, help="The path to a JSON file with test data")

args = parser.parse_args()

print("\nARGUMENTS:")
for k, v in vars(args).items():
  print(f"  {k} = {v}")
print("\n")

if not any([args.train, args.test]) or all([args.train and args.test]):
  raise ValueError("Must specify exactly one of --train or --test")


def main():
  """Main command line interface for training."""

  node_in_dim = (6, 3) # num scalar, num vector
  node_h_dim = (100, 16) # num scalar, num vector
  edge_in_dim = (32, 1) # num scalar, num vector
  edge_h_dim = (32, 10) # num scalar, num vector

  model = models.ProteinStructureClassificationModel(
    node_in_dim,
    node_h_dim,
    edge_in_dim,
    edge_h_dim,
    n_categories=10,
    num_layers=3,
    drop_rate=0.1
  )

if __name__ == "__main__":
  main()