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
import gvpgnn.datamodels as dm
import numpy as np
import torch_geometric
from sklearn.metrics import confusion_matrix

print = partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', metavar='PATH', default=paths.models_folder(),
                    help='Directory to save trained models')
parser.add_argument('--num-workers', metavar='N', type=int, default=2,
                    help='Number of threads for loading data')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='Max number of nodes per batch')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='Training epochs')
parser.add_argument('--train', action="store_true", help="Train a model from scratch or from a checkpoint.")
parser.add_argument('--test', action="store_true", help="Test a trained model")

parser.add_argument('--train-path', type=str, help="The path to a JSON file with TRAINING data. Only required with --train.")
parser.add_argument('--val-path', type=str, help="The path to a JSON file with VALIDATION data. Only required with --train.")
parser.add_argument('--test-path', type=str, help="The path to a JSON file with TEST data.")

args = parser.parse_args()

if not any([args.train, args.test]) or all([args.train and args.test]):
  raise ValueError("Must specify exactly one of --train or --test")

# Convenience function for intializing a data loader.
dataloader_factory = lambda dataset: torch_geometric.data.DataLoader(
  dataset,
  num_workers=args.num_workers,
  batch_sampler=datasets.BatchSampler(
    dataset.node_counts,
    max_nodes=args.max_nodes
  )
)

# Detect the best available device.
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
elif torch.backends.mps.is_available():
  device = "mps"

if not os.path.exists(args.models_dir):
  os.makedirs(args.models_dir)

# Set a unique model ID based on the current time.
model_id = int(datetime.timestamp(datetime.now()))


def train(model, params: dict, trainset, valset, testset):
  """Main coordinating function for training the model."""
  train_loader, val_loader, test_loader = map(
    dataloader_factory, (trainset, valset, testset)
  )

  optimizer = torch.optim.Adam(model.parameters())

  # Store the best model so far, and its loss on the validation set.
  best_path, best_val_loss = None, np.inf
  lookup = dm.num_to_readable_architecture

  for epoch in range(args.epochs):
    model.train()

    # Do one epoch of training:
    loss, acc, confusion = loop(model, train_loader, optimizer=optimizer)

    # Save the results after this epoch:
    path = f"{args.models_dir}/{model_id}_{epoch}.pt"
    torch.save(model.state_dict(), path)

    # Save the parameters that this model was trained with.
    with open(f"{args.models_dir}/{model_id}_params.json", "w") as f:
      json.dump(params, f, indent=2)

    print(f'EPOCH {epoch} TRAIN loss: {loss:.4f} acc: {acc:.4f}')
    print_confusion(confusion, lookup=lookup)

    model.eval()

    # Evaluate on the validation set:
    with torch.no_grad():
      loss, acc, confusion = loop(model, val_loader, optimizer=None)
    print(f'EPOCH {epoch} VAL loss: {loss:.4f} acc: {acc:.4f}')
    print_confusion(confusion, lookup=lookup)
    
    # If the latest model checkpoint performs better on the validation set, update.
    if loss < best_val_loss:
      best_path, best_val_loss = path, loss
      print(f'BEST {best_path} VAL loss: {best_val_loss:.4f}')
    
    # Evaluate performance of the best model on the held out test set:
    print(f"TESTING: loading from {best_path}")
    model.load_state_dict(torch.load(best_path))

    model.eval()
    with torch.no_grad():
      loss, acc, confusion = loop(model, test_loader, optimizer=None)
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')
    print_confusion(confusion, lookup=lookup)


def loop(model, dataloader, optimizer=None, n_categories: int = 10):
  """Perform one epoch of training or testing.
  
  Notes
  -----
  If the `optimizer` isn't passed in, we skip backpropagation.
  """
  confusion = np.zeros((n_categories, n_categories))
  t = tqdm.tqdm(dataloader)
  # TODO(milo): May want to weight based on prevalence of labels in the dataset...
  loss_fn = nn.CrossEntropyLoss()
  total_loss, total_correct, total_count = 0, 0, 0

  for batch in t:
    if optimizer:
      optimizer.zero_grad()

    batch = batch.to(device)
    h_V = (batch.node_s, batch.node_v)
    h_E = (batch.edge_s, batch.edge_v)

    # Generate unnormalized predictions from the network on this batch.
    # NOTE(milo): Important to pass in the batch indices so that nodes are
    # associated with the correct graphs in the batch!
    logits = model(h_V, batch.edge_index, h_E, graph_indices=batch.batch)

    loss_value = loss_fn(logits, batch.task_label)

    if optimizer:
      loss_value.backward()
      optimizer.step()

    num_nodes = int(batch.mask.sum())
    total_loss += float(loss_value) * num_nodes
    total_count += num_nodes

    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    true = batch.task_label.detach().cpu().numpy()
    total_correct += (pred == true).sum()
  
    confusion += confusion_matrix(true, pred, labels=range(n_categories))
    t.set_description("%.5f" % float(total_loss/total_count))
  
    torch.cuda.empty_cache()
      
  return (
    total_loss / total_count,
    total_correct / total_count,
    confusion
  )


def print_confusion(mat, lookup, n_categories: int = 10):
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


def main():
  """Main command line interface for training."""
  print(f"[INFO] Using device '{device}'")

  print("\nARGUMENTS:")
  for k, v in vars(args).items():
    print(f" {k} = {v}")
  print("\n")

  node_in_dim = (6, 3) # num scalar, num vector
  node_h_dim = (100, 16) # num scalar, num vector
  edge_in_dim = (32, 1) # num scalar, num vector
  edge_h_dim = (32, 10) # num scalar, num vector

  params = dict(
    node_in_dim=node_in_dim,
    node_h_dim=node_h_dim,
    edge_in_dim=edge_in_dim,
    edge_h_dim=edge_h_dim,
    n_categories=10,
    num_layers=3,
    drop_rate=0.1    
  )

  model = models.ProteinStructureClassificationModel(**params).to(device)

  if args.train:
    trainset = datasets.ProteinGraphDataset(args.train_path)
    valset = datasets.ProteinGraphDataset(args.val_path)
    testset = datasets.ProteinGraphDataset(args.test_path)
    train(model, params, trainset, valset, testset)
  else:
    testset = datasets.ProteinGraphDataset(args.test_path)


if __name__ == "__main__":
  main()