import sys; sys.path.append('..')

import argparse
from datetime import datetime
from functools import partial
import tqdm, os, json

import pandas as pd
import torch
import torch.nn as nn
import gvpgnn.datasets as datasets
import gvpgnn.models as models
import gvpgnn.paths as paths
import gvpgnn.data_models as dm
import gvpgnn.embeddings as embeddings
import gvpgnn.train_utils as train_utils
import numpy as np
import torch_geometric
from sklearn.metrics import confusion_matrix

print = partial(print, flush=True)

# You can set this to run through all the code quickly and make sure there aren't obvious bugs.
_FAST_MODE = False

parser = argparse.ArgumentParser()
parser.add_argument('--model-id',
                    default=int(datetime.timestamp(datetime.now())),
                    type=str,
                    help="The name of this model. Also determines the subfolder where checkpoints will go.")
parser.add_argument('--models-dir', metavar='PATH', default=paths.models_folder(),
                    help='Directory to save trained models')
parser.add_argument('--num-workers', metavar='N', type=int, default=2,
                    help='Number of threads for loading data')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='Max number of nodes per batch')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='Training epochs')

parser.add_argument('--plm', choices=[*embeddings.esm2_model_dictionary.keys(), None], 
                    default=None, help="Which pretrained protein language model to use (default None)")
parser.add_argument('--top-k', type=int, default=30, help="How many k-nearest neighbors to connect in the GNN")
parser.add_argument('--gnn-layers', type=int, default=4, help="Number of GVP layers in the GNN")

parser.add_argument('--train', action="store_true", help="Train a model from scratch or from a checkpoint.")
parser.add_argument('--test', action="store_true", help="Test a trained model")

parser.add_argument('--train-path', type=str, help="The path to a JSON file with TRAINING data. Only required with --train.")
parser.add_argument('--val-path', type=str, help="The path to a JSON file with VALIDATION data. Only required with --train.")
parser.add_argument('--test-path', type=str, help="The path to a JSON file with TEST data.")

args = parser.parse_args()

if not any([args.train, args.test]) or all([args.train and args.test]):
  raise ValueError("Must specify exactly one of --train or --test")

device = train_utils.get_best_system_device()


def train(
  model: nn.Module,
  params: dict,
  trainset: torch.utils.data.Dataset,
  valset: torch.utils.data.Dataset,
  testset: torch.utils.data.Dataset,
):
  """Main coordinating function for training the model."""
  model_id = str(params['model_id'])
  train_loader = train_utils.dataloader_factory(trainset, args)
  val_loader = train_utils.dataloader_factory(valset, args)
  test_loader = train_utils.dataloader_factory(testset, args)

  optimizer = torch.optim.Adam(model.parameters())

  # Store the best model so far, and its loss on the validation set.
  best_path, best_val_loss = None, np.inf
  lookup = dm.num_to_readable_architecture

  if os.path.exists(os.path.join(args.models_dir, model_id)):
    raise FileExistsError("The folder for this model exists already! You'll need to delete it and try again.")
  else:
    os.makedirs(os.path.join(args.models_dir, model_id), exist_ok=False)

  metrics = []

  for epoch in range(args.epochs):
    model.train()

    # Do one epoch of training:
    loss, acc, confusion = loop(model, train_loader, optimizer=optimizer)
    metrics.append(dict(epoch=epoch, split="train", loss=loss, accuracy=acc))

    # Save the results after this epoch:
    path = os.path.join(args.models_dir, model_id, f"checkpoint_{epoch}.pt")
    torch.save(model.state_dict(), path)

    # Save the parameters that this model was trained with.
    with open(os.path.join(args.models_dir, model_id, "params.json"), "w") as f:
      json.dump(params, f, indent=2)

    print('----------------------------------------------------')
    print(f'[EPOCH {epoch}] Training loss: {loss:.4f} acc: {acc:.4f}')
    train_utils.print_confusion(confusion, lookup=lookup)
    print('----------------------------------------------------')

    model.eval()

    # Evaluate on the VALIDATION set:
    with torch.no_grad():
      loss, acc, confusion = loop(model, val_loader, optimizer=None)
      metrics.append(dict(epoch=epoch, split="val", loss=loss, accuracy=acc))

    print('----------------------------------------------------')
    print(f'[EPOCH {epoch}] Validation loss: {loss:.4f} acc: {acc:.4f}')
    train_utils.print_confusion(confusion, lookup=lookup)
    print('----------------------------------------------------')
    
    # If the latest model checkpoint performs better on the validation set, update.
    if loss < best_val_loss:
      best_path, best_val_loss = path, loss
      print(f'[EPOCH {epoch}] Achieved better validation performance! The best model so far is at: {best_path}.')
    
    # Evaluate performance of the best model on the held out TEST set:
    print('----------------------------------------------------')
    print(f"TESTING: loading from {best_path}")
    model.load_state_dict(torch.load(best_path))

    model.eval()
    with torch.no_grad():
      loss, acc, confusion = loop(model, test_loader, optimizer=None)
      metrics.append(dict(epoch=epoch, split="test", loss=loss, accuracy=acc))

    print('----------------------------------------------------')
    print(f'TEST loss: {loss:.4f} acc: {acc:.4f}')
    train_utils.print_confusion(confusion, lookup=lookup)
    print('----------------------------------------------------')

    # Append (by overwriting) the metrics for this epoch.
    pd.DataFrame(metrics).to_csv(os.path.join(args.models_dir, model_id, "metrics.csv"), index=False)


def loop(
  model: nn.Module,
  dataloader: torch_geometric.data.DataLoader,
  optimizer=None,
  n_categories: int = 10,
) -> tuple[float, float, float]:
  """Perform one epoch of training or testing.
  
  Notes
  -----
  If the `optimizer` isn't passed in, we skip backpropagation.

  Returns
  -------
  The average loss, average % of correct labels (accuracy), and confusion matrix.
  """
  confusion = np.zeros((n_categories, n_categories))
  t = tqdm.tqdm(dataloader)

  loss_fn = nn.CrossEntropyLoss()
  total_loss, total_correct, total_count = 0, 0, 0

  for batch in t:
    if optimizer:
      optimizer.zero_grad()

    batch: torch_geometric.data.Batch = batch.to(device)
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

    # Weight the loss value by the size of this batch (we'll normalize later).
    total_loss += float(loss_value) * batch.num_graphs
    total_count += batch.num_graphs

    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    actual = batch.task_label.detach().cpu().numpy()

    total_correct += (pred == actual).sum()
  
    confusion += confusion_matrix(actual, pred, labels=range(n_categories))

    # Print out training stats next to progress bar.
    t.set_description("[L=%.3f|A1=%.2f]" % (float(total_loss/total_count), float(100*total_correct/total_count)))
  
    torch.cuda.empty_cache()

    if _FAST_MODE:
      break
      
  return (
    total_loss / total_count,
    total_correct / total_count,
    confusion
  )


def main():
  """Main command line interface for training."""
  if _FAST_MODE: print("[WARNING] Fast mode is ON! If you're doing a real training run, make sure this is off.")

  if args.plm is None:
    plm_embedding_dim = 0
  else:
    plm_embedding_dim=embeddings.esm2_embedding_dims[args.plm]

  # If using language model embeddings, these are concatenated with the scalar
  # features for each node.
  node_in_dim = (6 + plm_embedding_dim, 3) # num scalar, num vector
  node_h_dim = (100, 16) # num scalar, num vector
  edge_in_dim = (32, 1) # num scalar, num vector
  edge_h_dim = (32, 10) # num scalar, num vector

  # Store all training parameters for reproducibility.
  train_params = dict(
    plm=args.plm,
    plm_embedding_dim=plm_embedding_dim,
    node_in_dim=node_in_dim,
    node_h_dim=node_h_dim,
    edge_in_dim=edge_in_dim,
    edge_h_dim=edge_h_dim,
    top_k=args.top_k,
    n_categories=10,
    num_layers=args.gnn_layers,
    drop_rate=0.1,
    device=device,
    model_id=args.model_id,
  )

  print("\nPARAMETERS:")
  for k, v in train_params.items():
    print(f" {k} = {v}")

  # Strip out only the parameters relevant to the model.
  model_params = models.ClassifierGNNParams.model_validate(train_params).model_dump()
  model = models.ClassifierGNN(**model_params).to(device)

  if args.train:
    trainset = datasets.ProteinGraphDataset(args.train_path, top_k=args.top_k, plm=args.plm)
    valset = datasets.ProteinGraphDataset(args.val_path, top_k=args.top_k, plm=args.plm)
    testset = datasets.ProteinGraphDataset(args.test_path, top_k=args.top_k, plm=args.plm)
    train(model, train_params, trainset, valset, testset)

  else:
    testset = datasets.ProteinGraphDataset(args.test_path)


if __name__ == "__main__":
  main()