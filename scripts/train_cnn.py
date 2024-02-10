import sys; sys.path.append('..')

from functools import partial
import tqdm, os, json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import gvpgnn.datasets as datasets
import gvpgnn.cnn as cnn
import gvpgnn.data_models as dm
import gvpgnn.embeddings as embeddings
import gvpgnn.train_utils as train_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from scripts.parser import parser

print = partial(print, flush=True)

# You can set this to run through all the code quickly and make sure there aren't obvious bugs.
_FAST_MODE = False

args = parser.parse_args()

if not any([args.train, args.test]) or all([args.train and args.test]):
  raise ValueError("Must specify exactly one of --train or --test")

device = train_utils.get_best_system_device()
# Unfortunate hack due to limited support on MPS.
# https://github.com/pytorch/pytorch/issues/77764
if device == "mps":
  device = "cpu"


def make_dataloader(
  dataset: datasets.ProteinVoxelDataset,
  shuffle: bool = False,
) -> DataLoader:
  return DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=shuffle,
    batch_sampler=WeightedRandomSampler(dataset.sampler_weights)
  )


def train(
  model: nn.Module,
  params: dict,
  trainset: torch.utils.data.Dataset,
  valset: torch.utils.data.Dataset,
  testset: torch.utils.data.Dataset,
):
  """Main coordinating function for training the model."""
  model_id = str(params['model_id'])
  train_loader = make_dataloader(trainset, shuffle=True)
  val_loader = make_dataloader(valset, shuffle=False)
  test_loader = train_utils.dataloader_factory(testset, shuffle=False)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # Keep track of the best model so far, and its loss on the validation set.
  best_path, best_val_loss = None, np.inf
  lookup = dm.num_to_readable_architecture # label lookup for confusion matrix

  if os.path.exists(os.path.join(args.models_dir, model_id)):
    raise FileExistsError("The folder for this model exists already! You'll need to delete it and try again.")
  else:
    os.makedirs(os.path.join(args.models_dir, model_id), exist_ok=False)

  metrics = [] # store epoch training metrics
  
  # Save the parameters that this model was trained with.
  with open(os.path.join(args.models_dir, model_id, "params.json"), "w") as f:
    json.dump(params, f, indent=2)

  for epoch in range(args.epochs):
    model.train()

    # Do one epoch of training:
    loss, acc, confusion = loop(model, train_loader, optimizer=optimizer)
    metrics.append(dict(epoch=epoch, split="train", loss=loss, accuracy=acc))

    # Save the results after this epoch:
    path = os.path.join(args.models_dir, model_id, f"checkpoint_{epoch}.pt")
    torch.save(model.state_dict(), path)

    print('----------------------------------------------------')
    print(f'[EPOCH {epoch}] Training loss: {loss:.4f} acc: {acc:.4f}')
    train_utils.print_confusion(confusion, lookup=lookup)
    print('----------------------------------------------------')

    model.eval()

    # VALIDATION SET
    with torch.no_grad():
      loss, acc, confusion = loop(model, val_loader, optimizer=None)
      metrics.append(dict(epoch=epoch, split="val", loss=loss, accuracy=acc))

    print('----------------------------------------------------')
    print(f'[EPOCH {epoch}] Validation loss: {loss:.4f} acc: {acc:.4f}')
    train_utils.print_confusion(confusion, lookup=lookup)
    
    # If the latest model checkpoint performs better on the validation set, update.
    if loss < best_val_loss:
      best_path, best_val_loss = path, loss
      print(f'[EPOCH {epoch}] Achieved better validation performance! The best model so far is at: {best_path}.')
    print('----------------------------------------------------')

    # TEST SET
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


def test(
  model: nn.Module,
  testset: torch.utils.data.Dataset,
  weights_path: str,
  device: str = "cpu"
):
  """Main coordinating function for training the model."""
  test_loader = train_utils.dataloader_factory(testset, args)

  lookup = dm.num_to_readable_architecture # label lookup for confusion matrix

  if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find pretrained weights at '{weights_path}'! Make sure you've copied these to the right location.")

  print(f"Loading weights from {weights_path}")
  model.load_state_dict(torch.load(weights_path, map_location=device))

  model.eval()
  with torch.no_grad():
    loss, acc, confusion = loop(model, test_loader, optimizer=None)

  print('*** Test set results:')
  print(f'Loss: {loss:.4f} acc: {acc:.4f}')
  print(f'Accuracy (top1): {100*acc:.4f}%')

  print('----------------------------------------------------')
  train_utils.print_confusion(confusion, lookup=lookup)
  print('----------------------------------------------------')


def loop(
  model: nn.Module,
  dataloader: DataLoader,
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

    batch = batch.to(device)

    # Generate unnormalized predictions from the network on this batch.
    # NOTE(milo): Important to pass in the batch indices so that nodes are
    # associated with the correct graphs in the batch!
    logits = model(batch)
    loss_value = loss_fn(logits, batch.task_label)

    if optimizer:
      loss_value.backward()
      optimizer.step()

    # Weight the loss value by the size of this batch (we'll normalize later).
    total_loss += loss_value
    total_count += batch.shape[0]

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

  # Store all training parameters for reproducibility. These get written to the folder for the current run.
  train_params = dict(
    plm=args.plm,
    plm_embedding_dim=plm_embedding_dim,
    drop_rate=0.1,
    voxel_grid_dim=args.voxel_grid_dim,
    random_rotation=args.random_rotation,
    lr=args.lr,
    device=device,
    model_id=args.model_id,
    batch_size=args.batch_size,
  )

  print("\nPARAMETERS:")
  for k, v in train_params.items():
    print(f"* {k} = {v}")

  # Strip out only the parameters relevant to the model.
  model = cnn.CNN3DClassifier(
    in_dim=4,
    conv_out_dim=64,
    logits_out_dim=10,
    box_size=args.voxel_grid_dim,
    conv_hidden_dim=64,
    dense_hidden_dim=64
  ).to(device)

  if args.train:
    trainset = datasets.ProteinVoxelDataset(
      args.train_path, plm=args.plm, device=device,
      voxel_grid_dim=args.voxel_grid_dim, apply_random_rotation=args.random_rotation,
    )
    valset = datasets.ProteinVoxelDataset(
      args.train_path, plm=args.plm, device=device,
      voxel_grid_dim=args.voxel_grid_dim, apply_random_rotation=False,
    )
    testset = datasets.ProteinVoxelDataset(
      args.train_path, plm=args.plm, device=device,
      voxel_grid_dim=args.voxel_grid_dim, apply_random_rotation=False,
    )
    train(model, train_params, trainset, valset, testset)

  elif args.test:
    print("\n----- TESTING MODE -----\n")
    if args.test_path is None:
      raise ValueError("\
        You need to provide a `--test-path` that points to the folder with test\
        data in it! See the evaluation notebook for instructions.\
      ")
    if args.checkpoint is None:
      raise ValueError("\
        You need to provide a `--checkpoint` folder that points to a pretrained\
        model checkpoint. See the evaluation notebook for instructions.\
      ")
    # testset = datasets.ProteinGraphDataset(args.test_path, edge_algorithm=args.edge_algorithm,
    #   top_k=args.top_k, r_ball_radius=args.r_ball_radius, plm=args.plm)
    # test(model, testset, weights_path=args.checkpoint)


if __name__ == "__main__":
  main()