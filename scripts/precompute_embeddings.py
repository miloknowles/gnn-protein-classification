import sys; sys.path.append("..")
import os
import json
import glob
import math
import time

import torch

import gvpgnn.paths as paths
import gvpgnn.embeddings as plm


def run(name: str, in_dataset_folder: str, out_dataset_folder: str, device: str = "cpu"):
  """Adds precomputed language model embeddings to a dataset."""
  assert(in_dataset_folder != out_dataset_folder) # don't allow overwrite

  model, alphabet = plm.esm2_model_dictionary[name]()
  batch_converter = alphabet.get_batch_converter()

  model = model.to(device)
  model.eval()

  feat_dim = plm.esm2_embedding_dims[name]
  feat_layer = plm.esm2_embedding_layer[name]

  os.makedirs(out_dataset_folder, exist_ok=True)

  json_files = glob.glob(os.path.join(in_dataset_folder, "*.json"))

  N = len(json_files)
  batch_size = 16
  num_batches = math.ceil(N / batch_size)
  print(f"Found {N} examples, breaking into batches of size {batch_size}")

  for b in range(num_batches):
    t0 = time.time()
    chunk = json_files[b*batch_size : min(b*batch_size+batch_size, N)]

    data = []
    chunk_contents = []
    for fname in chunk:
      with open(fname, "r") as f:
        contents = json.load(f)
        data.append((contents["cath_id"], contents["seq"].replace("_", "<unk>")))
        chunk_contents.append(contents)

    with torch.no_grad():
      _, _, batch_tokens = batch_converter(data)
      batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

      results = model(batch_tokens.to(device), repr_layers=[feat_layer], return_contacts=False)
      token_representations = results["representations"][feat_layer] # get the features

      # Remove padding that was added to shorter sequences.
      sequence_representations = []
      for i, tokens_len in enumerate(batch_lens):
          sequence_representations.append(token_representations[i, 1 : tokens_len - 1])

      # Add the embedding for this model to the file contents.
      for j in range(len(chunk_contents)):
        # Write a copy of the original data file.
        with open(os.path.join(out_dataset_folder, f'{chunk_contents[j]["cath_id"]}.json'), "w") as f:
          json.dump(chunk_contents[j], f, indent=2)

        # Save the embedding to a different file.
        torch.save(
          sequence_representations[j].cpu(),
          os.path.join(out_dataset_folder, f'{chunk_contents[j]["cath_id"]}.pt')
        )

    print(f"Finished batch {b+1}/{num_batches} in {time.time() - t0:.2f} sec")

  print("\nDONE!")


def run_train_val_test(name: str, in_dataset_folder: str, out_dataset_folder: str, device: str = "cpu"):
  """Adds precomputed language model embeddings to a dataset."""
  for split_name in ("train", "test", "val"):
    print(f"\n\n-------- {split_name} --------")
    split_in_dataset_folder = os.path.join(in_dataset_folder, split_name)
    split_out_dataset_folder = os.path.join(out_dataset_folder, split_name)
    run(name, split_in_dataset_folder, split_out_dataset_folder, device=device)


if __name__ == "__main__":
  """
  Precompute language model embeddings across a dataset.

  Note that you should have already run `preprocess.py` to pre-process the
  dataset; this step comes after that.

  Usage
  -----
  python precompute_embeddings.py --in-dataset ../data/challenge_test_set

  You should see output like:

  Using device 'cpu'
  Loading model 'esm2_t33_650M_UR50D'
  Found 1261 examples, breaking into batches of size 16
  Finished batch 1/79 in 9.01 sec
  Finished batch 2/79 in 5.81 sec
  Finished batch 3/79 in 7.06 sec
  Finished batch 4/79 in 5.69 sec
  Finished batch 5/79 in 14.67 sec
  Finished batch 6/79 in 5.91 sec
  """
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument('--all', action="store_true", help="Process all of the splits at once.", default=False)

  parser.add_argument(
    '--in-dataset', type=str,
    help="The path to a pre-processed dataset (without embeddings)",
    default=None,
    required=False)

  args = parser.parse_args()
  
  # The language model to use:
  name = "esm2_t33_650M_UR50D"
  # name = "esm2_t6_8M_UR50D"

  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Using device '{device}'")
  print(f"Loading model '{name}'")

  if args.all:
    in_dataset_folder = paths.data_folder("cleaned_skip_missing")
    out_dataset_folder = paths.data_folder(f"cleaned_with_{name}")
    run_train_val_test(name, in_dataset_folder, out_dataset_folder, device=device)
  else:
    # Read in a dataset folder, and output a new one with embeddings included.
    out_dataset_folder = paths.data_folder(f"{args.in_dataset.split('/')[-1]}_with_{name}")
    run(name, args.in_dataset, out_dataset_folder)