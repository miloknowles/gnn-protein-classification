import sys; sys.path.append("..")
import os
import json
import glob
import math
import time

import torch

import gvpgnn.paths as paths
import gvpgnn.embeddings as plm


if __name__ == "__main__":
  """Adds precomputed language model embeddings to a dataset."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  name = "esm2_t33_650M_UR50D"
  # name = "esm2_t6_8M_UR50D" 
  print(f"Using device '{device}'")
  print(f"Loading model '{name}'")

  # Read in a dataset folder, and output a new one with embeddings included.
  in_dataset_folder = paths.data_folder("cleaned_skip_missing")
  out_dataset_folder = paths.data_folder(f"cleaned_with_{name}")
  assert(in_dataset_folder != out_dataset_folder) # don't allow overwrite

  model, alphabet = plm.esm2_model_dictionary[name]()
  batch_converter = alphabet.get_batch_converter()

  model = model.to(device)
  model.eval()

  feat_dim = plm.esm2_embedding_dims[name]
  feat_layer = plm.esm2_embedding_layer[name]

  for split_name in ("train", "test", "val"):
    print(f"\n\n-------- {split_name} --------")
    os.makedirs(os.path.join(out_dataset_folder, split_name), exist_ok=True)

    json_files = glob.glob(os.path.join(in_dataset_folder, split_name, "*.json"))

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
          chunk_contents[j][name] = sequence_representations[j].cpu().numpy().tolist()

          # Write the new file.
          with open(os.path.join(out_dataset_folder, split_name, f'{chunk_contents[j]["cath_id"]}.json'), "w") as f:
            json.dump(chunk_contents[j], f, indent=2)

      print(f"Finished batch {b+1}/{num_batches} in {time.time() - t0:.2f} sec")

    print("\nDONE!")