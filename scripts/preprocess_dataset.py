import sys; sys.path.append("..")
import os
from typing import Optional
import json

import pandas as pd

import Bio.PDB.PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

import gvpgnn.paths as paths
import gvpgnn.models as dm


def check_disjoint_dataset_splits():
  """Ensure that no examples are shared across splits!"""
  print("Checking that dataset splits are disjoint...")
  cath_ids = dict(train=set(), val=set(), test=set())

  for split_name in cath_ids:
    df_split = pd.read_csv(paths.data_folder(f"{split_name}_cath_w_seqs_share.csv"))
    cath_ids[split_name] = set(df_split.cath_id.unique())

  # Just to be really sure...
  assert(cath_ids["train"].isdisjoint(cath_ids["test"]))
  assert(cath_ids["train"].isdisjoint(cath_ids["val"]))
  assert(cath_ids["val"].isdisjoint(cath_ids["test"]))

  print("OK")


def preprocess(
  split_path: str,
  output_folder: str,
  pdb_folder: str = paths.data_folder("pdb_share"),
  limit: Optional[int] = None,
  handle_bad_residue: str = "drop",
) -> tuple:
  """Preprocess the raw training data into a more useful format for the DataLoader.
  
  Notes
  -----
  The dataset should already be split into train, test, and validation.

  We do the preprocessing ahead of time to reduce the complexity of the data
  loader code. Everything is put into a standardized JSON format before it gets
  to the data loader.

  The data model format is based on:
  https://github.com/drorlab/gvp-pytorch/blob/main/README.md

  TODOs
  ---------------
  * Impute missing backbone atom positions (currently using center of mass)
  * Deal with `UNK` residues
  * Deal with `PYL` residue
  * Impute missing residues that are in the CATH dataset but not the PDB dataset.
  """
  # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
  import warnings
  warnings.filterwarnings("ignore")

  df_split = pd.read_csv(split_path)

  data_list = []

  # Aggregate warnings to get a sense of missing data.
  warn = dict(
    unknown_residues=[],
    missing_N=0,
    missing_CA=0,
    missing_C=0,
    missing_O=0
  )

  for i in range(len(df_split)):
    if limit is not None and i >= limit:
      break

    print(f"-- ROW {i+1}/{len(df_split)}")
    row = df_split.iloc[i]

    pdb_filename = os.path.join(pdb_folder, row.cath_id)
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(row.cath_id, pdb_filename)

    # Expect only one model per structure.
    assert(len(structure) == 1)

    seq = []
    for model in structure:
      for chain in model:
        for residue in chain:
          if residue.get_id()[0] == " ":  # This checks if it's a standard residue
            # NOTE(milo): Need to handle unknown residues.
            if residue.get_resname() not in protein_letters_3to1:
              warn["unknown_residues"].append(residue.get_resname())
              seq.append("_")
            else:
              seq.append(protein_letters_3to1[residue.get_resname()])
          else:
            warn["nonstandard"] += 1
            print('nonstandard', residue.get_id())

    print("".join(seq))

    coords_each_residue = []
    for residue in structure.get_residues():
      atom_positions = dict([
        (a.get_fullname().strip(), list(a.get_coord())) for a in residue.get_atoms()
      ])

      # Get the coordinates of backbone atoms, using the ordering of GVP's code.
      coords = []
      for a in ('N', 'CA', 'C', 'O'):
        if a in atom_positions:
          coords.append(atom_positions[a])
        else:
          warn[f"missing_{a}"] += 1

          # NOTE(milo): When an atom is missing, just use the center of mass of the residue for now.
          # There are probably more sophisticated ways of imputing the atom's position, but we can
          # at least get it close to the other atoms in this residue.
          coords.append(residue.center_of_mass())

      coords_each_residue.append(coords)

    # Gather and standardize all of the data into our datamodel.
    item = dm.ProteinBackboneWithEmbedding(
      name=row.cath_id,
      seq="".join(seq),
      pdb_id=row.pdb_id,
      cath_id=row.cath_id,
      class_=row["class"],
      architecture=row.architecture,
      topology=row.topology,
      superfamily=row.superfamily,
      # This is the label that the model will learn to predict!
      task_label=dm.architecture_labels[(row["class"], row["architecture"])],
      coords=coords_each_residue,
    )

    assert(len(coords_each_residue) == len(seq))

    data_list.append(item)

    # Write the JSON output for this item:
    with open(os.path.join(output_folder, f"{item.cath_id}.json"), "w") as f:
      json.dump(item.model_dump(), f, indent=2)

  return data_list, warn


if __name__ == "__main__":
  """
  Preprocess data for training and/or inference.

  If you're just trying to evaluate a pre-trained model, you'll want to run a command like:

  ```
  python preprocess_dataset.py --csv challenge_test_set.csv \
    --output-folder ../data/challenge_test_set \
    --pdb-folder ../data/pdb_share
  ```
  where the CSV file follows the same format as `cath_w_seqs_share.csv`.

  You should see a printout like:
  ...
  -- ROW 494/1261
  SVKDPTLLRIKIVPVQPFIANSRKQLDLWASSHLLSMLMYKALEVIVDKFGPEHVIYPSLRDQPFFLKFYLGENIGDEILVANLPNKALAIVSGKEAEKIEEEIKKRIRDFLLQLYREAVDWAVENGVVKVDRSEKDSMLKEAYLKIVREYFTVSITWVSLAIYPLLVKILDSLGERKVTEEGWKCHVCGENLAIFGDMYDHDNLKSLWLDEEPLCPMCLIKRYYPVWIRSKTGQKIRFE
  -- ROW 495/1261
  WKEAHFQDAFSSFQAMYAKSYATEEEKQRRYAIFKNNLVYIHTHNQQGYSYSLKMNHFGDLSRDEFRRKYLGFKK
  -- ROW 496/1261
  ...
  """
  import argparse
  parser = argparse.ArgumentParser()

  parser.add_argument('--csv', type=str, help="The path to a `<split>_cath_w_seqs_share.csv` file", default=None)
  parser.add_argument('--output-folder', type=str, help="The output folder for preprocessed JSON files", default=None)
  parser.add_argument('--all', action="store_true", help="Process all of the splits at once.", default=False)
  parser.add_argument('--pdb-folder', type=str, help="The path to a folder with PDB files in it (e.g pdb_share/)",
                      default=paths.data_folder("pdb_share"))

  args = parser.parse_args()

  # NOTE(milo): This code is for my training process only; you shouldn't need to run it.
  if args.all:
    # check_disjoint_dataset_splits()
    dataset_version = "cleaned_skip_missing"

    for split_name in ("train", "test", "val"):
      print(f"\n\n-------- {split_name} --------")
      
      if not os.path.exists(paths.data_folder(f"{dataset_version}/{split_name}")):
        os.makedirs(paths.data_folder(f"{dataset_version}/{split_name}"))

      split_path = paths.data_folder(f"{split_name}_cath_w_seqs_share.csv")

      # NOTE(milo): You can set a small `limit` here for debugging purposes.
      data, warnings = preprocess(
        split_path,
        paths.data_folder(f"{dataset_version}/{split_name}"),
        pdb_folder=args.pdb_folder,
        limit=None
      )

      print("\nDONE!")
      print("\nWARNINGS:")
      print(warnings)

  # This is the code that you want to run for evaluation:
  else:
    assert(args.csv is not None and args.output_folder is not None)
    os.makedirs(args.output_folder, exist_ok=True)
    
    # NOTE(milo): You can set a small `limit` here for debugging purposes.
    data, warnings = preprocess(
      args.csv,
      args.output_folder,
      pdb_folder=args.pdb_folder,
      limit=None
    )

    print("\nDONE!")

    print("\nWARNINGS:")
    print(warnings)
