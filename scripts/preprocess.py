import sys; sys.path.append("..")
import os
from typing import Optional
import json

import pandas as pd

import Bio.PDB.PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

import gvpgnn.paths as paths
import gvpgnn.datamodels as dm


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
  pdb_folder: str = paths.data_folder("pdb_share"),
  limit: Optional[int] = None
) -> tuple:
  """Preprocess the raw training data into a more useful format for the DataLoader.
  
  Notes
  -----
  The dataset should already be split into train, test, and validation.

  We do the preprocessing ahead of time to reduce the complexity of the data
  loader code. Everything is put into a standardized JSON format before it gets
  to the data loader.

  Currently, all of the JSON data is stored in a single file per split. If the
  dataset was much larger, we'd probably want to output a single file per protein,
  and have the dataloader load files from disk as needed (instead of up-front).

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
          coords.append([-1, -1, -1]) # TODO

      coords_each_residue.append(coords)

    # Gather and standardize all of the data into our datamodel.
    item = dm.BackboneModel(
      name=row.cath_id,
      seq="".join(seq),
      pdb_id=row.pdb_id,
      class_=row["class"],
      architecture=row.architecture,
      topology=row.topology,
      superfamily=row.superfamily,
      coords=coords_each_residue
    )

    assert(len(coords_each_residue) == len(seq))

    data_list.append(item)

  return data_list, warn


if __name__ == "__main__":
  check_disjoint_dataset_splits()

  for split_name in ("train", "test", "val"):
    print(f"\n\n-------- {split_name} --------")
    split_path = paths.data_folder(f"{split_name}_cath_w_seqs_share.csv")

    # NOTE(milo): You can set a small `limit` here for debugging purposes.
    data, warnings = preprocess(split_path, pdb_folder=paths.data_folder("pdb_share"), limit=None)
    print("\nDONE!")

    print("\nWARNINGS:")
    print(warnings)

    # https://github.com/pydantic/pydantic/discussions/4091
    data_list = [item.dict() for item in data]

    # Write the JSON data to a file for each split:
    with open(paths.data_folder(f"{split_name}_cleaned.json"), "w") as f:
      json.dump(data_list, f, indent=2)