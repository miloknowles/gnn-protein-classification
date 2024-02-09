import sys; sys.path.append('..')
import glob
import os
import shutil
import gvpgnn.paths as paths


def write_pymol_script(input_folder: str, output_folder: str):
  """Write a script to execute using pymol."""
  pdb_files = [fn for fn in glob.glob(input_folder + "/*") if ".pdb" not in fn]
  print(f"Found {len(pdb_files)} files in the input folder '{input_folder}'")

  print(f"Output folder will be '{output_folder}'")
  os.makedirs(output_folder, exist_ok=True)

  with open(paths.data_folder("meshing.pml"), "w") as f:
    for l in pdb_files:
      input_file = l
      cath_id = l.split("/")[-1]
      file_w_extension = f"{l}.pdb"
      output_file = os.path.join(output_folder, f"{cath_id}.obj")
      shutil.copy(input_file, file_w_extension)
      f.write(f"load {file_w_extension}\n")
      f.write(f"save {output_file}\n")
      f.write("reinitialize\n")

if __name__ == "__main__":
  """
  Usage
  -----

  1. Add an alias for the `pymol` command:
  > alias pymol=/Applications/PyMOL.app/Contents/MacOS/PyMOL

  2. Run this script to generate a `meshing.pml` script in the `data` folder.

  3. The the generated script using `pymol`:
  > pymol -cq ../data/meshing.pml

  This will output `.obj` files for each protein found in the input folder.

  Notes
  -----
  https://pymol.org/pymol-command-ref.html#reinitialize
  """
  input_folder = paths.data_folder("pdb_share")
  output_folder = paths.data_folder("obj")

  print("Creating a script that you can pass to 'pymol'")
  write_pymol_script(input_folder, output_folder)