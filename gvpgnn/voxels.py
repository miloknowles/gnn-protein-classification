import Bio.PDB.PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

import numpy as np
import torch

import open3d as o3d


def extract_point_clouds_N_Ca_C_O(pdb_filename: str, cath_id: str):
  # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
  import warnings
  warnings.filterwarnings("ignore")

  pdb_parser = Bio.PDB.PDBParser()
  structure = pdb_parser.get_structure(cath_id, pdb_filename)

  # Expect only one model per structure.
  assert(len(structure) == 1)

  points_N = []
  points_Ca = []
  points_C = []
  points_O = []

  for residue in structure.get_residues():
    for a in residue.get_atoms():
      name = a.get_fullname().strip()

      if name == 'N':
        points_N.append(a.get_coord())
      elif name == 'CA':
        points_Ca.append(a.get_coord())
      elif name == 'C':
        points_C.append(a.get_coord())
      elif name == 'O':
        points_O.append(a.get_coord())

  return [torch.Tensor(l) for l in [points_N, points_Ca, points_C, points_O]]


def center_and_scale_unit_sphere(points: torch.Tensor) -> torch.Tensor:
  mu = points.mean(dim=0)
  vmax, _ = points.max(dim=0)
  vmin, _ = points.min(dim=0)
  max_dim = (vmax - vmin).norm()
  points = 2 * (points - mu) / max_dim
  return points + 0.5


def center_and_scale_unit_box(points: torch.Tensor) -> torch.Tensor:
  """
  Scales all of the points uniformly so that they are in the range [0, 1].
  """
  mu = points.mean(dim=0)
  # Min and max corners of the bounding box.
  vmax = torch.Tensor([points[:,0].amax(), points[:,1].amax(), points[:,2].amax()])
  vmin = torch.Tensor([points[:,0].amin(), points[:,1].amin(), points[:,2].amin()])
  sf = 1.0 / (vmax - vmin + 1).max() # largest bbox dimension
  centered = (points - mu)
  return (centered * sf) + 0.5


def create_occupancy_grid(points: torch.Tensor, G: int = 100) -> torch.Tensor:
  """Create a 3D occupancy grid from a collection of poins."""
  pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
  voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=(1/G))
  voxels = voxel_grid.get_voxels()
  indices = np.stack(list(vx.grid_index for vx in voxels))

  O = torch.zeros((G, G, G))

  for idx in indices:
    O[tuple(idx)] += 1

  return O