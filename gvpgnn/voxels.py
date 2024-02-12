import Bio.PDB.PDBParser
import torch


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


def get_voxel_indices(p: torch.Tensor, voxel_size: float, voxel_grid_dim: int) -> torch.Tensor:
  """Determine the indices that each point should map to in a voxel grid.
  
  All of the points should have been centered and scaled already. None of the
  coordinates should be less than zero.
  """
  assert(voxel_size > 1e-4) # nonnegative
  scale_factor = 1 / voxel_size
  indices = (p * scale_factor).floor().clamp_max(max=voxel_grid_dim - 1)
  return indices


def get_voxel_indices(p: torch.Tensor, voxel_size: float, voxel_grid_dim: int) -> torch.Tensor:
  """Determine the indices that each point should map to in a voxel grid.
  
  All of the points should have been centered and scaled already. None of the
  coordinates should be less than zero.
  """
  assert(voxel_size > 1e-4) # nonnegative
  scale_factor = 1 / voxel_size
  indices = (p * scale_factor).floor().long().clamp_max(max=voxel_grid_dim - 1)
  return indices


def create_occupancy_grid(points: torch.Tensor, G: int = 100) -> torch.Tensor:
  """Create a 3D occupancy grid from a collection of poins."""
  indices = get_voxel_indices(points, voxel_size=1/G, voxel_grid_dim=G)
  O = torch.zeros((G, G, G))

  for idx in indices:
    O[tuple(idx)] += 1

  return O