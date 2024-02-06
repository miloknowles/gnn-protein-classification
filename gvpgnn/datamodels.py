from pydantic import BaseModel


class BackboneModel(BaseModel):
  """The (augmented) protein backbone model used in the GVP code.

  See: https://github.com/drorlab/gvp-pytorch/blob/main/README.md
    """
  name: str # Use the CATH id here
  seq: str # Sequence of amino acid codes

  pdb_id: str

  class_: int
  architecture: int
  topology: int
  superfamily: int

  # Nested list of the positions of the backbone N, C-alpha, C, and O atoms, in that order.
  coords: list[list[list[float]]] # Shape: (num_residues, 4, 3)