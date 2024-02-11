from pydantic import BaseModel


architecture_names = {
  (1, 10): "Mainly Alpha: Orthogonal Bundle",
  (1, 20): "Mainly Alpha: Up-down Bundle",
  (2, 30): "Mainly Beta: Roll",
  (2, 40): "Mainly Beta: Beta Barrel",
  (2, 60): "Mainly Beta: Sandwich",
  (3, 10): "Alpha Beta: Roll",
  (3, 20): "Alpha Beta: Alpha-Beta Barrel",
  (3, 30): "Alpha Beta: 2-Layer Sandwich",
  (3, 40): "Alpha Beta: 3-Layer(aba) Sandwich",
  (3, 90): "Alpha Beta: Alpha-Beta Complex"
}


architecture_labels = {
  (1, 10): 0,
  (1, 20): 1,
  (2, 30): 2,
  (2, 40): 3,
  (2, 60): 4,
  (3, 10): 5,
  (3, 20): 6,
  (3, 30): 7,
  (3, 40): 8,
  (3, 90): 9,
}


num_to_readable_architecture = {v: str(k) for k, v in architecture_labels.items()}


class ProteinBackboneWithEmbedding(BaseModel):
  """The (augmented) protein backbone model used in the GVP code.

  See: https://github.com/drorlab/gvp-pytorch/blob/main/README.md
  """
  name: str # Use the CATH id here
  seq: str # Sequence of amino acid codes

  pdb_id: str
  cath_id: str

  class_: int
  architecture: int
  topology: int
  superfamily: int

  task_label: int # The labels for this learning task (0-9)

  # Nested list of the positions of the backbone N, C-alpha, C, and O atoms, in that order.
  coords: list[list[list[float]]] # Shape: (num_residues, 4, 3)

  # Optional precomputed language model embeddings.
  esm2_t6_8M_UR50D: list[list[float]] | None = None # esm2_t6_8M_UR50D
  esm2_t12_35M_UR50D: list[list[float]] | None = None # esm2_t12_35M_UR50D
  esm2_t30_150M_UR50D: list[list[float]] | None = None # esm2_t30_150M_UR50D
  esm2_t33_650M_UR50D: list[list[float]] | None = None # esm2_t33_650M_UR50D