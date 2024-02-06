# https://github.com/facebookresearch/esm/blob/main/README.md
import torch
import esm


esm2_model_dictionary = {
  "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
  "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
  "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
}


esm2_embedding_layer = {
  "esm2_t6_8M_UR50D": 6,
  "esm2_t12_35M_UR50D": 12,
  "esm2_t30_150M_UR50D": 33,
}


esm2_embedding_dims = {
  "esm2_t6_8M_UR50D": 320,
  "esm2_t12_35M_UR50D": 480,
  "esm2_t30_150M_UR50D": 1280,
}


def extract_embedding_single(
  model: torch.nn.Module,
  alphabet: esm.Alphabet,
  model_embedding_layer: int,
  sequence: str
) -> torch.Tensor:
  """Returns an embedding of shape `(num tokens, embedding dim)`."""
  model.eval()

  batch_converter = alphabet.get_batch_converter()

  with torch.no_grad():
    _, _, batch_tokens = batch_converter([("tmp", sequence)])

    results = model(batch_tokens, repr_layers=[model_embedding_layer], return_contacts=False)

    # Skip the first and last token, since this is a start and end.
    token_representations = results["representations"][model_embedding_layer][:,1:-1,:].squeeze(0)

  return token_representations