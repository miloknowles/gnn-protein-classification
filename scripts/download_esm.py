import sys; sys.path.append("..")

import gvpgnn.embeddings as plm

for name in plm.esm2_model_dictionary:
  print(name)
  # This will trigger a download if the pre-trained weights aren't found.
  model, alphabet = plm.esm2_model_dictionary[name]()