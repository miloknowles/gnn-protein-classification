import numpy as np


def print_confusion(mat, lookup, n_categories: int = 10):
  """Prints out a confusion matrix during training."""
  counts = mat.astype(np.int32)
  mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
  mat = np.round(mat * 1000).astype(np.int32)
  res = '\n'
  for i in range(n_categories):
    res += '\t{}'.format(lookup[i])
  res += '\tCount\n'
  for i in range(n_categories):
    res += '{}\t'.format(lookup[i])
    res += '\t'.join('{}'.format(n) for n in mat[i])
    res += '\t{}\n'.format(sum(counts[i]))
  print(res)
