import torch

Ca1 = torch.Tensor([
  [0, 0, 0],
  [0, 0, 1],
  [1, 2, 3]
])

T = torch.cdist(Ca1, Ca1)

print(T)