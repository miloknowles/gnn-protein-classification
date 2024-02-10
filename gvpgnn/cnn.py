# https://github.com/drorlab/atom3d/blob/master/atom3d/models/cnn.py

import torch
import torch.nn as nn


class CNN3D(nn.Module):
  """Base 3D convolutional neural network architecture for molecular data,
  consisting of six layers of 3D convolutions, each followed by batch
  normalization, ReLU activation, and optionally dropout.

  This network uses strided convolutions to downsample the input twice by half,
  so the original box size must be divisible by 4 (e.g. an atomic environment of
  side length 20 Å with 1 Å voxels). The final convolution reduces the 3D box to
  a single 1D vector of length ``out_dim``.
  
  The desired input and output dimensionality must be specified when
  instantiating the model.

  Parameters
  ----------
  * `in_dim` : The number of input channels
  * `out_dim` : The number of output channels
  * `box_size`: The size (edge length) of the voxelized 3D cube input
  * `hidden_dim` : The number of channels to use internally
  * `dropout` : The dropout probability
  """
  def __init__(self, in_dim: int, out_dim: int, box_size: int, hidden_dim=64, dropout=0.1):
    super(CNN3D, self).__init__()
    self.out_dim = out_dim

    self.model = nn.Sequential(
      nn.Conv3d(in_dim, hidden_dim, 4, 2, 1, bias=False),
      nn.BatchNorm3d(hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv3d(hidden_dim, hidden_dim * 2, 3, 1, 1, bias=False),
      nn.BatchNorm3d(hidden_dim * 2),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
      nn.BatchNorm3d(hidden_dim * 4),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, 1, 1, bias=False),
      nn.BatchNorm3d(hidden_dim * 8),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, 1, 1, bias=False),
      nn.BatchNorm3d(hidden_dim * 16),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout),

      nn.Conv3d(hidden_dim * 16, out_dim, int(box_size / 4), 1, 0, bias=False),
    )

  def forward(self, input):
    """Forward method.

    :param input: Input data, as voxelized 3D cube of shape (batch_size, in_dim, box_size, box_size, box_size).
    :type input: torch.FloatTensor
    :return: Output of network, of shape (batch_size, out_dim)
    :rtype: torch.FloatTensor
    """
    bs = input.size()[0]
    output = self.model(input)
    return output.view(bs, self.out_dim)
  


class CNN3DClassifier(nn.Module):
  """Base 3D convolutional neural network architecture for molecular data,
  consisting of six layers of 3D convolutions, each followed by batch
  normalization, ReLU activation, and optionally dropout.

  This network uses strided convolutions to downsample the input twice by half,
  so the original box size must be divisible by 4 (e.g. an atomic environment of
  side length 20 Å with 1 Å voxels). The final convolution reduces the 3D box to
  a single 1D vector of length ``out_dim``.
  
  The desired input and output dimensionality must be specified when
  instantiating the model.

  Parameters
  ----------
  * `in_dim` : The number of input channels
  * `out_dim` : The number of output channels
  * `box_size`: The size (edge length) of the voxelized 3D cube input
  * `hidden_dim` : The number of channels to use internally
  * `dropout` : The dropout probability
  """
  def __init__(
    self,
    in_dim: int,
    conv_out_dim: int,
    logits_out_dim: int,
    box_size: int,
    conv_hidden_dim=64,
    dense_hidden_dim=64,
    dropout=0.1
  ):
    super(CNN3DClassifier, self).__init__()
    self.logits_out_dim = logits_out_dim

    # 3DCNN for feature extraction
    self.conv = CNN3D(in_dim, conv_out_dim, box_size, conv_hidden_dim, dropout)

    # Final dense block, which makes label predictions.
    self.dense = nn.Sequential(
      nn.Linear(conv_out_dim, dense_hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout),
      nn.Linear(dense_hidden_dim, dense_hidden_dim),
      nn.ReLU(inplace=True),
      nn.Dropout(p=dropout),
      nn.Linear(dense_hidden_dim, logits_out_dim)
    )

  def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    """Forward method.

    :param input: Input data, as voxelized 3D cube of shape
      (batch_size, in_dim, box_size, box_size, box_size).
    :return: Output of network, of shape
      (batch_size, logits_out_dim)
    """
    bs = input.size()[0]
    x = self.conv(input)
    x = self.dense(x)
    return x.view(bs, self.logits_out_dim)  