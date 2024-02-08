"""The command line interface for training."""

import sys; sys.path.append('..')

import argparse
from datetime import datetime

import gvpgnn.paths as paths
import gvpgnn.embeddings as embeddings


parser = argparse.ArgumentParser()
parser.add_argument('--model-id',
                    default=int(datetime.timestamp(datetime.now())),
                    type=str,
                    help="The name of this model. Also determines the subfolder where checkpoints will go.")
parser.add_argument('--models-dir', metavar='PATH', default=paths.models_folder(),
                    help='Directory to save trained models')
parser.add_argument('--num-workers', metavar='N', type=int, default=2,
                    help='Number of threads for loading data')
parser.add_argument('--max-nodes', metavar='N', type=int, default=3000,
                    help='Max number of nodes per batch')
parser.add_argument('--epochs', metavar='N', type=int, default=100,
                    help='Training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help="The learning rate for Adam optimizer")
parser.add_argument('--drop-rate', type=float, default=0.1, help="The dropout rate")

parser.add_argument('--plm', choices=[*embeddings.esm2_model_dictionary.keys(), None], 
                    default=None, help="Which pretrained protein language model to use (default None)")
parser.add_argument('--edge-algorithm', choices=["knn_graph", "radius_graph"], default="knn_graph", help="How to determine graph connectivity")
parser.add_argument('--top-k', type=int, default=30, help="How many k-nearest neighbors to connect in the GNN")
parser.add_argument('--r-ball-radius', type=float, default=8, help="Radius for r-ball graph algorithm (in angstroms)")
parser.add_argument('--gnn-layers', type=int, default=4, help="Number of GVP layers in the GNN")
parser.add_argument('--node-h-scalar-dim', type=int, default=100, help="The dimensionality for node scalar embeddings in the GNN")
parser.add_argument('--pooling-op', choices=["naive", "conv", "topk"], default="conv", help="The pooling operation to use")
parser.add_argument('--n-conv-heads', type=int, default=1, help="The number of heads in Transformer layers")
parser.add_argument('--n-pool-layers', type=int, default=3, help="The number of pooling layers to apply")

parser.add_argument('--train', action="store_true", help="Train a model from scratch or from a checkpoint.")
parser.add_argument('--test', action="store_true", help="Test a trained model")

parser.add_argument('--train-path', type=str, help="The path to a folder with TRAINING data. Only required with --train.", default=None)
parser.add_argument('--val-path', type=str, help="The path to a folder with VALIDATION data. Only required with --train.", default=None)
parser.add_argument('--test-path', type=str, help="The path to a folder with TEST data. Only required with --test.", default=None)
parser.add_argument('--checkpoint', type=str, help="A path to pre-trained weights (e.g `path/to/checkpoint.pt`)", default=None)