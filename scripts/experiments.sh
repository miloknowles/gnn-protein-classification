# Simplest and least powerful model
# python train.py --train \
#   --model-id gvp_l2_k10_no_esm2 \
#   --train-path ../data/cleaned_skip_missing/train \
#   --val-path ../data/cleaned_skip_missing/val \
#   --test-path ../data/cleaned_skip_missing/test \
#   --max-nodes 3000 \
#   --num-workers 4 \
#   --gnn-layers 2 \
#   --top-k 10 \

# Try adding more layers to the GNN. The GVP authors use 4.
# python train.py --train \
#   --model-id gvp_l4_k30_no_esm2 \
#   --train-path ../data/cleaned_skip_missing/train \
#   --val-path ../data/cleaned_skip_missing/val \
#   --test-path ../data/cleaned_skip_missing/test \
#   --max-nodes 3000 \
#   --num-workers 4 \
#   --gnn-layers 4 \
#   --top-k 30 \

python train.py --train \
  --model-id gvp_l3_k10_esm2_t33_650M_UR50D \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --max-nodes 3000 \
  --num-workers 4 \
  --gnn-layers 3 \
  --top-k 10 \
  --plm esm2_t33_650M_UR50D \