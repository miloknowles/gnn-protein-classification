python train.py --train \
  --model-id gvp_l3_k30_s100_no_esm2_conv2 \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --max-nodes 3000 \
  --num-workers 6 \
  --gnn-layers 3 \
  --top-k 30 \
  --node-h-scalar-dim 100 \
  --lr 1e-4 \
  --plm esm2_t33_650M_UR50D \
  