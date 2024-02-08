python train.py --train \
  --model-id gvp_l3_k30_s100_no_esm2_virtual \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --max-nodes 3000 \
  --num-workers 6 \
  --gnn-layers 3 \
  --top-k 30 \
  --node-h-scalar-dim 100 \
  --lr 1e-4 \
  --pooling-op naive \
  --n-pool-layers 3 \
  --n-conv-heads 3 \