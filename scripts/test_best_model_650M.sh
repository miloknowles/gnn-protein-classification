# All of these configuration options should match those in `train_best_model.sh`!
python train_gnn.py --test \
  --model-id gvp_l3_k30_s100_650M_topk_pl3_ch3 \
  --test-path ../data/challenge_test_set_with_esm2_t33_650M_UR50D \
  --max-nodes 3000 \
  --num-workers 4 \
  --gnn-layers 3 \
  --top-k 30 \
  --node-h-scalar-dim 100 \
  --lr 1e-4 \
  --pooling-op topk \
  --n-pool-layers 3 \
  --n-conv-heads 3 \
  --plm esm2_t33_650M_UR50D \
  --checkpoint ../models/best_model_650M/checkpoint_1.pt