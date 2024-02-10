python train.py --train \
  --model-id cnn_test_v1024_b64 \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --num-workers 6 \
  --lr 1e-4 \
  --voxel-grid-dim 1024 \
  --batch-size 64 \
  --random-rotation \