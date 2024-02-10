python train_cnn.py --train \
  --model-id cnn_test_v512 \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --num-workers 1 \
  --lr 1e-4 \
  --voxel-grid-dim 64 \
  --batch-size 2 \
  --random-rotation \