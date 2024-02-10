python train.py --train \
  --model-id cnn_test_v0 \
  --train-path ../data/cleaned_skip_missing/train \
  --val-path ../data/cleaned_skip_missing/val \
  --test-path ../data/cleaned_skip_missing/test \
  --num-workers 4 \
  --lr 1e-4 \