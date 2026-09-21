#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=/mnt/cache/wanghanzhi/XK/WHU-Baseline
PYTHON=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/WHU-Baseline/configs/B_baseline_candidate.yml
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
OUTPUT=/mnt/cache/wanghanzhi/XK/WHU-Baseline_runs/B_baseline_candidate_retry1

cd "$CODE_DIR"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Refusing to reuse non-empty output directory: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

"$PYTHON" train.py --config_file "$CONFIG" \
  MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" \
  OUTPUT_DIR "$OUTPUT" 2>&1 | tee "$OUTPUT/train_stdout.log"

test -s "$OUTPUT/transformer_60.pth"
mkdir -p "$OUTPUT/eval_epoch60"
"$PYTHON" test.py --config_file "$CONFIG" \
  MODEL.PRETRAIN_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" \
  TEST.WEIGHT "$OUTPUT/transformer_60.pth" \
  OUTPUT_DIR "$OUTPUT/eval_epoch60" 2>&1 | tee "$OUTPUT/eval_epoch60/eval_stdout.log"
