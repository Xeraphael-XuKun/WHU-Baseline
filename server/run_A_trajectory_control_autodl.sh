#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1

CODE_DIR=/root/autodl-tmp/XK/WHU-Baseline
PYTHON=/root/autodl-tmp/envs/whu_mars/bin/python3
CONFIG=/root/autodl-tmp/XK/WHU-Baseline/configs/A_baseline_candidate.yml
DATA_ROOT=/root/autodl-tmp
PRETRAIN=/root/autodl-tmp/ViT-B-16.pt
OUTPUT=/root/autodl-tmp/XK/WHU-Baseline_runs/autodl/A_trajectory_control

cd "$CODE_DIR"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "拒绝复用非空输出目录： $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

echo "AutoDL 单卡前台实验，配置：$CONFIG，输出：$OUTPUT"
"$PYTHON" -m pip freeze --all > "$OUTPUT/pip_freeze.txt"
nvidia-smi > "$OUTPUT/nvidia_smi.txt"

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
