#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1

CODE_DIR=/mnt/cache/wanghanzhi/XK/WHU-Baseline-env-cudnn-e60
PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/WHU-Baseline-env-cudnn-e60/configs/G60_llmpar_benchmark_false.yml
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
OUTPUT=/mnt/cache/wanghanzhi/XK/WHU-Baseline_runs/diagnostic_e60/G60_llmpar_benchmark_false

cd "$CODE_DIR"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "拒绝复用非空输出目录：$OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

git rev-parse HEAD > "$OUTPUT/git_commit.txt"
"$PYTHON" -m pip freeze --all > "$OUTPUT/pip_freeze.txt"
nvidia-smi > "$OUTPUT/nvidia_smi.txt"

echo "G60：llmpar，CUDNN_BENCHMARK=False，完整训练60轮"
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

echo G60_TRAIN_AND_DISK_RELOAD_EVAL_OK
