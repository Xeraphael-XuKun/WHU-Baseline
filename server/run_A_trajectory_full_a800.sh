#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1
CODE_DIR=/mnt/cache/wanghanzhi/XK/WHU-Baseline
PYTHON=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
CONFIG=/mnt/cache/wanghanzhi/XK/WHU-Baseline/configs/A_trajectory_full.yml
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
OUTPUT=/mnt/cache/wanghanzhi/XK/WHU-Baseline_runs/A_trajectory_full
cd "$CODE_DIR"
if [ -d "$OUTPUT" ] && [ -n "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "拒绝复用非空输出目录：$OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
"$PYTHON" -m pip freeze --all > "$OUTPUT/pip_freeze.txt"
nvidia-smi > "$OUTPUT/nvidia_smi.txt"
"$PYTHON" -c "import sys, torch, torchvision, timm; print(sys.version); print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__,'CUDA',torch.version.cuda,'cuDNN',torch.backends.cudnn.version())" > "$OUTPUT/runtime.txt"
cp "$CONFIG" "$OUTPUT/config.yml"
echo "开始 A_trajectory_full：单卡60轮，RGB四格监督求和×5，完成后post→pre独立复评"
"$PYTHON" train.py --config_file "$CONFIG" \
  MODEL.PRETRAIN_PATH "$PRETRAIN" MODEL.FULL.TEXT_CLIP_PATH "$PRETRAIN" \
  DATASETS.ROOT_DIR "$DATA_ROOT" OUTPUT_DIR "$OUTPUT" 2>&1 | tee "$OUTPUT/train_stdout.log"
test -s "$OUTPUT/transformer_60.pth"
for NECK in after before; do
  if [ "$NECK" = after ]; then EVAL="$OUTPUT/eval_epoch60_postbn"; else EVAL="$OUTPUT/eval_epoch60_prebn"; fi
  mkdir -p "$EVAL"
  "$PYTHON" test.py --config_file "$CONFIG" \
    MODEL.PRETRAIN_PATH "$PRETRAIN" MODEL.FULL.TEXT_CLIP_PATH "$PRETRAIN" \
    DATASETS.ROOT_DIR "$DATA_ROOT" TEST.WEIGHT "$OUTPUT/transformer_60.pth" \
    TEST.NECK_FEAT "$NECK" OUTPUT_DIR "$EVAL" 2>&1 | tee "$EVAL/eval_stdout.log"
done
echo "A_trajectory_full：训练及post/pre独立复评全部完成"
