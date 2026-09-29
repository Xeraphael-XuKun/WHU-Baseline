#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1
cd /mnt/cache/wanghanzhi/XK/WHU-Baseline
PYTHON=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
test -s "$PRETRAIN"
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train
"$PYTHON" -m pip check
"$PYTHON" tests/test_trajectory.py
for NAME in A_trajectory_direct_full A_trajectory_full; do
  "$PYTHON" tools/audit_baseline.py \
    --config "/mnt/cache/wanghanzhi/XK/WHU-Baseline/configs/$NAME.yml" \
    --data-root /mnt/cache/wanghanzhi/Datasets --pretrain "$PRETRAIN"
done
echo "FULL_PREFLIGHT_OK：数据、模型和优化器构建通过；尚未正式训练。"
echo "可按手册单独运行真实CLIP合成小batch检查，再启动正式实验并确认首个正常iteration。"
