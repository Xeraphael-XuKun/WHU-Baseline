#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

cd /mnt/cache/wanghanzhi/XK/WHU-Baseline

test -s /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train
test -x /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3

/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 -c "import torch, torchvision, timm, yacs; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 tools/audit_candidates.py --data-root /mnt/cache/wanghanzhi/Datasets

echo PREFLIGHT_OK
