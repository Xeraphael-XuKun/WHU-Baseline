#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export PYTHONUNBUFFERED=1

cd /root/autodl-tmp/XK/WHU-Baseline
test -x /root/autodl-tmp/envs/whu_mars/bin/python3
test -s /root/autodl-tmp/ViT-B-16.pt
test -d /root/autodl-tmp/WHU-MARS/train
test -s configs/A_trajectory.yml
test -s model/backbones/trajectory.py

nvidia-smi
/root/autodl-tmp/envs/whu_mars/bin/python3 -m pip check
/root/autodl-tmp/envs/whu_mars/bin/python3 - <<'PY'
import sys
import torch, torchvision, timm, yacs
print('python',sys.version,'executable',sys.executable)
print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__)
print('CUDA',torch.version.cuda,'cuDNN',torch.backends.cudnn.version())
assert torch.cuda.is_available(), 'CUDA 不可用'
assert torch.cuda.device_count()==1, '应仅暴露一张GPU'
print('GPU',torch.cuda.get_device_name(0),'GiB',torch.cuda.get_device_properties(0).total_memory/1024**3)
PY
/root/autodl-tmp/envs/whu_mars/bin/python3 tests/test_trajectory.py
/root/autodl-tmp/envs/whu_mars/bin/python3 tools/audit_baseline.py \
  --config /root/autodl-tmp/XK/WHU-Baseline/configs/A_trajectory.yml \
  --data-root /root/autodl-tmp \
  --pretrain /root/autodl-tmp/ViT-B-16.pt
echo "AUTODL_PREFLIGHT_OK：环境、机制测试及真实数据/模型构建通过；尚未开始正式训练。"
