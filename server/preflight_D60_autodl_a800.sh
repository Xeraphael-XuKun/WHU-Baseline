#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=/root/autodl-tmp/XK/WHU-Baseline-env-cudnn-e60
PYTHON=/root/autodl-tmp/envs/whu_mars/bin/python3
DATA_ROOT=/root/autodl-tmp
PRETRAIN=/root/autodl-tmp/ViT-B-16.pt

cd "$CODE_DIR"
test -x "$PYTHON"
test -d "$DATA_ROOT/WHU-MARS"
test -s "$PRETRAIN"
test -z "$(git status --porcelain)"

echo "代码提交：$(git rev-parse HEAD)"

"$PYTHON" -c "import sys, torch, torchvision, timm; assert sys.version_info[:2] == (3, 10); assert torch.__version__ == '2.2.2+cu121'; assert torchvision.__version__ == '0.17.2+cu121'; assert timm.__version__ == '1.0.27'; assert torch.backends.cudnn.version() == 8902; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); assert p.total_memory >= 70*1024**3, p.total_memory; x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.t()).sum().backward(); print('D60_AUTODL_ENV_OK', sys.executable, torch.__version__, torchvision.__version__, timm.__version__, p.name, round(p.total_memory/1024**3,1))"

"$PYTHON" -m pip check
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo D60_AUTODL_A800_PREFLIGHT_OK
