#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=/mnt/cache/wanghanzhi/XK/WHU-Baseline-env-cudnn-e60
WHU_PYTHON=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
LLMPAR_PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt

cd "$CODE_DIR"
test -x "$WHU_PYTHON"
test -x "$LLMPAR_PYTHON"
test -d "$DATA_ROOT/WHU-MARS"
test -s "$PRETRAIN"
test -z "$(git status --porcelain)"

echo "代码提交：$(git rev-parse HEAD)"

"$WHU_PYTHON" -c "import sys, torch, torchvision, timm; assert sys.version_info[:2] == (3, 10); assert torch.__version__ == '2.2.2+cu121'; assert torchvision.__version__ == '0.17.2+cu121'; assert timm.__version__ == '1.0.27'; assert torch.backends.cudnn.version() == 8902; assert torch.cuda.is_available(); x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.t()).sum().backward(); print('WHU_ENV_OK', sys.executable, torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.get_device_name(0))"

"$LLMPAR_PYTHON" -c "import sys, torch, torchvision, timm; assert sys.version_info[:2] == (3, 8); assert torch.__version__ == '2.4.1+cu121'; assert torchvision.__version__.split('+')[0] == '0.13.1'; assert timm.__version__ == '1.0.19'; assert torch.cuda.is_available(); x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.t()).sum().backward(); print('LLMPAR_ENV_OK', sys.executable, torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.get_device_name(0))"

"$WHU_PYTHON" -m pip check
"$LLMPAR_PYTHON" -m pip check || echo "LLMPAR_PIP_CHECK_NONZERO_EXPECTED_DIAGNOSTIC_ONLY"

echo EFG60_A800_PREFLIGHT_OK
