#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

CODE_DIR=/mnt/cache/wanghanzhi/XK/WHU-Baseline-diagnostic
WHU_PYTHON=/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
TEACHER_PYTHON=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
DATA_ROOT=/mnt/cache/wanghanzhi/Datasets
PRETRAIN=/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt

cd "$CODE_DIR"
test -x "$WHU_PYTHON"
test -x "$TEACHER_PYTHON"
test -d "$DATA_ROOT/WHU-MARS"
test -s "$PRETRAIN"

for PYTHON in "$WHU_PYTHON" "$TEACHER_PYTHON"; do
  "$PYTHON" -c "import sys, torch, torchvision, timm, yacs; print('executable', sys.executable); print('python', sys.version); print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'timm', timm.__version__); print('cuda_build', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); assert torch.cuda.is_available(); print('gpu', torch.cuda.get_device_name(0)); x=torch.randn(32,32,device='cuda',requires_grad=True); (x@x.t()).sum().backward(); print('CUDA_BACKWARD_OK')"
  "$PYTHON" -m pip check || echo "PIP_CHECK_NONZERO_DIAGNOSTIC_ONLY: $PYTHON"
done

echo DIAGNOSTIC_E10_PREFLIGHT_OK
