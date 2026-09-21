#!/usr/bin/env bash
# One task -- CPU is enough, six images -- producing the Grad-CAM arrays for the
# qualitative figure.
#
#   bash diag/run_gradcam.sh            # CPU
#   bash diag/run_gradcam.sh gpu        # if a card happens to be free
#
# Identity 0298 is the one Figure 1 already shows, so the qualitative figure and
# the teaser describe the same person; both frames are tri-spectral and come
# from the training split, which is where the model actually saw this identity.
set -euo pipefail

DEV=${1:-cpu}
PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
LOG=$REPO/gradcam_log.txt

cd "$REPO"
exec > >(tee -a "$LOG") 2>&1
echo
echo "=== $(date '+%F %T')  gradcam  device=$DEV"

CFG=configs/hihr_whu_pemod_mtext_clip.yml
W=out_hihr_whu_pemod_mtext_clip/transformer_60.pth
for f in "$CFG" "$W"; do [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }; done

"$PY" -u diag/gradcam.py --config_file "$CFG" --weight "$W" \
  --pid 0298 --aerial c6:f005267 --ground c2:f005465 \
  --device "$DEV" --out "$REPO/fig_gradcam.json" \
  DATASETS.ROOT_DIR "$DATA" \
  MODEL.PRETRAIN_PATH "$DATA/ViT-B-16.pt" \
  MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt"

echo
echo "=== done $(date '+%F %T')"
ls -la "$REPO/fig_gradcam.json"
