#!/usr/bin/env bash
# One task: produce every number the paper's figures need.  GPU or CPU.
#
#   bash diag/fig_data.sh            # on a GPU worker
#   bash diag/fig_data.sh cpu        # CPU-only task: no card needed
#
# Three outputs, all small, all written straight to the shared drive so they
# survive the task (a successful task keeps no log and /tmp is wiped):
#
#   fig_cos_baseline.json   intra/inter cosine histogram, native CLIP baseline
#   fig_cos_mvpr.json       the same for the full method
#   fig_pe_norms.json       ||delta[m, l]|| per spectrum and block
#
# The two cosine passes are the expensive part -- one feature extraction over
# 6,405 query and 93,609 gallery images each -- and they are what the figure
# comparing "baseline vs ours" is made of.  pe_norms costs nothing and only
# reads the checkpoint, but it rides along rather than earning its own
# submission.
#
# On a GPU that extraction is about four minutes.  On CPU it is over an hour
# per checkpoint, which is why the `cpu` mode keeps a fixed-seed 20% sample of
# the GALLERY: the output is a histogram over ~10^8 pairs either way, and a
# fifth of them still pins every reported statistic well past the digits the
# figure shows.  The query side is never sampled.  Nothing here feeds a
# retrieval metric, so dropping gallery images costs nothing that is reported;
# `bash diag/fig_data.sh cpu full` sets the fraction back to 1.0 if the time is
# available.
set -euo pipefail

MODE=${1:-gpu}
FULL=${2:-}
case "$MODE" in
  gpu) DEV=(--device cuda) ;;
  cpu) DEV=(--device cpu --threads "$(nproc)"
            --gallery_frac "$([ "$FULL" = full ] && echo 1.0 || echo 0.2)") ;;
  *)   echo "usage: bash diag/fig_data.sh [gpu|cpu] [full]" >&2; exit 1 ;;
esac

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
LOG=$REPO/fig_data_log.txt

cd "$REPO"
exec > >(tee -a "$LOG") 2>&1          # opened FIRST: a pre-flight failure that
                                      # prints nothing is how the WHU-MARS
                                      # launches were lost once already.
echo
echo "=== $(date '+%F %T')  fig_data  mode=$MODE  ${DEV[*]}"

BASE_CFG=configs/hihr_whu_recipe_clip.yml
BASE_W=out_hihr_whu_recipe_clip/transformer_60.pth
MVPR_CFG=configs/hihr_whu_pemod_mtext_clip.yml
MVPR_W=out_hihr_whu_pemod_mtext_clip/transformer_60.pth

for f in "$BASE_CFG" "$BASE_W" "$MVPR_CFG" "$MVPR_W"; do
  [ -e "$f" ] || { echo "MISSING: $f"; exit 1; }
done

# Same content-based checks reeval.sh uses: a name glob would get these wrong,
# and MODEL.PRETRAIN_PATH has to be loadable even though the weight overwrites
# it a moment later, because make_model reads it during construction.
EXTRA=()
extra_for() {                          # fills the global EXTRA for one config
  local cfg=$1
  EXTRA=(DATASETS.ROOT_DIR "$DATA")
  if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$cfg"; then
    local self
    self=$(grep -aE "^ *PRETRAIN_PATH:" "$cfg" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
    EXTRA+=(MODEL.PRETRAIN_PATH "$REPO/${self#./}")
  else
    EXTRA+=(MODEL.PRETRAIN_PATH "$DATA/ViT-B-16.pt")
  fi
  if grep -q '^ *TEXT_ALIGN: *True' "$cfg"; then
    EXTRA+=(MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt")
  fi
}

echo
echo "--- 1/3  cosine histogram, native CLIP baseline"
extra_for "$BASE_CFG"
"$PY" -u diag/cos_hist.py --config_file "$BASE_CFG" --weight "$BASE_W" \
  --label '原生 CLIP 基线' "${DEV[@]}" --out "$REPO/fig_cos_baseline.json" "${EXTRA[@]}"

echo
echo "--- 2/3  cosine histogram, MVPR"
extra_for "$MVPR_CFG"
"$PY" -u diag/cos_hist.py --config_file "$MVPR_CFG" --weight "$MVPR_W" \
  --label 'MVPR（本文）' "${DEV[@]}" --out "$REPO/fig_cos_mvpr.json" "${EXTRA[@]}"

echo
echo "--- 3/3  positional-residual magnitudes"
"$PY" -u diag/pe_norms.py --weight "$MVPR_W" --label 'MVPR' \
  --out "$REPO/fig_pe_norms.json"

echo
echo "=== done $(date '+%F %T').  On the dev machine:"
ls -la "$REPO"/fig_cos_baseline.json "$REPO"/fig_cos_mvpr.json "$REPO"/fig_pe_norms.json
