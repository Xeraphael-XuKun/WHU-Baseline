#!/usr/bin/env bash
# The twin-InfoNCE feasibility probe, as one GPU task.
#
#   bash diag/run_twin_probe.sh                 # on the whu_recipe_hihr baseline
#   bash diag/run_twin_probe.sh whu_pe_indep    # on some other run
#
# No training happens here.  What costs time is the shift search -- 400 Adam
# steps over every sampled batch, twice over -- which is minutes on a CPU and
# seconds on the A800, and that is the only reason this is a GPU task.
#
# Features are dumped first if there is no usable npz on the shared drive, so
# this works whether or not feats_<mode>.npz survived the last cleanup.  That
# dump is the expensive half (one pass over ~100k images, about 3 minutes);
# a usable npz is reused untouched.
set -euo pipefail

MODE=${1:-whu_recipe_hihr}

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
CFG=$REPO/configs/hihr_$MODE.yml
NPZ=$REPO/feats_$MODE.npz

# The checkpoint name follows SOLVER.MAX_EPOCHS: the 120-epoch runs write
# transformer_120.pth, and a hardcoded 60 would send this looking for a file
# that was never written.
EPOCHS=$(grep -aE '^ *MAX_EPOCHS:' "$CFG" | head -1 | tr -dc '0-9')
CKPT=$REPO/out_hihr_$MODE/transformer_${EPOCHS:-60}.pth

[ -f "$CFG" ] || { echo "missing config: $CFG" >&2; exit 1; }

cd "$REPO"
LOG=$REPO/twin_probe_$MODE.txt
{
  echo "=== $(date '+%F %T')  twin InfoNCE probe on $MODE"
  echo "=== config : $CFG"
  echo "=== npz    : $NPZ"
} > "$LOG"

# "Usable" is stronger than "exists": the npz files dumped before 2026-08-07
# carry no frame index, and without it there are no twins to probe.  Checking
# here rather than letting the probe exit means the re-dump happens inside the
# same task instead of costing a second submission.
usable=0
if [ -f "$NPZ" ]; then
  if "$PY" -c "
import numpy as np, sys
d = np.load(sys.argv[1])
sys.exit(0 if 'g_frames' in d and (d['g_frames'] >= 0).any() else 1)" "$NPZ" 2>/dev/null; then
    usable=1
  else
    echo "$NPZ has no frame index -- re-dumping" | tee -a "$LOG"
  fi
fi

if [ "$usable" = 1 ]; then
  echo "reusing $NPZ" | tee -a "$LOG"
else
  [ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2
                      echo "train $MODE first, or pass another run name." >&2; exit 1; }

  # Same content-based branch as dump_all.sh.  PRETRAIN_PATH still has to be
  # loadable even though --weight overwrites it, because make_model reads it
  # while constructing and the WHU configs' 'self' path rejects the raw CLIP
  # archive.
  EXTRA=()
  if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$CFG"; then
    SELF=$(grep -aE "^ *PRETRAIN_PATH:" "$CFG" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
    EXTRA+=(MODEL.PRETRAIN_PATH "$REPO/${SELF#./}")
  else
    EXTRA+=(MODEL.PRETRAIN_PATH "$DATA/ViT-B-16.pt")
  fi
  if grep -q '^ *TEXT_ALIGN: *True' "$CFG"; then
    EXTRA+=(MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt")
  fi

  # /tmp first, then copy: /tmp is wiped when the allocation ends, and a
  # half-written npz on the shared drive is worse than no npz at all.
  TMP=/tmp/feats_$MODE.npz
  echo "dumping features from $CKPT" | tee -a "$LOG"
  "$PY" -u diag/dump_features.py \
    --config_file "$CFG" --weight "$CKPT" --out "$TMP" \
    DATASETS.ROOT_DIR "$DATA" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
  cp "$TMP" "$NPZ"
  echo "copied -> $NPZ" | tee -a "$LOG"
fi

"$PY" -u diag/twin_infonce_probe.py "$NPZ" --device cuda 2>&1 | tee -a "$LOG"

echo
echo "log: $LOG"
