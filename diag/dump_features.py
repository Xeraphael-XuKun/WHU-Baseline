"""Dump the retrieval features of one checkpoint to an .npz.  One GPU task.

    python diag/dump_features.py --config_file configs/hihr_whu_pe_indep.yml \
        --weight out_hihr_whu_pe_indep/transformer_60.pth \
        --out feats_whu_pe_indep.npz  [KEY VALUE]...

Why this exists: every question we now have about the thermal modality -- is
the gap a translation or a different manifold, is identity still recoverable
across it, how much mAP is it actually costing -- is answered by arithmetic on
the features.  None of it needs a GPU twice.  So the GPU runs once, writes the
features to the shared drive, and every analysis after that is free and can be
redone on the dev machine as often as we like.

The extraction loop below is copied from `do_inference`, not re-derived: the
diagnostic is worthless if it measures a feature the evaluator never saw.  The
one deliberate difference is that features are saved *before* L2
normalisation, because `feat_norm` is a switch the analysis wants to control.
`TEST.FEAT_NORM` is stored alongside so it can be reproduced exactly.

Size: query 6,405 x 768 and gallery 93,609 x 768 in float32 is about 290 MB per
checkpoint.  float16 would halve that, but distances between near-duplicate
768-d vectors are exactly where float16 starts rounding, and this file exists
to measure such distances.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg                              # noqa: E402
from datasets import make_dataloader                # noqa: E402
from model import make_model                        # noqa: E402
from utils.metrics import as_bool                   # noqa: E402

# WHU-MARS names files `{pid}_c{cam}_m{modality}_f{frame}.jpg`, and the frame is
# shared by the sensors that recorded the same instant.  Without it a distance
# between two same-identity images cannot be split into "differs in spectrum"
# and "differs in pose as well", which is the one decomposition that separates
# the two costs.  CARGO has no such field; those rows get -1 and the analysis
# reports the section as unavailable rather than inventing pairings.
FRAME_RE = re.compile(r'_f(\d+)')


def parse_frames(names):
    out = []
    for n in names:
        hit = FRAME_RE.search(str(n))
        out.append(int(hit.group(1)) if hit else -1)
    return np.asarray(out, dtype=np.int64)


def extract(model, val_loaders, num_querys, device='cuda', gallery_frac=1.0, seed=1234):
    """Returns (query, gallery) dicts of concatenated arrays.

    Modes are the three modality-specific loaders; within each, the first
    `num_query` rows are the query side.  That split is exactly what
    `R1_mAP_eval.split_all` does, and it has to stay that way or the query and
    gallery sides silently swap for one modality.

    `gallery_frac` below 1.0 keeps a random subset of the GALLERY side and runs
    the backbone only on what it keeps -- the point is to skip the forward, not
    to throw features away afterwards.  It exists for the distribution figures,
    which are shape estimates over hundreds of millions of pairs and do not need
    every gallery image, and it matters only when there is no GPU: 93,609
    forwards on CPU is over an hour.  The mask is drawn from a fixed seed rather
    than taken as a stride, because the gallery list is ordered by identity and
    a stride would sample cameras unevenly.  Retrieval metrics must never use
    this: dropping gallery images changes ranks.  The query side is always kept
    whole.
    """
    model.eval()
    q = {k: [] for k in ('feat', 'pid', 'camid', 'mid', 'frame')}
    g = {k: [] for k in ('feat', 'pid', 'camid', 'mid', 'frame')}

    for mode, (loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        feats, pids, camids, mids, frames = [], [], [], [], []
        total = len(loader.dataset)
        if gallery_frac < 1.0:
            rng = np.random.RandomState(seed + mode)
            keep_all = np.ones(total, dtype=bool)
            gal = np.arange(num_query, total)
            drop = rng.rand(gal.size) >= gallery_frac
            keep_all[gal[drop]] = False
            print('  mode {}: keeping {} of {} gallery images ({:.0%})'
                  .format(mode, int(keep_all[num_query:].sum()), gal.size, gallery_frac),
                  flush=True)
        else:
            keep_all = None
        cursor = 0

        for img, vid, camid, camidt, modid, imgpath in loader:
            n = img.shape[0]
            sl = slice(cursor, cursor + n)
            cursor += n
            if keep_all is not None:
                m = keep_all[sl]
                if not m.any():
                    continue
                if not m.all():
                    img = img[torch.from_numpy(m)]
                    camidt = camidt[torch.from_numpy(m)]
                    vid = np.asarray(vid)[m]
                    camid = np.asarray(camid)[m]
                    modid = np.asarray(modid)[m]
                    imgpath = [p for p, k in zip(imgpath, m) if k]
            with torch.no_grad():
                # MODEL.PE_VIEW_GATE reads these; every other config ignores
                # them.  Without it a gated checkpoint cannot be dumped at all.
                feat = model(img.to(device), mode=mode, camids=camidt)
            feats.append(feat.detach().float().cpu())
            pids.extend(np.asarray(vid))
            camids.extend(np.asarray(camid))
            mids.extend(np.asarray(modid))
            frames.extend(parse_frames(imgpath))

        feats = torch.cat(feats, dim=0).numpy()
        pids = np.asarray(pids, dtype=np.int64)
        camids = np.asarray(camids, dtype=np.int64)
        mids = np.asarray(mids, dtype=np.int64)
        frames = np.asarray(frames, dtype=np.int64)
        unparsed = int((frames < 0).sum())
        if unparsed:
            print('  mode {}: {} of {} filenames carry no _f<frame> field'
                  .format(mode, unparsed, len(frames)), flush=True)
        assert len(feats) == len(pids) == len(camids) == len(mids) == len(frames)
        assert num_query < len(feats), (mode, num_query, len(feats))

        print('  mode {}: {} images, {} query / {} gallery, modality ids {}'
              .format(mode, len(feats), num_query, len(feats) - num_query,
                      sorted(set(mids.tolist()))), flush=True)

        for dst, sl in ((q, slice(None, num_query)), (g, slice(num_query, None))):
            dst['feat'].append(feats[sl])
            dst['pid'].append(pids[sl])
            dst['camid'].append(camids[sl])
            dst['mid'].append(mids[sl])
            dst['frame'].append(frames[sl])

    return ({k: np.concatenate(v, axis=0) for k, v in q.items()},
            {k: np.concatenate(v, axis=0) for k, v in g.items()})


def main():
    parser = argparse.ArgumentParser(description='Dump retrieval features')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--weight', required=True,
                        help='checkpoint to score; overrides TEST.WEIGHT')
    parser.add_argument('--out', required=True, help='.npz path on the shared drive')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    print('config : {}'.format(args.config_file))
    print('weight : {}'.format(args.weight))
    print('feature: {} (TEST.NECK_FEAT), FEAT_NORM={}'
          .format(cfg.TEST.NECK_FEAT, cfg.TEST.FEAT_NORM), flush=True)

    _tr, _trn, val_loaders, num_querys, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)
    model.load_param(args.weight)
    model.to('cuda')

    q, g = extract(model, val_loaders, num_querys)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez(
        out,
        qf=q['feat'], q_pids=q['pid'], q_camids=q['camid'], q_mids=q['mid'],
        gf=g['feat'], g_pids=g['pid'], g_camids=g['camid'], g_mids=g['mid'],
        q_frames=q['frame'], g_frames=g['frame'],
        # Everything the analysis needs to reproduce the reported numbers
        # exactly.  Storing them removes the chance of analysing under a
        # different convention than the one the mAP was computed with.
        # as_bool, not bool(): bool('no') is True, and this field is what the
        # analysis reads to decide whether to normalise.  It had been recording
        # True for every run regardless of the setting.
        feat_norm=np.array(as_bool(cfg.TEST.FEAT_NORM, 'TEST.FEAT_NORM')),
        metric=np.array(str(cfg.TEST.METRIC)),
        neck_feat=np.array(str(cfg.TEST.NECK_FEAT)),
        aerial_cams=np.asarray(list(cfg.DATASETS.AERIAL_CAMS), dtype=np.int64),
        modalities=np.asarray([str(m) for m in cfg.DATASETS.MODALITIES]),
        weight=np.array(str(args.weight)),
    )
    print('saved {}  ({:.0f} MB)  query {} / gallery {}'
          .format(out, os.path.getsize(out) / 1e6, len(q['pid']), len(g['pid'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
