"""Cosine-similarity distributions for intra- and inter-identity pairs.

    python diag/cos_hist.py --config_file configs/hihr_whu_recipe_clip.yml \
        --weight out_hihr_whu_recipe_clip/transformer_60.pth \
        --out cos_whu_recipe_clip.json  [KEY VALUE]...

This is the figure UAD reports as its Figure 4, and the plot in
`ZYK100/LLCM/Visualization/intra_inter-distance.py` is the implementation both
that paper and this one are following.  Two histograms over the same axis --
same-identity pairs and different-identity pairs -- and the gap between their
means is the number the figure exists to show.

Four deliberate differences from the LLCM script:

  * It plots 1 - cos and calls it distance; UAD's figure plots cos.  We follow
    UAD, because that is the figure being replicated, and because "similarity
    of same-identity pairs should be HIGHER" reads the right way round.

  * LLCM materialises the full query x gallery matrix.  Here that is
    6,405 x 93,609 = 600 million pairs, which is 2.4 GB in float32 before the
    two boolean masks.  We accumulate a histogram in chunks instead: the output
    is 400 bin counts per class, a few kilobytes, and it goes to the shared
    drive rather than dying with the task.

  * The junk rule is applied.  WHU-MARS scores under the SYSU rule -- the whole
    query camera is dropped from the gallery -- so a distribution computed over
    ALL pairs would be dominated by same-camera same-identity pairs that the
    evaluator never ranks.  Those pairs are trivially similar and would inflate
    the intra-identity mode of every arm equally, which is exactly the kind of
    free shift that makes a figure look better than the method is.
    `--junk none` reproduces the LLCM/UAD convention for comparison.

  * Both the raw cosine and the mean-centred cosine are reported.  CLIP's
    space is strongly anisotropic -- unrelated images already sit at 0.97 --
    so the raw histograms pile up against 1 and, worse, concentrate to
    DIFFERENT degrees in different arms, which makes the raw gap
    incomparable across models.  See `accumulate`.

Written as one number-producing pass with no plotting: matplotlib is not
installed on the dev machine, and the figure is drawn from this JSON afterwards.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as FN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg                              # noqa: E402
from datasets import make_dataloader                # noqa: E402
from model import make_model                        # noqa: E402
from diag.dump_features import extract              # noqa: E402

NBINS = 400
LO, HI = -1.0, 1.0


class Bag(object):
    """Histogram plus exact first and second moments for one class of pairs.

    The moments are kept apart from the histogram because the figure's headline
    is a difference of two means and 400 bins over [-1, 1] would quantise it to
    0.005 -- which for CLIP features is larger than the effect.  The variance is
    what makes two arms comparable at all: see `accumulate` below.
    """

    def __init__(self):
        self.h = np.zeros(NBINS, dtype=np.int64)
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add(self, v):                      # v: 1-D float32 torch tensor on GPU
        if v.numel() == 0:                 # a query whose whole camera is junk
            return
        self.h += torch.histc(v, bins=NBINS, min=LO, max=HI).cpu().numpy().astype(np.int64)
        self.n += v.numel()
        self.s += float(v.sum())
        self.s2 += float((v.double() ** 2).sum())

    def stats(self):
        m = self.s / max(self.n, 1)
        var = max(self.s2 / max(self.n, 1) - m * m, 0.0)
        return m, var ** 0.5

    def as_dict(self):
        m, sd = self.stats()
        return {'counts': self.h.tolist(), 'mean': m, 'std': sd, 'pairs': int(self.n)}


def accumulate(qf, qpid, qcam, gf, gpid, gcam, junk='sysu', chunk=256, device='cuda'):
    """Cosine histograms split by identity match, raw and mean-centred.

    Both variants come out of one pass because the expensive part is the
    feature extraction that happened before this function was called; the
    second histogram costs one extra matmul on data already resident.

    Why the centred variant exists.  CLIP's feature space is strongly
    anisotropic: every embedding carries a large common component, so two
    unrelated images already sit at cosine 0.97 and BOTH distributions pile up
    against 1.  That compression is not equal across arms -- the two models
    concentrate to different degrees -- so a raw gap of 0.009 against a raw gap
    of 0.004 is a comparison between two differently scaled axes and says
    nothing about which space separates identities better.  Subtracting the
    mean direction (computed over query and gallery together, then
    renormalising) removes the shared component that carries no identity, and
    is the standard correction for this property.  Both sets of numbers are
    written out; the figure states which it draws.
    """
    qt = torch.from_numpy(qf).to(device)
    gt = torch.from_numpy(gf).to(device)
    mu = torch.cat([qt, gt], 0).mean(0, keepdim=True)
    qc_ = FN.normalize(qt - mu, dim=1)
    gc_ = FN.normalize(gt - mu, dim=1)
    print('  anisotropy: ||mean direction|| = {:.4f}'.format(float(mu.norm())), flush=True)

    qpid_t = torch.from_numpy(qpid).to(device)
    gpid_t = torch.from_numpy(gpid).to(device)
    qcam_t = torch.from_numpy(qcam).to(device)
    gcam_t = torch.from_numpy(gcam).to(device)

    bags = {'raw_intra': Bag(), 'raw_inter': Bag(),
            'cen_intra': Bag(), 'cen_inter': Bag()}

    for a in range(0, len(qf), chunk):
        b = min(a + chunk, len(qf))
        same = qpid_t[a:b, None] == gpid_t[None, :]
        if junk == 'sysu':
            keep = qcam_t[a:b, None] != gcam_t[None, :]
        elif junk == 'market':
            keep = ~(same & (qcam_t[a:b, None] == gcam_t[None, :]))
        elif junk == 'none':
            keep = torch.ones_like(same)
        else:
            raise ValueError('unknown junk rule: {}'.format(junk))
        pos_m, neg_m = same & keep, (~same) & keep

        for tag, Q, G in (('raw', qt, gt), ('cen', qc_, gc_)):
            sim = Q[a:b] @ G.t()
            bags[tag + '_intra'].add(sim[pos_m])
            bags[tag + '_inter'].add(sim[neg_m])

        if (a // chunk) % 5 == 0:
            print('  {}/{} queries'.format(b, len(qf)), flush=True)

    return bags


def main():
    ap = argparse.ArgumentParser(description='Intra/inter cosine histograms')
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--out', required=True, help='.json path on the shared drive')
    ap.add_argument('--label', default=None, help='name shown on the figure')
    ap.add_argument('--junk', default='sysu', choices=('sysu', 'market', 'none'))
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--threads', type=int, default=0,
                    help='torch CPU threads; 0 leaves the default')
    ap.add_argument('--gallery_frac', type=float, default=1.0,
                    help='keep this fraction of gallery images (CPU runs only; '
                         'the histogram is a shape estimate over ~10^8 pairs, '
                         'so 0.2 changes the third decimal at most)')
    ap.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda but no CUDA is visible; pass --device cpu')
    print('config : {}'.format(args.config_file))
    print('weight : {}'.format(args.weight))
    print('junk   : {}   device: {}   threads: {}   gallery_frac: {}'
          .format(args.junk, args.device, torch.get_num_threads(), args.gallery_frac),
          flush=True)

    _tr, _trn, val_loaders, num_querys, num_classes, camera_num, view_num = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)
    model.load_param(args.weight)
    model.to(args.device)

    q, g = extract(model, val_loaders, num_querys, device=args.device,
                   gallery_frac=args.gallery_frac)

    # The evaluator retrieves with L2-normalised features whenever FEAT_NORM is
    # on, and cosine similarity is only cosine similarity once they are, so this
    # is not a choice the figure gets to make.
    qf = q['feat'] / np.linalg.norm(q['feat'], axis=1, keepdims=True)
    gf = g['feat'] / np.linalg.norm(g['feat'], axis=1, keepdims=True)
    print('query {}, gallery {}'.format(qf.shape, gf.shape), flush=True)

    bags = accumulate(qf.astype(np.float32), q['pid'].astype(np.int64), q['camid'].astype(np.int64),
                      gf.astype(np.float32), g['pid'].astype(np.int64), g['camid'].astype(np.int64),
                      junk=args.junk, device=args.device)

    payload = {
        'label': args.label or os.path.basename(args.weight),
        'config': args.config_file, 'weight': args.weight, 'junk': args.junk,
        'nbins': NBINS, 'lo': LO, 'hi': HI,
        'gallery_frac': args.gallery_frac, 'device': args.device,
    }
    for tag in ('raw', 'cen'):
        mi, si = bags[tag + '_intra'].stats()
        mo, so = bags[tag + '_inter'].stats()
        payload[tag] = {
            'intra': bags[tag + '_intra'].as_dict(),
            'inter': bags[tag + '_inter'].as_dict(),
            'delta': mi - mo,
            # Scale-free separation: the gap in units of the inter-identity
            # spread.  This is the only one of the three numbers that can be
            # compared between two models whose feature spaces concentrate to
            # different degrees, which is exactly our situation.
            'dprime': (mi - mo) / so if so > 0 else float('nan'),
        }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(payload, fh)

    for tag, name in (('raw', 'raw cosine'), ('cen', 'mean-centred cosine')):
        d = payload[tag]
        print('\n== {} =='.format(name))
        print('  intra : {:>14,} pairs  mean {:+.4f}  sd {:.4f}'
              .format(d['intra']['pairs'], d['intra']['mean'], d['intra']['std']))
        print('  inter : {:>14,} pairs  mean {:+.4f}  sd {:.4f}'
              .format(d['inter']['pairs'], d['inter']['mean'], d['inter']['std']))
        print('  delta : {:+.4f}      d-prime : {:+.3f}'.format(d['delta'], d['dprime']))
    print('\nwrote {}'.format(out), flush=True)


if __name__ == '__main__':
    main()
