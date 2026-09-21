"""Grad-CAM for the positional correction: where the model looks, with the
layer-wise increments switched on and switched off.

    python diag/gradcam.py --config_file configs/hihr_whu_pemod_mtext_clip.yml \
        --weight out_hihr_whu_pemod_mtext_clip/transformer_60.pth \
        --pid 0298 --aerial c6:f005267 --ground c2:f005465 \
        --out fig_gradcam.json  [KEY VALUE]...

Why this shape and not a generic Grad-CAM.  A saliency map of one image says
little on its own -- any trained ReID model attends to the torso.  What this
paper needs to show is the DIFFERENCE the positional residual makes, so every
image is scored twice: once with `delta_gate = 1` (the deployed model) and once
with `delta_gate = 0`, which zeroes the twelve increments and leaves the forward
pass bit-identical to the original CLIP tower's positional behaviour.  Same
weights, same image, same target; the only thing that changed is the 3.57M
residual parameters.  That gating path is not invented here -- it is the same
one TGVA uses for its "before" pass during training (`make_model.forward`).

Why the target is a retrieval similarity and not a classifier logit.  The
classifier only spans training identities, and its logit is not what retrieval
optimises.  We instead differentiate the cosine similarity between the query
feature and a fixed reference feature of the SAME identity from the other view,
which is exactly the quantity ranking is based on.  The reference is computed
once with the increments on and detached, so it is identical across the two
passes and cannot itself explain any difference between them.

Output is the CAMs only -- 16x8 floats per image per gate, a few kilobytes of
JSON on the shared drive.  The figure is drawn from it afterwards, so no
plotting library is needed on the worker.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as FN
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg                              # noqa: E402
from model import make_model                        # noqa: E402
from datasets import make_dataloader                # noqa: E402

MODS = [('RGB', 'RGB', 'm1'), ('NIR', 'IR', 'm2'), ('TIR', 'Thermal', 'm3')]


def load(root, split, pid, cam, frame, tf):
    """WHU-MARS stores one directory per spectrum and shares the frame index,
    so the three files of one instant differ only in the directory and the
    `_m{k}_` field."""
    out, paths = [], []
    for _tag, d, m in MODS:
        p = os.path.join(root, split, d, '%s_%s_%s_%s.jpg' % (pid, cam, m, frame))
        if not os.path.exists(p):
            raise SystemExit('missing: {}'.format(p))
        out.append(tf(Image.open(p).convert('RGB')))
        paths.append(p)
    return torch.stack(out), paths


class Tap(object):
    """Captures the tokens ENTERING the tapped block, and their gradients.

    Not the tokens leaving the last block.  `forward_features` ends with
    ``x = self.norm(x); return x[:, 0]``: only the class token reaches the
    output, and LayerNorm is per-token, so the patch tokens of the FINAL block
    have an analytically zero gradient.  Tapping there yields an all-zero map
    for every image -- which is what the first version of this script produced.
    The tokens on the way IN to that block still have to pass through its
    attention before they reach the class token, so their gradient is the
    quantity Grad-CAM actually wants.
    """

    def __init__(self, block):
        self.a = self.g = None
        self.h = block.register_forward_pre_hook(self._pre)

    def _pre(self, _m, inp):
        x = inp[0]
        self.a = x
        x.register_hook(self._bwd)

    def _bwd(self, grad):
        self.g = grad

    def close(self):
        self.h.remove()

    def cam(self, ny, nx):
        """Channel weights are the token-mean gradients; the class token is
        dropped because it has no location on the image."""
        a, g = self.a[:, 1:], self.g[:, 1:]                 # [B, N, D]
        w = g.mean(dim=1, keepdim=True)                     # [B, 1, D]
        raw = (a * w).sum(dim=-1)                           # [B, N], pre-ReLU
        c = FN.relu(raw).reshape(raw.shape[0], ny, nx)
        hi = c.amax(dim=(1, 2), keepdim=True)
        # A degenerate map is a bug, not a result: say so rather than writing
        # zeros that look like data.
        for i, m in enumerate(hi.reshape(-1).tolist()):
            if m <= 0:
                raise RuntimeError(
                    'row {}: every activation-gradient product is <= 0, so the '
                    'CAM is empty (pre-ReLU range {:.3e} .. {:.3e}). The tapped '
                    'layer is probably downstream of everything that carries '
                    'spatial information.'.format(
                        i, float(raw[i].min()), float(raw[i].max())))
        lo = c.amin(dim=(1, 2), keepdim=True)
        stats = (float(raw.min()), float(raw.max()))
        return ((c - lo) / (hi - lo + 1e-12)).detach().cpu().numpy(), stats


def features(model, x, modal, gate):
    """One forward through the tower with the increments scaled by `gate`."""
    return model.base(x, modal_label=modal,
                      delta_gate=None if gate is None
                      else torch.full((x.shape[0],), float(gate), device=x.device))


def main():
    ap = argparse.ArgumentParser(description='Grad-CAM with and without the residual')
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--weight', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pid', required=True)
    ap.add_argument('--aerial', required=True, help='cam:frame, e.g. c6:f005267')
    ap.add_argument('--ground', required=True, help='cam:frame, e.g. c2:f005465')
    ap.add_argument('--split', default='train',
                    help='the directory the crops come from; train keeps the '
                         'identity inside the model that was trained on it')
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--layer', type=int, default=-1,
                    help='which block to tap; the map is built from the tokens '
                         'ENTERING it, so -1 means "just before the last block"')
    ap.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda but no CUDA is visible; pass --device cpu')
    dev = args.device

    h, w = cfg.INPUT.SIZE_TEST
    ny = h // cfg.MODEL.STRIDE_SIZE[0]
    nx = w // cfg.MODEL.STRIDE_SIZE[1]
    tf = T.Compose([T.Resize((h, w), interpolation=3), T.ToTensor(),
                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    print('input {}x{} -> {} x {} patches; tapping the input of block {}'
          .format(h, w, ny, nx, args.layer), flush=True)

    # make_dataloader is called only for the class count the checkpoint expects;
    # the crops below are read straight from disk so the figure can name them.
    _tr, _trn, _vl, _nq, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num,
                       view_num=view_num)
    model.load_param(args.weight)
    model.to(dev).eval()

    root = cfg.DATASETS.ROOT_DIR
    sub = getattr(cfg.DATASETS, 'SUBDIR', '') or 'WHU-MARS'
    root = os.path.join(root, sub)
    acam, aframe = args.aerial.split(':')
    gcam, gframe = args.ground.split(':')
    xa, pa = load(root, args.split, args.pid, acam, aframe, tf)
    xg, pg = load(root, args.split, args.pid, gcam, gframe, tf)
    xa, xg = xa.to(dev), xg.to(dev)
    modal = (torch.arange(3, device=dev) if cfg.MODEL.PE_PER_MODALITY else None)

    # the reference each CAM is differentiated against: the other view of the
    # same identity, with the increments ON, detached
    with torch.no_grad():
        ref_g = FN.normalize(features(model, xg, modal, None), dim=1)
        ref_a = FN.normalize(features(model, xa, modal, None), dim=1)

    out = {'pid': args.pid, 'ny': ny, 'nx': nx,
           'aerial': {'cam': acam, 'frame': aframe, 'files': pa},
           'ground': {'cam': gcam, 'frame': gframe, 'files': pg},
           'cams': {}, 'similarity': {}}

    for view, x, ref in (('aerial', xa, ref_g), ('ground', xg, ref_a)):
        for gname, gate in (('after', None), ('before', 0.0)):
            tap = Tap(model.base.blocks[args.layer])
            model.zero_grad(set_to_none=True)
            f = FN.normalize(features(model, x, modal, gate), dim=1)
            sim = (f * ref).sum(dim=1)                      # [3], one per spectrum
            sim.sum().backward()
            cam, stats = tap.cam(ny, nx)
            tap.close()
            for i, (tag, _d, _m) in enumerate(MODS):
                out['cams'].setdefault('%s_%s' % (view, tag), {})[gname] = \
                    [round(float(v), 5) for v in cam[i].reshape(-1)]
                out['similarity'].setdefault('%s_%s' % (view, tag), {})[gname] = \
                    round(float(sim[i]), 5)
            print('  {:<7} gate={:<7} sim = {}  pre-ReLU {:.2e}..{:.2e}'.format(
                view, gname, ['%.4f' % v for v in sim.tolist()], *stats), flush=True)

    p = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'w') as fh:
        json.dump(out, fh)
    print('\nwrote {}'.format(p), flush=True)


if __name__ == '__main__':
    main()
