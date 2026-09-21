"""Magnitude of the learned positional residual, per spectrum and per block.

    python diag/pe_norms.py --weight out_hihr_whu_pemod_mtext_clip/transformer_60.pth \
        --out pe_norms_pemod_mtext.json

Reads a checkpoint's `pos_delta` tensor and nothing else: no config, no data,
no model construction, no GPU.  It is the one diagnostic in this directory that
costs nothing, which is why it exists -- the figure it feeds answers "did the
three spectra actually learn different corrections, and where in the network"
without a second forward pass.

Two quantities come out of the same tensor, and they are not the same figure:

  * `state` -- ||delta[m, l]||, the total correction the position table carries
    at block l.  Under `indep` this is what block l sees added to P.
  * `inject` -- ||delta[m, l] - delta[m, l-1]||, the increment actually written
    into the stream before block l.  This is the thing the method calls a
    layer-wise residual, and a flat `state` curve with a large `inject` curve
    would mean the layers are undoing each other.

Both are reported per spectrum.  Norms are RMS per element rather than the raw
Frobenius norm, so the numbers are comparable against the pretrained table's own
scale, which is printed alongside when the checkpoint carries it.
"""
import argparse
import json
import os

import torch


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser(description='Positional-residual magnitudes')
    ap.add_argument('--weight', required=True)
    ap.add_argument('--out', required=True, help='.json path on the shared drive')
    ap.add_argument('--label', default=None)
    args = ap.parse_args()

    sd = torch.load(args.weight, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']

    keys = [k for k in sd if k.endswith('pos_delta')]
    if not keys:
        raise SystemExit('no pos_delta in {} -- keys like: {}'
                         .format(args.weight, sorted(sd)[:5]))
    d = sd[keys[0]].float()
    print('{} {}'.format(keys[0], tuple(d.shape)))

    # [M, L, 1, N+1, D] when per-spectrum, [L, 1, N+1, D] when shared.  The
    # shared case is reported as a single unnamed row rather than special-cased
    # downstream, so the same figure code draws both.
    if d.dim() == 5:
        spectra = ['RGB', 'NIR', 'TIR'][:d.shape[0]]
    else:
        d = d.unsqueeze(0)
        spectra = ['shared']
    depth = d.shape[1]

    state, inject = [], []
    for m in range(d.shape[0]):
        state.append([rms(d[m, l]) for l in range(depth)])
        inject.append([rms(d[m, l] - (d[m, l - 1] if l else 0)) for l in range(depth)])

    base = None
    for k in sd:
        if k.endswith('pos_embed') or k.endswith('positional_embedding'):
            base = rms(sd[k].float())
            break

    payload = {
        'label': args.label or os.path.basename(os.path.dirname(args.weight)),
        'weight': args.weight, 'key': keys[0], 'shape': list(d.shape),
        'spectra': spectra, 'depth': depth,
        'state': state, 'inject': inject, 'pos_embed_rms': base,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(payload, fh)

    hdr = '        ' + ''.join('  L{:<5d}'.format(l + 1) for l in range(depth))
    for name, tbl in (('state ||d[l]||', state), ('inject ||d[l]-d[l-1]||', inject)):
        print('\n{}   (RMS per element{})'.format(
            name, '; pretrained table {:.4f}'.format(base) if base else ''))
        print(hdr)
        for s, row in zip(spectra, tbl):
            print('{:<8}'.format(s) + ''.join('{:8.4f}'.format(v) for v in row))
    print('\nwrote {}'.format(out))


if __name__ == '__main__':
    main()
