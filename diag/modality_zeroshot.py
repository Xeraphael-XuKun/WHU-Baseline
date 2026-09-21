"""Stage 0.3 -- can a CLIP text anchor tell our three spectra apart at all?

    python diag/modality_zeroshot.py --config_file configs/hihr_whu_recipe_hihr.yml \
        --clip   /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
        --weight out_hihr_whu_recipe_hihr/transformer_60.pth

The document's 0.3 asks for a zero-shot RETRIEVAL R-1 and checks whether it
moves when the template changes.  That question cannot be asked of this
codebase: retrieval here is image-to-image (TEST.NECK_FEAT 'before', the 768-d
pre-BNNeck feature), the text tower is a training-time loss and is not even
constructed at evaluation, so changing the template moves R-1 by exactly zero
and always would.  The underlying worry is real though -- "is the text branch
connected to anything?" -- so it is asked in the form this pipeline can answer:

    take an image, push it through the visual tower and CLIP's own visual.proj
    into the 512-d joint space, score it against the three spectrum sentences,
    and see whether the argmax is its true spectrum.

That is exactly the quantity `processor.modality_align` computes on
`cos_before` -- same feature, same projection, same anchors -- so a number near
chance here means the "before" cells of that loss are supervising noise.

TWO ARMS, and the difference between them is the result:

    A  original CLIP        MODEL.PRETRAIN_CHOICE 'imagenet' + ViT-B-16.pt
    B  our baseline         MODEL.PRETRAIN_CHOICE 'self'     + transformer_60.pth

Both are built from the SAME config and therefore the same architecture, the
same 256x128 input, the same resized positional embedding and the same eval
transform.  Only the weights differ.  Feeding arm A through CLIP's official
224x224 preprocessing instead would have made the comparison confound the
weights with the input pipeline.

    B > A   fine-tuning sharpened the spectrum signal -> the anchors have
            something to hold on to, and prompt wording is worth tuning
    B < A   fine-tuning washed it out -> CE+triplet are already removing
            modality, and a modality prompt is pushing against them
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag.prompt_probe import GROUPS, encode                   # noqa: E402

# Which sentence groups to score images against.  Every group with one sentence
# per spectrum qualifies; the view group has no spectrum axis and is skipped.
SPECTRUM_KEYS = ('RGB', 'NIR', 'TIR')


def spectrum_groups():
    return [(label, s) for label, s in GROUPS
            if tuple(s) == SPECTRUM_KEYS]


def collect(model, val_loaders, per_modality, device='cuda'):
    """[N, 768] features and [N] true spectrum indices, balanced per spectrum.

    Sampled from the head of each modality's loader rather than at random: the
    two arms must see the SAME images, and the loaders are deterministic in
    eval mode, so taking the first `per_modality` of each gives that for free
    without carrying an index list between two separate model builds.
    """
    model.eval()
    feats, mods = [], []
    for mode, loader in enumerate(val_loaders, start=1):
        got = 0
        for img, _vid, _camid, camidt, _modid, _path in loader:
            take = min(per_modality - got, img.shape[0])
            with torch.no_grad():
                # Sliced alongside the images: MODEL.PE_VIEW_GATE checks that
                # the camera ids cover exactly as many rows as the batch.
                f = model(img[:take].to(device), mode=mode,
                          camids=camidt[:take])
            feats.append(f.detach().float().cpu())
            mods.append(torch.full((take,), mode - 1, dtype=torch.long))
            got += take
            if got >= per_modality:
                break
        print('    spectrum %d: %d images' % (mode - 1, got), flush=True)
    return torch.cat(feats, dim=0), torch.cat(mods, dim=0)


def confusion(feat, mods, proj, text_vecs, n_spectrum):
    """[n_spectrum, n_spectrum] counts, row = true spectrum, col = predicted."""
    v = F.normalize(feat.float() @ proj.float().cpu(), dim=-1)
    pred = (v @ text_vecs.t()).argmax(dim=1)
    mat = np.zeros((n_spectrum, n_spectrum), dtype=np.int64)
    for t, p in zip(mods.tolist(), pred.tolist()):
        mat[t, p] += 1
    return mat


def mean_pairwise_cos(vecs):
    """Mean cosine over every distinct pair, without building [N, N].

    For unit vectors, ``sum_{i != j} <u_i, u_j> = ||sum u||^2 - N``.  A 6000 x
    6000 matrix would only be 144 MB, but the identity is exact, costs O(N*D),
    and removes any temptation to subsample the thing being measured.

    Read it as "how much is this cloud one direction": 1.0 means every vector
    points the same way, 0.0 means they are on average orthogonal.
    """
    u = F.normalize(vecs.double(), dim=-1)
    n = u.shape[0]
    if n < 2:
        return float('nan')
    return (float(u.sum(dim=0).pow(2).sum()) - n) / (n * (n - 1))


def centroid_cos(vecs, mods, n_spectrum):
    """[n_spectrum, n_spectrum] cosine between per-spectrum mean directions.

    Rows are L2-normalised BEFORE averaging, because that is what the classifier
    sees: a spectrum whose images are individually distinguishable but happen to
    have larger norms would otherwise dominate its own centroid.
    """
    u = F.normalize(vecs.double(), dim=-1)
    cents = []
    for m in range(n_spectrum):
        sel = mods == m
        cents.append(F.normalize(u[sel].mean(dim=0), dim=-1))
    c = torch.stack(cents)
    return (c @ c.t()).clamp(-1.0, 1.0).numpy()


def concentration(feat, mods, proj, n_spectrum):
    """Is the collapse in the features, or in the projection?

    The confusion matrix alone cannot say.  Everything predicted as one anchor
    is consistent with two very different worlds:

      * the 768-d features are already one direction -- then nothing about
        CLIP is involved and the fault is upstream, in what the backbone
        learned;
      * the 768-d features are spread out and `clip_proj` flattens them --
        then the features have moved into that matrix's near-null space, and
        the text loss has been talking down a dead channel all along.

    A linear map cannot make distinct directions coincide unless it annihilates
    what distinguishes them, so measuring the same two quantities on both sides
    of the projection separates the cases outright.
    """
    p = feat.double() @ proj.double()
    n = feat.double().norm(dim=-1).clamp_min(1e-12)
    return {
        'cos768': mean_pairwise_cos(feat),
        'cos512': mean_pairwise_cos(p),
        'gain': float((p.norm(dim=-1) / n).mean()),
        'centroid768': centroid_cos(feat, mods, n_spectrum),
        'centroid512': centroid_cos(p, mods, n_spectrum),
    }


def print_concentration(label, c, names):
    print('\n    [%s]  投影前后的集中度' % label)
    print('      平均两两余弦   768d %.4f   ->  512d %.4f   (1.0 = 塌成一个方向)'
          % (c['cos768'], c['cos512']))
    print('      投影增益       ||f @ proj|| / ||f||  =  %.4f' % c['gain'])
    for tag in ('768', '512'):
        mat = c['centroid' + tag]
        print('      %sd 各光谱质心之间的余弦:' % tag)
        print('        %-10s' % '' + ''.join('%10s' % n for n in names))
        for i, n in enumerate(names):
            row = '        %-10s' % n
            for j in range(mat.shape[1]):
                row += '%10s' % ('--' if i == j else '%.4f' % mat[i, j])
            print(row)


def print_confusion(label, mat, names):
    total = mat.sum()
    correct = int(np.trace(mat))
    print('\n    [%s]  accuracy %.1f%%  (%d / %d, chance %.1f%%)'
          % (label, 100.0 * correct / max(total, 1), correct, total,
             100.0 / mat.shape[0]))
    print('      %-10s' % 'true\\pred' + ''.join('%10s' % n for n in names))
    for i, n in enumerate(names):
        row = '      %-10s' % n
        for j in range(mat.shape[1]):
            row += '%10s' % ('%d (%.0f%%)' % (mat[i, j],
                                              100.0 * mat[i, j] / max(mat[i].sum(), 1)))
        print(row)


def build(cfg_path, opts, num_classes, camera_num, view_num):
    """A fresh cfg per arm.  cfg is a module-level singleton that gets frozen
    on first use, so the two arms cannot share one."""
    from config import cfg
    from model import make_model
    c = cfg.clone()
    c.merge_from_file(cfg_path)
    c.merge_from_list(opts)
    c.freeze()
    return make_model(c, num_class=num_classes, camera_num=camera_num,
                      view_num=view_num), c


def main():
    p = argparse.ArgumentParser(description='Stage 0.3 zero-shot modality probe')
    p.add_argument('--config_file', required=True)
    p.add_argument('--clip', required=True, help='the ORIGINAL CLIP release, ViT-B-16.pt')
    p.add_argument('--weight', required=True, help='our checkpoint, for arm B')
    p.add_argument('--per-modality', type=int, default=2000)
    p.add_argument('--root', default=None, help='override DATASETS.ROOT_DIR')
    args = p.parse_args()

    from config import cfg
    from datasets import make_dataloader
    from model.backbones.clip_text import CLIPTextEncoder, build_tokenizer
    from model.make_model import _read_clip_checkpoint

    base_opts = ['MODEL.TEXT_ALIGN', 'False']     # arm A has no text tower to load
    if args.root:
        base_opts += ['DATASETS.ROOT_DIR', args.root]

    # One dataloader build, shared by both arms -- and read from a throwaway cfg
    # so the two model builds each get their own.
    c0 = cfg.clone()
    c0.merge_from_file(args.config_file)
    c0.merge_from_list(base_opts)
    c0.freeze()
    _tr, _trn, val_loaders, _nq, num_classes, camera_num, view_num = make_dataloader(c0)
    names = list(c0.DATASETS.MODALITIES)

    print('\n' + '=' * 78)
    print('Stage 0.3  零样本模态分类 -- 图像特征 @ clip_proj  vs  文本锚点')
    print('=' * 78)

    text = CLIPTextEncoder()
    text.load_clip(_read_clip_checkpoint(args.clip))
    text.eval()
    tok = build_tokenizer()

    groups = spectrum_groups()
    print('\n  scoring against %d sentence groups: %s'
          % (len(groups), ', '.join(g[0].split()[0] for g in groups)))

    arms = [('A  original CLIP',
             base_opts + ['MODEL.PRETRAIN_CHOICE', 'imagenet',
                          'MODEL.PRETRAIN_PATH', args.clip]),
            ('B  our baseline',
             base_opts + ['MODEL.PRETRAIN_CHOICE', 'self',
                          'MODEL.PRETRAIN_PATH', args.weight])]

    results, conc = {}, {}
    for arm, opts in arms:
        print('\n' + '-' * 78)
        print('  arm %s' % arm)
        print('-' * 78, flush=True)
        model, _c = build(args.config_file, opts, num_classes, camera_num, view_num)
        model.to('cuda')
        proj = model.base.clip_proj
        if proj is None:
            raise SystemExit('this backbone has no clip_proj; '
                             'MODEL.TRANSFORMER_TYPE must be vit_base_clip')
        feat, mods = collect(model, val_loaders, args.per_modality)
        proj = proj.detach().cpu()
        conc[arm] = concentration(feat, mods, proj, len(names))
        print_concentration(arm, conc[arm], names)
        for label, sentences in groups:
            vecs, _counts = encode(text, tok, [sentences[k] for k in SPECTRUM_KEYS])
            mat = confusion(feat, mods, proj, vecs, len(names))
            print_confusion('%s  |  %s' % (arm, label), mat, names)
            results[(arm, label)] = mat
        del model, feat
        torch.cuda.empty_cache()

    print('\n' + '=' * 78)
    print('  汇总：每组模板下两臂的准确率')
    print('  %-34s %10s %10s %10s' % ('sentence group', 'A (CLIP)', 'B (ours)', 'B - A'))
    print('-' * 78)
    for label, _s in groups:
        a = results[(arms[0][0], label)]
        b = results[(arms[1][0], label)]
        pa = 100.0 * np.trace(a) / max(a.sum(), 1)
        pb = 100.0 * np.trace(b) / max(b.sum(), 1)
        print('  %-34s %9.1f%% %9.1f%% %+9.1f' % (label, pa, pb, pb - pa))
    print('  %-34s %9.1f%%' % ('(chance)', 100.0 / len(names)))

    print('\n  集中度：投影这一步到底做了什么')
    print('  %-20s %12s %12s %12s' % ('', '768d 两两cos', '512d 两两cos', '投影增益'))
    print('-' * 78)
    for arm, _o in arms:
        c = conc[arm]
        print('  %-20s %12.4f %12.4f %12.4f'
              % (arm, c['cos768'], c['cos512'], c['gain']))
    print('-' * 78)
    print('  判读：')
    print('    768d 已接近 1  -> 骨干本身就把一切压成了一个方向，与 CLIP 无关')
    print('    768d 低、512d 接近 1 -> 特征落进 clip_proj 的近似零空间，')
    print('                            文本损失一直在对着一条死通道说话')
    print('    两个都低       -> 通道是通的，塌陷另有原因，回去查混淆矩阵')
    print('=' * 78)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
