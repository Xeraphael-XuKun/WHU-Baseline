"""Where did the TEXT loss actually move the weights?

    python diag/weight_shift.py A.pth B.pth          # A = without, B = with

CPU only, a few seconds, no GPU and no data.

THE QUESTION.  We say VTC drives the positional delta.  The evidence so far is
GRADIENT norms measured mid-training (backbone 26.9, pos_delta 54.9, clip_proj
26.0) -- instantaneous push, at one moment, on tensors of wildly different
sizes.  What the claim is really about is DISPLACEMENT: after sixty epochs, what
did the text loss change?  Two checkpoints answer that directly, because
whu_pe_indep_clip and whu_text_lam50_clip differ in exactly one config key:

    whu_pe_indep_clip     PE_LAYERWISE indep, TEXT_ALIGN False   13.29
    whu_text_lam50_clip   PE_LAYERWISE indep, TEXT_ALIGN True    14.11

Same seed, same data order, same everything else.  So B - A is what adding the
text loss did.

WHAT THIS IS NOT.  Not a clean attribution.  Adding a loss changes the gradient
from step one, so the two runs diverge chaotically as well as systematically --
the weight-space version of the +-0.2 mAP sensitivity band.  A group that moves
a LOT is not proof the text loss targeted it; a group that moves very LITTLE
while others move a lot is the informative direction, because chaotic
divergence alone would not spare one group.  Read the table for contrast
between groups, not for absolute magnitudes.

`pos_delta` is the exception and is reported separately: it is zero at
initialisation, so its norm IS its displacement, in both runs, with nothing to
subtract.
"""

import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def group_of(name):
    """Bucket a parameter name into something worth reasoning about.

    Deliberately coarse.  Per-tensor numbers for an 86M-parameter tower are
    noise you have to average yourself; the question is about mechanisms --
    the residual, the trunk, the readout into text space -- so the buckets are
    those.  Order matters: `pos_delta` and `pos_embed` both start with `pos`,
    and `clip_proj` sits under `base.` like the trunk does.
    """
    if 'pos_delta' in name:
        return 'pos_delta      (逐层残差)'
    if 'clip_proj' in name:
        return 'clip_proj      (进文本空间的投影)'
    if name.startswith('prompts.text.'):
        return 'text tower     (冻结)'
    if name.startswith('prompts.'):
        return 'prompts        (可学习提示词)'
    if 'pos_embed' in name:
        return 'pos_embed      (继承的位置表)'
    if 'cls_token' in name:
        return 'cls_token'
    if 'patch_embed' in name:
        return 'patch_embed'
    if name.startswith('base.blocks.'):
        return 'blocks         (ViT 主干 12 层)'
    if name.startswith('base.'):
        return 'base 其它      (ln_pre / norm)'
    return 'head           (BNNeck + 分类头)'


def norms(state, keys):
    """(L2 norm over the group, parameter count).

    Norms combine as the square root of the sum of squares -- concatenating the
    tensors and taking one norm gives the same number, without the memory.
    """
    sq = 0.0
    n = 0
    for k in keys:
        t = state[k].float()
        sq += float(t.pow(2).sum())
        n += t.numel()
    return sq ** 0.5, n


def load(path):
    sd = torch.load(path, map_location='cpu')
    # Checkpoints from this codebase are a bare state_dict; be tolerant of the
    # wrapped form rather than failing with a KeyError three screens later.
    for wrapper in ('state_dict', 'model'):
        if wrapper in sd and isinstance(sd[wrapper], dict):
            sd = sd[wrapper]
            break
    return OrderedDict((k, v) for k, v in sd.items() if torch.is_floating_point(v))


def main(path_a, path_b):
    a, b = load(path_a), load(path_b)
    print('A (无文本损失): {}'.format(path_a))
    print('B (有文本损失): {}'.format(path_b))
    print('  A 有 {} 个浮点张量, B 有 {} 个'.format(len(a), len(b)))

    shared = [k for k in a if k in b and a[k].shape == b[k].shape]
    only_a = [k for k in a if k not in shared]
    only_b = [k for k in b if k not in shared]

    groups = OrderedDict()
    for k in shared:
        groups.setdefault(group_of(k), []).append(k)

    print()
    print('=== 两个 run 都有的张量: 加上文本损失之后改动了多少 ===')
    print('{:<34} {:>12} {:>11} {:>11} {:>9}'.format(
        '分组', '参数量', '||A||', '||B-A||', '相对'))
    rows = []
    for g, keys in groups.items():
        na, count = norms(a, keys)
        diff_sq = sum(float((b[k].float() - a[k].float()).pow(2).sum()) for k in keys)
        nd = diff_sq ** 0.5
        rel = nd / na if na > 0 else float('nan')
        rows.append((g, count, na, nd, rel))
    # Largest relative change first: the ordering IS the finding.
    for g, count, na, nd, rel in sorted(rows, key=lambda r: -r[4]):
        print('{:<34} {:>12,} {:>11.3f} {:>11.3f} {:>8.2%}'.format(
            g, count, na, nd, rel))

    print()
    print('=== pos_delta: 零初始化, 所以范数本身就是位移 ===')
    for tag, sd in (('A 无文本', a), ('B 有文本', b)):
        keys = [k for k in sd if 'pos_delta' in k]
        if not keys:
            print('  {}: 没有 pos_delta'.format(tag))
            continue
        n, count = norms(sd, keys)
        print('  {}: ||pos_delta|| = {:.4f}   ({:,} 参数)'.format(tag, n, count))
        for k in keys:
            t = sd[k].float()
            if t.dim() >= 3 and t.shape[0] > 1:
                per = ['{:.3f}'.format(float(t[i].norm())) for i in range(t.shape[0])]
                print('     逐层  {}'.format('  '.join(per)))

    if only_a or only_b:
        print()
        print('=== 只在一边出现的张量 (没有可比对象, 只报范数) ===')
        for tag, keys, sd in (('只在 A', only_a, a), ('只在 B', only_b, b)):
            if not keys:
                continue
            per_group = OrderedDict()
            for k in keys:
                per_group.setdefault(group_of(k), []).append(k)
            for g, ks in per_group.items():
                n, count = norms(sd, ks)
                print('  {} {:<32} {:>12,} 参数   ||.|| = {:.3f}'.format(
                    tag, g, count, n))

    print()
    print('怎么读:')
    print('  这一列有绝对刻度。两个互不相关、量级相当的张量, ||B-A||/||A|| 约为')
    print('  sqrt(2) = 141%; 所以 ~141% 意味着"这一组已经和随机重画没区别",')
    print('  0% 意味着"一步没动", 中间的值是保留了多少原有结构。')
    print('  两个 run 从第一步就开始分岔, 所以任何一组都会有改动 —— 有意义的是')
    print('  组与组之间的落差: 混沌分岔不会单独放过某一组。')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
