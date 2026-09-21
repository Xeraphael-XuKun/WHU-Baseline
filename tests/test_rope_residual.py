# encoding: utf-8
"""Rotary position encoding as a zero-initialised residual.

The claim this file has to protect is one sentence: with ROPE_GATE on, the model
at step 0 IS the CLIP baseline, bit for bit, so every later difference belongs
to the rotation.  If that is false the whole arm is uninterpretable -- it would
be measuring "rotary plus whatever else changed" and reporting it as rotary.

The existing tests/test_chartpe.py covers the rotary machinery on the ViT-B line
but never builds it with clip_style=True, which is the only tower these four
arms run on.  That gap is closed here.
"""
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.chartpe import ChartRotaryEmbedding, build_centered_grid
from model.backbones.vit_pytorch import TransReID

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('PASS  %s' % name)
    else:
        FAIL += 1
        print('FAIL  %s  %s' % (name, detail))


def tiny(**kw):
    """A small CLIP-style tower: two blocks, two heads, the real 16x8 grid ratio."""
    torch.manual_seed(0)
    args = dict(img_size=(32, 16), stride_size=16, patch_size=16, embed_dim=32,
                depth=2, num_heads=2, mlp_ratio=1.0, qkv_bias=True,
                clip_style=True, num_classes=0)
    args.update(kw)
    return TransReID(**args)


# ---------------------------------------------------------------- the theorem
def test_zero_gain_is_the_baseline_bit_for_bit():
    """The whole arm rests on this: alpha = 0 must reproduce 'learnable' exactly.

    Not "close to" -- exactly.  A 1e-7 drift would still be a different model,
    and the project has already been burned once by attributing a difference to
    a design choice when it came from somewhere else.
    """
    x = torch.randn(2, 3, 32, 16)
    base = tiny(pe_type='learnable')
    rope = tiny(pe_type='rope_fixed', pe_keep_ape=True, rope_gate=True)
    # Same weights everywhere they overlap; only the rotary module is extra.
    missing = rope.load_state_dict(base.state_dict(), strict=False)
    extra = [k for k in missing.missing_keys if 'rope' not in k]
    check('zero-gain: nothing but rope is left uninitialised', not extra, extra)
    base.eval(), rope.eval()
    with torch.no_grad():
        a, b = base(x, None, None, None), rope(x, None, None, None)
    check('zero-gain: forward is bit-identical to the baseline',
          torch.equal(a, b), (a - b).abs().max().item() if a.shape == b.shape else 'shape')


def test_a_nonzero_gain_actually_changes_something():
    """The mirror of the theorem: if alpha never mattered the test above would
    pass for a rotary path that was silently disconnected."""
    x = torch.randn(2, 3, 32, 16)
    m = tiny(pe_type='rope_fixed', pe_keep_ape=True, rope_gate=True)
    m.eval()
    with torch.no_grad():
        before = m(x, None, None, None)
        for blk in m.blocks:
            blk.attn.rope.alpha.fill_(0.7)
        after = m(x, None, None, None)
    check('a non-zero gain moves the output', not torch.equal(before, after))


def test_the_gain_is_one_scalar_per_layer():
    """'Same skeleton as pe_indep' means per layer and independent.  Twelve
    numbers that can be read off after training, not one global knob."""
    m = tiny(depth=4, pe_type='rope_fixed', pe_keep_ape=True, rope_gate=True)
    alphas = [k for k, _ in m.named_parameters() if k.endswith('rope.alpha')]
    check('one gain per layer', len(alphas) == 4, alphas)
    check('every gain starts at zero',
          all(float(p.abs().max()) == 0.0 for k, p in m.named_parameters()
              if k.endswith('rope.alpha')))
    check('every gain is trainable',
          all(p.requires_grad for k, p in m.named_parameters()
              if k.endswith('rope.alpha')))
    # Independent, not shared: moving one must not move another.
    m.blocks[0].attn.rope.alpha.data.fill_(1.0)
    check('the gains are separate tensors',
          float(m.blocks[1].attn.rope.alpha.abs().max()) == 0.0)


def test_gain_off_leaves_the_old_arms_untouched():
    """ROPE_GATE defaults to False, and the two rope configs that predate it
    must keep behaving exactly as before -- no alpha parameter at all."""
    m = tiny(pe_type='rope_fixed', pe_keep_ape=True, rope_gate=False)
    check('gate off: no alpha in the state dict',
          not any('alpha' in k for k in m.state_dict()))
    check('gate off: the rotary module still exists',
          m.blocks[0].attn.rope is not None)


# ------------------------------------------------------- the rotation itself
def test_scaling_the_angle_preserves_the_norm():
    """Why a gain on the phase and not a side branch `q + g*rope(q)`.

    A rotation cannot change |q|, so attention logits never inflate because of
    where a patch sits.  The side-branch alternative expands to
    q*(1 + g cos) + rot90(q)*(g sin), whose norm carries a first-order position
    term -- this test is the reason that alternative was rejected, so it pins
    the property rather than the implementation.
    """
    torch.manual_seed(0)
    rope = ChartRotaryEmbedding(8, 2, gate=True)
    rope.alpha.data.fill_(0.6)
    q = torch.randn(2, 2, 5, 8)
    chart = torch.randn(2, 5, 2)
    q_out, _ = rope(q, q.clone(), chart)
    check('rotation preserves |q| for every token',
          torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5),
          (q_out.norm(dim=-1) - q.norm(dim=-1)).abs().max().item())
    # The rejected alternative, measured on the same input, for the record.
    side = q + 0.6 * q_out
    spread = (side.norm(dim=-1) / q.norm(dim=-1))
    check('the side-branch alternative would NOT preserve it',
          float(spread.max() - spread.min()) > 1e-3,
          float(spread.max() - spread.min()))


def test_the_gain_scales_the_angle_linearly():
    """alpha multiplies the phase, so doubling it doubles every rotation angle.
    Checked through the cosine between the rotated and original vector, which
    is cos(phase) and therefore reads the angle directly."""
    torch.manual_seed(0)
    rope = ChartRotaryEmbedding(8, 2, gate=True)
    q = torch.randn(1, 2, 4, 8)
    chart = torch.full((1, 4, 2), 0.05)   # small angles -> the linear regime
    outs = {}
    for a in (0.0, 0.5, 1.0):
        rope.alpha.data.fill_(a)
        outs[a], _ = rope(q, q.clone(), chart)
    # At alpha = 0 the question is exactness, so ask it directly.  Reading the
    # angle through arccos would not do: near cos = 1 its derivative blows up,
    # and a 1.8e-8 float error shows up as a 1.9e-4 "rotation" that is not there.
    check('alpha = 0 gives no rotation at all', torch.equal(outs[0.0], q))
    angles = []
    for a in (0.5, 1.0):
        cos = torch.nn.functional.cosine_similarity(outs[a], q, dim=-1).clamp(-1, 1)
        angles.append(float(cos.arccos().mean()))
    check('the angle is linear in alpha',
          abs(angles[1] - 2 * angles[0]) < 1e-3 * angles[1], angles)


def test_cls_is_never_rotated():
    """The CLS row of the chart is zero, and CLS is the retrieval feature.  If
    the gain ever reached it, the arm would be changing the very vector the mAP
    is computed from by a route that has nothing to do with position."""
    torch.manual_seed(0)
    rope = ChartRotaryEmbedding(8, 2, gate=True)
    rope.alpha.data.fill_(1.3)
    q = torch.randn(1, 2, 5, 8)
    grid = build_centered_grid(2, 2)
    chart = torch.cat([torch.zeros(1, 2), grid], dim=0).unsqueeze(0)
    out, _ = rope(q, q.clone(), chart)
    check('CLS comes out untouched', torch.allclose(out[:, :, 0], q[:, :, 0], atol=1e-6))
    check('the patch rows do move', not torch.allclose(out[:, :, 1:], q[:, :, 1:], atol=1e-4))


# ------------------------------------------------------------- the plumbing
def test_the_gain_is_off_the_pretrained_clock():
    """alpha lives inside `base.`, so the default rule would file it with the
    pretrained weights at 5e-6.  A zero-initialised scalar on that clock barely
    leaves the floor, and the arm would report "the model does not want
    rotation" when the truth is that alpha never got to move.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'solver', 'make_optimizer.py'), encoding='utf-8').read()
    head = src[src.index('PRETRAINED_LR > 0'):src.index('lr = cfg.SOLVER.PRETRAINED_LR')]
    check('alpha is excluded from the pretrained bucket', "'rope.alpha' not in key" in head)
    check('alpha shares the delta learning-rate multiplier',
          'if "pos_delta" in key or "rope.alpha" in key:' in src)
    check('alpha is not weight-decayed toward the baseline',
          src.index('if "rope.alpha" in key:') < src.index('if "rope.freqs" in key:'))


def test_the_four_configs_differ_only_where_intended():
    """Four arms generated from one base; the diff is the experiment."""
    import yaml

    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    base = load('whu_pe_indep_clip')
    added = {'MODEL.PE_TYPE', 'MODEL.PE_KEEP_APE', 'MODEL.ROPE_GATE',
             'MODEL.ROPE_FREQ_TRAINABLE'}
    expect = {
        'whu_rope_indep_clip': ('rope_fixed', True, False),
        'whu_rope_full_clip': ('rope_fixed', False, False),
        'whu_rope_freq_clip': ('rope_fixed', True, True),
        'whu_rope_chart_clip': ('chartpe', True, False),
        'whu_rope_chart_freq_clip': ('chartpe', True, True),
    }
    for name, (pe_type, gate, freq) in expect.items():
        c = load(name)
        check('%s: adds only the rotary keys' % name,
              set(c) - set(base) == added, sorted(set(c) - set(base)))
        diff = {k for k in base if k in c and base[k] != c[k]}
        check('%s: changes only PE_LAYERWISE and OUTPUT_DIR' % name,
              diff == {'MODEL.PE_LAYERWISE', 'OUTPUT_DIR'}, sorted(diff))
        check('%s: the additive delta is OFF (replaced, not stacked)' % name,
              c['MODEL.PE_LAYERWISE'] == 'none', c['MODEL.PE_LAYERWISE'])
        check('%s: the pretrained position table is kept' % name,
              c['MODEL.PE_KEEP_APE'] is True)
        check('%s: rotary settings are as designed' % name,
              (c['MODEL.PE_TYPE'], c['MODEL.ROPE_GATE'],
               c['MODEL.ROPE_FREQ_TRAINABLE']) == (pe_type, gate, freq),
              (c['MODEL.PE_TYPE'], c['MODEL.ROPE_GATE'], c['MODEL.ROPE_FREQ_TRAINABLE']))
        check('%s: still raw CLIP, not the CARGO tower' % name,
              c['MODEL.PRETRAIN_CHOICE'] == 'imagenet' and 'ViT-B-16' in c['MODEL.PRETRAIN_PATH'])


def test_chartpe_arm_is_also_zero_at_init():
    """Two zero-initialised things in series (TPCG's chart and the gain).  The
    arm is only interpretable if the pair still reproduces the baseline."""
    x = torch.randn(2, 3, 32, 16)
    base = tiny(pe_type='learnable')
    chart = tiny(pe_type='chartpe', pe_keep_ape=True, rope_gate=True)
    chart.load_state_dict(base.state_dict(), strict=False)
    base.eval(), chart.eval()
    with torch.no_grad():
        a, b = base(x, None, None, None), chart(x, None, None, None)
    check('chartpe + zero gain is the baseline', torch.equal(a, b),
          (a - b).abs().max().item() if a.shape == b.shape else 'shape')


def main():
    for fn in (test_zero_gain_is_the_baseline_bit_for_bit,
               test_a_nonzero_gain_actually_changes_something,
               test_the_gain_is_one_scalar_per_layer,
               test_gain_off_leaves_the_old_arms_untouched,
               test_scaling_the_angle_preserves_the_norm,
               test_the_gain_scales_the_angle_linearly,
               test_cls_is_never_rotated,
               test_the_gain_is_off_the_pretrained_clock,
               test_the_four_configs_differ_only_where_intended,
               test_chartpe_arm_is_also_zero_at_init):
        fn()
    print('\n%d/%d passed' % (PASS, PASS + FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
