"""Unit tests for MODEL.TEXT_SUPERVISION -- the 2x2 contrast vs one direct target.

CPU-only and self-contained.  Run directly::

    python tests/test_text_supervision.py

WHAT IS BEING PROTECTED.

  * 'contrast' must stay bit-identical to the code before this key existed --
    every recorded run used it, and a four-cell sum quietly becoming a two-cell
    sum would rescale the loss by ~2 without a word in the log;
  * 'direct' must produce NO second forward and no push/drift, because both are
    defined as after-minus-before and reporting them as 0.0 would read as "the
    delta moved nothing" rather than "there is nothing to compare";
  * TEXT_ALIGN with PE_LAYERWISE 'none' under 'contrast' must be refused.  It
    runs happily otherwise and reports a plausible mAP: before and after are
    one tensor, A_before -> aerial and A_after -> ground cancel at p = 0.5, and
    what actually trains is "sit equidistant from the anchors".  The codebase
    already refuses the identical pathology for PE_VIEW_GATE 'ground'; this is
    the same guard, on the other route in.
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss.text_align import AERIAL, GROUND  # noqa: E402
from processor.processor import view_align  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def aux(supervision='contrast', n=8, dim=6, n_anchor=2, seed=0):
    """The dict make_model hands the processor, small enough to reason about."""
    torch.manual_seed(seed)
    feat_after = torch.randn(n, dim, requires_grad=True)
    before = None if supervision == 'direct' else torch.randn(n, dim,
                                                              requires_grad=True)
    is_aerial = torch.tensor([True] * (n // 2) + [False] * (n - n // 2))
    return {
        'feat_after': feat_after,
        'feat_before': before,
        'is_aerial': is_aerial,
        'modality': torch.zeros(n, dtype=torch.long),
        'n_view': 2,
        'n_modality': 1,
        'text': torch.randn(n_anchor, 4),
        'logit_scale': torch.tensor(100.0),
        'proj': torch.randn(dim, 4),
        'supervision': supervision,
        'target': 'view',
    }


# ---- the loss shape ------------------------------------------------------

def test_contrast_sums_four_cells_and_direct_sums_two():
    _, c = view_align(aux('contrast'))
    _, d = view_align(aux('direct'))
    cells = lambda s: sorted(k[4:] for k in s if k.startswith('acc_'))
    assert cells(c) == ['A_after', 'A_before', 'G_after', 'G_before'], cells(c)
    assert cells(d) == ['A_after', 'G_after'], cells(d)


def test_direct_reports_no_push_and_no_drift():
    """Absent, not zero: 0.0 would read as 'the delta moved nothing'."""
    _, d = view_align(aux('direct'))
    assert 'push' not in d and 'drift' not in d, sorted(d)
    assert d['direct'] is True
    _, c = view_align(aux('contrast'))
    assert 'push' in c and 'drift' in c and c['direct'] is False


def test_direct_never_touches_feat_before():
    """feat_before is None there; anything reading it would raise, not slip."""
    a = aux('direct')
    assert a['feat_before'] is None
    view_align(a)          # must not raise


def test_direct_targets_the_ground_anchor_for_both_groups():
    """Aerial reads as ground, ground reads as ground -- one target, not two.

    Verified by construction rather than by reading the source: features placed
    exactly on the ground anchor must give a loss at the floor for BOTH groups,
    which is only true if both groups target that anchor.
    """
    a = aux('direct', dim=4)
    a['proj'] = torch.eye(4)
    text = torch.zeros(2, 4)
    text[AERIAL, 0] = 1.0
    text[GROUND, 1] = 1.0
    a['text'] = text
    a['feat_after'] = text[GROUND].repeat(8, 1).clone().requires_grad_(True)

    loss, s = view_align(a)
    assert s['acc_A_after'] == 1.0 and s['acc_G_after'] == 1.0
    assert float(loss) < 1e-4, float(loss)

    # ...and on the AERIAL anchor both groups must be wrong, which is what
    # distinguishes "one target" from "each row keeps its own view".
    a2 = aux('direct', dim=4)
    a2['proj'] = torch.eye(4)
    a2['text'] = text
    a2['feat_after'] = text[AERIAL].repeat(8, 1).clone().requires_grad_(True)
    _, s2 = view_align(a2)
    assert s2['acc_A_after'] == 0.0 and s2['acc_G_after'] == 0.0


def test_contrast_is_unchanged_by_the_key_existing():
    """No `supervision` entry at all must behave exactly like 'contrast'."""
    a, b = aux('contrast', seed=3), aux('contrast', seed=3)
    del b['supervision']
    la, sa = view_align(a)
    lb, sb = view_align(b)
    assert torch.equal(la, lb)
    for k in sa:
        if k == 'direct':
            continue
        assert sa[k] == sb[k], k


def test_gradient_reaches_the_features_in_direct_mode():
    a = aux('direct')
    loss, _ = view_align(a)
    loss.backward()
    assert a['feat_after'].grad is not None
    assert float(a['feat_after'].grad.abs().sum()) > 0


def test_per_modality_push_is_skipped_in_direct_mode():
    """It is defined as after-minus-before and would dereference None."""
    a = aux('direct', n=12)
    a['n_modality'] = 3
    a['modality'] = torch.arange(12) % 3
    a['text'] = torch.randn(6, 4)
    a['n_view'] = 2
    _, s = view_align(a)
    assert 'push_per_modality' not in s


# ---- the guards ----------------------------------------------------------

def _src(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def test_no_actuator_is_refused_by_default_and_opt_in_exists():
    src = _src('model', 'make_model.py')
    # Anchor on the opt-in, not on the words: "has no actuator" also appears in
    # the older PE_VIEW_GATE 'ground' guard, which is the same pathology by the
    # other route in, and searching for the phrase finds that one first.
    assert 'not cfg.MODEL.TEXT_ALLOW_NO_ACTUATOR' in src, \
        'the opt-in must be able to switch the guard off, or the degenerate ' \
        'arm is unreachable rather than merely guarded'
    at = src.index('not cfg.MODEL.TEXT_ALLOW_NO_ACTUATOR')
    condition = src[at - 400:at]
    assert "cfg.MODEL.PE_LAYERWISE == 'none'" in condition, condition
    assert "self.text_supervision == 'contrast'" in condition, condition
    # ...and 'direct' must NOT be caught by it: that mode has no before pass,
    # so there is no contradiction to guard against.
    message = src[at:at + 1400]
    assert "'direct'" in message, 'the message must point at the way out'


def test_direct_is_view_only():
    src = _src('model', 'make_model.py')
    assert "'direct' is defined for TEXT_TARGET" in src


def test_unknown_supervision_is_refused():
    src = _src('model', 'make_model.py')
    assert "must be 'contrast' or " in src


def test_make_model_skips_the_before_pass_in_direct_mode():
    """The second forward is the cost this mode exists to drop."""
    src = _src('model', 'make_model.py')
    body = src[src.index("if self.text_supervision == 'direct':"):]
    body = body[:body.index('feat_before = self.base(')]
    assert "'feat_before': None" in body
    assert "'supervision': 'direct'" in body
    assert 'self.base(' not in body.split('return')[1], \
        'the direct branch must return before the second forward'


# ---- config wiring -------------------------------------------------------

def test_the_keys_exist_in_defaults():
    src = _src('config', 'defaults.py')
    assert "_C.MODEL.TEXT_SUPERVISION = 'contrast'" in src
    assert '_C.MODEL.TEXT_ALLOW_NO_ACTUATOR = False' in src


def _keys(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        m = re.match(r'^(\s*)([A-Z_]+):\s*(.+)$', line)
        if m and m.group(3).strip():
            out[m.group(2)] = m.group(3).strip()
    return out


def test_the_arm_differs_in_exactly_the_two_intended_keys():
    cfgs = os.path.join(ROOT, 'configs')
    control = _keys(os.path.join(cfgs, 'hihr_whu_text_lam50_clip.yml'))
    arm = _keys(os.path.join(cfgs, 'hihr_whu_direct_nopld_clip.yml'))
    differs = {k for k in set(control) | set(arm) if control.get(k) != arm.get(k)}
    assert differs == {'PE_LAYERWISE', 'TEXT_SUPERVISION', 'OUTPUT_DIR'}, sorted(differs)
    assert arm['PE_LAYERWISE'] == "'none'"
    assert arm['TEXT_SUPERVISION'] == "'direct'"
    assert arm['TEXT_ALIGN'] == 'True', 'the text loss is the whole point'


def test_the_arm_does_not_quietly_opt_into_the_degenerate_path():
    for name in ('hihr_whu_direct_nopld_clip.yml', 'hihr_whu_direct_pld_clip.yml'):
        arm = _keys(os.path.join(ROOT, 'configs', name))
        assert 'TEXT_ALLOW_NO_ACTUATOR' not in arm, \
            '{} avoids the pathology by design, not by suppressing the guard'.format(name)


def test_the_pld_arm_differs_from_the_main_method_in_one_key():
    """Single-factor against 14.11: only the target SHAPE changes, delta stays.

    This is what makes the grid attribute -- without it, whu_direct_nopld_clip
    differs from the main method in two things at once and cannot say which
    moved the number.
    """
    cfgs = os.path.join(ROOT, 'configs')
    control = _keys(os.path.join(cfgs, 'hihr_whu_text_lam50_clip.yml'))
    arm = _keys(os.path.join(cfgs, 'hihr_whu_direct_pld_clip.yml'))
    differs = {k for k in set(control) | set(arm) if control.get(k) != arm.get(k)}
    assert differs == {'TEXT_SUPERVISION', 'OUTPUT_DIR'}, sorted(differs)
    assert arm['PE_LAYERWISE'] == "'indep'", 'the delta has to stay'
    assert arm['TEXT_SUPERVISION'] == "'direct'"


def test_the_two_direct_arms_differ_only_in_the_delta():
    """The other edge of the grid: same target shape, delta on vs off."""
    cfgs = os.path.join(ROOT, 'configs')
    a = _keys(os.path.join(cfgs, 'hihr_whu_direct_nopld_clip.yml'))
    b = _keys(os.path.join(cfgs, 'hihr_whu_direct_pld_clip.yml'))
    differs = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differs == {'PE_LAYERWISE', 'OUTPUT_DIR'}, sorted(differs)
    assert a['TEXT_SUPERVISION'] == b['TEXT_SUPERVISION'] == "'direct'"
    assert a['TEXT_LOSS_WEIGHT'] == b['TEXT_LOSS_WEIGHT'], \
        'a different lambda would put a third factor in the grid'


def test_the_runner_knows_both_arms():
    src = _src('run_hihr.sh')
    for mode in ('whu_direct_nopld_clip', 'whu_direct_pld_clip'):
        assert '{})'.format(mode) in src, '{} has no branch in run_hihr.sh'.format(mode)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL  {}: {}: {}'.format(fn.__name__, type(exc).__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
