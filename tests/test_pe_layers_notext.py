"""Unit tests for the PE_LAYERS sweep moved onto the plain residual baseline.

CPU-only and self-contained.  Run directly::

    python tests/test_pe_layers_notext.py

WHY THIS SET EXISTS.  Three tables ask about the positional residual itself --
where to inject it, how to inject it (indep vs chain), and what to inject
(additive vs rotary).  Two of them already sit on the plain baseline
`whu_pe_indep_clip` (13.29), with no text loss anywhere.  The third, the
PE_LAYERS sweep, was built on `whu_text_lam50_clip` and therefore carried the
two-anchor TGVA, which put it on a base of 14.11 -- neither the plain baseline
nor the full method.  This set moves it down so all three share one control.

The failure modes are silent, which is why they are asserted:

  * an arm that kept TEXT_ALIGN would still train and still report a number,
    just against the wrong control -- exactly the defect being repaired;
  * an arm differing from the base in a second key measures two things at once;
  * a duplicated or mistyped PE_LAYERS list changes which blocks get an
    increment while the filename still says otherwise.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = os.path.join(ROOT, 'configs')

BASE = 'hihr_whu_pe_indep_clip.yml'
ARMS = {
    'hihr_whu_pe_first_notext_clip.yml': '[0]',
    'hihr_whu_pe_last_notext_clip.yml': '[11]',
    'hihr_whu_pe_every3_notext_clip.yml': '[0, 3, 6, 9]',
    'hihr_whu_pe_every2_notext_clip.yml': '[0, 2, 4, 6, 8, 10]',
}
# The older sweep, kept: it becomes the appendix's robustness check.
WITH_TGVA = {
    'hihr_whu_pe_first_clip.yml': '[0]',
    'hihr_whu_pe_last_clip.yml': '[11]',
    'hihr_whu_pe_every3_clip.yml': '[0, 3, 6, 9]',
    'hihr_whu_pe_every2_clip.yml': '[0, 2, 4, 6, 8, 10]',
}


def keys(name):
    out = {}
    for line in open(os.path.join(CFGS, name), encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        m = re.match(r'^(\s*)([A-Z_]+):\s*(.+)$', line)
        if m and m.group(3).strip():
            out[m.group(2)] = m.group(3).strip()
    return out


# ---- single-factor against the plain baseline ---------------------------

def test_each_arm_differs_from_the_plain_baseline_in_one_key():
    base = keys(BASE)
    for name, layers in ARMS.items():
        arm = keys(name)
        differs = {k for k in set(base) | set(arm) if base.get(k) != arm.get(k)}
        assert differs == {'PE_LAYERS', 'OUTPUT_DIR'}, (name, sorted(differs))
        assert arm['PE_LAYERS'] == layers, (name, arm['PE_LAYERS'])


def test_no_arm_carries_the_text_loss():
    """The whole point of the move.  An arm with TEXT_ALIGN would read against
    14.11 while the table says 13.29."""
    for name in [BASE] + list(ARMS):
        arm = keys(name)
        assert arm.get('TEXT_ALIGN') != 'True', \
            '{} still carries TGVA; it belongs on the 14.11 base, not here'.format(name)


def test_the_base_is_the_control_the_tables_quote():
    """13.29 / 34.10 comes from whu_pe_indep_clip: layer-wise residual at every
    block, indep differencing, no text.  If any of those moved, every number in
    the three tables is being read against something else."""
    base = keys(BASE)
    assert base['PE_LAYERWISE'] == "'indep'"
    assert 'PE_LAYERS' not in base, 'the control must inject at every block'
    assert base.get('TEXT_ALIGN') != 'True'
    assert base.get('PE_PER_MODALITY') != 'True', \
        'the control for these three tables is the SHARED residual'


def test_the_two_sweeps_agree_on_which_blocks_each_arm_uses():
    """The appendix version and the main version must differ ONLY in TGVA."""
    for notext, withtgva in zip(sorted(ARMS), sorted(WITH_TGVA)):
        assert ARMS[notext] == WITH_TGVA[withtgva], (notext, withtgva)
        a, b = keys(notext), keys(withtgva)
        assert a['PE_LAYERS'] == b['PE_LAYERS']
        assert b.get('TEXT_ALIGN') == 'True', \
            '{} is supposed to be the TGVA-enabled version'.format(withtgva)


def test_the_matched_pair_is_matched():
    """[0] and [11] must hold the parameter count fixed -- one insertion point
    each -- or the 'position matters at equal capacity' claim has no basis."""
    first = eval(ARMS['hihr_whu_pe_first_notext_clip.yml'])   # noqa: S307
    last = eval(ARMS['hihr_whu_pe_last_notext_clip.yml'])     # noqa: S307
    assert len(first) == len(last) == 1
    assert first == [0] and last == [11]


def test_the_sweep_covers_four_distinct_counts():
    counts = sorted(len(eval(v)) for v in ARMS.values())      # noqa: S307
    assert counts == [1, 1, 4, 6], counts


def test_every_index_is_a_real_block():
    """ViT-B/16 has twelve blocks, numbered 0..11."""
    for name, layers in ARMS.items():
        idx = eval(layers)                                     # noqa: S307
        assert all(0 <= i < 12 for i in idx), (name, idx)
        assert len(set(idx)) == len(idx), (name, idx)
        assert idx == sorted(idx), (name, idx)


def test_output_dirs_are_distinct_from_the_tgva_sweep():
    outs = [keys(n)['OUTPUT_DIR'] for n in list(ARMS) + list(WITH_TGVA) + [BASE]]
    assert len(set(outs)) == len(outs), outs


def test_the_runner_knows_all_four():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    for name in ARMS:
        mode = name[len('hihr_'):-len('.yml')]
        assert '{})'.format(mode) in src, \
            '{} has no branch in run_hihr.sh'.format(mode)


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
