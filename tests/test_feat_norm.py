"""Unit tests for the TEST.FEAT_NORM switch.

Why this got its own file.  On 2026-08-13 a re-evaluation was launched as
`bash reeval.sh whu_recipe_hihr TEST.FEAT_NORM no` and returned 10.23 / 26.65
-- the normalised baseline to the digit.  yacs had accepted the key, the config
dump showed it, and `utils/metrics.py` then tested it with a bare
`if self.feat_norm:`.  TEST.FEAT_NORM is the STRING 'no', and every non-empty
string is truthy, so the features were normalised anyway and nothing anywhere
said so.

The same shape has now cost this project three times: MODEL.ID_LOSS_TYPE is set
in configs and read by nobody, and TEXT_MODALITY was ignored throughout the
mtext runs.  What makes it expensive is that the run completes, the log is
clean, and the result is filed under the wrong label.

So: one coercion at the boundary, and anything unrecognised raises rather than
picking a side.

Run:  python tests/test_feat_norm.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.metrics import R1_mAP_eval, as_bool  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


# --------------------------------------------------------------------------
# 1. the coercion itself
# --------------------------------------------------------------------------
def test_the_string_no_is_false():
    """The whole bug in one line.  bool('no') is True."""
    assert as_bool('no') is False
    assert bool('no') is True, 'the premise of this test has changed'


def test_the_spellings_that_appear_in_configs_work():
    for v in ('yes', 'Yes', 'YES', ' yes ', 'true', 'on', '1', True):
        assert as_bool(v) is True, v
    for v in ('no', 'No', 'NO', ' no ', 'false', 'off', '0', '', False):
        assert as_bool(v) is False, v


def test_an_unrecognised_value_raises_and_names_the_key():
    """Defaulting is what made the original bug invisible for months.  A typo
    has to stop the run, and the message has to say which key."""
    for bad in ('nope', 'yes please', 'None', 2, None, [], 0.5):
        try:
            as_bool(bad, 'TEST.FEAT_NORM')
        except ValueError as exc:
            assert 'TEST.FEAT_NORM' in str(exc), (bad, exc)
            assert repr(bad) in str(exc), (bad, exc)
        else:
            raise AssertionError('accepted %r' % (bad,))


def test_a_real_bool_survives_unchanged():
    """R1_mAP_eval's own default is the bool True, and several call sites pass
    bools.  Coercing must not turn those into a string comparison."""
    assert as_bool(True) is True
    assert as_bool(False) is False


# --------------------------------------------------------------------------
# 2. the evaluator honours it
# --------------------------------------------------------------------------
def test_the_evaluator_stores_a_bool_not_the_string():
    for value, want in (('no', False), ('yes', True), (False, False), (True, True)):
        ev = R1_mAP_eval(feat_norm=value)
        assert ev.feat_norm is want, (value, ev.feat_norm)


def _rank1(feat_norm, q, gallery):
    """Drive the evaluator the way processor.py does: update per mode, declare
    the query count, split, compute."""
    ev = R1_mAP_eval(feat_norm=feat_norm, metric='market')
    ev.reset()
    ev.set_query_num(1, 1)                       # mode 1, first row is the query
    feats = torch.cat([q] + [g for g, _p, _c in gallery], dim=0)
    pids = np.array([1] + [p for _g, p, _c in gallery])
    camids = np.array([0] + [c for _g, _p, c in gallery])
    ev.update((feats, pids, camids, np.zeros_like(pids)), 1)
    ev.split_all()
    cmc, _mAP, *_ = ev.compute()
    return float(cmc[0])


def test_turning_it_off_actually_changes_the_ranking():
    """The property that was missing.  A source grep cannot see it: the flag
    could be coerced correctly and still not reach the distance.

    Constructed so the two settings MUST disagree.  The query points along e0.
    The right answer is a short vector exactly along e0 -- cosine 1, but far in
    L2.  The distractor is long and 45 degrees off -- cosine 0.707, but nearer
    in L2.  Normalised retrieval ranks by direction and finds the right one;
    unnormalised ranks by L2 and does not.
    """
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    right = torch.tensor([[0.10, 0.00, 0.0, 0.0]])
    wrong = torch.tensor([[0.75, 0.75, 0.0, 0.0]])
    # the premise, checked rather than assumed
    assert torch.norm(q - wrong) < torch.norm(q - right), 'L2 order is not as set up'
    cos = lambda a, b: float((a @ b.T) / (a.norm() * b.norm()))
    assert cos(q, right) > cos(q, wrong), 'cosine order is not as set up'

    gallery = [(right, 1, 1), (wrong, 2, 1)]      # (feat, pid, camid)
    assert _rank1('yes', q, gallery) == 1.0, \
        'normalised retrieval should rank the aligned vector first'
    assert _rank1('no', q, gallery) == 0.0, \
        'unnormalised retrieval still ranked by direction -- the switch is ' \
        'not reaching the distance'


# --------------------------------------------------------------------------
# 3. the other consumers, and the log
# --------------------------------------------------------------------------
def test_dump_features_uses_the_coercion_too():
    """It records feat_norm in the npz and the analysis reads that field to
    decide whether to normalise, so bool('no') there mislabels the file."""
    src = _read('diag', 'dump_features.py')
    assert 'as_bool(cfg.TEST.FEAT_NORM' in src
    assert 'np.array(bool(cfg.TEST.FEAT_NORM))' not in src
    assert 'from utils.metrics import as_bool' in src


def test_the_evaluator_says_which_way_it_went():
    """Silence used to mean both 'normalised' and 'your setting was dropped'."""
    src = _read('utils', 'metrics.py')
    assert 'The test feature is normalized' in src
    assert 'NOT normalized' in src


def test_reeval_no_longer_filters_that_line_away():
    """The line existed all along; reeval.sh's grep did not pass it through,
    which is why the ignored setting left no trace in reeval_log.txt."""
    src = _read('reeval.sh')
    assert 'feature is (NOT )?normalized' in src


def test_no_bare_truthiness_on_the_string_survives():
    src = _read('utils', 'metrics.py')
    assert 'self.feat_norm  = as_bool(' in src, 'the coercion was removed'
    # rindex, not index: the docstring of as_bool quotes the old broken line,
    # and matching that instead would make this test pass on a reverted fix.
    i = src.rindex('        if self.feat_norm:')
    j = src.index('self.feat_norm  = as_bool(')
    assert j < i, 'the flag is used before it is coerced'


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d FEAT_NORM tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
