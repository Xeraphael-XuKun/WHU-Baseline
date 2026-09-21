# encoding: utf-8
"""mINP, and the ranks the paper's protocol tables ask for.

WHU-MARS reports R-1 / mAP / mINP for the VI protocol and the 3x3 modality
table, and mAP / R-1 / R-10 for the AG protocol.  Two of those we already had;
mINP did not exist, and R-5 / R-10 sat in the CMC array without ever being
printed.

mINP (Ye et al., TPAMI 2021) is decided by the LAST true match, not the first:
INP = num_rel / rank_of_hardest_match.  mAP is dominated by the easy hits near
the top, so a model can hold its mAP while stranding one spectrum of an identity
far down the list -- mINP is the number that notices.  The tests below pin that
property, not just the arithmetic.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import cmc_at, eval_func, eval_func_top_k  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('PASS  %s' % name)
    else:
        FAIL += 1
        print('FAIL  %s  %s' % (name, detail))


def ranking(order_pids, q_pid=1, q_cam=0):
    """One query against a gallery laid out in the given rank order.

    Every gallery item sits on a camera different from the query's, so the junk
    rule removes nothing and the ranking reaching eval_func is exactly the list
    given here -- otherwise these hand-computed expectations would not hold.
    """
    g_pids = np.asarray(order_pids)
    n = len(g_pids)
    return (np.arange(n)[None, :].astype(np.int32),
            np.asarray([q_pid]), g_pids,
            np.asarray([q_cam]), np.full(n, q_cam + 1))


# ------------------------------------------------------------ the definition
def test_perfect_ranking_gives_one():
    """All true matches at the very top: the hardest one sits at rank num_rel,
    so INP = num_rel / num_rel = 1."""
    idx, qp, gp, qc, gc = ranking([1, 1, 1, 2, 2, 3])
    _, _, minp = eval_func(idx, qp, gp, qc, gc, metric='market')
    check('a perfect ranking scores mINP = 1', abs(minp - 1.0) < 1e-6, minp)


def test_one_straggler_dominates():
    """Three true matches, two at the top and one buried at rank 6.
    INP = 3/6 = 0.5 -- and mAP stays high, which is the whole point of
    reporting mINP alongside it."""
    idx, qp, gp, qc, gc = ranking([1, 1, 2, 2, 2, 1])
    _, mAP, minp = eval_func(idx, qp, gp, qc, gc, metric='market')
    check('one straggler halves mINP', abs(minp - 0.5) < 1e-6, minp)
    check('...while mAP stays high', mAP > 0.75, mAP)


def test_minp_reports_something_rank1_cannot():
    """Move the LAST true match further down while leaving the first one alone.

    Rank-1 cannot see the difference by construction, and mAP barely can; mINP
    has to. This is the reason the column is worth a place in the table -- if it
    ever failed, mINP would be decoration.
    """
    near = eval_func(*ranking([1, 1, 2, 2, 2, 2, 2, 2]), metric='market')
    far = eval_func(*ranking([1, 2, 2, 2, 2, 2, 2, 1]), metric='market')
    check('Rank-1 is blind to where the last hit sits',
          near[0][0] == far[0][0] == 1.0, (near[0][0], far[0][0]))
    check('mINP is not: 2/2 against 2/8',
          abs(near[2] - 1.0) < 1e-6 and abs(far[2] - 0.25) < 1e-6,
          (near[2], far[2]))
    check('and mAP moves far less than mINP does',
          (near[1] - far[1]) < (near[2] - far[2]),
          (near[1] - far[1], near[2] - far[2]))


def test_minp_never_leaves_the_unit_interval():
    rng = np.random.RandomState(0)
    for _ in range(20):
        gp = rng.randint(1, 4, size=30)
        if not np.any(gp == 1):
            gp[0] = 1
        _, _, minp = eval_func(*ranking(list(gp)), metric='market')
        if not (0.0 < minp <= 1.0 + 1e-9):
            return check('mINP stays in (0, 1]', False, minp)
    check('mINP stays in (0, 1]', True)


def test_averaging_is_over_valid_queries_only():
    """A query with no true match in the gallery is skipped for mAP and CMC;
    mINP has to skip the same ones or the three columns describe different
    query sets."""
    idx = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
    q_pids = np.asarray([1, 9])          # pid 9 appears nowhere
    g_pids = np.asarray([1, 2, 2, 1])
    q_camids = np.asarray([0, 0])
    g_camids = np.asarray([1, 1, 1, 1])
    cmc, mAP, minp = eval_func(idx, q_pids, g_pids, q_camids, g_camids,
                               metric='market')
    # Only the first query counts: hits at ranks 1 and 4 -> INP = 2/4.
    check('a query absent from the gallery does not dilute mINP',
          abs(minp - 0.5) < 1e-6, minp)
    check('...and mAP agrees it was skipped', mAP > 0.5, mAP)


# --------------------------------------------------------------- the ranks
def test_cmc_at_reads_the_right_positions():
    cmc = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    check('cmc_at is 1-based', cmc_at(cmc, 1) == 0.1 and cmc_at(cmc, 5) == 0.5
          and cmc_at(cmc, 10) == 1.0)


def test_cmc_at_saturates_on_a_short_curve():
    """A thin subset -- one modality pair inside one view -- can leave fewer
    than ten gallery items, and eval_func trims max_rank to match.  Asking for
    R-10 there must not raise in the middle of an otherwise fine evaluation."""
    short = np.asarray([0.4, 0.7, 0.9])
    check('R-10 on a 3-long curve returns the last entry',
          cmc_at(short, 10) == 0.9)
    check('an empty curve gives NaN, not an exception',
          np.isnan(cmc_at(np.asarray([]), 5)) and np.isnan(cmc_at(None, 5)))


# ------------------------------------------------------------ the top-k path
def test_top_k_refuses_to_guess_minp():
    """mINP is decided by the hardest true match, which is exactly what a top-k
    cut is most likely to have discarded.  Reporting a number computed from the
    survivors would be optimistic while wearing the right name, so this path
    returns NaN.  TEST.TOP_K_EVAL is 0 in every config we run.
    """
    idx = np.asarray([[0, 1, 2]], dtype=np.int32)
    q_pids, g_pids = np.asarray([1]), np.asarray([1, 2, 1])
    q_camids, g_camids = np.asarray([0]), np.asarray([1, 1, 1])
    mids = np.asarray([0]), np.asarray([0, 0, 0])
    cmc, mAP, minp = eval_func_top_k(idx, q_pids, g_pids, q_camids, g_camids,
                                     mids[0], mids[1], metric='market')
    check('top-k returns NaN for mINP rather than an optimistic number',
          np.isnan(minp), minp)
    check('...and still returns real CMC / mAP', cmc[0] == 1.0 and mAP > 0)


# --------------------------------------------------------- the two matrices
def test_both_matrices_carry_minp_through():
    """The view and modality matrices are what the AG and VI protocols are read
    off, so the extra column has to survive the trip out of compute()."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'utils', 'metrics.py'), encoding='utf-8').read()
    check('view rows carry mINP',
          'view_pair_results.append((names[qa], names[ga], c, m, i))' in src)
    check('modality rows carry mINP',
          '(q_mod_target, g_mod_target, pair_cmc, pair_mAP, pair_mINP))' in src)
    check('the view line prints Rank-10 (the AG table needs it)',
          "'Rank-5: %.2f%%, Rank-10: %.2f%%, mINP: %.2f%%'" in src)
    check('the modality line prints mINP (the VI table needs it)',
          'mINP: {pair_mINP:.2%}' in src)
    check('pair_eval returns three values in every branch',
          src.count('return None, None, None') == 2, src.count('return None, None, None'))


def main():
    for fn in (test_perfect_ranking_gives_one,
               test_one_straggler_dominates,
               test_minp_reports_something_rank1_cannot,
               test_minp_never_leaves_the_unit_interval,
               test_averaging_is_over_valid_queries_only,
               test_cmc_at_reads_the_right_positions,
               test_cmc_at_saturates_on_a_short_curve,
               test_top_k_refuses_to_guess_minp,
               test_both_matrices_carry_minp_through):
        fn()
    print('\n%d/%d passed' % (PASS, PASS + FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
