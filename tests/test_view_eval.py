"""The view matrix: aerial->ground retrieval, scored separately.

Every WHU-MARS number so far was split by *modality* (RGB / IR / thermal) and
never by *view*.  So the question direction one and direction two both exist to
answer -- does an aerial query find the same person on the ground -- has never
actually been measured.  This adds that readout, and these tests pin the two
things that make it trustworthy:

  * it changes no existing number (aerial_cams empty -> byte-identical output),
  * it is not a constant: a synthetic case where cross-view retrieval is good
    and one where it collapses have to come out that way round.

A note on the fixtures: eval_func stacks the per-query CMC rows with
np.asarray, so they all have to come out the same length -- but the sysu junk
rule removes a different number of gallery entries per query, and the rows are
only equalised by the truncation to max_rank.  On a toy gallery smaller than
max_rank that truncation does nothing and the stack raises.  Real galleries are
far larger, so this never bites in a run; here it would only produce a
confusing failure that says nothing about views.  Hence the deliberately
oversized galleries below.

Run:  python tests/test_view_eval.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import R1_mAP_eval, eval_func, junk_mask  # noqa: E402

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


# --------------------------------------------------------------------------
# Fixture.  Cameras 5 and 6 are aerial (WHU-MARS c6/c7, zero based); 0..4 are
# ground.  Every identity is seen once from the air and twice from the ground
# on both sides, so all four cells are non-degenerate by construction.
# --------------------------------------------------------------------------
AERIAL = (5, 6)
# The gallery has to exceed eval_func's default max_rank of 50, because the
# headline call does not pass self.max_rank -- with a smaller gallery it
# truncates to the gallery size and the per-query CMC rows, which the sysu junk
# rule leaves at different lengths, can no longer be stacked.  30 identities x
# 3 gallery images = 90, and the thinnest cell still keeps 60.
N_ID = 30
DIM = 32


def build(cross_view_good=True, aerial_gallery_cam=6, drop_aerial_gallery=False):
    """Queries: (aerial cam5, ground cam0) per id.
    Gallery: (aerial cam6, ground cam1, ground cam2) per id.

    Identity i lives on direction e_i.  Ground images sit exactly on it.
    Aerial images sit on e_i too when `cross_view_good`; otherwise they pile
    onto one shared view direction and carry the *wrong* identity residual
    (a derangement, i -> i+1), so cross-view retrieval confidently returns the
    wrong person while same-view retrieval stays perfect.  That is precisely
    the failure the positional correction targets.  A plant that merely shrank
    the identity component would not do: any positive coefficient leaves the
    ranking intact, so A->G would still score 100%.
    """
    rng = np.random.RandomState(0)
    # A pinch of noise on every feature.  Without it the wrong-identity
    # gallery entries are all exactly equidistant, and numpy's argsort and
    # torch's break that tie differently -- which would make the hand-built
    # cross-check below fail over tie order rather than over anything real.
    jitter = lambda: 1e-3 * rng.randn(DIM)                       # noqa: E731

    view_dir = np.zeros(DIM)
    view_dir[N_ID] = 1.0
    q_f, q_p, q_c, g_f, g_p, g_c = [], [], [], [], [], []
    for i in range(N_ID):
        base = np.zeros(DIM)
        base[i] = 1.0
        wrong = np.zeros(DIM)
        wrong[(i + 1) % N_ID] = 1.0
        aerial = base if cross_view_good else view_dir + 0.05 * wrong

        q_f.append(aerial + jitter()); q_p.append(i); q_c.append(5)
        q_f.append(base + jitter());   q_p.append(i); q_c.append(0)

        if not drop_aerial_gallery:
            g_f.append(aerial + jitter()); g_p.append(i); g_c.append(aerial_gallery_cam)
        g_f.append(base + jitter()); g_p.append(i); g_c.append(1)
        g_f.append(base + jitter()); g_p.append(i); g_c.append(2)
    return np.asarray(q_f), q_p, q_c, np.asarray(g_f), g_p, g_c


def decoy_case():
    """A fixture whose A->A cell scores differently under the two junk rules.

    Every query is aerial (cam 5).  Each identity has a true aerial match on
    cam 6 at distance 0.6, and there is a decoy on cam 5 -- labelled as a
    *different* identity -- sitting at distance 0.3.

      sysu   drops the query's entire camera, decoys included  -> Rank-1 = 1.0
      market drops only the query's own identity on that camera, so the decoy
             survives and outranks the true match                -> Rank-1 = 0.0

    That is the sharpest available statement that the cells honour TEST.METRIC
    rather than a hardcoded rule.
    """
    n = 60
    dim = n + 2
    q_f, q_p, q_c, g_f, g_p, g_c = [], [], [], [], [], []
    for i in range(n):
        base = np.zeros(dim); base[i] = 1.0
        true_off = np.zeros(dim); true_off[n] = 0.6
        decoy_off = np.zeros(dim); decoy_off[n + 1] = 0.3
        q_f.append(base);             q_p.append(i);           q_c.append(5)
        g_f.append(base + true_off);  g_p.append(i);           g_c.append(6)
        g_f.append(base + decoy_off); g_p.append((i + 1) % n); g_c.append(5)
    return np.asarray(q_f), q_p, q_c, np.asarray(g_f), g_p, g_c


def make_eval(qf, q_pids, q_camids, gf, g_pids, g_camids,
              aerial_cams=AERIAL, metric='sysu', max_rank=1):
    ev = R1_mAP_eval(max_rank=max_rank, feat_norm=False, reranking=False,
                     top_k=0, logger=None, metric=metric, aerial_cams=aerial_cams)
    feats = torch.cat([torch.as_tensor(qf, dtype=torch.float32),
                       torch.as_tensor(gf, dtype=torch.float32)], dim=0)
    pids = np.asarray(list(q_pids) + list(g_pids))
    cams = np.asarray(list(q_camids) + list(g_camids))
    mods = np.zeros(len(pids), dtype=np.int64)
    ev.update((feats, pids, cams, mods), mode=0)
    ev.set_query_num(0, len(q_pids))
    ev.split_all()
    return ev


def cells(view_pair_results):
    return {(q, g): (m, c[0]) for q, g, c, m, _ in view_pair_results}


# --------------------------------------------------------------------------
# 1. The readout is purely additive.
# --------------------------------------------------------------------------
def test_off_by_default_and_headline_identical():
    args = build()
    c0, m0, mod0, *_r0, view0 = make_eval(*args, aerial_cams=()).compute()
    c1, m1, mod1, *_r1, view1 = make_eval(*args, aerial_cams=AERIAL).compute()
    assert view0 == [], view0
    assert len(view1) == 4, view1
    assert m0 == m1, (m0, m1)
    assert np.array_equal(c0, c1)
    # the modality matrix must be untouched too
    assert len(mod0) == len(mod1) and len(mod0) > 0
    for a, b in zip(mod0, mod1):
        assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3]


def test_compute_returns_eight_values_and_star_unpack_still_works():
    out = make_eval(*build()).compute()
    assert len(out) == 8, len(out)
    cmc, mAP, *_ = out          # the exact form both processor.py call sites use
    assert cmc.shape[0] >= 1 and 0.0 <= mAP <= 1.0


def test_default_constructor_still_takes_no_aerial_cams():
    assert R1_mAP_eval().aerial_cams == ()


def test_aerial_cams_accepts_a_yacs_list():
    assert R1_mAP_eval(aerial_cams=[5, 6]).aerial_cams == (5, 6)
    assert R1_mAP_eval(aerial_cams=(6, 5)).aerial_cams == (6, 5)


# --------------------------------------------------------------------------
# 2. The four cells are the right four cells, computed the right way.
# --------------------------------------------------------------------------
def test_four_cells_named_AA_AG_GA_GG():
    *_, view = make_eval(*build()).compute()
    assert [(q, g) for q, g, _c, _m, _i in view] == [('A', 'A'), ('A', 'G'),
                                                 ('G', 'A'), ('G', 'G')], view


def test_cells_match_a_hand_built_subset_evaluation():
    """Each cell must equal eval_func run on that subset by hand.

    This is the whole correctness claim: the loop slices the right rows and
    columns, and the junk rule inside still sees the real camera ids -- so
    restricting the gallery does not smuggle the query's own camera back in.
    """
    qf, q_pids, q_cams, gf, g_pids, g_cams = build(cross_view_good=False)
    got = cells(make_eval(qf, q_pids, q_cams, gf, g_pids, g_cams).compute()[-1])

    qf_t = torch.as_tensor(qf, dtype=torch.float32)
    gf_t = torch.as_tensor(gf, dtype=torch.float32)
    q_cams_a, g_cams_a = np.asarray(q_cams), np.asarray(g_cams)
    q_pids_a, g_pids_a = np.asarray(q_pids), np.asarray(g_pids)
    q_is_a = np.isin(q_cams_a, AERIAL)
    g_is_a = np.isin(g_cams_a, AERIAL)

    for qa in (True, False):
        for ga in (True, False):
            qs = np.where(q_is_a == qa)[0]
            gs = np.where(g_is_a == ga)[0]
            idx = np.argsort(torch.cdist(qf_t[qs], gf_t[gs]).numpy(),
                             axis=1).astype(np.int32)
            cmc, mAP, _ = eval_func(idx, q_pids_a[qs], g_pids_a[gs],
                                 q_cams_a[qs], g_cams_a[gs],
                                 max_rank=1, metric='sysu')
            gm, gc = got[('A' if qa else 'G', 'A' if ga else 'G')]
            assert abs(gm - mAP) < 1e-6, (qa, ga, gm, mAP)
            assert abs(gc - cmc[0]) < 1e-6, (qa, ga, gc, cmc[0])


def test_a_cell_is_dropped_when_the_junk_rule_empties_it():
    """A->A is not a free win.

    Put the aerial gallery on the same camera as the aerial queries.  Under
    'sysu' the whole camera is junk, so that cell has no scorable query left --
    it must be reported as absent, never as a perfect 100%.  The headline
    number is unaffected, because those queries still have ground gallery.
    """
    ev = make_eval(*build(aerial_gallery_cam=5))
    cmc, mAP, *_, view = ev.compute()
    got = cells(view)
    assert ('A', 'A') not in got, got
    assert ('A', 'G') in got and ('G', 'A') in got and ('G', 'G') in got
    assert mAP > 0.0                       # headline still computed


def test_cells_honour_test_metric_rather_than_a_hardcoded_rule():
    """Same data, two junk rules, opposite Rank-1 in the same cell.

    If the cells silently used a fixed rule they would not be comparable to
    the mAP printed above them -- and on CARGO, which scores with 'market',
    the matrix would be quietly wrong.
    """
    data = decoy_case()
    sysu = cells(make_eval(*data, metric='sysu').compute()[-1])
    market = cells(make_eval(*data, metric='market').compute()[-1])
    assert sysu[('A', 'A')][1] == 1.0, sysu       # decoys dropped with the camera
    assert market[('A', 'A')][1] == 0.0, market   # decoy survives and outranks
    # and the rule itself is the one being described
    assert junk_mask(np.asarray([1]), np.asarray([5]), 1, 5, 'market')[0]
    assert not junk_mask(np.asarray([2]), np.asarray([5]), 1, 5, 'market')[0]


def test_empty_cell_is_skipped_not_crashed():
    """No aerial gallery at all -> the two (*, A) cells simply do not appear."""
    *_, view = make_eval(*build(drop_aerial_gallery=True)).compute()
    assert [(q, g) for q, g, _c, _m, _i in view] == [('A', 'G'), ('G', 'G')], view


# --------------------------------------------------------------------------
# 3. The readout actually discriminates -- the point of adding it.
# --------------------------------------------------------------------------
def test_matrix_separates_good_from_bad_cross_view():
    good = cells(make_eval(*build(cross_view_good=True)).compute()[-1])
    bad = cells(make_eval(*build(cross_view_good=False)).compute()[-1])
    # ground -> ground is untouched by the plant and near perfect in both
    assert good[('G', 'G')][0] > 0.99 and bad[('G', 'G')][0] > 0.99, (good, bad)
    # aerial -> ground is what collapses when the view direction dominates
    assert good[('A', 'G')][0] > 0.99, good
    assert bad[('A', 'G')][0] < 0.5, bad


def test_overall_map_can_hide_a_dead_cross_view_cell():
    """Why this readout is needed at all.

    The two runs below differ enormously in A->G, yet the headline mAP moves
    much less, because the same-view cells dominate the average.  That is
    exactly the ambiguity in "text supervision bought +0.09 mAP" that the
    matrix resolves.
    """
    _, good_map, *_r, good_view = make_eval(*build(cross_view_good=True)).compute()
    _, bad_map, *_r, bad_view = make_eval(*build(cross_view_good=False)).compute()
    cell_gap = cells(good_view)[('A', 'G')][0] - cells(bad_view)[('A', 'G')][0]
    overall_gap = good_map - bad_map
    assert cell_gap > overall_gap > 0, (cell_gap, overall_gap)


def test_cross_view_cells_are_the_hard_ones_in_the_bad_case():
    """Sanity on the direction of the effect: A->G and G->A both suffer."""
    bad = cells(make_eval(*build(cross_view_good=False)).compute()[-1])
    assert bad[('A', 'G')][0] < bad[('G', 'G')][0]
    assert bad[('G', 'A')][0] < bad[('G', 'G')][0]
    assert bad[('A', 'A')][0] > bad[('A', 'G')][0]   # same-view stays easy


# --------------------------------------------------------------------------
# 4. Plumbing.
# --------------------------------------------------------------------------
def _read(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(here, '..', *parts), encoding='utf-8').read()


def test_processor_passes_aerial_cams_at_both_sites():
    src = _read('processor', 'processor.py')
    assert src.count('aerial_cams=cfg.DATASETS.AERIAL_CAMS') == 2, \
        'do_train and do_inference must both pass it'


def test_non_rank0_return_arity_matches_compute():
    """DDP ranks other than 0 return a tuple of Nones; it has to be 8 long now,
    or `cmc, mAP, *_ = compute()` would still work but any positional consumer
    of the last element would silently read the wrong slot."""
    src = _read('utils', 'metrics.py')
    assert '(None,)*8' in src or '(None,) * 8' in src, 'stale arity on the DDP path'


def test_config_probes_are_anchored_to_real_keys():
    """Found 2026-08-13, in the results rather than in review.

    reeval.sh decided whether to inject DATASETS.AERIAL_CAMS with an unanchored
    `grep -q 'AERIAL_CAMS'`, and hihr_whu_ce_mod_g2.yml has the words
    "AERIAL_CAMS stays empty" in its header comment.  The check matched the
    COMMENT, skipped the injection, and that run came back with no view matrix
    -- while whu_ce_mod_g2_tri2, whose header happens not to contain those
    letters, got one.  Two runs of the same family scored differently because
    of a sentence in a comment.
    """
    import glob
    import re
    for path in ['reeval.sh', 'run_hihr.sh',
                 os.path.join('diag', 'dump_all.sh'),
                 os.path.join('diag', 'run_twin_probe.sh')]:
        for line in _read(*path.split(os.sep)).splitlines():
            if 'grep -q' not in line:
                continue
            for pat in re.findall(r"grep -q ['\"]([^'\"]+)['\"]", line):
                assert pat.startswith('^'), (path, pat)

    # and the config that triggered it still says the words, so the anchoring
    # is what is being relied on rather than a rewritten comment
    g2 = _read('configs', 'hihr_whu_ce_mod_g2.yml')
    assert 'AERIAL_CAMS' in g2, 'the comment was edited instead of the grep'
    assert not any(l.strip().startswith('AERIAL_CAMS:') for l in g2.splitlines())


def test_reeval_keeps_its_output():
    """A task that succeeds on this platform keeps no downloadable log, so a
    re-evaluation that only printed to stdout produced nothing at all.  Append,
    not truncate: several re-evaluations share one task."""
    src = _read('reeval.sh')
    assert 'REEVAL_LOG:-$REPO/reeval_log.txt' in src
    assert 'tee -a "$LOG"' in src
    assert src.count('tee -a "$LOG"') == 2, 'the results themselves are not teed'
    assert '> "$LOG"' not in src.replace('>> "$LOG"', ''), 'the log is truncated'


def test_reeval_records_the_setting_it_used():
    """Its whole purpose is comparing test-time settings.  Two blocks that
    differ only in NECK_FEAT are indistinguishable once the header is gone, and
    a mislabelled result is worse than a missing one."""
    src = _read('reeval.sh')
    assert 'overrides  : ${*:-<none, config defaults>}' in src
    assert 'checkpoint : $CKPT' in src


def test_reeval_injects_aerial_cams_for_older_whu_runs():
    """The pre-direction-two configs do not set AERIAL_CAMS, and they are the
    control the new runs are compared against -- so reeval has to supply it."""
    src = _read('reeval.sh')
    assert 'DATASETS.AERIAL_CAMS' in src
    assert 'View' in src, 'the output filter would swallow the matrix'


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d view-matrix tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
