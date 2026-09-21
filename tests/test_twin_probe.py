"""Unit tests for the twin-InfoNCE feasibility probe.

The probe's whole job is to answer one question before a GPU day is spent on
it: can a single constant per spectrum still buy a large reduction in this
loss?  If the answer it gives is wrong, we either throw away a design that
would have worked or run a design that cannot.  So the tests plant both
answers and check the probe recovers each:

    a displacement that IS a shared constant   -> the search must find it and
                                                  the loss must collapse
    a displacement that is pure per-capture     -> the search must find almost
    scatter                                       nothing

plus the ordinary correctness of the masking and the ranking, where an error
would show up as a plausible number rather than as a crash.

Run:  python tests/test_twin_probe.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diag.analyse_modality import group_captures                  # noqa: E402
from diag.twin_infonce_probe import (batch_logits, infonce,        # noqa: E402
                                     mean_shift, run, sample_batches,
                                     search_shift)

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


# --------------------------------------------------------------------------
# fixture: 16 identities x 4 captures x 3 spectra, laid out the way the probe
# indexes -- row = capture * n_mod + modality
# --------------------------------------------------------------------------
DIM, N_PID, N_INST, N_MOD = 24, 16, 4, 3
N_CAP = N_PID * N_INST


# The fixture has to live in the real regime, not just be "some vectors".
# WHU-MARS features sit in a narrow cone: two different people are at cosine
# 0.982 and a twin pair at 0.988.  With base vectors drawn isotropically the
# twin is trivially the nearest of 64 candidates, the loss starts at 0.0003,
# and a "40% reduction" of that is noise -- which is exactly how the first
# version of these tests reported a false positive.
#
#   CONE  spread of the identity cloud;  cos(stranger) ~ 1 / (1 + CONE^2)
#   SCALE length of the displacement;    cos(twin)     ~ 1 / sqrt(1 + SCALE^2)
#
# 0.135 and 0.156 put those at 0.982 and 0.988, matching the measurement.
CONE = 0.135


def planted(mode, scale=0.156, seed=0, n_batch=8, cone=None):
    """Build (gf, batches) with a known kind of displacement.

    'constant'  every capture's IR/TIR sits at its RGB plus one fixed vector
    'scatter'   the displacement is a fresh random direction per capture, of
                the same length, so the two cases are equally far apart per
                image and differ only in whether the offsets agree
    """
    g = torch.Generator().manual_seed(seed)
    cone = CONE if cone is None else cone
    # cone=0 puts the cloud at the origin, which removes the centring
    # bonus entirely -- see RECOVERY below for why that matters.
    mu = (F.normalize(torch.randn(DIM, generator=g), dim=-1) if cone else
          torch.zeros(DIM))
    const = F.normalize(torch.randn(N_MOD, DIM, generator=g), dim=-1) * scale
    const[0] = 0.0

    # n_batch batches' worth of captures.  The shift search fits D free
    # parameters per spectrum on these, so too few captures lets it overfit --
    # with a single batch of 64 the search "found" a 0.177 offset in data that
    # had none.
    rows, pids, feats = [], [], []
    for cap in range(N_CAP * n_batch):
        base = mu + max(cone, 1.0 - bool(cone)) * F.normalize(
            torch.randn(DIM, generator=g), dim=-1)
        row = []
        for m in range(N_MOD):
            if m == 0:
                d = torch.zeros(DIM)
            elif mode == 'constant':
                d = const[m]
            else:
                d = F.normalize(torch.randn(DIM, generator=g), dim=-1) * scale
            row.append(len(feats))
            feats.append(base + d)
        rows.append(row)
        pids.append(cap // N_INST)
    gf = torch.stack(feats)
    batches = [(np.asarray(rows[k:k + N_CAP], dtype=np.int64),
                np.asarray(pids[k:k + N_CAP], dtype=np.int64))
               for k in range(0, len(rows), N_CAP)]
    return gf, batches


# --------------------------------------------------------------------------
# 1. the decisive property
# --------------------------------------------------------------------------
TAU = 0.006          # the temperature the cone geometry calls for


def test_the_fixture_lands_in_the_regime_the_real_data_is_in():
    """If the loss starts near zero the fixture is trivial and every other
    test below is measuring rounding.  Pinned rather than assumed: this is the
    bug the first version of this file had."""
    for mode in ('constant', 'scatter'):
        gf, b = planted(mode)
        loss, rank1, _m = run(gf, b, TAU)
        assert 0.3 < float(loss) < 4.2, (mode, float(loss))
        assert rank1 < 0.999, (mode, rank1)


# The recovery tests below deliberately do NOT use the cone fixture.  With the
# cloud sitting near one direction, plain centring drives the loss to almost
# zero on its own, the per-spectrum search is left with no gradient, and what
# it returns is drift -- measured at 0.071 in data with no offset planted at
# all, and at cosine -0.39 against an offset that was.  An isotropic cloud has
# no centring bonus to collect, so the search is judged on the only thing these
# tests can hold it to: does it find what was planted.
RECOVERY = dict(cone=0.0, scale=1.0)
TAU_REC = 0.2


def differential(mode, seed=0, **kw):
    """What the search recovers as the PER-SPECTRUM part, i.e. c_m - c_0.

    Judged on this rather than on how much loss the constant buys, because a
    constant lowers this loss two different ways and only one is the failure
    mode.  Subtracting something near the cloud's centre decompresses the cone
    and makes the twin the nearest candidate almost for free, whatever the
    spectra are doing -- an earlier version of these tests measured exactly
    that and reported ~100% for both planted cases.  The common part cancels
    in c_m - c_0, so this quantity sees only the modality.
    """
    kw = dict(RECOVERY, **kw)
    gf, b = planted(mode, seed=seed, **kw)
    sh = search_shift(gf, b, TAU_REC, N_MOD, steps=300, lr=0.05, shared=True)
    ss = search_shift(gf, b, TAU_REC, N_MOD, steps=300, lr=0.05, init=sh)
    return torch.stack([ss[m] - ss[0] for m in range(1, N_MOD)])


def planted_constants(seed=0):
    """The `const` rows planted by `planted('constant')`, rebuilt from the same
    generator sequence so the test compares against the real target."""
    g = torch.Generator().manual_seed(seed)
    const = F.normalize(torch.randn(N_MOD, DIM, generator=g), dim=-1) * RECOVERY['scale']
    const[0] = 0.0
    return const


def test_a_planted_per_spectrum_offset_is_recovered():
    """The property the whole probe rests on: if the displacement really is a
    shared constant, the search finds it."""
    got = differential('constant')
    want = planted_constants()
    for m in range(1, N_MOD):
        cos = float(F.cosine_similarity(got[m - 1], want[m], dim=0))
        assert cos > 0.8, (m, cos)
        ratio = float(got[m - 1].norm()) / float(want[m].norm())
        assert 0.3 < ratio < 3.0, (m, ratio)


def test_pure_scatter_yields_almost_no_per_spectrum_offset():
    """The case the probe exists to detect: the same per-image displacement
    length, but the directions disagree, so there is no constant to find."""
    got = differential('scatter')
    for m in range(N_MOD - 1):
        assert float(got[m].norm()) < 0.2 * RECOVERY['scale'], (m, float(got[m].norm()))


def test_the_two_cases_are_actually_distinguishable():
    """Guards against a probe that reports 'nothing found' for everything,
    which would pass the scatter test for the wrong reason."""
    c = float(differential('constant').norm(dim=-1).mean())
    s = float(differential('scatter').norm(dim=-1).mean())
    assert c > 4 * s, (c, s)


def test_the_shared_search_cannot_beat_the_per_spectrum_one():
    """(c) is warm-started from (b), so it can only improve on it.  A negative
    'modal share' would mean the warm start was dropped."""
    gf, b = planted('constant')
    sh = search_shift(gf, b, TAU, N_MOD, steps=200, lr=0.02, shared=True)
    ss = search_shift(gf, b, TAU, N_MOD, steps=200, lr=0.02, init=sh)
    l_sh, _r, _m = run(gf, b, TAU, shift=sh)
    l_ss, _r, _m = run(gf, b, TAU, shift=ss)
    assert float(l_ss) <= float(l_sh) + 1e-4, (float(l_ss), float(l_sh))
    assert sh.shape == ss.shape == (N_MOD, DIM)
    assert torch.allclose(sh[0], sh[1]), 'shared=True must give one vector'


def test_the_shared_constant_alone_already_decompresses_the_cone():
    """Names the confound rather than leaving it to be rediscovered: plain
    centring lowers this loss a lot in BOTH planted cases, which is why the
    verdict is read off the per-spectrum part alone."""
    for mode in ('constant', 'scatter'):
        gf, b = planted(mode)
        base, _r, _m = run(gf, b, TAU)
        sh = search_shift(gf, b, TAU, N_MOD, steps=300, lr=0.02, shared=True)
        l_sh, _r, _m = run(gf, b, TAU, shift=sh)
        assert float(l_sh) < 0.6 * float(base), (mode, float(base), float(l_sh))


def test_the_searched_shift_is_no_worse_than_the_mean_one():
    """The mean displacement minimises squared distance, not this loss.  The
    search exists so the 'lazy constant' gets its best shot, and it has to be
    at least as good as the closed-form answer or the search is broken."""
    gf, batches = planted('constant')
    l_mean, _r, _m = run(gf, batches, TAU, shift=mean_shift(gf, batches, N_MOD))
    sh = search_shift(gf, batches, TAU, N_MOD, steps=300, lr=0.02, shared=True)
    l_srch, _r, _m = run(gf, batches, TAU,
                         shift=search_shift(gf, batches, TAU, N_MOD, steps=300,
                                            lr=0.02, init=sh))
    assert float(l_srch) <= float(l_mean) + 1e-3, (float(l_srch), float(l_mean))


# --------------------------------------------------------------------------
# 2. masking and ranking
# --------------------------------------------------------------------------
def test_same_identity_non_twins_are_excluded_from_the_denominator():
    """The other captures of the same person are neither positive nor
    negative.  Counting them as negatives would push a person's own frames
    apart, in direct conflict with the ID loss."""
    q = F.normalize(torch.randn(6, 5), dim=-1)
    r = F.normalize(torch.randn(6, 5), dim=-1)
    pid = np.array([0, 0, 0, 1, 1, 1])
    logits = batch_logits(q, r, pid, 0.1)
    for i in range(6):
        for j in range(6):
            same = pid[i] == pid[j]
            if same and i != j:
                assert logits[i, j] == float('-inf'), (i, j)
            else:
                assert torch.isfinite(logits[i, j]), (i, j)


def test_infonce_matches_a_hand_computation():
    q = F.normalize(torch.randn(4, 5), dim=-1)
    r = F.normalize(torch.randn(4, 5), dim=-1)
    pid = np.array([0, 1, 2, 3])                  # all distinct: nothing masked
    tau = 0.2
    loss, _r1, _m = infonce(q, r, pid, tau)
    logits = (q @ r.t()) / tau
    want = float(torch.nn.functional.cross_entropy(
        logits, torch.arange(4)))
    assert abs(float(loss) - want) < 1e-6, (float(loss), want)


def test_rank1_is_one_when_every_twin_is_the_nearest():
    v = F.normalize(torch.randn(5, 6), dim=-1)
    pid = np.arange(5)
    _l, rank1, margin = infonce(v, v, pid, 0.1)   # each row's twin is itself
    assert rank1 == 1.0, rank1
    assert margin > 0, margin


def test_rank1_falls_when_the_twins_are_shuffled():
    v = F.normalize(torch.randn(8, 6), dim=-1)
    pid = np.arange(8)
    _l, good, _m = infonce(v, v, pid, 0.1)
    _l, bad, _m = infonce(v, v.flip(0), pid, 0.1)
    assert good == 1.0 and bad < 0.5, (good, bad)


def test_margin_is_reported_in_cosine_units():
    """The log prints it next to cosines measured elsewhere, so it must not
    carry the temperature."""
    v = F.normalize(torch.randn(6, 7), dim=-1)
    pid = np.arange(6)
    _l, _r, m1 = infonce(v, v, pid, 0.01)
    _l, _r, m2 = infonce(v, v, pid, 0.1)
    assert abs(m1 - m2) < 1e-6, (m1, m2)


# --------------------------------------------------------------------------
# 3. batch construction
# --------------------------------------------------------------------------
def _by_pid(n_pid=20, n_cap=6, complete=True):
    by, row = {}, 0
    for pid in range(n_pid):
        by[pid] = {}
        for c in range(n_cap):
            mods = {}
            for m in range(N_MOD if complete or c else N_MOD - 1):
                mods[m] = row
                row += 1
            by[pid][(0, c)] = mods
    return by


def test_sample_batches_has_the_shape_the_sampler_produces():
    b = sample_batches(_by_pid(), np.random.RandomState(0), 3)
    assert len(b) == 3
    for rows, pid in b:
        assert rows.shape == (N_PID * N_INST, N_MOD), rows.shape
        assert len(set(pid.tolist())) == N_PID, set(pid.tolist())
        for p in set(pid.tolist()):
            assert int((pid == p).sum()) == N_INST


def test_sample_batches_drops_incomplete_captures():
    """A capture missing a spectrum has no twin.  Including it would pair the
    query with some other capture's reference row and quietly measure the
    wrong thing."""
    by = _by_pid(complete=False)                  # capture 0 of every pid short
    b = sample_batches(by, np.random.RandomState(0), 1)
    rows = b[0][0]
    bad = {min(v.values()) for pid in by for k, v in by[pid].items()
           if len(v) < N_MOD}
    assert not (set(rows.flatten().tolist()) & bad)


def test_sample_batches_refuses_a_dataset_it_cannot_fill():
    try:
        sample_batches(_by_pid(n_pid=4), np.random.RandomState(0), 1)
    except ValueError as e:
        assert 'identities' in str(e), e
        return
    raise AssertionError('a too-small dataset was accepted')


def test_group_captures_is_the_one_shared_with_the_decomposition():
    """Both scripts define a twin as an entry of this grouping; two copies
    would be two chances to disagree about what a twin is."""
    d = {'g_frames': np.array([0, 0, 0, 1, -1]),
         'g_pids': np.array([7, 7, 7, 7, 7]),
         'g_camids': np.array([2, 2, 2, 2, 2]),
         'g_mids': np.array([1, 2, 3, 1, 1])}
    by = group_captures(d)
    assert set(by) == {7}
    assert by[7][(2, 0)] == {1: 0, 2: 1, 3: 2}
    assert by[7][(2, 1)] == {1: 3}
    assert len(by[7]) == 2                        # the frame -1 row is dropped


# --------------------------------------------------------------------------
# 4. the probe follows its input's device
# --------------------------------------------------------------------------
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def test_every_helper_stays_on_the_device_it_was_given():
    """The probe is meant to be launched as a GPU task -- the shift search is
    400 optimiser steps over every sampled batch, which is minutes on a CPU.
    One `torch.eye` or `torch.arange` without a device would raise only on
    CUDA, i.e. only on the machine we cannot debug on interactively."""
    for dev in DEVICES:
        gf, b = planted('constant', n_batch=2)
        gf = gf.to(dev)
        loss, _r, _m = run(gf, b, TAU)
        assert loss.device.type == dev, (dev, loss.device)
        ms = mean_shift(gf, b, N_MOD)
        assert ms.device.type == dev, (dev, ms.device)
        ss = search_shift(gf, b, TAU, N_MOD, steps=5, shared=True)
        assert ss.device.type == dev, (dev, ss.device)
        after, _r, _m = run(gf, b, TAU, shift=ss)
        assert torch.isfinite(after), (dev, after)


def test_cpu_and_cuda_agree():
    """Only meaningful where a GPU exists; on the dev machine it is a no-op.
    Tolerance is loose because the chunked reductions reorder differently, and
    a real logic error is orders of magnitude larger than that."""
    if 'cuda' not in DEVICES:
        return
    gf, b = planted('constant', n_batch=2)
    a, _r, _m = run(gf, b, TAU)
    c, _r, _m = run(gf.cuda(), b, TAU)
    assert abs(float(a) - float(c)) < 1e-3, (float(a), float(c))


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d twin-probe tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
