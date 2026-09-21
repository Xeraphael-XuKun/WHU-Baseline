"""Unit tests for the thermal-gap diagnostics.

The diagnostics decide which project we do next -- centring a translation, or a
real cross-modal method -- so a plausible-looking wrong answer is expensive.
Each test below plants a geometry whose correct answer is known in advance and
checks the code recovers it:

  * a pure translation between modality clouds, which centring must remove,
  * two clouds with the same mean but different shape, which centring must NOT
    help, so the diagnostic cannot claim a translation that is not there,
  * a noise-only case, where the measured mean offset must be reported as
    indistinguishable from the sampling floor.

Run:  python tests/test_diag_modality.py
"""
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag.analyse_modality import (CONDITIONS, ap_by_gallery_modality,  # noqa: E402
                                   capture_decomposition, cell, eval_stream,
                                   features, mean_offsets, pair_distances,
                                   pooled, report)

# Every equivalence test runs on both devices when a GPU is present: the whole
# point of the vectorised evaluator is that it is used on the GPU, and a CPU
# only check would not exercise the path that produces the reported numbers.
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def _tol(dev):
    """Device-aware comparison tolerance.

    The CUDA path computes distances and rankings in float32 and its chunked
    reductions reorder the summations, so the same computation differs from
    the CPU reference by up to ~5e-5 on the A800 (measured 2026-08-06).  CPU
    keeps the strict float64-style bound; CUDA gets 1e-3 so float noise never
    fails the test while a real logic error (orders of magnitude larger)
    still does.
    """
    return 1e-3 if dev == 'cuda' else 1e-9


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
# fixture
# --------------------------------------------------------------------------
DIM = 64
N_ID = 40
CAMS = (0, 1, 2)
MODS = (1, 2, 3)

# One fixed direction per modality, shared by every build() call so the plants
# are comparable.  Crucially these are *random* directions, not private axes:
# a shift orthogonal to the identity subspace is common to every gallery
# candidate in a cell and therefore cannot reorder anything, so it would be a
# translation that does no harm and proves nothing.
_U = np.random.RandomState(999).randn(len(MODS), DIM)
_U /= np.linalg.norm(_U, axis=1, keepdims=True)


def build(offset=0.0, shape_noise=None, seed=0):
    """Identity i sits on direction e_i (no two identities share one).

    Modality m adds `offset` along its own random direction, and/or extra
    isotropic noise (`shape_noise[m]`).  The offset is a translation and
    centring must remove it; the noise leaves the means equal and centring
    must not help.
    """
    rng = np.random.RandomState(seed)
    shape_noise = shape_noise or {}
    q, g = [], []
    for pid in range(N_ID):
        base = np.zeros(DIM)
        base[pid] = 1.0
        for mi, m in enumerate(MODS):
            shift = offset * _U[mi]
            for cam in CAMS:
                jitter = 0.02 * rng.randn(DIM)
                extra = shape_noise.get(m, 0.0) * rng.randn(DIM)
                f = base + shift + jitter + extra
                (q if cam == 0 else g).append((f, pid, cam, m))
            # A gallery copy on the query's own camera, near-identical to the
            # query so it ranks first and is then junked.  Without it nothing
            # is ever removed from the top of the ranking, keep.cumsum starts
            # at 1 everywhere, and a division by zero in the AP goes unnoticed
            # -- which is precisely how nan reached the real data.
            g.append((base + shift + 0.001 * rng.randn(DIM), pid, 0, m))
    def pack(rows):
        return (np.asarray([r[0] for r in rows], dtype=np.float32),
                np.asarray([r[1] for r in rows], dtype=np.int64),
                np.asarray([r[2] for r in rows], dtype=np.int64),
                np.asarray([r[3] for r in rows], dtype=np.int64))
    qf, q_pids, q_camids, q_mids = pack(q)
    gf, g_pids, g_camids, g_mids = pack(g)
    return dict(qf=qf, q_pids=q_pids, q_camids=q_camids, q_mids=q_mids,
                gf=gf, g_pids=g_pids, g_camids=g_camids, g_mids=g_mids,
                feat_norm=np.array(True), metric='sysu',
                neck_feat=np.array('after'), label='synthetic',
                modalities=['RGB', 'IR', 'Thermal'],
                aerial_cams=np.asarray([], dtype=np.int64))


def centred(d, means):
    # features() normalises in place and hands back the only copy, so a test
    # that wants a before/after comparison has to clone here.
    qf, gf = features(d, feat_norm=True)
    qf, gf = qf.clone(), gf.clone()
    for m, mu in means.items():
        mu = torch.from_numpy(mu).float()
        qf[torch.from_numpy(d['q_mids'] == m)] -= mu
        gf[torch.from_numpy(d['g_mids'] == m)] -= mu
    return (torch.nn.functional.normalize(qf, dim=1, p=2),
            torch.nn.functional.normalize(gf, dim=1, p=2))


def cross_cells(qf, gf, d):
    out = []
    for a in MODS:
        for b in MODS:
            if a == b:
                continue
            _c, m = cell(qf, gf, d, np.where(d['q_mids'] == a)[0],
                         np.where(d['g_mids'] == b)[0])
            if m is not None:
                out.append(m)
    return float(np.mean(out))


# --------------------------------------------------------------------------
# 1. a planted translation
# --------------------------------------------------------------------------
def test_translation_is_detected_above_the_noise_floor():
    d = build(offset=3.0)
    means, floors, mods = mean_offsets(d, np.random.RandomState(0))
    assert mods == list(MODS), mods
    for a in MODS:
        for b in MODS:
            if a >= b:
                continue
            off = np.linalg.norm(means[a] - means[b])
            # 3x is the threshold the report itself prints "real" at; the test
            # has to agree with it or the two would disagree on live data.
            assert off > 3 * max(floors[a], floors[b]), (a, b, off, floors)


def test_centring_removes_a_planted_translation():
    # 3.0 rather than something small: a per-modality translation is *common
    # to every candidate inside a cell*, so it only reorders through its
    # interaction with the identity spread.  Measured on this fixture, an
    # offset of 1.0 costs nothing at all and 2.0 costs 8 points -- it takes 3.0
    # to break retrieval properly.  Worth remembering when reading live data:
    # a translation has to be large before a restricted-gallery cell notices.
    d = build(offset=3.0)
    qf, gf = features(d)
    before = cross_cells(qf, gf, d)
    means, _f, _m = mean_offsets(d, np.random.RandomState(0))
    after = cross_cells(*centred(d, means), d)
    assert after > before + 0.10, (before, after)


# --------------------------------------------------------------------------
# 2. same mean, different shape -- centring must NOT be credited
# --------------------------------------------------------------------------
def test_shape_difference_is_not_reported_as_a_translation():
    d = build(offset=0.0, shape_noise={3: 0.8})
    means, floors, _m = mean_offsets(d, np.random.RandomState(0))
    off = np.linalg.norm(means[1] - means[3])
    assert off < 3 * max(floors[1], floors[3]), (off, floors)


def test_centring_does_not_rescue_a_shape_difference():
    d = build(offset=0.0, shape_noise={3: 0.8})
    qf, gf = features(d)
    before = cross_cells(qf, gf, d)
    means, _f, _m = mean_offsets(d, np.random.RandomState(0))
    after = cross_cells(*centred(d, means), d)
    assert abs(after - before) < 0.05, (before, after)


def test_the_two_cases_are_actually_distinguishable():
    """Both plants must hurt cross-modality retrieval in the first place,
    otherwise the two tests above would pass for the trivial reason that
    nothing was broken."""
    clean = build(offset=0.0)
    shifted = build(offset=3.0)
    shaped = build(offset=0.0, shape_noise={3: 0.8})
    base = cross_cells(*features(clean), clean)
    assert base > 0.9, base
    assert cross_cells(*features(shifted), shifted) < base - 0.10
    assert cross_cells(*features(shaped), shaped) < base - 0.10


# --------------------------------------------------------------------------
# 3. noise floor honesty
# --------------------------------------------------------------------------
def test_no_offset_reads_as_indistinguishable_from_noise():
    d = build(offset=0.0)
    means, floors, _m = mean_offsets(d, np.random.RandomState(0))
    for a in MODS:
        for b in MODS:
            if a >= b:
                continue
            off = np.linalg.norm(means[a] - means[b])
            assert off < 3 * max(floors[a], floors[b]), \
                ('would be reported as a real offset', a, b, off, floors)


def test_noise_floor_shrinks_as_the_sample_grows():
    """The floor is a sampling quantity: more images per modality, smaller
    floor.  If it did not move with n it would not be a floor."""
    small = build(offset=0.0, seed=1)
    _m, f_small, _ = mean_offsets(small, np.random.RandomState(0))
    big = build(offset=0.0, seed=2)
    for k in ('gf', 'g_pids', 'g_camids', 'g_mids'):
        big[k] = np.concatenate([big[k]] * 4, axis=0)
    _m2, f_big, _ = mean_offsets(big, np.random.RandomState(0))
    assert np.mean(list(f_big.values())) < np.mean(list(f_small.values())) * 0.8, \
        (f_small, f_big)


# --------------------------------------------------------------------------
# 4. d_same / d_cross / d_diffpid
# --------------------------------------------------------------------------
def test_pair_distances_order_on_a_clean_case():
    d = build(offset=0.0)
    same, cross, diff, n = pair_distances(d, np.random.RandomState(0), num_pids=30)
    assert n > 0
    ds = np.mean(list(same.values()))
    assert max(cross.values()) < min(diff.values()), (cross, diff)
    assert ds < min(diff.values())


def test_pair_distances_flags_a_lost_identity():
    """With a large enough modality shift, two images of the same person in
    different spectra are further apart than two different people in the same
    spectrum -- the condition under which no ranking method can help."""
    d = build(offset=3.0)
    _same, cross, diff, _n = pair_distances(d, np.random.RandomState(0), num_pids=30)
    assert max(cross.values()) > min(diff.values()), (cross, diff)


# --------------------------------------------------------------------------
# 4b. spectrum cost vs pose cost
# --------------------------------------------------------------------------
def _capture_gallery(spectrum_cost, pose_cost, n_id=12, n_cam=2, n_frame=3,
                     dim=32, seed=0):
    """A gallery with the two costs planted separately and known exactly.

    Identity i sits on e_i.  Changing the FRAME adds `pose_cost` along a
    frame-specific direction, changing the CAMERA along a camera-specific one,
    changing the SPECTRUM adds `spectrum_cost` along a spectrum-specific one.
    The decomposition has to tell them apart.
    """
    rng = np.random.RandomState(seed)
    axes = rng.randn(3, 8, dim)
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    feats, pids, cams, mids, frames = [], [], [], [], []
    for pid in range(n_id):
        base = np.zeros(dim)
        base[pid % (dim - 8)] = 1.0
        for cam in range(n_cam):
            for fr in range(n_frame):
                for m in (1, 2, 3):
                    v = (base + pose_cost * axes[0][cam] + pose_cost * axes[1][fr]
                         + spectrum_cost * axes[2][m])
                    feats.append(v + 1e-4 * rng.randn(dim))
                    pids.append(pid); cams.append(cam); mids.append(m)
                    frames.append(fr)
    dim_ = dim
    return dict(qf=np.zeros((1, dim_), dtype=np.float32),
                q_pids=np.zeros(1, dtype=np.int64),
                q_camids=np.zeros(1, dtype=np.int64),
                q_mids=np.ones(1, dtype=np.int64),
                gf=np.asarray(feats, dtype=np.float32),
                g_pids=np.asarray(pids, dtype=np.int64),
                g_camids=np.asarray(cams, dtype=np.int64),
                g_mids=np.asarray(mids, dtype=np.int64),
                g_frames=np.asarray(frames, dtype=np.int64),
                feat_norm=np.array(True), metric='sysu',
                neck_feat=np.array('after'), label='synthetic',
                modalities=['RGB', 'IR', 'Thermal'],
                aerial_cams=np.asarray([], dtype=np.int64))


def _repo_file(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(here, '..', *parts), encoding='utf-8').read()


def _decomp(**kw):
    d = _capture_gallery(**kw)
    return capture_decomposition(d, np.random.RandomState(0), num_pids=12, per_pid=6)


def test_decomposition_isolates_a_spectrum_only_cost():
    """Spectrum expensive, pose free: every cross-modality cell lights up and
    the same-modality row stays at zero in every condition."""
    r = _decomp(spectrum_cost=0.6, pose_cost=0.0)
    assert r is not None
    for c in CONDITIONS:
        assert r[((1, 3), c)] > 0.3, (c, r[((1, 3), c)])
        if c == 'twin':
            continue          # same modality + same capture is the same image
        assert r[('same', c)] < 0.02, (c, r[('same', c)])


def test_decomposition_isolates_a_pose_only_cost():
    """The reverse, and the case that matters here: with the spectrum free the
    twin cell must be ~0 while the other-camera cells are large, and crossing
    the spectrum must cost nothing on top of the pose."""
    r = _decomp(spectrum_cost=0.0, pose_cost=0.6)
    assert r[((1, 3), 'twin')] < 0.02, r[((1, 3), 'twin')]
    assert r[((1, 3), 'diff_cam')] > 0.3, r[((1, 3), 'diff_cam')]
    assert r[('same', 'diff_cam')] > 0.3, r[('same', 'diff_cam')]
    assert abs(r[((1, 3), 'diff_cam')] - r[('same', 'diff_cam')]) < 0.05, r


def test_twin_cells_exist_and_the_same_modality_twin_does_not():
    """A same-modality twin would be the identical image, so that cell must be
    absent; every cross-modality twin cell must be populated."""
    r = _decomp(spectrum_cost=0.3, pose_cost=0.3)
    assert ('same', 'twin') not in r['counts'], r['counts']
    for pair in ((1, 2), (1, 3), (2, 3)):
        assert r['counts'][(pair, 'twin')] > 0, (pair, r['counts'])


def test_decomposition_reports_unavailable_without_frames():
    """Older npz files carry no frame index; the section has to say so rather
    than invent a pairing from row order."""
    d = _capture_gallery(0.3, 0.3)
    del d['g_frames']
    assert capture_decomposition(d, np.random.RandomState(0)) is None
    d2 = _capture_gallery(0.3, 0.3)
    d2['g_frames'] = np.full_like(d2['g_frames'], -1)
    assert capture_decomposition(d2, np.random.RandomState(0)) is None


def test_different_identity_stays_the_ceiling():
    r = _decomp(spectrum_cost=0.2, pose_cost=0.2)
    same_id = [v for k, v in r.items()
               if isinstance(k, tuple) and len(k) == 2 and isinstance(v, float)
               and v == v]
    assert r['d_diffpid'] > max(same_id), (r['d_diffpid'], max(same_id))


def test_a_single_identity_does_not_crash_the_ceiling():
    """--num-pids 1 leaves no different-identity pair to sample.  Report nan and
    still produce the table, rather than dying inside numpy's sampler."""
    r = _decomp(spectrum_cost=0.2, pose_cost=0.2)
    assert r['d_diffpid'] == r['d_diffpid']
    d = _capture_gallery(0.2, 0.2)
    one = capture_decomposition(d, np.random.RandomState(0), num_pids=1, per_pid=2)
    assert one is not None and one['d_diffpid'] != one['d_diffpid']
    assert one[((1, 3), 'twin')] == one[((1, 3), 'twin')]


def test_dump_stores_the_frame_index():
    src = _repo_file('diag', 'dump_features.py')
    assert 'g_frames' in src and 'FRAME_RE' in src


# --------------------------------------------------------------------------
# 5. the AP decomposition
# --------------------------------------------------------------------------
def test_ap_decomposition_penalises_the_shifted_modality():
    d = build(offset=3.0)
    qf, gf = features(d)
    ap = ap_by_gallery_modality(qf, gf, d)
    for qm in MODS:
        same_mod = ap[(qm, qm)][0]
        for gm in MODS:
            if gm == qm:
                continue
            assert ap[(qm, gm)][0] < same_mod, (qm, gm, ap[(qm, gm)], same_mod)


def test_ap_decomposition_counts_every_query_that_has_a_positive():
    d = build(offset=0.0)
    qf, gf = features(d)
    ap = ap_by_gallery_modality(qf, gf, d)
    for qm in MODS:
        for gm in MODS:
            _v, n = ap[(qm, gm)]
            assert n == N_ID * 1, (qm, gm, n)   # one query per (pid, modality)


def test_ap_is_bounded_and_perfect_when_retrieval_is_perfect():
    """Every cell must read 1.00 on clean data.

    This is the test that caught the original definition, which left the
    query's other-modality positives in the ranking: six correct images cannot
    all sit in the top two ranks, so perfect retrieval scored 0.53 and every
    real reading would have been compared against the wrong baseline.
    """
    d = build(offset=0.0)
    qf, gf = features(d)
    ap = ap_by_gallery_modality(qf, gf, d)
    for k, (v, _n) in ap.items():
        assert 0.0 <= v <= 1.0, (k, v)
        assert v > 0.999, (k, v)


# --------------------------------------------------------------------------
# 6. the reproduction gate
# --------------------------------------------------------------------------
def test_the_fixture_really_junks_the_top_of_the_ranking():
    """Guards the guard: if this stops holding, the finiteness test below
    passes for the wrong reason."""
    from diag.analyse_modality import _kept, ranking_chunks
    d = build(offset=1.0)
    qf, gf = features(d)
    g_pid = torch.as_tensor(d['g_pids'], dtype=torch.int32)
    g_cam = torch.as_tensor(d['g_camids'], dtype=torch.int32)
    q_pid = torch.as_tensor(d['q_pids'], dtype=torch.int32)
    q_cam = torch.as_tensor(d['q_camids'], dtype=torch.int32)
    for start, order in ranking_chunks(qf, gf, 32, torch.device('cpu')):
        n = order.shape[0]
        _match, keep = _kept(order, g_pid, g_cam,
                             q_pid[start:start + n], q_cam[start:start + n], 'sysu')
        assert not bool(keep[:, 0].all()), 'no query has a junked top-1'
        return


def test_ap_is_finite_when_the_top_of_the_ranking_is_junked():
    d = build(offset=1.0)
    qf, gf = features(d)
    for dev in DEVICES:
        cmc, mAP = eval_stream(qf, gf, d['q_pids'], d['g_pids'], d['q_camids'],
                               d['g_camids'], metric='sysu', chunk=16, device=dev)
        assert np.isfinite(mAP), (dev, mAP)
        assert np.isfinite(cmc).all(), dev
        assert 0.0 < mAP <= 1.0, (dev, mAP)


def test_pooled_matches_a_hand_built_eval():
    from utils.metrics import compute_indices_chunked, eval_func
    d = build(offset=0.5)
    qf, gf = features(d)
    mAP, cmc = pooled(qf, gf, d)
    idx = compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=512)
    cmc2, mAP2, _ = eval_func(idx, d['q_pids'], d['g_pids'], d['q_camids'],
                           d['g_camids'], max_rank=50, metric='sysu')
    assert abs(mAP - mAP2) < 1e-9 and abs(cmc[0] - cmc2[0]) < 1e-9


def test_features_honours_feat_norm():
    # Two builds, not two calls on one: the normalisation is in place, so a
    # second call on the same dict cannot undo it and never claims to.
    qn, _g = features(build(), feat_norm=True)
    assert torch.allclose(qn.norm(dim=1), torch.ones(qn.shape[0]), atol=1e-5)
    qr, _g = features(build(), feat_norm=False)
    assert not torch.allclose(qr.norm(dim=1), torch.ones(qr.shape[0]), atol=1e-3)


def test_features_shares_storage_with_the_loaded_array():
    """The dev machine ran out of memory on the first attempt; the fix was to
    stop copying.  If this ever silently starts copying again, the analysis
    will OOM on the real 290 MB gallery rather than here."""
    d = build()
    qf, _gf = features(d, feat_norm=True)
    assert qf.data_ptr() == d['qf'].__array_interface__['data'][0]


def test_eval_stream_matches_eval_func_exactly():
    """The vectorised evaluator exists to make re-analysis cheap.  If it
    disagreed with eval_func by even a little, every number this script prints
    would be incomparable with the ones in view_matrix.txt -- which is exactly
    what section [0] claims to check.

    Both junk rules, several chunk sizes, and every available device: the
    uncompacted cumsum trick and the float64 accumulation are the two places
    this could silently drift, and neither is device specific, but the reported
    numbers will come off the GPU.  CUDA floats are float32, so that device is
    compared with _tol('cuda') instead of the CPU's bit-exact bound.
    """
    from utils.metrics import compute_indices_chunked, eval_func
    for metric in ('sysu', 'market'):
        for offset in (0.0, 1.0, 3.0):
            d = build(offset=offset)
            qf, gf = features(d)
            idx = compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=4096)
            want_cmc, want_mAP, _ = eval_func(idx, d['q_pids'], d['g_pids'],
                                           d['q_camids'], d['g_camids'],
                                           max_rank=50, metric=metric)
            for dev in DEVICES:
                for chunk in (7, 64):
                    got_cmc, got_mAP = eval_stream(
                        qf, gf, d['q_pids'], d['g_pids'], d['q_camids'],
                        d['g_camids'], metric=metric, max_rank=50,
                        chunk=chunk, device=dev)
                    assert abs(got_mAP - want_mAP) < _tol(dev),                         (metric, offset, dev, chunk, got_mAP, want_mAP)
                    assert np.allclose(got_cmc, want_cmc, atol=_tol(dev)),                         (metric, offset, dev, chunk)


def test_eval_stream_is_chunk_invariant():
    """Chunking is a memory device, not a parameter.  Any dependence on it
    would mean a query's score leaked across chunk boundaries."""
    d = build(offset=1.0)
    qf, gf = features(d)
    ref = None
    for dev in DEVICES:
        for chunk in (1, 3, 40, 10000):
            cmc, mAP = eval_stream(qf, gf, d['q_pids'], d['g_pids'],
                                   d['q_camids'], d['g_camids'],
                                   chunk=chunk, device=dev)
            if ref is None:
                ref = (cmc, mAP)
            assert abs(mAP - ref[1]) < _tol(dev), (dev, chunk, mAP, ref[1])
            assert np.array_equal(cmc, ref[0]), (dev, chunk)


def _ap_reference(qf, gf, d):
    """The straightforward per-query loop the vectorised version replaced.

    Kept here rather than in the script: it is the definition, and the fast
    path is only trustworthy while it agrees with it.
    """
    from utils.metrics import compute_indices_chunked, junk_mask
    mods = sorted(set(d['g_mids'].tolist()))
    idx = compute_indices_chunked(qf, gf, top_k=0, q_chunk_size=4096)
    acc = {(qm, gm): [] for qm in sorted(set(d['q_mids'].tolist())) for gm in mods}
    for qi in range(len(d['q_pids'])):
        order = idx[qi]
        keep = ~junk_mask(d['g_pids'][order], d['g_camids'][order],
                          d['q_pids'][qi], d['q_camids'][qi], d['metric'])
        ranked_pid = d['g_pids'][order][keep]
        ranked_mid = d['g_mids'][order][keep]
        hit = ranked_pid == d['q_pids'][qi]
        if not hit.any():
            continue
        for gm in mods:
            rel = hit & (ranked_mid == gm)
            n_rel = int(rel.sum())
            if n_rel == 0:
                continue
            sub = rel | ~hit
            pos = np.flatnonzero(rel[sub])
            acc[(int(d['q_mids'][qi]), gm)].append(
                float((np.arange(1, n_rel + 1) / (pos + 1.0)).mean()))
    return {k: (float(np.mean(v)) if v else float('nan'), len(v))
            for k, v in acc.items()}


def test_ap_decomposition_matches_the_reference_loop():
    for offset in (0.0, 3.0):
        d = build(offset=offset)
        qf, gf = features(d)
        want = _ap_reference(qf, gf, d)
        for dev in DEVICES:
            got = ap_by_gallery_modality(qf, gf, d, chunk=13, device=dev)
            assert set(got) == set(want)
            for k in want:
                w, wn = want[k]
                g, gn = got[k]
                assert gn == wn, (offset, dev, k, gn, wn)
                assert abs(g - w) < _tol(dev), (offset, dev, k, g, w)


# --------------------------------------------------------------------------
# 6. the whole report, every section on
# --------------------------------------------------------------------------
def _with_frames(d):
    """build() plus a frame index, so the three spectra of one (pid, cam) are
    a twin triple and section [2b] has something to decompose."""
    d = dict(d)
    for side in ('q', 'g'):
        d[side + '_frames'] = d[side + '_camids'].copy()
    return d


def test_report_runs_every_section_without_crashing():
    """The regression that motivated this test: [2b] bound a local named
    `base`, which [0] had already bound to the 3x3 matrix and [4] indexes --
    Python has no block scope, so the rebind reached them and [4] died with
    "'float' object is not subscriptable".  The earlier smoke check passed
    --skip-centering, so [4] never ran and the collision was invisible.  Any
    test of the sections in isolation would miss this again; only running them
    in sequence, in one call, catches a name leaking between them.
    """
    import argparse
    import contextlib

    d = _with_frames(build(offset=0.5))
    args = argparse.Namespace(seed=0, chunk=64, num_pids=20,
                              skip_centering=False, skip_ap=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(d, args, torch.device('cpu'))
    out = buf.getvalue()
    for section in ('[0]', '[1]', '[2]', '[2b]', '[3]', '[4]'):
        assert section in out, (section, out[-2000:])
    # [4] prints its delta table last; if it raised, the header would be the
    # final thing in the buffer.
    assert out.rstrip().splitlines()[-1].strip().startswith(('RGB', 'IR', 'Thermal')),         out[-800:]


def test_the_3x3_matrix_survives_the_decomposition():
    """Narrower and blunter: hold section [0]'s result, run [2b], check the
    binding is still a matrix.  Stated as a fact about names rather than about
    output text, so it keeps working if the report is reformatted."""
    import argparse
    import contextlib

    d = _with_frames(build(offset=0.5))
    args = argparse.Namespace(seed=0, chunk=64, num_pids=20,
                              skip_centering=False, skip_ap=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(d, args, torch.device('cpu'))
    text = buf.getvalue()
    assert 'as multiples of' in text, text[-1500:]     # [2b] really ran
    assert 'change per cell' in text, text[-1500:]     # and [4] got past it


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d diagnostics tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
