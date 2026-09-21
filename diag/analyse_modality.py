"""What exactly is wrong with the thermal modality.

    python diag/analyse_modality.py feats_whu_recipe_hihr.npz [more.npz ...]
    python diag/analyse_modality.py --device cpu feats_*.npz      # no GPU

Reads what `dump_features.py` wrote and answers, in order:

  0. Does this file reproduce the numbers we reported?  If the 3x3 matrix here
     does not match view_matrix.txt, nothing below means anything.
  1. Is the thermal gap a *translation* -- one constant offset between the
     modality clouds -- or are the clouds shaped differently?
  2. Is identity still recoverable across thermal at all?  If two images of the
     same person in different spectra are further apart than two different
     people in the same spectrum, no ranking method can fix it.
  3. How much of the pooled mAP is thermal actually costing?
  4. What happens to the 3x3 if the per-modality mean is simply subtracted?

The split matters because it decides the method.  A translation is removed by
centring, which is nearly free.  Differently shaped clouds need a real
cross-modal method.  These are not the same project, and right now we cannot
tell which one we are in.

One prior worth carrying: on the old ChartPE line, `CHART_INPUT_NORM` removed
85% of the modality offset in the chart and mAP moved by 0.00.  That was an
intermediate quantity, not the retrieval feature, so it does not settle this --
but it is a standing warning that a smaller offset does not imply a better mAP.
Section 4 measures the mAP directly for that reason.

-- On the evaluator ------------------------------------------------------

`R1_mAP_eval` materialises the whole 6,405 x 93,609 ranking (2.4 GB) and then
walks it one query at a time in Python.  That is affordable on the worker and
it is what the reported numbers came from, but it makes re-analysis expensive:
this script gets run every time we think of a new question.

Everything below is instead expressed as whole-tensor operations over one
chunk of queries, so it runs on whatever device it is given.  The obstacle is
that the junk rule removes a different number of gallery entries per query, so
the "compacted" ranking is ragged and cannot be a rectangular tensor.  The way
around it is to never compact: `keep.cumsum(1)` gives, at every original
position, the rank that position *would* have after compaction, which is all
the AP and CMC formulas actually need.

`eval_stream` is arithmetically identical to `utils.metrics.eval_func` -- the
AP accumulates in float64 for exactly that reason -- and a unit test pins the
agreement to 1e-9 on both devices.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CPU_CHUNK = 128
GPU_CHUNK = 512


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def default_chunk(device):
    return GPU_CHUNK if device.type == 'cuda' else CPU_CHUNK


def ranking_chunks(qf, gf, chunk, device):
    """Yield (start, order) with order[r] ranking the whole gallery for query r.

    The arithmetic mirrors `utils.metrics.iter_euclidean_distance_chunks`
    exactly, `.contiguous()` included.  A transposed view would work and save a
    copy of the gallery, but BLAS may take a different code path for a strided
    operand, and the rounding difference would show up as flipped near-ties --
    that is, as a section [0] that does not reproduce.
    """
    gf = gf.to(device)
    gf_t = gf.t().contiguous()
    gf_sq = gf.pow(2).sum(dim=1, keepdim=True).t()
    for start in range(0, qf.shape[0], chunk):
        q = qf[start:min(start + chunk, qf.shape[0])].to(device)
        dist = q.pow(2).sum(dim=1, keepdim=True) + gf_sq
        dist.addmm_(q, gf_t, beta=1, alpha=-2)
        order = torch.argsort(dist, dim=1)
        del dist, q
        yield start, order
        del order
    del gf, gf_t, gf_sq


def _kept(order, g_pid, g_cam, q_pid, q_cam, metric):
    """Per ranked position: does it match the query, and does it survive the
    junk rule.  Both as [Q, G] boolean tensors, uncompacted."""
    gp = g_pid[order]
    gc = g_cam[order]
    match = gp == q_pid.unsqueeze(1)
    same_cam = gc == q_cam.unsqueeze(1)
    # Mirrors utils.metrics.junk_mask exactly.
    if metric == 'market':
        remove = match & same_cam
    elif metric == 'sysu':
        remove = same_cam
    else:
        raise ValueError("metric must be 'sysu' or 'market', got %r" % metric)
    del gp, gc, same_cam
    return match, ~remove


def eval_stream(qf, gf, q_pids, g_pids, q_camids, g_camids,
                metric='sysu', max_rank=50, chunk=None, device=None):
    """CMC / mAP, identical to `utils.metrics.eval_func`, vectorised per chunk.

    The identity that makes this work: for a hit at original ranked position
    j, its rank among the *kept* entries is ``keep.cumsum(1)[j]`` and the
    number of hits at or before it is ``hit.cumsum(1)[j]``.  eval_func's

        AP = sum_p orig_cmc[p] * cumsum(orig_cmc)[p] / p   /  num_rel

    is then the same sum taken over uncompacted positions.
    """
    device = pick_device(device) if not isinstance(device, torch.device) else device
    chunk = chunk or default_chunk(device)

    g_pid = torch.as_tensor(np.asarray(g_pids), dtype=torch.int32, device=device)
    g_cam = torch.as_tensor(np.asarray(g_camids), dtype=torch.int32, device=device)
    q_pid = torch.as_tensor(np.asarray(q_pids), dtype=torch.int32, device=device)
    q_cam = torch.as_tensor(np.asarray(q_camids), dtype=torch.int32, device=device)

    num_g = gf.shape[0]
    if num_g < max_rank:
        max_rank = num_g

    ap_sum, n_valid, first_ranks = 0.0, 0, []
    for start, order in ranking_chunks(qf, gf, chunk, device):
        n = order.shape[0]
        match, keep = _kept(order, g_pid, g_cam,
                            q_pid[start:start + n], q_cam[start:start + n], metric)
        hit = keep & match
        del match
        num_rel = hit.sum(dim=1)
        valid = num_rel > 0
        if not bool(valid.any()):
            del hit, keep, num_rel, valid
            continue

        n_keep = keep.sum(dim=1)
        if bool((n_keep[valid] < max_rank).any()):
            # eval_func would build a ragged list of CMC rows here and die in
            # np.asarray.  Say so, rather than silently reporting a curve
            # computed over a different number of entries per query.
            raise ValueError(
                'a query keeps only %d gallery entries but max_rank is %d; the '
                'CMC rows would not be comparable'
                % (int(n_keep[valid].min()), max_rank))

        # cumsum on bool promotes to int64.  The counts top out at the
        # gallery size, so int32 says the same thing in half the memory -- and
        # this is the largest transient in the whole script.
        kpos = keep.to(torch.int32).cumsum(dim=1)
        hpos = hit.to(torch.int32).cumsum(dim=1)
        # float64: eval_func sums ~150 float64 terms per query, and float32
        # here would drift by ~1e-6 relative -- three orders of magnitude
        # above the tolerance the equivalence test holds us to.
        # clamp_min: before the first kept entry kpos is 0, and there the
        # division gives inf.  hit is false there too, so `inf * 0 = nan`
        # poisons the whole sum -- which is exactly what happened on the real
        # data, where the sysu rule junks the query's own camera and those are
        # the nearest neighbours, so rank 0 is essentially always dropped.
        prec = hpos.to(torch.float64).div_(kpos.to(torch.float64).clamp_min(1))
        prec.mul_(hit)
        ap = prec.sum(dim=1).div_(num_rel.to(torch.float64).clamp_min(1))
        del prec, hpos

        big = torch.full_like(kpos, num_g + 1)
        first = torch.where(hit, kpos, big).min(dim=1).values
        del big, kpos, hit, keep, n_keep

        ap_sum += float(ap[valid].sum())
        n_valid += int(valid.sum())
        first_ranks.append(first[valid].to('cpu'))
        del ap, first, num_rel, valid

    if n_valid == 0:
        raise AssertionError('Error: all query identities do not appear in gallery')

    first = torch.cat(first_ranks).to(torch.int64)
    ranks = torch.arange(1, max_rank + 1, dtype=torch.int64)
    cmc = (first.unsqueeze(1) <= ranks.unsqueeze(0)).to(torch.float64).mean(dim=0)
    return cmc.numpy().astype(np.float32), ap_sum / n_valid


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['label'] = os.path.splitext(os.path.basename(path))[0].replace('feats_', '')
    d['metric'] = str(d['metric'])
    d['modalities'] = [str(m) for m in d['modalities']]
    return d


def features(d, feat_norm=None):
    """Query and gallery as float32 CPU tensors sharing storage with the loaded
    arrays, normalised in place the way the evaluator normalises them.

    They stay on the CPU: sections 1 and 2 want numpy views of the same memory,
    and `ranking_chunks` uploads what it needs.  The consequence is that this
    must be called once -- the tensors it returns are the only copy, and
    section 4 mutates them.
    """
    norm = bool(d['feat_norm']) if feat_norm is None else feat_norm
    out = []
    for k in ('qf', 'gf'):
        t = torch.from_numpy(d[k])
        if t.dtype != torch.float32:
            t = t.float()
        if norm and not d.get('_normed_' + k):
            t.div_(t.norm(dim=1, keepdim=True).clamp_min(1e-12))
            d['_normed_' + k] = True
        out.append(t)
    return out[0], out[1]


def mod_name(d, mid):
    """Modality ids are 1-based (`whu_mars.py`: enumerate(modality_ls, 1))."""
    names = d['modalities']
    return names[mid - 1] if 1 <= mid <= len(names) else str(mid)


# ---------------------------------------------------------------------------
# section 0 / 4: the 3x3 matrix
# ---------------------------------------------------------------------------
def cell(qf, gf, d, q_sel, g_sel, chunk=None, device=None):
    if q_sel.size == 0 or g_sel.size == 0:
        return None, None
    qs = qf.index_select(0, torch.from_numpy(q_sel).long())
    gs = gf.index_select(0, torch.from_numpy(g_sel).long())
    try:
        return eval_stream(qs, gs, d['q_pids'][q_sel], d['g_pids'][g_sel],
                           d['q_camids'][q_sel], d['g_camids'][g_sel],
                           metric=d['metric'], chunk=chunk, device=device)
    except AssertionError:
        return None, None
    finally:
        del qs, gs


def matrix_3x3(qf, gf, d, chunk=None, device=None, indent='    '):
    mods = sorted(set(d['q_mids'].tolist()))
    print(indent + '%-8s' % 'Q\\G' + ''.join('%12s' % mod_name(d, m) for m in mods),
          flush=True)
    out = {}
    for qm in mods:
        row = indent + '%-8s' % mod_name(d, qm)
        for gm in mods:
            cmc, mAP = cell(qf, gf, d,
                            np.where(d['q_mids'] == qm)[0],
                            np.where(d['g_mids'] == gm)[0],
                            chunk=chunk, device=device)
            out[(qm, gm)] = (mAP, None if cmc is None else cmc[0])
            row += '%12s' % ('--' if mAP is None else '%.2f' % (mAP * 100))
        print(row, flush=True)
    return out


def pooled(qf, gf, d, chunk=None, device=None):
    cmc, mAP = eval_stream(qf, gf, d['q_pids'], d['g_pids'],
                           d['q_camids'], d['g_camids'],
                           metric=d['metric'], chunk=chunk, device=device)
    return mAP, cmc


# ---------------------------------------------------------------------------
# section 1: is it a translation?
# ---------------------------------------------------------------------------
def mean_offsets(d, rng, gf=None):
    """Distance between per-modality mean features, against a noise floor.

    In 768 dimensions a mean estimated from n samples still carries noise of
    order sigma/sqrt(n), and that noise alone produces a non-zero distance
    between any two means.  Reporting the raw distance without the floor is how
    you convince yourself of an offset that is not there.

    The floor is measured by splitting one modality at random into two halves
    and taking the distance between the two half-means.  Each half-mean uses
    n/2 samples, so that distance is sqrt(2) times larger than the noise in a
    difference of two full n-sample means -- hence the division below.
    """
    if gf is None:
        _q, gf = features(d, feat_norm=True)
    g = gf.numpy()                       # shares storage, no copy
    mods = sorted(set(d['g_mids'].tolist()))

    means, floors = {}, {}
    for m in mods:
        sel = np.where(d['g_mids'] == m)[0]
        means[m] = g[sel].mean(axis=0)
        perm = rng.permutation(sel)
        half = len(perm) // 2
        floors[m] = np.linalg.norm(g[perm[:half]].mean(axis=0) -
                                   g[perm[half:2 * half]].mean(axis=0)) / np.sqrt(2)
    return means, floors, mods


# ---------------------------------------------------------------------------
# section 2: is identity still recoverable across thermal?
# ---------------------------------------------------------------------------
def pair_distances(d, rng, num_pids=200, per_cell=2, gf=None):
    """d_same / d_cross / d_diffpid, on gallery features.

    All three conditions hold the camera *different*, so none of them is
    inflated by near-duplicate frames from one camera:

      d_same(m)        same person, modality m, different camera
      d_cross(m1,m2)   same person, m1 vs m2, different camera
      d_diffpid(m)     different people, modality m

    Everything is reported as a ratio to d_same, because the feature norm
    differs between checkpoints and raw distances would not be comparable.
    """
    if gf is None:
        _q, gf = features(d, feat_norm=True)
    g = gf.numpy()
    pids, cams, mids = d['g_pids'], d['g_camids'], d['g_mids']
    mods = sorted(set(mids.tolist()))

    # index[pid][modality] -> row ids
    by = {}
    for i, (p, m) in enumerate(zip(pids, mids)):
        by.setdefault(int(p), {}).setdefault(int(m), []).append(i)
    usable = [p for p, mm in by.items() if len(mm) == len(mods)]
    chosen = rng.permutation(usable)[:num_pids]

    def sample(rows, k):
        rows = np.asarray(rows)
        return rows if len(rows) <= k else rng.choice(rows, k, replace=False)

    same = {m: [] for m in mods}
    cross = {(a, b): [] for a in mods for b in mods if a < b}
    for p in chosen:
        for m in mods:
            r = sample(by[p][m], 6)
            for i in r:
                for j in r:
                    if i < j and cams[i] != cams[j]:
                        same[m].append(np.linalg.norm(g[i] - g[j]))
        for a, b in cross:
            ra, rb = sample(by[p][a], per_cell + 2), sample(by[p][b], per_cell + 2)
            for i in ra:
                for j in rb:
                    if cams[i] != cams[j]:
                        cross[(a, b)].append(np.linalg.norm(g[i] - g[j]))

    diff = {m: [] for m in mods}
    for m in mods:
        pool = [p for p in chosen if m in by[p]]
        for _ in range(4000):
            p1, p2 = rng.choice(len(pool), 2, replace=False)
            i = rng.choice(by[pool[p1]][m])
            j = rng.choice(by[pool[p2]][m])
            diff[m].append(np.linalg.norm(g[i] - g[j]))

    mean = lambda v: float(np.mean(v)) if len(v) else float('nan')   # noqa: E731
    return ({m: mean(v) for m, v in same.items()},
            {k: mean(v) for k, v in cross.items()},
            {m: mean(v) for m, v in diff.items()},
            len(chosen))


# ---------------------------------------------------------------------------
# section 2b: how much of a same-identity distance is the SPECTRUM?
# ---------------------------------------------------------------------------
CONDITIONS = ('twin', 'same_cam', 'diff_cam')
COND_LABEL = {'twin': 'same cam, same frame', 'same_cam': 'same cam, other frame',
              'diff_cam': 'other camera'}


def group_captures(d):
    """Gallery rows grouped as ``pid -> {(cam, frame): {modality: row}}``.

    A "capture" is one instant recorded by one camera: WHU-MARS fires the three
    sensors together, so that inner dict normally holds all three spectra and
    any two of its entries differ ONLY in spectrum.  Rows with no frame index
    (CARGO, or a filename that did not parse) are dropped rather than lumped
    together, which would silently invent capture pairings.

    Shared with diag/twin_infonce_probe.py: both the spectrum/pose split and
    the twin loss are defined on exactly this grouping, and two copies of it
    would be two chances to disagree about what a twin is.
    """
    frames = d['g_frames']
    pids, cams, mids = d['g_pids'], d['g_camids'], d['g_mids']
    by_pid = {}
    for i in range(len(pids)):
        if frames[i] < 0:
            continue
        by_pid.setdefault(int(pids[i]), {}).setdefault(
            (int(cams[i]), int(frames[i])), {})[int(mids[i])] = i
    return by_pid


def capture_decomposition(d, rng, gf=None, num_pids=200, per_pid=8):
    """Split every same-identity distance by what changed BESIDES the spectrum.

    Every number this project has quoted about the modality gap -- d_cross(RGB,
    Thermal) = 1.125 against d_same = 1.0 -- was measured on pairs that differ
    in camera as well as in spectrum, because `pair_distances` requires
    ``cams[i] != cams[j]``.  So "the cost of crossing to thermal" has always
    been the cost of crossing to thermal *and* changing viewpoint, and the two
    have never been separated.

    They can be, because WHU-MARS records the three sensors in sync: a gallery
    capture (pid, camera, frame) exists in all three spectra, so a pair that
    differs ONLY in spectrum is available.  This function reports, for each
    modality pair, three conditions:

        twin       same camera, same frame -- spectrum alone
        same_cam   same camera, other frame -- spectrum + pose/time
        diff_cam   different camera        -- spectrum + viewpoint

    with the same-modality rows as the reference for what each condition costs
    without any spectrum change at all.  Reading down a column gives the price
    of the spectrum; reading across a row gives the price of the pose.

    What prompted it: whu_twin's first diagnostic line put same-capture RGB/IR
    features at cosine 0.996 and RGB/Thermal at 0.993 -- essentially aligned
    already -- while the different-camera measurement says crossing thermal
    costs 12.5%.  If that holds here, the gap is not a displacement between
    spectra but a difference in how well each spectrum absorbs a viewpoint
    change, and an alignment method is aimed at the wrong thing.

    Sampling is by capture rather than by image, so the twin cells cannot come
    out under-populated: `per_pid` captures are drawn per identity and all
    their modalities are taken.
    """
    if 'g_frames' not in d:
        return None
    frames = d['g_frames']
    if (frames < 0).all():
        return None

    if gf is None:
        _q, gf = features(d, feat_norm=True)
    g = gf.numpy()
    pids, cams, mids = d['g_pids'], d['g_camids'], d['g_mids']
    mods = sorted(set(mids.tolist()))

    by_pid = group_captures(d)
    chosen = rng.permutation(sorted(by_pid.keys()))[:num_pids]
    buckets = {}
    n_capture = 0
    for pid in chosen:
        caps = sorted(by_pid[int(pid)].keys())
        if len(caps) > per_pid:
            caps = [caps[k] for k in rng.choice(len(caps), per_pid, replace=False)]
        rows = [(cam, frame, m, r) for (cam, frame) in caps
                for m, r in by_pid[int(pid)][(cam, frame)].items()]
        n_capture += len(caps)
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                cam_a, fr_a, m_a, i = rows[a]
                cam_b, fr_b, m_b, j = rows[b]
                if cam_a != cam_b:
                    cond = 'diff_cam'
                elif fr_a != fr_b:
                    cond = 'same_cam'
                else:
                    cond = 'twin'
                pair = 'same' if m_a == m_b else tuple(sorted((m_a, m_b)))
                buckets.setdefault((pair, cond), []).append(
                    float(np.linalg.norm(g[i] - g[j])))

    # Ceiling: two different identities, same modality, captures drawn at
    # random -- so mostly the `diff_cam` condition, matching how d_diffpid is
    # measured in section [2].  It is a reference for the whole table, not a
    # per-condition one.
    pool = [int(p) for p in chosen]
    diff = []
    for _ in range(4000 if len(pool) >= 2 else 0):
        p1, p2 = rng.choice(len(pool), 2, replace=False)
        for who, pid in ((0, pool[p1]), (1, pool[p2])):
            pass
        c1 = list(by_pid[pool[p1]].keys()); c2 = list(by_pid[pool[p2]].keys())
        k1 = c1[rng.randint(len(c1))]; k2 = c2[rng.randint(len(c2))]
        shared = set(by_pid[pool[p1]][k1]) & set(by_pid[pool[p2]][k2])
        if not shared:
            continue
        m = sorted(shared)[rng.randint(len(shared))]
        diff.append(float(np.linalg.norm(g[by_pid[pool[p1]][k1][m]]
                                         - g[by_pid[pool[p2]][k2][m]])))

    mean = lambda v: float(np.mean(v)) if len(v) else float('nan')   # noqa: E731
    out = {k: mean(v) for k, v in buckets.items()}
    out['counts'] = {k: len(v) for k, v in buckets.items()}
    # A single identity cannot produce a different-identity pair; report nan
    # rather than crashing, so a tiny --num-pids still gives the table.
    out['d_diffpid'] = mean(diff)
    out['n_pids'] = len(chosen)
    out['n_captures'] = n_capture
    out['modalities'] = mods
    return out


# ---------------------------------------------------------------------------
# section 3: what is thermal costing in the pooled ranking?
# ---------------------------------------------------------------------------
def ap_by_gallery_modality(qf, gf, d, chunk=None, device=None):
    """Decompose the pooled AP by the modality of the positive.

    Unlike the 3x3 matrix, the *distractors* are not restricted: a thermal
    positive here has to beat all 93k images of other people, in every
    spectrum, not just the thermal ones.  That is what the pooled mAP is
    actually made of, and it is the number a deployment sees.

    What is removed for each subgroup is the query's *other* positives.  Left
    in, they compete with the subgroup being scored and a perfect retrieval
    scores 0.53 rather than 1.0 -- purely because six correct images cannot all
    occupy the top two ranks.  That artefact would swamp the effect we are
    looking for, so a cell here reads 1.00 exactly when this modality's
    positives outrank every genuine distractor.
    """
    device = pick_device(device) if not isinstance(device, torch.device) else device
    chunk = chunk or default_chunk(device)
    mods = sorted(set(d['g_mids'].tolist()))
    q_mids = d['q_mids']

    g_pid = torch.as_tensor(d['g_pids'], dtype=torch.int32, device=device)
    g_cam = torch.as_tensor(d['g_camids'], dtype=torch.int32, device=device)
    g_mid = torch.as_tensor(d['g_mids'], dtype=torch.int32, device=device)
    q_pid = torch.as_tensor(d['q_pids'], dtype=torch.int32, device=device)
    q_cam = torch.as_tensor(d['q_camids'], dtype=torch.int32, device=device)
    q_mid_t = torch.as_tensor(q_mids, dtype=torch.int32, device=device)

    sums = {(qm, gm): 0.0 for qm in sorted(set(q_mids.tolist())) for gm in mods}
    counts = dict.fromkeys(sums, 0)

    for start, order in ranking_chunks(qf, gf, chunk, device):
        n = order.shape[0]
        match, keep = _kept(order, g_pid, g_cam,
                            q_pid[start:start + n], q_cam[start:start + n],
                            d['metric'])
        hit = keep & match
        gm_ranked = g_mid[order]
        qm_chunk = q_mid_t[start:start + n]
        not_pos = keep & ~match
        for gm in mods:
            rel = hit & (gm_ranked == gm)
            n_rel = rel.sum(dim=1)
            ok = n_rel > 0
            if not bool(ok.any()):
                continue
            sub = rel | not_pos               # keep, minus other-modality hits
            kpos = sub.to(torch.int32).cumsum(dim=1)
            rpos = rel.to(torch.int32).cumsum(dim=1)
            prec = rpos.to(torch.float64).div_(kpos.to(torch.float64).clamp_min(1))
            prec.mul_(rel)
            ap = prec.sum(dim=1).div_(n_rel.to(torch.float64).clamp_min(1))
            del prec, kpos, rpos, sub
            for qm in sorted(set(q_mids.tolist())):
                sel = ok & (qm_chunk == qm)
                c = int(sel.sum())
                if c:
                    sums[(qm, gm)] += float(ap[sel].sum())
                    counts[(qm, gm)] += c
            del rel, n_rel, ok, ap
        del match, keep, hit, gm_ranked, not_pos

    return {k: (sums[k] / counts[k] if counts[k] else float('nan'), counts[k])
            for k in sums}


# ---------------------------------------------------------------------------
def report(d, args, device):
    rng = np.random.RandomState(args.seed)
    chunk = args.chunk or default_chunk(device)
    bar = '=' * 74
    print('\n' + bar)
    print('%s   (%s feature, FEAT_NORM=%s, metric=%s, device=%s, chunk=%d)'
          % (d['label'], str(d['neck_feat']), bool(d['feat_norm']), d['metric'],
             device, chunk))
    print(bar, flush=True)

    qf, gf = features(d)
    mods = sorted(set(d['g_mids'].tolist()))
    kw = dict(chunk=chunk, device=device)

    print('\n[0] reproduce the reported numbers  '
          '-- must match view_matrix.txt or stop here', flush=True)
    mAP, cmc = pooled(qf, gf, d, **kw)
    print('    pooled mAP %.2f%%   Rank-1 %.2f%%   Rank-5 %.2f%%   Rank-10 %.2f%%'
          % (mAP * 100, cmc[0] * 100, cmc[4] * 100, cmc[9] * 100), flush=True)
    print('    3x3 mAP (%):')
    base = matrix_3x3(qf, gf, d, **kw)

    print('\n[1] is the modality gap a translation?', flush=True)
    means, floors, _m = mean_offsets(d, rng, gf=gf)
    print('    distance between per-modality mean features'
          '   (noise floor in brackets)')
    for a in mods:
        for b in mods:
            if a >= b:
                continue
            off = float(np.linalg.norm(means[a] - means[b]))
            fl = max(floors[a], floors[b])
            print('      %-8s <-> %-8s  %.4f   [floor %.4f]   %s'
                  % (mod_name(d, a), mod_name(d, b), off, fl,
                     'real' if off > 3 * fl else 'INDISTINGUISHABLE FROM NOISE'))

    print('\n[2] is identity still recoverable across modalities?', flush=True)
    same, cross, diff, n_pid = pair_distances(d, rng, num_pids=args.num_pids, gf=gf)
    ds = float(np.mean([v for v in same.values() if v == v]))
    print('    %d identities sampled; all distances divided by the mean '
          'd_same (%.4f)' % (n_pid, ds))
    for m in mods:
        print('      d_same    %-8s  %.3f' % (mod_name(d, m), same[m] / ds))
    for (a, b), v in sorted(cross.items()):
        print('      d_cross   %-8s %-8s  %.3f'
              % (mod_name(d, a), mod_name(d, b), v / ds))
    for m in mods:
        print('      d_diffpid %-8s  %.3f' % (mod_name(d, m), diff[m] / ds))
    worst = max(cross.items(), key=lambda kv: kv[1])
    ceiling = min(diff.values())
    print('    VERDICT: worst cross-modality pair %s-%s is %.3f, the easiest '
          'different-person\n             distance is %.3f -- %s'
          % (mod_name(d, worst[0][0]), mod_name(d, worst[0][1]),
             worst[1] / ds, ceiling / ds,
             'IDENTITY IS LOST, ranking cannot fix this'
             if worst[1] > ceiling else 'identity survives, this is fixable'),
          flush=True)

    print('\n[2b] how much of a same-identity distance is the SPECTRUM?', flush=True)
    decomp = capture_decomposition(d, rng, gf=gf, num_pids=args.num_pids)
    if decomp is None:
        print('    unavailable: this npz carries no frame index.  Re-dump with a '
              'diag/dump_features.py\n    that stores g_frames.')
    else:
        print('    %d identities, %d captures.  Absolute distances on L2-normalised '
              'features;' % (decomp['n_pids'], decomp['n_captures']))
        print('    the same-modality row is that same condition with NO spectrum '
              'change.')
        pairs = ['same'] + [(a, b) for a in decomp['modalities']
                            for b in decomp['modalities'] if a < b]

        def label(pair):
            return ('same modality' if pair == 'same'
                    else '%s <-> %s' % (mod_name(d, pair[0]), mod_name(d, pair[1])))

        print('      %-22s' % 'same identity'
              + ''.join('%22s' % COND_LABEL[c] for c in CONDITIONS))
        for pair in pairs:
            line = '      %-22s' % label(pair)
            for c in CONDITIONS:
                v = decomp.get((pair, c), float('nan'))
                n = decomp['counts'].get((pair, c), 0)
                line += '%22s' % ('--' if v != v else '%.4f (n=%d)' % (v, n))
            print(line)
        print('      %-22s%22s' % ('different identity', '%.4f' % decomp['d_diffpid']))

        # NOT `base`: section [0] binds that name to the 3x3 matrix and
        # section [4] indexes it.  Python has no block scope, so a rebind here
        # reaches them.
        ref_dist = decomp.get(('same', 'diff_cam'), float('nan'))
        if ref_dist == ref_dist and ref_dist > 0:
            print('    as multiples of "same modality, other camera" (%.4f):'
                  % ref_dist)
            for pair in pairs:
                line = '      %-22s' % label(pair)
                for c in CONDITIONS:
                    v = decomp.get((pair, c), float('nan'))
                    line += '%22s' % ('--' if v != v else '%.3f' % (v / ref_dist))
                print(line)
            print('      %-22s%22.3f' % ('different identity',
                                         decomp['d_diffpid'] / ref_dist))
        print('    READ: down a column is the price of the SPECTRUM, across a row is')
        print('          the price of the POSE.  A flat twin column beside a steep')
        print('          other-camera column means the gap is pose robustness, not a')
        print('          displacement between spectra -- and an alignment method is')
        print('          then aimed at the wrong thing.')

    if not args.skip_ap:
        print('\n[3] where the pooled mAP goes  '
              '-- AP counting only positives of one modality,', flush=True)
        print('    but ranked against every distractor in the 93k gallery')
        ap = ap_by_gallery_modality(qf, gf, d, **kw)
        print('    %-8s' % 'Q\\G' + ''.join('%12s' % mod_name(d, m) for m in mods))
        for qm in sorted(set(d['q_mids'].tolist())):
            row = '    %-8s' % mod_name(d, qm)
            for gm in mods:
                v, _n = ap[(qm, gm)]
                row += '%12s' % ('--' if v != v else '%.2f' % (v * 100))
            print(row)

    if not args.skip_centering:
        # Last, because it mutates the features in place -- there is no reason
        # to keep a second copy of the gallery around for the earlier sections.
        print('\n[4] subtract the per-modality mean, then re-score', flush=True)
        print('    NOTE: the means come from the test set, so this is '
              'transductive.  It is a\n          diagnostic, not a method -- '
              'it answers "is it a translation", nothing more.')
        for m in mods:
            mu = torch.from_numpy(means[m]).float()
            qf[torch.from_numpy(d['q_mids'] == m)] -= mu
            gf[torch.from_numpy(d['g_mids'] == m)] -= mu
        qf.div_(qf.norm(dim=1, keepdim=True).clamp_min(1e-12))
        gf.div_(gf.norm(dim=1, keepdim=True).clamp_min(1e-12))
        mAP_c, cmc_c = pooled(qf, gf, d, **kw)
        print('    pooled mAP %.2f%% -> %.2f%%   (%+.2f)   Rank-1 %.2f%% -> %.2f%%'
              % (mAP * 100, mAP_c * 100, (mAP_c - mAP) * 100,
                 cmc[0] * 100, cmc_c[0] * 100), flush=True)
        print('    3x3 mAP (%) after centring:')
        cen = matrix_3x3(qf, gf, d, **kw)
        print('    change per cell (%):')
        print('    %-8s' % 'Q\\G' + ''.join('%12s' % mod_name(d, m) for m in mods))
        for qm in sorted(set(d['q_mids'].tolist())):
            row = '    %-8s' % mod_name(d, qm)
            for gm in mods:
                a, b = base[(qm, gm)][0], cen[(qm, gm)][0]
                row += '%12s' % ('--' if a is None or b is None
                                 else '%+.2f' % ((b - a) * 100))
            print(row)


def main():
    p = argparse.ArgumentParser(description='Thermal-gap diagnostics')
    p.add_argument('npz', nargs='+')
    p.add_argument('--device', default=None,
                   help="'cuda' or 'cpu'; defaults to cuda when available")
    p.add_argument('--chunk', type=int, default=None,
                   help='query rows per ranking chunk; lower it if memory is tight')
    p.add_argument('--num-pids', type=int, default=200)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--skip-centering', action='store_true')
    p.add_argument('--skip-ap', action='store_true')
    args = p.parse_args()

    device = pick_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda asked for, but torch.cuda is unavailable')

    for path in args.npz:
        d = load(path)
        report(d, args, device)
        del d                      # the gallery is ~290 MB; do not keep two
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
