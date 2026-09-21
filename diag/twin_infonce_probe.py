"""Would the twin contrastive loss have anything to learn?  Zero training.

    python diag/twin_infonce_probe.py feats_whu_recipe_hihr.npz

Before spending a GPU day on a loss, three things about it can be settled from
features we already have.  All of them have burned us before:

  1. Is the task already satisfied at the start?  whu_sync's triplet was --
     d_cross 1.125 was already below d_diffpid 1.273, the soft margin decided
     the constraint held, and sixty epochs bought +0.10.  mod_lam50's six-way
     cross entropy reached accuracy 1.00 and loss 0.015 by epoch 9 and taught
     nothing after that.  A loss that starts near zero is not worth running.

  2. What temperature makes it produce gradient?  Our features sit in a narrow
     cone -- a twin pair is at cosine 0.988 and two different people at 0.982,
     a gap of 0.006 -- so the 0.05-0.07 that contrastive papers use would map
     every candidate onto the same logit and the softmax would be flat.  tau
     has to be read off that gap, not copied.

  3. THE DECISIVE ONE: how much of the loss can a single shared shift remove?
     The whole argument for this design is that CE+triplet already took the
     shared component (the three spectrum centroids are at cosine 0.9999) and
     that only per-image structure is left.  If a constant can still buy a
     large reduction here, the optimiser will take it, learn nothing new, and
     the run is wasted -- exactly what happened to every "one anchor" method
     so far.  So the constant is not assumed to be the mean displacement; it is
     SEARCHED by gradient descent, one vector per spectrum, to give the lazy
     solution its best possible shot.

One limitation, stated because it cannot be removed: these are TEST-set
features and training happens on the train split, where the twin cosine is
known to differ (0.9932 against 0.9880).  Absolute loss values are therefore
estimates.  Question 3 should still transfer, because it turns on the geometry
of the displacements rather than on their scale.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag.analyse_modality import features, group_captures, load  # noqa: E402

TAUS = (0.003, 0.006, 0.01, 0.02, 0.05, 0.07)


def sample_batches(by_pid, rng, n_batch, n_pid=16, n_inst=4, n_mod=3):
    """Batches shaped like the real ones: n_pid identities, n_inst captures
    each, every capture complete in all spectra.

    Returns a list of [n_pid * n_inst, n_mod] row-index arrays, so column m is
    modality m and row c is one capture -- which is exactly the layout
    SYNC_FRAMES produces, with the twin of row c in column m sitting at row c
    of column 0.
    """
    full = {}
    for pid, caps in by_pid.items():
        ok = [k for k, v in caps.items() if len(v) == n_mod]
        if len(ok) >= n_inst:
            full[pid] = sorted(ok)
    pids = sorted(full)
    if len(pids) < n_pid:
        raise ValueError('only %d identities have %d complete captures'
                         % (len(pids), n_inst))

    out = []
    for _ in range(n_batch):
        chosen = [pids[k] for k in rng.choice(len(pids), n_pid, replace=False)]
        rows, labels = [], []
        for pid in chosen:
            caps = full[pid]
            pick = rng.choice(len(caps), n_inst, replace=False)
            for k in pick:
                cap = by_pid[pid][caps[k]]
                rows.append([cap[m] for m in sorted(cap)])
                labels.append(pid)
        out.append((np.asarray(rows, dtype=np.int64),
                    np.asarray(labels, dtype=np.int64)))
    return out


def batch_logits(qf, rf, pid, tau):
    """[Nq, Nr] logits with same-identity non-twin columns masked out.

    Row i of `qf` and row i of `rf` are the two spectra of ONE capture, so the
    positive is the diagonal.  The other three captures of the same identity
    are neither positive nor negative: pushing them away would fight the ID
    loss, which wants every image of a person together.  They are set to -inf
    so logsumexp drops them.
    """
    logits = (qf @ rf.t()) / tau
    dev = logits.device
    same = torch.as_tensor(pid[:, None] == pid[None, :], device=dev)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=dev)
    logits = logits.masked_fill(same & ~eye, float('-inf'))
    return logits


def infonce(qf, rf, pid, tau):
    """(mean loss, rank-1 rate, mean margin over the strongest negative)."""
    logits = batch_logits(qf, rf, pid, tau)
    n = logits.shape[0]
    tgt = torch.arange(n, device=logits.device)
    loss = F.cross_entropy(logits, tgt)
    with torch.no_grad():
        pos = logits[tgt, tgt].clone()
        neg = logits.clone()
        neg[tgt, tgt] = float('-inf')
        hardest = neg.max(dim=1).values
        rank1 = (pos > hardest).double().mean()
        margin = ((pos - hardest) * tau).mean()      # back to cosine units
    return loss, float(rank1), float(margin)


def run(gf, batches, tau, shift=None):
    """Mean loss / rank-1 / margin over the sampled batches.

    `shift` is [n_mod, D]; it is subtracted before renormalising, the same
    order of operations section [4] of analyse_modality.py uses, so a shift
    measured here means the same thing there.
    """
    losses, rank1s, margins = [], [], []
    for rows, pid in batches:
        f = gf[torch.from_numpy(rows).to(gf.device)]        # [C, M, D]
        if shift is not None:
            f = f - shift.unsqueeze(0)
        f = F.normalize(f, dim=-1)
        ref = f[:, 0]
        for m in range(1, f.shape[1]):
            l, r, g = infonce(f[:, m], ref, pid, tau)
            losses.append(l)
            rank1s.append(r)
            margins.append(g)
    return (torch.stack(losses).mean(), float(np.mean(rank1s)),
            float(np.mean(margins)))


def mean_shift(gf, batches, n_mod):
    """The displacement averaged over captures -- one vector per spectrum.

    Optimal for squared distance, and identical to the difference between the
    spectrum centroids, which is the 0.0103 already reported in section [1].
    Reported alongside the searched shift as a cross-check on that search.
    """
    acc = torch.zeros(n_mod, gf.shape[1], dtype=torch.float64, device=gf.device)
    count = 0
    for rows, _pid in batches:
        f = F.normalize(gf[torch.from_numpy(rows).to(gf.device)].double(), dim=-1)
        acc += f.sum(dim=0)
        count += f.shape[0]
    centroid = acc / count
    return (centroid - centroid[0:1]).float()


def search_shift(gf, batches, tau, n_mod, steps=400, lr=0.02,
                 shared=False, init=None):
    """The best constant, by gradient descent on the loss it is judged by.

    Deliberately not the mean displacement: the mean minimises squared
    distance, not this loss, and the point is to give the "settle on a
    constant" solution every chance.  If even a directly optimised constant
    cannot move the loss, training will not find one either.

    `shared=True` optimises ONE vector applied to every spectrum, which is the
    reason this parameter exists.  A constant can lower this loss two entirely
    different ways, and only the second is the failure mode we care about:

      * by decompressing the cone.  Our features all point near one common
        direction, so subtracting something near the cloud's centre and
        renormalising blows the relative differences up and makes the twin the
        nearest candidate almost for free.  This does nothing about modality --
        it is plain centring, it is equally available whatever the spectra are
        doing, and section [4] already measured what it buys on retrieval: an
        even +0.50 mAP across every cell of the 3x3.
      * by cancelling a per-spectrum displacement.  THIS is what every method
        so far has actually learned, and what CE+triplet has already exhausted.

    So the two are searched in sequence -- the shared one first, the
    per-spectrum one starting from it -- and the verdict is read off the
    SECOND drop alone.  Searching only the per-spectrum form would fold the
    decompression into the number and make a hopeless design look promising.
    """
    d = gf.shape[1]
    if init is None:
        shift = torch.zeros(1 if shared else n_mod, d, device=gf.device)
    else:
        shift = init.clone() if not shared else init[:1].clone()
        if not shared and shift.shape[0] == 1:
            shift = shift.expand(n_mod, d).clone()
    shift.requires_grad_(True)
    opt = torch.optim.Adam([shift], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = shift.expand(n_mod, d) if shared else shift
        loss = torch.stack([
            infonce(F.normalize(f[:, m] - s[m], dim=-1),
                    F.normalize(f[:, 0] - s[0], dim=-1), pid, tau)[0]
            for rows, pid in batches
            for f in [gf[torch.from_numpy(rows).to(gf.device)]]
            for m in range(1, f.shape[1])]).mean()
        loss.backward()
        opt.step()
    out = shift.detach()
    return out.expand(n_mod, d).contiguous() if shared else out


def main():
    p = argparse.ArgumentParser(description='twin InfoNCE feasibility probe')
    p.add_argument('npz')
    p.add_argument('--batches', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--tau', type=float, default=0.006,
                   help='the temperature the shift search is run at')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--device', default=None,
                   help="'cuda' or 'cpu'; defaults to cuda when available")
    args = p.parse_args()

    d = load(args.npz)
    if 'g_frames' not in d or (d['g_frames'] < 0).all():
        raise SystemExit('this npz carries no frame index; re-dump with the '
                         'current diag/dump_features.py')
    _qf, gf = features(d)
    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda asked for, but torch.cuda is unavailable')
    gf = gf.to(device)
    rng = np.random.RandomState(args.seed)
    by_pid = group_captures(d)
    names = [str(m) for m in d['modalities']] if 'modalities' in d else None

    # Two disjoint halves.  The shift is D free parameters per spectrum --
    # 768 of them against ~1300 captures -- so fitting and reporting on the
    # same batches would let it memorise them and overstate what a constant can
    # do, which biases the verdict towards "do not run".  Measured on the
    # synthetic fixture: a search on one batch of 64 "found" an offset of 0.177
    # in data that had none planted.
    both = sample_batches(by_pid, rng, 2 * args.batches)
    fit, batches = both[:args.batches], both[args.batches:]
    n_mod = batches[0][0].shape[1]
    n_cap = batches[0][0].shape[0]
    print('\n' + '=' * 78)
    print('孪生 InfoNCE 可行性预检   %s' % d['label'])
    print('  %d 个 batch x %d 次拍摄 x %d 个光谱；每张非参考图对 1 个孪生 + %d 个陌生人'
          % (args.batches, n_cap, n_mod, n_cap - 4))
    print('  位移在另外 %d 个不相交的 batch 上拟合，此处报的全部是留出集读数'
          % args.batches)
    print('  device = %s' % device)
    if names:
        print('  参考光谱 = %s' % names[0])
    print('=' * 78)

    ceiling = float(np.log(n_cap - 4 + 1))
    print('\n[1] 温度扫描   (损失上限 log %d = %.3f，越接近上限说明起点越难)'
          % (n_cap - 3, ceiling))
    print('  %-8s %10s %10s %12s %14s'
          % ('tau', '初始损失', '完成度', '孪生rank-1', 'logit间隔(cos)'))
    for tau in TAUS:
        loss, rank1, margin = run(gf, batches, tau)
        print('  %-8.3f %10.3f %9.1f%% %11.1f%% %14.4f'
              % (tau, float(loss), 100 * (ceiling - float(loss)) / ceiling,
                 100 * rank1, margin))
    print('  读法：完成度接近 100%% = 起点上任务已满足，跑了白跑（whu_sync 的下场）；')
    print('        接近 0%% = 起点上完全做不到，多半是 tau 太小。')

    print('\n[2] 常数位移能买到多少   tau = %.3f' % args.tau)
    base, base_r1, _m = run(gf, batches, args.tau)
    print('    (a) 不做任何位移                       损失 %.4f   rank-1 %.1f%%'
          % (float(base), 100 * base_r1))
    if float(base) < 0.1:
        print('\n    起点损失已经接近 0 —— 这个损失在训练开始时就没有可学的东西，')
        print('    和 whu_sync 的三元组是同一种失效。不必往下看，也不必跑。')
        print('=' * 78 + '\n')
        return 0

    # (b) ONE vector for every spectrum: plain centring, nothing to do with
    # modality.  Measured first so its effect can be taken out of (c).
    sh = search_shift(gf, fit, args.tau, n_mod, steps=args.steps, shared=True)
    l_sh, r_sh, _m = run(gf, batches, args.tau, shift=sh)
    print('    (b) + 整体常数（与模态无关，= 去中心）  损失 %.4f   rank-1 %.1f%%   (%+.4f)'
          % (float(l_sh), 100 * r_sh, float(l_sh - base)))
    print('        ‖c‖ = %.4f' % float(sh[0].norm()))

    # (c) per-spectrum, warm-started from (b).  The EXTRA drop is the only part
    # that is about modality, and the only part CE+triplet could have taken.
    ss = search_shift(gf, fit, args.tau, n_mod, steps=args.steps, init=sh)
    l_ss, r_ss, _m = run(gf, batches, args.tau, shift=ss)
    print('    (c) + 逐光谱的那一份                   损失 %.4f   rank-1 %.1f%%   (%+.4f)'
          % (float(l_ss), 100 * r_ss, float(l_ss - l_sh)))
    for m in range(1, n_mod):
        print('        逐光谱差量 ‖c_%d − c_0‖ = %.4f'
              % (m, float((ss[m] - ss[0]).norm())))

    ms = mean_shift(gf, fit, n_mod)
    l_mean, _r, _m = run(gf, batches, args.tau, shift=ms)
    print('    (参考) 平均位移（= [1] 节的质心差）     损失 %.4f   (%+.4f)   ‖c_%d − c_0‖ = %.4f'
          % (float(l_mean), float(l_mean - base), n_mod - 1,
             float((ms[-1] - ms[0]).norm())))

    print('\n' + '=' * 78)
    total = 100 * (float(base) - float(l_ss)) / float(base)
    modal = 100 * (float(l_sh) - float(l_ss)) / float(base)
    print('  常数位移合计降低损失 %.1f%%，其中：' % total)
    print('    与模态无关的去中心   %5.1f%%   这一条训练时也拿得到，但 [4] 已经测过，'
          % (total - modal))
    print('                                它在检索上只值 +0.50 mAP')
    print('    模态特有的那一份     %5.1f%%   这才是「退化成一个共享偏移」' % modal)
    print('-' * 78)
    if modal < 5:
        print('  判定：模态特有的常数只值 %.1f%% —— 这条捷径已经被 CE+triplet 走完了。' % modal)
        print('        训练要么学到逐图的东西，要么什么也学不到，两种结局都是干净的结论。')
        print('        可以跑。')
    else:
        print('  判定：模态特有的常数还值 %.1f%% —— 优化器会先去拿它，' % modal)
        print('        跑完大概率只是又学了一个共享偏移。设计应当在此处停下。')
    print('=' * 78)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
