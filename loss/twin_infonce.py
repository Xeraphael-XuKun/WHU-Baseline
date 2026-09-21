"""Contrastive alignment to the same-capture reference frame.

Each non-reference image is asked to pick its own twin -- the reference-spectrum
image of the SAME capture -- out of the batch's reference images.  Positive is
the twin, negatives are the reference images of every OTHER identity.

Why contrastive rather than the plain cosine regression `1 - cos(f, sg(r))`
that loss/twin_align.py implements.  That form was safe while the backbone was
frozen, because 2.4M zero-initialised parameters cannot collapse a feature
space.  With the tower trainable it stops being safe: pushing every feature of
every person and every spectrum onto one point makes the cosine identically 1
and the loss identically 0, so collapse becomes the OPTIMUM.  The contrastive
form inverts that -- if all features coincide, numerator and denominator
match, every candidate is equally likely, and the loss sits at its ceiling
log(1 + n_neg).  Collapse becomes the worst answer available.

Two details are not arbitrary.

*The other captures of the same identity are excluded from the denominator.*
They are neither positive nor negative: counting them as negatives would push
a person's own frames apart, in direct conflict with the ID loss that wants
them together.  They are masked to -inf so logsumexp drops them.

*The reference features are detached.*  Not for anti-collapse -- the negatives
already handle that -- but so the reference spectrum stays a coordinate frame
rather than meeting the others half way.  It makes "the RGB diagonal must not
move" a readable failure signal instead of an expected side effect.

On tau, measured rather than copied (diag/twin_infonce_probe.py, 2026-08-10 on
feats_whu_recipe_hihr): the cosine gap between a twin and the HARDEST of 60
strangers is 0.0026, so tau has to be around 0.003 for that gap to become a
logit gap of order one.  The 0.05-0.07 usual in contrastive papers put the
loss at 4.5% of its range from the ceiling -- a softmax so flat that there is
essentially no gradient, which is the same way whu_sync's triplet failed.
"""

import torch
import torch.nn.functional as F


def twin_infonce_loss(feat, pids, n_modalities, per_modality, tau, ref_index=0):
    """(loss, stats) for a modality-major batch under DATALOADER.SYNC_FRAMES.

    feat          [n_modalities * per_modality, D], row i + m * per_modality is
                  capture i in modality m -- the layout train_collate_fn and
                  the PKM sampler produce.  Row i of block m and row i of block
                  `ref_index` are therefore the same instant, which is the
                  whole premise; without SYNC_FRAMES they are different frames
                  and this loss silently means something else.
    pids          [per_modality] identity label per capture.
    tau           temperature.  See the module docstring -- it is read off the
                  measured hardest-negative gap, not chosen by convention.

    stats carries one entry per modality (nan at `ref_index`):
      rank1   fraction of images whose twin beats every stranger.  THE number
              to read: the probe measured 70.1% before training, while plain
              centring moves the loss by 57% and rank1 by 0.1 points.  The
              loss curve is therefore not a progress report and rank1 is.
      cos     mean cosine to the twin.
      margin  mean cosine gap over the strongest negative, in cosine units so
              it is comparable with everything else we quote.
    """
    total = n_modalities * per_modality
    if feat.shape[0] != total:
        raise ValueError(
            'twin_infonce_loss got {} rows but n_modalities * per_modality = {}'
            .format(feat.shape[0], total))
    if pids.shape[0] != per_modality:
        raise ValueError(
            'twin_infonce_loss got {} pids for {} captures'
            .format(pids.shape[0], per_modality))
    if not tau > 0:
        raise ValueError('tau must be positive, got {!r}'.format(tau))

    f = F.normalize(feat.float(), dim=-1).view(n_modalities, per_modality, -1)
    ref = f[ref_index].detach()

    pids = pids.reshape(-1).to(f.device)
    same = pids[:, None] == pids[None, :]
    eye = torch.eye(per_modality, dtype=torch.bool, device=f.device)
    drop = same & ~eye                       # same identity, not the twin
    tgt = torch.arange(per_modality, device=f.device)

    loss = feat.new_zeros((), dtype=torch.float32)
    stats = {'rank1': [], 'cos': [], 'margin': []}
    for m in range(n_modalities):
        if m == ref_index:
            for k in stats:
                stats[k].append(float('nan'))
            continue
        cos = f[m] @ ref.t()
        logits = (cos / tau).masked_fill(drop, float('-inf'))
        loss = loss + F.cross_entropy(logits, tgt)
        with torch.no_grad():
            pos = cos[tgt, tgt]
            neg = cos.masked_fill(drop, float('-inf')).clone()
            neg[tgt, tgt] = float('-inf')
            hardest = neg.max(dim=1).values
            stats['rank1'].append((pos > hardest).float().mean().item())
            stats['cos'].append(pos.mean().item())
            stats['margin'].append((pos - hardest).mean().item())

    return loss / max(n_modalities - 1, 1), stats
