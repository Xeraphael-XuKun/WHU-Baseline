"""Twin-anchored modality alignment.

Every non-reference image is asked to land on the feature of its *twin*: the
same person, same camera, same instant, photographed by the reference sensor.
With DATALOADER.SYNC_FRAMES on, that twin is already in the batch -- rows are
modality-major, so row ``i`` and row ``i + k * per_modality`` are the same
capture -- which is why this costs no extra forward pass.

Why the twin and not a text anchor.  The text form was tried both ways and both
ends failed for the same structural reason: a sentence is ONE vector shared by
every identity, so "get closer to it" and "keep different people apart" are in
direct conflict.  Stopping early (cross entropy) bought +0.1 mAP; not stopping
(a hinge) drove every projected feature to cosine 0.998 with the anchor and
cost -1.22.  The twin target is a different object entirely: it is a different
vector for every person and every frame, and it already carries the identity,
so satisfying it perfectly is exactly what we want rather than a collapse.

Why cosine and not L2.  Retrieval compares directions -- TEST.FEAT_NORM
normalises before the distance matrix -- so matching the norm as well would
spend capacity on something no metric ever reads.

The target is detached.  The reference modality has no bank of its own, and in
v1 the backbone is frozen, so a gradient into the twin would have nowhere to go
anyway; the stop-gradient states that rather than relying on it.
"""

import torch
import torch.nn.functional as F


def twin_align_loss(feat, n_modalities, per_modality, ref_index=0):
    """1 - cos(f_m, sg(f_ref)) over every non-reference row.

    ``feat``          [n_modalities * per_modality, D], modality-major.
    ``ref_index``     which block is the reference (0 = the first modality).

    Returns ``(loss, per_modality_cos)`` where the second is a list of length
    ``n_modalities`` holding the mean cosine to the twin, ``nan`` at the
    reference position.  Those numbers are the diagnostic: the loss itself is a
    weighted sum and says nothing about which spectrum is moving.
    """
    total = n_modalities * per_modality
    if feat.shape[0] != total:
        raise ValueError(
            'twin_align_loss got {} rows for {} modalities x {} per modality; '
            'the batch is not modality-major or SYNC_FRAMES is off'
            .format(feat.shape[0], n_modalities, per_modality))
    if not 0 <= ref_index < n_modalities:
        raise ValueError('ref_index {} outside [0, {})'.format(ref_index, n_modalities))

    f = F.normalize(feat.float(), dim=-1).view(n_modalities, per_modality, -1)
    ref = f[ref_index].detach()

    loss = feat.new_zeros((), dtype=torch.float32)
    stats = []
    for m in range(n_modalities):
        if m == ref_index:
            stats.append(float('nan'))
            continue
        cos = (f[m] * ref).sum(dim=-1)
        loss = loss + (1.0 - cos).mean()
        stats.append(cos.detach().mean().item())
    return loss / max(n_modalities - 1, 1), stats
