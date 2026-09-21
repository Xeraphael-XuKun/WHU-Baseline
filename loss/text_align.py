"""View alignment against CLIP text anchors.

Two sentences give two fixed points in CLIP's joint space -- one reading as an
aerial viewpoint, one as a ground viewpoint.  Every supervised image feature is
then asked which of the two it is closer to, and scored with a two-class cross
entropy.  The full supervision is a 2x2:

                    before (no delta)     after (delta applied)
    aerial image      -> aerial                -> ground
    ground image      -> ground                -> ground

Only the top-right cell asks for a *change*; the other three ask for things to
stay put.  That asymmetry is deliberate -- the net pressure on the positional
delta comes from the aerial images, and the rest keeps it from simply
translating every feature.

Contrastive rather than "pull towards the target": pushing away from the other
anchor is what stops the degenerate solution where every feature collapses onto
one point and trivially satisfies its target.
"""

import torch
import torch.nn.functional as F

AERIAL, GROUND = 0, 1          # row order of the text anchors


def view_logits(feat, text, logit_scale):
    """[N, D] image features and [2, D] anchors -> [N, 2] logits.

    Both sides are L2-normalised, so the product is a cosine and the only thing
    setting the scale is `logit_scale` (CLIP's learned temperature, exp(4.6052)
    ~ 100, kept frozen).  Without the normalisation the loss could be minimised
    by growing feature norms rather than by moving anything.
    """
    feat = F.normalize(feat.float(), dim=-1)
    text = F.normalize(text.float(), dim=-1)
    return feat @ text.t() * logit_scale


def text_align_loss(feat, text, target, logit_scale, margin_pair=None):
    """Cross entropy of `feat` against the view anchors.

    `text` holds 2 anchors (one per view) or, when the template carries a
    modality slot, 2 x n_modality of them.  With six anchors the task becomes
    six-way, which is the point: at two classes it saturates almost at once --
    all four accuracies reach 1.00 and the loss 0.000 -- and a saturated loss
    has nothing left to teach.

    target      int, or a [N] long tensor of anchor rows.
    margin_pair None to read the margin off rows (AERIAL, GROUND), or a [N, 2]
                long tensor giving each sample its own (aerial, ground) pair.
                With modality anchors a sample has to be compared against its
                *own* modality's two rows; averaging across modalities would
                mix three different directions into one number.

    Returns ``(loss, accuracy, margin)``; the last two are detached.

    ``margin`` is ``mean(cos(feat, T_ground) - cos(feat, T_aerial))``, in
    [-2, 2].  It is reported instead of the softmax probability because at
    CLIP's temperature (~100) that probability saturates immediately: a cosine
    gap of 0.63 already puts it at 1.0 in float32, so it can say which side a
    feature is on but not how far.  The margin does not saturate, and how far
    is exactly the question -- comparing it before and after the delta is the
    only direct measure of whether the delta moved the viewpoint reading at
    all.  Without it, a run in which nothing was learned looks identical to one
    in which the idea does not work.
    """
    if feat.numel() == 0:
        zero = feat.new_zeros(())
        return zero, zero.clone(), zero.clone()

    logits = view_logits(feat, text, logit_scale)
    n = logits.shape[0]
    if not torch.is_tensor(target):
        target = torch.full((n,), int(target), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == target).float().mean()
        cos = logits / logit_scale
        if margin_pair is None:
            margin = (cos[:, GROUND] - cos[:, AERIAL]).mean()
        else:
            pair = margin_pair.to(cos.device)
            margin = (cos.gather(1, pair[:, 1:2]) - cos.gather(1, pair[:, 0:1])).mean()
    return loss, acc, margin


# ---------------------------------------------------------------------------
# modality target: the same cross entropy, with the axis swapped
# ---------------------------------------------------------------------------
def anchor_cos(feat, text):
    """[N, D] features and [K, D] anchors -> [N, K] cosines, no temperature.

    Deliberately not `view_logits`: these cosines are read straight into the
    log, and multiplying them by ~100 would only make them harder to compare
    with each other.  The training loss applies the temperature itself.
    """
    return F.normalize(feat.float(), dim=-1) @ F.normalize(text.float(), dim=-1).t()
