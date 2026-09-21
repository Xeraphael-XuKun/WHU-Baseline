"""Unit tests for the view-alignment loss.

CPU-only, no CLIP weights, no tokenizer.  Run directly::

    python tests/test_text_align.py

The loss is a two-class cross entropy, which is easy to get subtly wrong in
ways that never raise: a flipped target trains the delta backwards, a missing
normalisation lets the loss be minimised by growing feature norms instead of
moving anything, and a detached diagnostic that is *not* detached silently
doubles the graph.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss.text_align import AERIAL, GROUND, text_align_loss, view_logits  # noqa: E402

SCALE = 100.0


def anchors():
    """Two orthogonal anchors, so cosines are unambiguous."""
    t = torch.zeros(2, 8)
    t[AERIAL, 0] = 1.0
    t[GROUND, 1] = 1.0
    return t


def test_perfect_match_costs_almost_nothing():
    t = anchors()
    feat = t[GROUND].unsqueeze(0).repeat(4, 1)          # exactly on the ground anchor
    loss, acc, margin = text_align_loss(feat, t, GROUND, SCALE)
    assert loss.item() < 1e-6, loss.item()
    assert acc.item() == 1.0
    assert abs(margin.item() - 1.0) < 1e-5      # cos 1 with ground, 0 with aerial


def test_wrong_side_is_expensive_and_the_target_is_not_flipped():
    """A flipped target would train the delta in the wrong direction."""
    t = anchors()
    feat = t[AERIAL].unsqueeze(0)
    on_target, _, _ = text_align_loss(feat, t, AERIAL, SCALE)
    off_target, acc, margin = text_align_loss(feat, t, GROUND, SCALE)
    assert on_target.item() < 1e-6
    assert off_target.item() > 50, off_target.item()   # scale 100, cosine gap 1.0
    assert acc.item() == 0.0
    assert abs(margin.item() + 1.0) < 1e-5


def test_normalisation_makes_the_loss_scale_invariant():
    """Otherwise the cheapest way to cut the loss is to grow ||feat||."""
    t = anchors()
    feat = torch.randn(6, 8)
    base = text_align_loss(feat, t, GROUND, SCALE)[0]
    for k in (0.01, 3.0, 1000.0):
        scaled = text_align_loss(feat * k, t, GROUND, SCALE)[0]
        assert torch.allclose(base, scaled, atol=1e-5), k
    # the anchors too -- they are learnable and could otherwise inflate
    assert torch.allclose(base, text_align_loss(feat, t * 7.0, GROUND, SCALE)[0], atol=1e-5)


def test_margin_is_the_graded_diagnostic_we_rely_on():
    """It must move monotonically all the way from aerial to ground.

    The softmax probability cannot do this job: at scale 100 a cosine gap of
    0.63 already saturates it to 1.0 in float32, so it reports which side a
    feature is on but not how far -- and "how far" is what tells us whether the
    delta moved anything.  This test pins the graded behaviour, and also pins
    that the saturating quantity is NOT what gets returned.
    """
    t = anchors()
    prev = -2.0
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):               # interpolate aerial -> ground
        feat = ((1 - w) * t[AERIAL] + w * t[GROUND]).unsqueeze(0)
        m = text_align_loss(feat, t, GROUND, SCALE)[2].item()
        assert m > prev + 1e-4, (w, m, prev)
        prev = m
    assert abs(prev - 1.0) < 1e-5

    # ... and confirm the probability really would have been useless here.
    import torch.nn.functional as _F
    sat = []
    for w in (0.75, 1.0):
        feat = ((1 - w) * t[AERIAL] + w * t[GROUND]).unsqueeze(0)
        sat.append(view_logits(feat, t, SCALE).softmax(dim=1)[0, GROUND].item())
    assert sat[0] == sat[1] == 1.0, sat


def test_per_sample_targets_are_supported():
    t = anchors()
    feat = torch.stack([t[AERIAL], t[GROUND]])
    tgt = torch.tensor([AERIAL, GROUND])
    loss, acc, _ = text_align_loss(feat, t, tgt, SCALE)
    assert loss.item() < 1e-6 and acc.item() == 1.0
    flipped, acc2, _ = text_align_loss(feat, t, torch.tensor([GROUND, AERIAL]), SCALE)
    assert flipped.item() > 50 and acc2.item() == 0.0


def test_gradient_reaches_features_and_anchors():
    t = anchors().clone().requires_grad_(True)
    feat = torch.randn(5, 8, requires_grad=True)
    text_align_loss(feat, t, GROUND, SCALE)[0].backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    assert t.grad is not None and t.grad.abs().sum() > 0


def test_diagnostics_carry_no_graph():
    """acc and p_ground are reported every LOG_PERIOD; if they held a graph the
    backward would traverse it twice."""
    t = anchors()
    feat = torch.randn(4, 8, requires_grad=True)
    loss, acc, p = text_align_loss(feat, t, GROUND, SCALE)
    assert loss.requires_grad
    assert not acc.requires_grad and not p.requires_grad


def test_empty_subset_is_a_no_op():
    """A batch can happen to hold no aerial RGB images; that must not crash and
    must not inject a spurious gradient."""
    t = anchors()
    feat = torch.zeros(0, 8, requires_grad=True)
    loss, acc, p = text_align_loss(feat, t, GROUND, SCALE)
    assert loss.item() == 0.0 and acc.item() == 0.0 and p.item() == 0.0
    assert loss.shape == ()


def test_logits_match_a_hand_computation():
    t = anchors()
    feat = torch.tensor([[3.0, 4.0, 0, 0, 0, 0, 0, 0]])          # norm 5
    got = view_logits(feat, t, SCALE)
    want = torch.tensor([[0.6 * SCALE, 0.8 * SCALE]])            # cos with e0, e1
    assert torch.allclose(got, want, atol=1e-4), (got, want)


def test_matches_plain_cross_entropy_on_normalised_inputs():
    t = anchors()
    feat = torch.randn(7, 8)
    got = text_align_loss(feat, t, GROUND, SCALE)[0]
    logits = F.normalize(feat, dim=-1) @ F.normalize(t, dim=-1).t() * SCALE
    want = F.cross_entropy(logits, torch.full((7,), GROUND))
    assert torch.allclose(got, want, atol=1e-6)


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL  {}: {}: {}'.format(fn.__name__, type(exc).__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
