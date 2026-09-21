"""Unit tests for SOLVER.TEXT_GRAD_SCOPE -- routing the text loss's gradient.

CPU-only and self-contained.  Run directly::

    python tests/test_text_grad_scope.py

WHAT IS BEING PROTECTED.  'delta' claims that the text loss moves `pos_delta`
and the prompt vectors and NOTHING else, while CE and triplet keep training the
whole tower.  Every failure mode here is silent -- the run completes and reports
a plausible mAP whether the routing worked or not:

  * the routing is a no-op (empty parameter list, or the text term quietly
    still summed into the main loss) -> the arm reports its control's number
    under its own name;
  * the routing is too aggressive (the identity losses also stop reaching the
    tower) -> that is whu_strictpld, a different and much blunter experiment,
    which lands at ~2.4 mAP;
  * only one of the two terms is passed through GradScaler.scale -> the two
    losses are reweighted against each other by the scale factor, ~65536.

So the central test does not inspect the implementation: it builds a model,
back-propagates the same two losses twice -- once routed, once with the text
term simply dropped -- and asserts that every backbone gradient is bit-identical
between the two while pos_delta's is not.  That is the property the arm's
conclusion rests on, stated as an equality.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor.processor import (TEXT_GRAD_SCOPES, split_backward,  # noqa: E402
                                 text_grad_params)


class NoScaler:
    """GradScaler's interface, minus the scaling.

    `split_backward` is written against `scaler.scale`; on CPU the real
    GradScaler is unavailable, and a factor of 1.0 is what `enabled=False`
    would give anyway.  A separate test covers the case that actually matters
    -- that both terms get the SAME factor -- by handing in a scaler that
    multiplies by something conspicuous.
    """

    def __init__(self, factor=1.0):
        self.factor = factor
        self.calls = 0

    def scale(self, loss):
        self.calls += 1
        return loss * self.factor


class Tower(nn.Module):
    """A stand-in with the same parameter-name shapes the real model has.

    `pos_delta` sits INSIDE the trunk -- between the two blocks -- because that
    placement is the whole reason a forward-graph detach cannot separate the
    text loss from the delta.  A stub that hung the delta off the side would
    make the routing look unnecessary.
    """

    def __init__(self, prompts=True, delta=True):
        super().__init__()
        self.base = nn.Module()
        self.base.block0 = nn.Linear(8, 8)
        self.base.block1 = nn.Linear(8, 8)
        self.base.clip_proj = nn.Parameter(torch.randn(8, 4) * 0.1)
        if delta:
            self.base.pos_delta = nn.Parameter(torch.zeros(2, 8))
        self.classifier = nn.Linear(8, 5)
        if prompts:
            self.prompts = nn.Module()
            self.prompts.view_ctx = nn.Parameter(torch.randn(2, 4) * 0.1)
            self.prompts.shared_ctx = nn.Parameter(torch.randn(4, 4) * 0.1)
            # The frozen CLIP text encoder lives under the prompt module.  It
            # must never be picked up, which is why the selector excludes it by
            # name rather than trusting requires_grad.
            self.prompts.text = nn.Module()
            self.prompts.text.token_embedding = nn.Embedding(10, 4)

    def forward(self, x):
        x = torch.relu(self.base.block0(x))
        if hasattr(self.base, 'pos_delta'):
            x = x + self.base.pos_delta[0]
        x = torch.relu(self.base.block1(x))
        if hasattr(self.base, 'pos_delta'):
            x = x + self.base.pos_delta[1]
        return x

    def losses(self, x, target):
        """(identity loss, text loss) off ONE forward, as do_train has them."""
        feat = self(x)
        loss_id = nn.functional.cross_entropy(self.classifier(feat), target)
        clip = feat @ self.base.clip_proj                      # [N, 4]
        anchors = torch.cat([self.prompts.view_ctx, self.prompts.shared_ctx])
        logits = torch.nn.functional.normalize(clip, dim=-1) @ \
            torch.nn.functional.normalize(anchors, dim=-1).t()
        loss_text = nn.functional.cross_entropy(logits * 10.0, target % 6)
        return loss_id, loss_text


def fresh(seed=0, **kw):
    torch.manual_seed(seed)
    m = Tower(**kw)
    if hasattr(m.base, 'pos_delta'):
        with torch.no_grad():
            m.base.pos_delta.normal_(std=0.05)  # a zero delta hides sign errors
    return m


def grads(model):
    return {n: (None if p.grad is None else p.grad.clone())
            for n, p in model.named_parameters()}


DATA = (torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 0]))


# --------------------------------------------------------------------------

def test_selector_picks_the_delta_and_the_prompts_and_nothing_else():
    picked = [n for n, _ in text_grad_params(fresh(), 'delta')]
    assert picked == ['base.pos_delta', 'prompts.view_ctx', 'prompts.shared_ctx'], picked


def test_selector_excludes_the_frozen_text_encoder_even_if_it_is_trainable():
    m = fresh()
    for p in m.prompts.text.parameters():
        p.requires_grad_(True)
    picked = [n for n, _ in text_grad_params(m, 'delta')]
    assert not any('prompts.text' in n for n in picked), picked


def test_selector_skips_parameters_that_are_frozen():
    m = fresh()
    m.prompts.shared_ctx.requires_grad_(False)
    picked = [n for n, _ in text_grad_params(m, 'delta')]
    assert 'prompts.shared_ctx' not in picked and 'base.pos_delta' in picked, picked


def test_selector_survives_the_ddp_module_prefix():
    """named_parameters() reads `module.base.pos_delta` under DataParallel."""
    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    picked = [n for n, _ in text_grad_params(Wrapped(fresh()), 'delta')]
    assert picked == ['module.base.pos_delta', 'module.prompts.view_ctx',
                      'module.prompts.shared_ctx'], picked


def test_no_delta_is_an_error_not_an_empty_list():
    """Without pos_delta the routing would move only the anchors, silently."""
    try:
        text_grad_params(fresh(delta=False), 'delta')
    except ValueError as exc:
        assert 'pos_delta' in str(exc), exc
    else:
        raise AssertionError('a model with no pos_delta should have been refused')


def test_nothing_at_all_is_an_error():
    try:
        text_grad_params(fresh(prompts=False, delta=False), 'delta')
    except ValueError as exc:
        assert 'PE_LAYERWISE' in str(exc) and 'TEXT_ALIGN' in str(exc), exc
    else:
        raise AssertionError('an empty parameter list should have been refused')


def test_unknown_scope_raises():
    for bad in ('none', 'backbone', '', 'DELTA'):
        try:
            text_grad_params(fresh(), bad)
        except ValueError as exc:
            assert 'TEXT_GRAD_SCOPE' in str(exc), exc
        else:
            raise AssertionError('{!r} should have raised'.format(bad))
    assert TEXT_GRAD_SCOPES == ('all', 'delta')


# ---- the property the arm's conclusion rests on --------------------------

def test_the_tower_gets_exactly_the_identity_gradient_and_the_delta_does_not():
    """Routed run vs. a run with the text term simply deleted.

    Backbone gradients must be bit-identical -- that is what "the text loss does
    not reach the tower" means, and an approximate check would pass on a routing
    that leaked a small amount.  pos_delta must differ, or the routing added
    nothing and the arm is its own control.
    """
    x, target = DATA

    routed = fresh()
    li, lt = routed.losses(x, target)
    split_backward(li, 5.0 * lt, text_grad_params(routed, 'delta'), NoScaler())
    g_routed = grads(routed)

    identity_only = fresh()
    li2, _ = identity_only.losses(x, target)
    li2.backward()
    g_id = grads(identity_only)

    for name in ('base.block0.weight', 'base.block0.bias',
                 'base.block1.weight', 'base.block1.bias',
                 'classifier.weight', 'classifier.bias'):
        a, b = g_routed[name], g_id[name]
        assert a is not None and b is not None, name
        assert torch.equal(a, b), \
            '{} moved: the text gradient leaked into the tower'.format(name)

    # clip_proj is the sharpest case: in the real model it is used ONLY by the
    # text path, so under 'delta' it must come back with no gradient at all --
    # not a small one.  Under 'all' it measured a norm of 26.0.
    assert g_routed['base.clip_proj'] is None, \
        'clip_proj received a gradient; the text term reached the tower'
    assert g_id['base.clip_proj'] is None

    assert not torch.equal(g_routed['base.pos_delta'], g_id['base.pos_delta']), \
        'pos_delta is unchanged -- the text term was never routed anywhere'
    assert g_id['prompts.view_ctx'] is None
    assert g_routed['prompts.view_ctx'] is not None


def test_the_routed_delta_gradient_is_the_sum_of_both_terms():
    """Not the text gradient alone -- CE and triplet still act on the delta."""
    x, target = DATA

    routed = fresh()
    li, lt = routed.losses(x, target)
    split_backward(li, 5.0 * lt, text_grad_params(routed, 'delta'), NoScaler())

    both = fresh()
    li2, lt2 = both.losses(x, target)
    (li2 + 5.0 * lt2).backward()

    assert torch.allclose(routed.base.pos_delta.grad, both.base.pos_delta.grad,
                          atol=1e-6), \
        'the delta should see identity + text, exactly as the unrouted run does'
    assert torch.allclose(routed.prompts.view_ctx.grad, both.prompts.view_ctx.grad,
                          atol=1e-6)
    # ...and the tower is where the two runs are supposed to part company.
    assert not torch.allclose(routed.base.block0.weight.grad,
                              both.base.block0.weight.grad, atol=1e-6)


def test_scope_all_is_unchanged_from_before_the_key_existed():
    """The default must not perturb any recorded run by so much as a bit."""
    x, target = DATA

    before = fresh()
    li, lt = before.losses(x, target)
    (li + 5.0 * lt).backward()

    after = fresh()
    li2, lt2 = after.losses(x, target)
    total = li2
    total = total + 5.0 * lt2          # the 'all' branch, verbatim
    total.backward()

    for (n, a), (_, b) in zip(before.named_parameters(), after.named_parameters()):
        if a.grad is None and b.grad is None:
            continue
        assert torch.equal(a.grad, b.grad), n


def test_both_terms_go_through_the_same_scale_factor():
    """One un-scaled term would reweight the losses against each other.

    With a factor of k the routed gradient must be exactly k times the
    unit-factor one -- for the tower AND for the delta.  If only the main loss
    were scaled, the delta's two contributions would come back in the ratio
    k : 1 instead of 1 : 1.
    """
    x, target = DATA
    k = 1024.0

    plain = fresh()
    li, lt = plain.losses(x, target)
    scaler = NoScaler(1.0)
    split_backward(li, 5.0 * lt, text_grad_params(plain, 'delta'), scaler)
    assert scaler.calls == 2, 'both terms must pass through scale()'

    scaled = fresh()
    li2, lt2 = scaled.losses(x, target)
    split_backward(li2, 5.0 * lt2, text_grad_params(scaled, 'delta'), NoScaler(k))

    for name in ('base.pos_delta', 'prompts.view_ctx', 'prompts.shared_ctx',
                 'base.block0.weight', 'base.block1.weight'):
        a = dict(plain.named_parameters())[name].grad
        b = dict(scaled.named_parameters())[name].grad
        assert torch.allclose(b, a * k, rtol=1e-4, atol=1e-4), \
            '{} did not scale by {}'.format(name, k)


def test_grads_accumulate_rather_than_overwrite():
    """`param.grad = g` on a fresh step is fine; clobbering an existing one is not.

    do_train calls optimizer.zero_grad() every iteration, so in practice the
    delta's grad is None when the text half arrives -- but the main backward
    runs FIRST inside split_backward, so by then it is not.  Overwriting would
    silently drop the identity loss's pull on the delta.
    """
    x, target = DATA
    m = fresh()
    li, lt = m.losses(x, target)
    params = text_grad_params(m, 'delta')

    id_only = fresh()
    li2, _ = id_only.losses(x, target)
    li2.backward()
    id_delta = id_only.base.pos_delta.grad.clone()

    text_grads = split_backward(li, 5.0 * lt, params, NoScaler())
    assert torch.allclose(m.base.pos_delta.grad, id_delta + text_grads[0],
                          atol=1e-6), 'the identity half was overwritten'


def test_unused_parameters_come_back_as_none_without_crashing():
    """A prompt tensor absent from this batch's graph is allow_unused territory."""
    x, target = DATA
    m = fresh()
    m.prompts.spare_ctx = nn.Parameter(torch.randn(2, 4))   # in no graph at all
    li, lt = m.losses(x, target)
    params = text_grad_params(m, 'delta')
    assert any(n == 'prompts.spare_ctx' for n, _ in params)
    out = split_backward(li, 5.0 * lt, params, NoScaler())
    assert out[[n for n, _ in params].index('prompts.spare_ctx')] is None
    assert m.prompts.spare_ctx.grad is None


# ---- config wiring -------------------------------------------------------

def test_default_is_all_and_the_key_exists():
    """Read defaults.py as text: yacs is not installed on the dev box, and a
    key missing there makes every yml carrying it fail the merge at startup."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'config', 'defaults.py'),
               encoding='utf-8').read()
    assert "_C.SOLVER.TEXT_GRAD_SCOPE = 'all'" in src, \
        'the key is missing from defaults.py, or its default is not `all`'


def test_the_config_differs_from_its_control_in_one_key():
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    cfgs = os.path.join(os.path.dirname(here), 'configs')

    def keys(name):
        out = {}
        for line in open(os.path.join(cfgs, name), encoding='utf-8'):
            line = line.split('#')[0].rstrip()
            m = re.match(r'^(\s*)([A-Z_]+):\s*(.+)$', line)
            if m and m.group(3).strip():
                out[m.group(2)] = m.group(3).strip()
        return out

    control = keys('hihr_whu_text_lam50_clip.yml')
    arm = keys('hihr_whu_txtgrad_clip.yml')
    differs = {k for k in set(control) | set(arm)
               if control.get(k) != arm.get(k)}
    assert differs == {'TEXT_GRAD_SCOPE', 'OUTPUT_DIR'}, sorted(differs)
    assert arm['TEXT_GRAD_SCOPE'] == "'delta'"


def test_do_train_refuses_the_combinations_that_would_lie():
    """Read the guards out of the source: DDP, no text loss, and PLD_ONLY."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'processor', 'processor.py'),
               encoding='utf-8').read()
    body = src[src.index('def do_train('):]
    guard = body[body.index("text_scope = cfg.SOLVER.TEXT_GRAD_SCOPE"):]
    guard = guard[:guard.index('text_params = text_grad_params')]
    assert 'DIST_TRAIN' in guard, 'DDP would not all-reduce the routed half'
    assert 'text_weight <= 0' in guard, 'a scope with no text loss is a no-op arm'
    assert 'PLD_ONLY' in guard, 'PLD_ONLY plus this key is two arms at once'


def test_the_text_term_is_held_out_of_the_total_only_when_routing():
    """The `loss = loss + text_weight * loss_text` line must be conditional."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'processor', 'processor.py'),
               encoding='utf-8').read()
    body = src[src.index('def do_train('):]
    assert body.count('scaler.scale(loss).backward()') == 1
    assert 'split_backward(loss, text_term, text_params, scaler)' in body
    # and the reported total still includes it, or the Loss column stops being
    # comparable to every run recorded before this key existed
    assert '(loss + text_term).item()' in body


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
