"""Unit tests for the layer-wise positional residuals (MODEL.PE_LAYERWISE).

CPU-only and self-contained: no dataset, no pretrained weights, no yacs.
Run directly::

    python tests/test_pe_layerwise.py

The design contract is narrow and worth pinning down precisely:

  * 'none' must stay bit-for-bit the old model, including the RNG stream;
  * 'indep' and 'chain' must start *exactly* at the baseline (zero init), so
    that any mAP difference is attributable to the residuals and nothing else;
  * 'chain' must be the running sum of 'indep', which is the only thing that
    distinguishes the two strategies;
  * the inherited pretrained table must survive load_param under both.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.vit_pytorch import TransReID, vit_base_in  # noqa: E402


SMALL = dict(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=32,
             depth=4, num_heads=4, num_classes=0, drop_path_rate=0.0)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    opts = dict(SMALL)
    opts.update(kw)
    return TransReID(**opts).eval()


def test_none_is_bit_identical_regression():
    """The default must not perturb the existing model in any way."""
    torch.manual_seed(0)
    ref = TransReID(**SMALL).eval()
    ref_next = torch.rand(1)

    torch.manual_seed(0)
    new = TransReID(pe_layerwise='none', **SMALL).eval()
    new_next = torch.rand(1)

    assert set(ref.state_dict()) == set(new.state_dict()), 'key set changed'
    for k in ref.state_dict():
        assert torch.equal(ref.state_dict()[k], new.state_dict()[k]), k
    # A stray random draw during construction would silently decorrelate every
    # later run from the recorded baseline.
    assert torch.equal(ref_next, new_next), 'RNG stream consumed'

    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        assert torch.equal(ref(x), new(x))


def test_zero_init_starts_at_baseline():
    """Both strategies reproduce the single-injection baseline at step 0."""
    x = torch.randn(2, 3, 256, 128)
    base = build(pe_layerwise='none')
    with torch.no_grad():
        want = base(x)
    for mode in ('indep', 'chain'):
        m = build(pe_layerwise=mode)
        assert torch.count_nonzero(m.pos_delta) == 0, mode
        with torch.no_grad():
            got = m(x)
        assert torch.allclose(want, got, atol=0, rtol=0), mode


def test_new_parameters_are_exactly_the_deltas():
    base = build(pe_layerwise='none')
    m = build(pe_layerwise='indep')
    extra = set(m.state_dict()) - set(base.state_dict())
    assert extra == {'pos_delta'}, extra
    depth, num_patches, dim = SMALL['depth'], m.patch_embed.num_patches, SMALL['embed_dim']
    assert m.pos_delta.shape == (depth, 1, num_patches + 1, dim)


def test_stream_carries_what_each_strategy_promises():
    """The contract is what the residual stream HOLDS, not what gets added.

    Pre-norm blocks add x back untouched, so every increment persists (see
    test_residual_stream_is_never_normalised below).  Adding delta[l] at each
    block therefore accumulates by itself -- which is 'chain'.  Getting
    'indep' requires cancelling the previous increment.

    Getting this backwards costs nothing at runtime and everything at
    reporting time: the two runs would be mislabelled as the advisor's two
    strategies while actually measuring accumulation against double
    accumulation.
    """
    depth = SMALL['depth']
    for mode in ('indep', 'chain'):
        m = build(pe_layerwise=mode)
        with torch.no_grad():
            m.pos_delta.normal_(std=0.1)
        # The stream at block l holds the running total of everything added
        # at blocks 0..l, because nothing in between normalises it.
        carried = torch.cumsum(m._layerwise_deltas(), dim=0)
        for l in range(depth):
            want = m.pos_delta[l] if mode == 'indep' else m.pos_delta[:l + 1].sum(dim=0)
            assert torch.allclose(carried[l], want, atol=1e-6), \
                '{} block {}: stream carries the wrong thing'.format(mode, l)


def test_residual_stream_is_never_normalised():
    """The fact the two strategies are built on.  Pin it here, not in prose.

    If someone ever switches these blocks to post-norm (x = norm(x + attn(x))),
    increments would decay layer by layer and both strategies above would
    silently stop meaning what they say.
    """
    from model.backbones.vit_pytorch import Block
    torch.manual_seed(0)
    blk = Block(dim=32, num_heads=4, qkv_bias=True, drop_path=0.).eval()
    with torch.no_grad():                       # zero both branch outputs
        blk.attn.proj.weight.zero_(); blk.attn.proj.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        x = torch.randn(2, 5, 32) * 7 + 3       # non-zero mean, large variance
        assert torch.equal(blk(x), x), 'the residual stream is being normalised'


def test_the_two_strategies_actually_differ():
    """They agree at block 0 and diverge from block 1 on."""
    ref = build(pe_layerwise='indep')
    with torch.no_grad():
        ref.pos_delta.normal_(std=0.1)
    out = {}
    for mode in ('indep', 'chain'):
        m = build(pe_layerwise=mode)
        with torch.no_grad():
            m.pos_delta.copy_(ref.pos_delta)
        out[mode] = torch.cumsum(m._layerwise_deltas(), dim=0)
    assert torch.allclose(out['indep'][0], out['chain'][0], atol=1e-6)   # same at block 0
    for l in range(1, SMALL['depth']):
        assert not torch.allclose(out['indep'][l], out['chain'][l], atol=1e-4), l


def test_deltas_reach_the_output():
    """A non-zero delta must change the features -- otherwise it is dead code."""
    x = torch.randn(2, 3, 256, 128)
    for mode in ('indep', 'chain'):
        m = build(pe_layerwise=mode)
        with torch.no_grad():
            before = m(x)
            m.pos_delta.normal_(std=0.1)
            after = m(x)
        assert not torch.allclose(before, after, atol=1e-5), mode

    # Only the last block's delta is applied after every earlier one, so under
    # 'indep' touching a single layer must move the output too.
    m = build(pe_layerwise='indep')
    with torch.no_grad():
        before = m(x)
        m.pos_delta[SMALL['depth'] - 1].normal_(std=0.1)
        after = m(x)
    assert not torch.allclose(before, after, atol=1e-5)


def test_gradient_flows_to_every_layer_delta():
    x = torch.randn(2, 3, 256, 128)
    for mode in ('indep', 'chain'):
        m = build(pe_layerwise=mode)
        m.train()
        m(x).square().mean().backward()
        g = m.pos_delta.grad
        assert g is not None, mode
        for l in range(SMALL['depth']):
            assert g[l].abs().sum() > 0, '{} layer {} got no gradient'.format(mode, l)


def test_freeze_base_stops_gradient_at_the_table_only():
    x = torch.randn(2, 3, 256, 128)
    m = build(pe_layerwise='indep', pe_freeze_base=True)
    assert m.pos_embed.requires_grad is False
    m.train()
    m(x).square().mean().backward()
    assert m.pos_embed.grad is None
    assert m.pos_delta.grad is not None and m.pos_delta.grad.abs().sum() > 0

    # ... and the default leaves it trainable, matching the baseline.
    m2 = build(pe_layerwise='indep')
    assert m2.pos_embed.requires_grad is True


def test_load_param_fills_the_table_and_leaves_deltas_at_zero():
    """The inherited weights are the whole point; the deltas must stay new."""
    torch.manual_seed(0)
    donor = TransReID(**SMALL)
    with torch.no_grad():
        donor.pos_embed.normal_(std=0.5)
    ckpt = {k: v for k, v in donor.state_dict().items() if k != 'pos_delta'}
    path = os.path.join(tempfile.mkdtemp(), 'fake.pth')
    torch.save(ckpt, path)

    for freeze in (False, True):
        m = build(seed=1, pe_layerwise='chain', pe_freeze_base=freeze)
        m.load_param(path)
        assert torch.allclose(m.pos_embed, donor.pos_embed, atol=0, rtol=0), freeze
        assert torch.count_nonzero(m.pos_delta) == 0, freeze
        # Freezing must not be undone by the load.
        assert m.pos_embed.requires_grad is (not freeze)


def test_rejects_bad_mode_and_missing_table():
    try:
        build(pe_layerwise='cumulative')
    except ValueError as e:
        assert 'pe_layerwise' in str(e), str(e)
    else:
        raise AssertionError('bad mode should have raised')

    # Rotary modes drop the table, so there is nothing to inherit; that has to
    # fail loudly rather than silently training deltas off a zero base.
    try:
        build(pe_type='rope_fixed', pe_layerwise='indep')
    except ValueError as e:
        assert 'PE_KEEP_APE' in str(e), str(e)
    else:
        raise AssertionError('missing pos_embed should have raised')

    # With the table explicitly kept, the combination is legal.
    m = build(pe_type='rope_fixed', pe_keep_ape=True, pe_layerwise='indep')
    assert m.pos_delta is not None


def test_full_size_smoke():
    """The real backbone, at the real input size, in both modes."""
    for mode in ('indep', 'chain'):
        m = vit_base_in(img_size=(256, 128), pe_layerwise=mode, drop_path_rate=0.0).eval()
        assert m.pos_delta.shape == (12, 1, 129, 768)
        with torch.no_grad():
            out = m(torch.randn(2, 3, 256, 128))
        assert out.shape == (2, 768) and torch.isfinite(out).all(), mode


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
