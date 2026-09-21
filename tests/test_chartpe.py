"""Unit tests for the ChartPE positional-encoding path.

CPU-only and self-contained: no dataset, no pretrained weights, no yacs.
Run directly::

    python tests/test_chartpe.py

Every test targets one specific way the implementation could be silently wrong
(wrong flatten order, clobbered identity init, wrong rotary pairing, ...), so a
failure here should point straight at the cause.
"""

import os
import sys
import math
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.chartpe import (  # noqa: E402
    ChartRotaryEmbedding, TopologyPreservingChart, build_centered_grid, init_2d_freqs, rotate_half,
)
from model.backbones.vit_pytorch import TransReID, vit_base_in  # noqa: E402


def randomize_(module):
    """Break identity init so monotonicity/det tests see non-trivial charts."""
    for head in (module.row_head, module.col_head, module.global_head):
        torch.nn.init.normal_(head.weight, std=0.5)
        torch.nn.init.normal_(head.bias, std=0.5)


def test_flatten_order():
    """PatchEmbed's token order must match how we reshape the chart back."""
    x = torch.randn(2, 5, 16, 8)  # [B, C, H, W]
    from_patch_embed = x.flatten(2).transpose(1, 2)          # what PatchEmbed does
    from_chart_view = x.permute(0, 2, 3, 1).reshape(2, 128, 5)  # row-major, y outer
    assert torch.equal(from_patch_embed, from_chart_view)


def test_identity_init():
    """Zero-initialised TPCG must reproduce the fixed grid exactly."""
    tpcg = TopologyPreservingChart(64, hidden=32, grid_h=16, grid_w=8).eval()
    q, stats = tpcg(torch.randn(3, 16, 8, 64))
    grid = build_centered_grid(16, 8)
    assert torch.allclose(q.reshape(3, 128, 2), grid.unsqueeze(0).expand(3, -1, -1), atol=1e-5)
    assert stats['theta'].abs().max().item() == 0.0
    assert stats['scale'].abs().max().item() == 0.0

    # Regression guard for the real landmine: TransReID ends __init__ with
    # self.apply(self._init_weights), which re-inits every nn.Linear. If the
    # chart generator were built before that call, these would be non-zero.
    model = vit_base_in(img_size=(256, 128), pe_type='chartpe', drop_path_rate=0.0)
    for name in ('row_head', 'col_head', 'global_head'):
        head = getattr(model.chart_generator, name)
        assert head.weight.abs().max().item() == 0.0, '{} was clobbered by self.apply'.format(name)
        assert head.bias.abs().max().item() == 0.0


def test_monotonic_positive():
    """Spacings stay strictly positive and axes strictly increasing."""
    for grid_h, grid_w in [(16, 8), (18, 9), (24, 12)]:
        tpcg = TopologyPreservingChart(64, hidden=32, grid_h=16, grid_w=8).eval()
        randomize_(tpcg)
        q, stats = tpcg(torch.randn(2, grid_h, grid_w, 64))
        assert q.shape == (2, grid_h, grid_w, 2)
        assert torch.isfinite(q).all()
        assert stats['dx'].min().item() > 0.0
        assert stats['dy'].min().item() > 0.0
        # Column x increases left to right, row y increases top to bottom.
        # (Read off row 0 / column 0 before the global rotation mixes the axes.)
        col_axis = stats['dx'].cumsum(dim=1)
        row_axis = stats['dy'].cumsum(dim=1)
        assert (col_axis[:, 1:] > col_axis[:, :-1]).all()
        assert (row_axis[:, 1:] > row_axis[:, :-1]).all()


def test_det_A():
    """The global transform is area preserving by construction."""
    tpcg = TopologyPreservingChart(64, hidden=32, grid_h=16, grid_w=8).eval()
    randomize_(tpcg)
    _, stats = tpcg(torch.randn(8, 16, 8, 64))
    det = torch.det(stats['A'])
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)
    # Limits must hold too, otherwise the chart could deform without bound.
    assert stats['theta'].abs().max().item() <= math.pi / 6 + 1e-6
    assert stats['scale'].abs().max().item() <= 0.35 + 1e-6


def test_rotary_pairing():
    """Guards the classic bug: adjacent-pair vs half-split rotary convention."""
    rope = ChartRotaryEmbedding(8, 2, theta=10.0).eval()
    q = torch.randn(1, 2, 3, 8)
    chart = torch.randn(1, 3, 2)
    q_out, k_out = rope(q, q.clone(), chart)
    assert torch.allclose(q_out, k_out)

    phase = torch.einsum('bnd,hcd->bhnc', chart, rope.freqs)
    phase_rep = torch.repeat_interleave(phase, 2, dim=-1)
    # A phase is shared by the two channels of a pair.
    assert torch.allclose(phase_rep[..., 0::2], phase_rep[..., 1::2])

    # Compare one pair against an explicit 2x2 rotation matrix.
    for c in range(4):
        a, b = q[0, 0, 0, 2 * c], q[0, 0, 0, 2 * c + 1]
        ang = phase[0, 0, 0, c]
        expect = torch.stack([a * ang.cos() - b * ang.sin(), a * ang.sin() + b * ang.cos()])
        got = q_out[0, 0, 0, 2 * c:2 * c + 2]
        assert torch.allclose(expect, got, atol=1e-5)

    # Rotation preserves length.
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-4)

    # rotate_half itself: (a, b) -> (-b, a)
    v = torch.tensor([1., 2., 3., 4.])
    assert torch.equal(rotate_half(v), torch.tensor([-2., 1., -4., 3.]))


def test_phase_formula():
    """phase = omega_x * x + omega_y * y, per head and channel pair."""
    rope = ChartRotaryEmbedding(8, 3, theta=10.0).eval()
    chart = torch.randn(2, 4, 2)
    phase = torch.einsum('bnd,hcd->bhnc', chart, rope.freqs)
    assert phase.shape == (2, 3, 4, 4)
    for b in range(2):
        for h in range(3):
            for n in range(4):
                for c in range(4):
                    expect = chart[b, n, 0] * rope.freqs[h, c, 0] + chart[b, n, 1] * rope.freqs[h, c, 1]
                    assert torch.allclose(phase[b, h, n, c], expect, atol=1e-6)


def test_shift_invariance():
    """Only relative position reaches attention (patch-patch block).

    The CLS row is pinned at phase zero, so CLS-patch scores legitimately do
    depend on absolute coordinates -- hence the [1:, 1:] slice.
    """
    torch.manual_seed(0)
    rope = ChartRotaryEmbedding(16, 2, theta=10.0).eval()
    q, k = torch.randn(1, 2, 9, 16), torch.randn(1, 2, 9, 16)
    chart = torch.cat([torch.zeros(1, 1, 2), torch.randn(1, 8, 2)], dim=1)

    def logits(c):
        qr, kr = rope(q, k, c)
        return qr @ kr.transpose(-2, -1)

    shifted = chart + torch.tensor([1.7, -2.3])
    shifted[:, 0] = 0.0  # CLS keeps its zero phase
    assert torch.allclose(logits(chart)[..., 1:, 1:], logits(shifted)[..., 1:, 1:], atol=1e-4)


def test_cls_untouched():
    """A zero chart row leaves the CLS token bit-exactly unchanged."""
    rope = ChartRotaryEmbedding(16, 2, theta=10.0).eval()
    q, k = torch.randn(2, 2, 5, 16), torch.randn(2, 2, 5, 16)
    chart = torch.cat([torch.zeros(2, 1, 2), torch.randn(2, 4, 2)], dim=1)
    q_out, k_out = rope(q, k, chart)
    assert torch.equal(q_out[:, :, 0], q[:, :, 0])
    assert torch.equal(k_out[:, :, 0], k[:, :, 0])
    assert not torch.equal(q_out[:, :, 1], q[:, :, 1])  # patches did move


def test_forward_smoke():
    """All three PE modes run end to end and produce finite features."""
    for pe_type in ('learnable', 'rope_fixed', 'chartpe'):
        torch.manual_seed(0)
        model = vit_base_in(img_size=(256, 128), pe_type=pe_type, drop_path_rate=0.0).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 256, 128))
        assert out.shape == (2, 768), pe_type
        assert torch.isfinite(out).all(), pe_type
        if pe_type == 'chartpe':
            stats = model.last_chart_stats
            assert stats is not None
            # Identity init: chart still sits on the fixed grid (up to fp32 noise),
            # spacings are softplus(0) + eps, and the global transform is off.
            assert stats['q_dev'] < 1e-5, stats
            assert abs(stats['min_dx'] - (math.log(2.0) + 1e-4)) < 1e-5, stats
            assert stats['theta_max'] == 0.0 and stats['s_max'] == 0.0, stats
            assert stats['s_mean'] == 0.0, stats
            # Identity init: every spacing equal, so the skew ratio is exactly 1.
            assert abs(stats['dx_skew'] - 1.0) < 1e-5, stats
            assert abs(stats['dy_skew'] - 1.0) < 1e-5, stats
        if pe_type == 'learnable':
            assert model.pos_embed is not None
        else:
            assert model.pos_embed is None


def test_keep_ape():
    """Hybrid modes keep the pos_embed table AND rotate Q/K."""
    for pe_type in ('rope_fixed', 'chartpe'):
        torch.manual_seed(0)
        plain = vit_base_in(img_size=(256, 128), pe_type=pe_type, drop_path_rate=0.0).eval()
        torch.manual_seed(0)
        hybrid = vit_base_in(img_size=(256, 128), pe_type=pe_type, pe_keep_ape=True,
                             drop_path_rate=0.0).eval()

        assert plain.pos_embed is None, pe_type
        assert hybrid.pos_embed is not None, pe_type
        assert hybrid.pos_embed.shape == (1, 129, 768)
        # Rotary machinery is present in both.
        assert hybrid.blocks[0].attn.rope is not None
        assert 'pos_embed' in hybrid.state_dict()

        x = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out_plain, out_hybrid = plain(x), hybrid(x)
        assert torch.isfinite(out_hybrid).all()
        # The extra additive table must actually change the output.
        assert not torch.allclose(out_plain, out_hybrid)

    # A hybrid checkpoint carries pos_embed and reloads through the resize path.
    small = dict(img_size=(256, 128), embed_dim=32, depth=1, num_heads=2, num_classes=0)
    path = os.path.join(tempfile.mkdtemp(), 'ape.pth')
    torch.save({'pos_embed': torch.randn(1, 197, 32), 'cls_token': torch.randn(1, 1, 32)}, path)
    m = TransReID(pe_type='chartpe', pe_keep_ape=True, **small)
    m.load_param(path)
    assert m.pos_embed.shape == (1, 129, 32)


def test_learnable_regression():
    """'learnable' must be byte-identical to the untouched implementation."""
    torch.manual_seed(123)
    ref = vit_base_in(img_size=(256, 128))
    ref_next = torch.rand(1)

    torch.manual_seed(123)
    new = vit_base_in(img_size=(256, 128), pe_type='learnable')
    new_next = torch.rand(1)

    # No extra RNG draws: anything else would silently shift the whole init.
    assert torch.equal(ref_next, new_next)

    ref_sd, new_sd = ref.state_dict(), new.state_dict()
    assert set(ref_sd) == set(new_sd)
    assert not any('rope' in k or 'chart' in k for k in new_sd)
    for k in ref_sd:
        assert torch.equal(ref_sd[k], new_sd[k]), k

    ref.eval(), new.eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        assert torch.equal(ref(x), new(x))


def test_load_param_skip():
    """A checkpoint carrying pos_embed must not break a rotary model."""
    small = dict(img_size=(256, 128), embed_dim=32, depth=1, num_heads=2, num_classes=0)
    fake = {
        'pos_embed': torch.randn(1, 197, 32),
        'cls_token': torch.randn(1, 1, 32),
    }
    path = os.path.join(tempfile.mkdtemp(), 'fake.pth')
    torch.save(fake, path)

    for pe_type in ('rope_fixed', 'chartpe'):
        model = TransReID(pe_type=pe_type, **small)
        model.load_param(path)  # must not raise
        assert torch.equal(model.cls_token, fake['cls_token'])

    # The learnable model still takes the resize path.
    model = TransReID(pe_type='learnable', **small)
    model.load_param(path)
    assert model.pos_embed.shape == (1, 129, 32)


def test_chart_input_norm():
    """Instance norm on TPCG's input: identity-init survives, and a per-image
    per-channel offset -- the shape a modality difference takes -- stops moving
    the chart once the flag is on."""
    torch.manual_seed(0)
    plain = TopologyPreservingChart(24, hidden=16, grid_h=6, grid_w=4).eval()
    torch.manual_seed(0)
    normed = TopologyPreservingChart(24, hidden=16, grid_h=6, grid_w=4, input_norm=True).eval()

    assert plain.input_norm is False and normed.input_norm is True
    # The flag adds no parameters, so the two share a state_dict shape for shape.
    assert [t.shape for t in plain.state_dict().values()] == \
           [t.shape for t in normed.state_dict().values()]

    x = torch.randn(3, 6, 4, 24)
    with torch.no_grad():
        # Identity initialisation must still reproduce the fixed grid exactly.
        grid = build_centered_grid(6, 4).reshape(6, 4, 2)
        for m in (plain, normed):
            q, stats = m(x)
            assert torch.allclose(q, grid.expand(3, -1, -1, -1), atol=1e-5)
            assert stats['theta'].abs().max() == 0.0

        # Randomise the heads so the chart actually depends on the input.
        for m in (plain, normed):
            for head in (m.row_head, m.col_head, m.global_head):
                nn.init.normal_(head.weight, std=0.5)
                nn.init.normal_(head.bias, std=0.5)

        # A per-image, per-channel shift and gain is exactly what distinguishes
        # one sensor from another; instance norm must make the chart blind to it.
        shift = torch.randn(3, 1, 1, 24) * 3.0
        gain = torch.rand(3, 1, 1, 24) * 2.0 + 0.5
        x2 = x * gain + shift

        q_plain_a, _ = plain(x)
        q_plain_b, _ = plain(x2)
        q_norm_a, _ = normed(x)
        q_norm_b, _ = normed(x2)

    moved = (q_plain_a - q_plain_b).abs().mean().item()
    stayed = (q_norm_a - q_norm_b).abs().mean().item()
    assert moved > 1e-3, 'the probe must actually perturb the un-normalised chart'
    assert stayed < 1e-5, 'normalised chart moved by {} (plain moved {})'.format(stayed, moved)


def test_chart_input_norm_wiring():
    """MODEL.CHART_INPUT_NORM reaches the chart generator and nothing else."""
    torch.manual_seed(0)
    off = vit_base_in(img_size=(256, 128), pe_type='chartpe', pe_keep_ape=True,
                      drop_path_rate=0.0).eval()
    torch.manual_seed(0)
    on = vit_base_in(img_size=(256, 128), pe_type='chartpe', pe_keep_ape=True,
                     chart_input_norm=True, drop_path_rate=0.0).eval()

    assert off.chart_generator.input_norm is False
    assert on.chart_generator.input_norm is True
    # Same parameters, same init -- the flag is pure behaviour, not capacity.
    a, b = off.state_dict(), on.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)

    with torch.no_grad():
        out = on(torch.randn(2, 3, 256, 128))
    assert out.shape == (2, 768) and torch.isfinite(out).all()
    # Identity init is untouched by the flag: chart still on the grid at step 0.
    assert on.last_chart_stats['q_dev'] < 1e-5, on.last_chart_stats


def test_layer_scale_off_is_a_noop():
    """With MODEL.LAYER_SCALE off, the model is bit-identical to before."""
    torch.manual_seed(0)
    plain = vit_base_in(img_size=(256, 128), pe_type='learnable', drop_path_rate=0.0).eval()
    torch.manual_seed(0)
    explicit = vit_base_in(img_size=(256, 128), pe_type='learnable', layer_scale=False,
                           drop_path_rate=0.0).eval()
    a, b = plain.state_dict(), explicit.state_dict()
    assert a.keys() == b.keys()
    assert not any('gamma' in k for k in a), sorted(k for k in a if 'gamma' in k)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert plain.blocks[0].gamma_1 is None


def test_layer_scale_shapes_and_identity():
    """LayerScale adds exactly 2 vectors per block, and gamma == 1 is a no-op."""
    torch.manual_seed(0)
    off = vit_base_in(img_size=(256, 128), pe_type='learnable', drop_path_rate=0.0).eval()
    torch.manual_seed(0)
    # init 1.0 makes the scaling an identity, so the two models must agree exactly.
    on = vit_base_in(img_size=(256, 128), pe_type='learnable', layer_scale=True,
                     layer_scale_init=1.0, drop_path_rate=0.0).eval()

    extra = set(on.state_dict()) - set(off.state_dict())
    assert len(extra) == 2 * len(on.blocks), sorted(extra)
    assert all(k.endswith(('gamma_1', 'gamma_2')) for k in extra)
    assert on.state_dict()['blocks.0.gamma_1'].shape == (768,)

    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        assert torch.allclose(off(x), on(x), atol=1e-5)

    # ... and a gamma that is not 1 must actually change the output, otherwise
    # the parameter is dead weight that silently does nothing.
    with torch.no_grad():
        on.blocks[0].gamma_1.fill_(0.5)
        assert not torch.allclose(off(x), on(x), atol=1e-4)


def test_zero_based_grid():
    """PE_ZERO_BASED reproduces RoPE-ViT's t_x = (t % end_x) exactly."""
    g = build_centered_grid(16, 8, zero_based=True)
    assert torch.equal(g[:, 0].unique(), torch.arange(8, dtype=torch.float32))
    assert torch.equal(g[:, 1].unique(), torch.arange(16, dtype=torch.float32))
    # Row-major: the first 8 tokens are y=0, x=0..7 -- same order as PatchEmbed.
    assert torch.equal(g[:8, 0], torch.arange(8, dtype=torch.float32))
    assert (g[:8, 1] == 0).all()
    # Centred and zero-based differ by a constant, nothing else.
    c = build_centered_grid(16, 8)
    assert torch.allclose(g - c, torch.tensor([3.5, 7.5]).expand_as(g))

    # An identity-initialised TPCG must land on it, or the pretrained rotary
    # phases would be offset from step 0.
    tpcg = TopologyPreservingChart(dim=16, hidden=8, grid_h=16, grid_w=8, zero_based=True)
    q, _ = tpcg(torch.randn(2, 16, 8, 16))
    assert torch.allclose(q.reshape(2, 128, 2), g.expand(2, -1, -1), atol=1e-5)


def test_deit3_freqs_scatter():
    """RoPE-ViT's single `freqs` tensor lands in the right block, head and pair."""
    torch.manual_seed(0)
    model = vit_base_in(img_size=(256, 128), pe_type='rope_fixed', pe_keep_ape=True,
                        drop_path_rate=0.0).eval()
    depth, heads = len(model.blocks), 12
    pairs = model.blocks[0].attn.rope.freqs.shape[1]
    # Their layout: stack(per_layer [2, heads, pairs], dim=1).view(2, depth, -1)
    per_layer = [torch.randn(2, heads, pairs) for _ in range(depth)]
    official = torch.stack(per_layer, dim=1).view(2, depth, -1)
    assert official.shape == (2, depth, heads * pairs)

    n = model._load_rope_freqs(official)
    assert n == depth
    for i, blk in enumerate(model.blocks):
        assert torch.allclose(blk.attn.rope.freqs, per_layer[i].permute(1, 2, 0))

    # A depth mismatch must raise rather than load a scrambled subset.
    try:
        model._load_rope_freqs(torch.randn(2, depth + 1, heads * pairs))
    except RuntimeError:
        pass
    else:
        raise AssertionError('depth mismatch should have raised')


def test_deit3_checkpoint_load():
    """A DeiT-III-shaped checkpoint loads: LayerScale, CLS-less pos_embed, freqs."""
    torch.manual_seed(0)
    model = TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=32,
                      depth=2, num_heads=4, pe_type='rope_fixed', pe_keep_ape=True,
                      layer_scale=True, pe_zero_based=True, drop_path_rate=0.0).eval()
    heads, pairs = model.blocks[0].attn.rope.freqs.shape[:2]
    ckpt = {
        'model': {
            # 14x14 patches, no CLS row -- DeiT-III gives CLS no position at all.
            'pos_embed': torch.randn(1, 196, 32),
            'cls_token': torch.randn(1, 1, 32),
            'blocks.0.gamma_1': torch.full((32,), 0.3),
            'blocks.1.gamma_2': torch.full((32,), 0.7),
            'freqs': torch.randn(2, 2, heads * pairs),
            'freqs_t_x': torch.arange(196.0),      # must be ignored, not loaded
            'freqs_t_y': torch.arange(196.0),
            'head.weight': torch.randn(1000, 32),  # must be skipped
        }
    }
    path = os.path.join(tempfile.mkdtemp(), 'deit3.pth')
    torch.save(ckpt, path)
    model.load_param(path)

    assert torch.allclose(model.blocks[0].gamma_1, torch.full((32,), 0.3))
    assert torch.allclose(model.blocks[1].gamma_2, torch.full((32,), 0.7))
    # 196 patch rows resized to 16x8 = 128, plus a CLS row that must stay zero.
    assert model.pos_embed.shape == (1, 129, 32)
    assert torch.equal(model.pos_embed[0, 0], torch.zeros(32))
    assert model.pos_embed[0, 1:].abs().sum() > 0
    exp = ckpt['model']['freqs'][:, 0, :].view(2, heads, pairs).permute(1, 2, 0)
    assert torch.allclose(model.blocks[0].attn.rope.freqs, exp)

    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 128))
    assert out.shape == (2, 32) and torch.isfinite(out).all()

    # Loading the same checkpoint without LayerScale must fail loudly: the rest
    # of those weights are only valid with gamma_1/gamma_2 in place.
    torch.manual_seed(0)
    no_ls = TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=32,
                      depth=2, num_heads=4, pe_type='rope_fixed', pe_keep_ape=True,
                      drop_path_rate=0.0).eval()
    try:
        no_ls.load_param(path)
    except RuntimeError as e:
        assert 'LAYER_SCALE' in str(e), str(e)
    else:
        raise AssertionError('missing LayerScale should have raised')


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
