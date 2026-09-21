"""Unit tests for loading OpenAI CLIP's image tower (TRANSFORMER_TYPE vit_base_clip).

CPU-only and self-contained: no dataset, no real CLIP download, no yacs.
Run directly::

    python tests/test_clip_backbone.py

Loading a checkpoint under the wrong architecture is the failure mode that does
not announce itself -- every tensor copies cleanly, the model trains, and it
just lands a point or two low.  So the central test here does not check names:
it builds a reference implementation of CLIP's own VisionTransformer.forward,
puts the same weights in both, and demands the outputs agree.  That covers
QuickGELU vs GELU, the extra ln_pre, the q/k/v packing order and the residual
ordering in one shot.

The key names and shapes come from the real ViT-B-16.pt on the cluster
(sha256 5806e77c..., 305 tensors, 152 visual, 12 resblocks), not from memory.
"""

import os
import sys
import tempfile
from functools import partial

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.vit_pytorch import (  # noqa: E402
    QuickGELU, TransReID, vit_base_clip, vit_base_in,
)

DIM, DEPTH, HEADS = 64, 2, 4


def fake_clip_state_dict(dim=DIM, depth=DEPTH, grid=14, text=True):
    """A checkpoint with CLIP's exact key names, at test dimensions."""
    g = torch.Generator().manual_seed(7)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    sd = {
        'visual.conv1.weight': r(dim, 3, 16, 16),
        'visual.class_embedding': r(dim),
        'visual.positional_embedding': r(grid * grid + 1, dim),
        'visual.ln_pre.weight': r(dim) + 1.0,
        'visual.ln_pre.bias': r(dim),
        'visual.ln_post.weight': r(dim) + 1.0,
        'visual.ln_post.bias': r(dim),
        'visual.proj': r(dim, 512),
    }
    for n in range(depth):
        p = 'visual.transformer.resblocks.{}.'.format(n)
        sd.update({
            p + 'attn.in_proj_weight': r(3 * dim, dim),
            p + 'attn.in_proj_bias': r(3 * dim),
            p + 'attn.out_proj.weight': r(dim, dim),
            p + 'attn.out_proj.bias': r(dim),
            p + 'ln_1.weight': r(dim) + 1.0,
            p + 'ln_1.bias': r(dim),
            p + 'ln_2.weight': r(dim) + 1.0,
            p + 'ln_2.bias': r(dim),
            p + 'mlp.c_fc.weight': r(4 * dim, dim),
            p + 'mlp.c_fc.bias': r(4 * dim),
            p + 'mlp.c_proj.weight': r(dim, 4 * dim),
            p + 'mlp.c_proj.bias': r(dim),
        })
    if text:  # the text tower rides along in the real file; we must ignore it
        sd['token_embedding.weight'] = r(49408, 512)
        sd['text_projection'] = r(512, 512)
        sd['logit_scale'] = torch.tensor(4.6052)
    return sd


def build_clip_vit(img_size=(224, 224), dim=DIM, depth=DEPTH, heads=HEADS):
    return TransReID(img_size=img_size, patch_size=16, stride_size=16, embed_dim=dim,
                     depth=depth, num_heads=heads, mlp_ratio=4, qkv_bias=True,
                     num_classes=0, drop_path_rate=0.0,
                     norm_layer=partial(nn.LayerNorm, eps=1e-5),
                     clip_style=True, act_layer=QuickGELU).eval()


def save(sd):
    path = os.path.join(tempfile.mkdtemp(), 'clip.pt')
    torch.save(sd, path)
    return path


# --------------------------------------------------------------------------
# the reference: OpenAI's clip/model.py VisionTransformer, transcribed
# --------------------------------------------------------------------------

class RefBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
        self.ln_1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), QuickGELU(), nn.Linear(dim * 4, dim))
        self.ln_2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class RefVisual(nn.Module):
    """Mirrors CLIP's VisionTransformer.forward, including the NLD<->LND flips."""

    def __init__(self, dim, depth, heads, grid):
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, 16, 16, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(dim))
        self.positional_embedding = nn.Parameter(torch.zeros(grid * grid + 1, dim))
        self.ln_pre = nn.LayerNorm(dim)
        self.resblocks = nn.ModuleList([RefBlock(dim, heads) for _ in range(depth)])
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND
        for blk in self.resblocks:
            x = blk(x)
        x = x.permute(1, 0, 2)          # LND -> NLD
        return self.ln_post(x[:, 0, :])


def load_ref(ref, sd):
    own = ref.state_dict()
    own['conv1.weight'].copy_(sd['visual.conv1.weight'])
    own['class_embedding'].copy_(sd['visual.class_embedding'])
    own['positional_embedding'].copy_(sd['visual.positional_embedding'])
    for a, b in (('ln_pre', 'visual.ln_pre'), ('ln_post', 'visual.ln_post')):
        own[a + '.weight'].copy_(sd[b + '.weight'])
        own[a + '.bias'].copy_(sd[b + '.bias'])
    for n in range(len(ref.resblocks)):
        p, q = 'resblocks.{}.'.format(n), 'visual.transformer.resblocks.{}.'.format(n)
        for suffix in ('attn.in_proj_weight', 'attn.in_proj_bias',
                       'attn.out_proj.weight', 'attn.out_proj.bias',
                       'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias'):
            own[p + suffix].copy_(sd[q + suffix])
        own[p + 'mlp.0.weight'].copy_(sd[q + 'mlp.c_fc.weight'])
        own[p + 'mlp.0.bias'].copy_(sd[q + 'mlp.c_fc.bias'])
        own[p + 'mlp.2.weight'].copy_(sd[q + 'mlp.c_proj.weight'])
        own[p + 'mlp.2.bias'].copy_(sd[q + 'mlp.c_proj.bias'])


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_matches_reference_clip_forward():
    """The one that matters: same weights in, same features out."""
    sd = fake_clip_state_dict()
    model = build_clip_vit()
    model.load_param(save(sd))

    ref = RefVisual(DIM, DEPTH, HEADS, grid=14).eval()
    load_ref(ref, sd)

    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        got, want = model(x), ref(x)
    err = (got - want).abs().max().item()
    assert err < 1e-5, 'max abs diff {:.3e} -- the architectures disagree'.format(err)


def test_wrong_activation_or_missing_lnpre_would_have_been_caught():
    """Guard on the guard: prove the reference test is sensitive to both bugs."""
    sd = fake_clip_state_dict()
    path = save(sd)
    ref = RefVisual(DIM, DEPTH, HEADS, grid=14).eval()
    load_ref(ref, sd)
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        want = ref(x)

    # standard GELU instead of QuickGELU
    m = TransReID(img_size=(224, 224), patch_size=16, stride_size=16, embed_dim=DIM,
                  depth=DEPTH, num_heads=HEADS, mlp_ratio=4, qkv_bias=True, num_classes=0,
                  drop_path_rate=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-5),
                  clip_style=True, act_layer=nn.GELU).eval()
    m.load_param(path)
    with torch.no_grad():
        assert not torch.allclose(m(x), want, atol=1e-5), 'GELU/QuickGELU swap went unnoticed'

    # ln_pre present but bypassed
    m2 = build_clip_vit()
    m2.load_param(path)
    m2.ln_pre = None
    with torch.no_grad():
        assert not torch.allclose(m2(x), want, atol=1e-5), 'missing ln_pre went unnoticed'


def test_every_mapped_tensor_is_the_source_tensor():
    sd = fake_clip_state_dict()
    model = build_clip_vit()
    model.load_param(save(sd))
    own = model.state_dict()

    assert torch.equal(own['patch_embed.proj.weight'], sd['visual.conv1.weight'])
    assert torch.equal(own['cls_token'].reshape(-1), sd['visual.class_embedding'])
    assert torch.equal(own['pos_embed'][0], sd['visual.positional_embedding'])
    assert torch.equal(own['clip_proj'], sd['visual.proj'])
    assert torch.equal(own['norm.weight'], sd['visual.ln_post.weight'])
    for n in range(DEPTH):
        p = 'visual.transformer.resblocks.{}.'.format(n)
        assert torch.equal(own['blocks.{}.attn.qkv.weight'.format(n)], sd[p + 'attn.in_proj_weight'])
        assert torch.equal(own['blocks.{}.attn.proj.bias'.format(n)], sd[p + 'attn.out_proj.bias'])
        assert torch.equal(own['blocks.{}.mlp.fc1.weight'.format(n)], sd[p + 'mlp.c_fc.weight'])
        assert torch.equal(own['blocks.{}.mlp.fc2.weight'.format(n)], sd[p + 'mlp.c_proj.weight'])
        assert torch.equal(own['blocks.{}.norm2.bias'.format(n)], sd[p + 'ln_2.bias'])


def test_patch_conv_bias_is_zeroed():
    """CLIP's conv1 has no bias; ours does.  Zero makes them identical."""
    model = build_clip_vit()
    with torch.no_grad():
        model.patch_embed.proj.bias.fill_(3.0)
    model.load_param(save(fake_clip_state_dict()))
    assert torch.count_nonzero(model.patch_embed.proj.bias) == 0
    assert model.patch_embed.proj.bias.requires_grad is True


def test_pos_embed_is_interpolated_to_the_reid_grid():
    """224x224 -> 14x14 = 197 rows; 256x128 -> 16x8 = 129."""
    model = build_clip_vit(img_size=(256, 128))
    assert model.pos_embed.shape == (1, 129, DIM)
    model.load_param(save(fake_clip_state_dict()))
    assert model.pos_embed.shape == (1, 129, DIM)
    assert torch.isfinite(model.pos_embed).all()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 128))
    assert out.shape == (2, DIM) and torch.isfinite(out).all()


def test_plain_vit_refuses_clip_weights():
    """Loading CLIP into a GELU/no-ln_pre ViT must fail loudly, not silently."""
    plain = TransReID(img_size=(224, 224), patch_size=16, stride_size=16, embed_dim=DIM,
                      depth=DEPTH, num_heads=HEADS, num_classes=0, drop_path_rate=0.0)
    try:
        plain.load_param(save(fake_clip_state_dict()))
    except RuntimeError as e:
        assert 'vit_base_clip' in str(e), str(e)
    else:
        raise AssertionError('plain ViT should have refused CLIP weights')


def test_reads_a_torchscript_archive():
    """OpenAI ships a JIT archive, and that path has to work end to end.

    torch.load does *not* raise on one: since torch 2.0 it detects the zip
    format and dispatches to torch.jit.load, returning a module rather than a
    dict.  So load_param cannot rely on an exception -- it has to notice it was
    handed an nn.Module.  This test exists because the first version did rely
    on the exception, and nothing else would have caught it.
    """

    class Holder(nn.Module):
        """Parameter holder whose state_dict keys are CLIP's, dots and all."""

        def __init__(self, sd):
            super().__init__()
            for k, v in sd.items():
                path, leaf = k.split('.')[:-1], k.split('.')[-1]
                mod = self
                for part in path:
                    if part not in mod._modules:
                        mod.add_module(part, nn.Module())
                    mod = mod._modules[part]
                mod.register_parameter(leaf, nn.Parameter(v.clone(), requires_grad=False))

        def forward(self, x):
            return x

    sd = fake_clip_state_dict(text=False)
    path = os.path.join(tempfile.mkdtemp(), 'jit.pt')
    torch.jit.save(torch.jit.script(Holder(sd)), path)

    loaded = torch.load(path, map_location='cpu', weights_only=False)
    assert isinstance(loaded, nn.Module), 'torch.load stopped dispatching JIT archives'
    assert not isinstance(loaded, dict)
    assert set(loaded.state_dict()) == set(sd), 'archive lost or renamed tensors'

    # ... and the real loader swallows it whole.
    model = build_clip_vit()
    model.load_param(path)
    assert torch.equal(model.state_dict()['patch_embed.proj.weight'], sd['visual.conv1.weight'])
    assert torch.equal(model.state_dict()['clip_proj'], sd['visual.proj'])


def test_factory_pins_the_three_clip_differences():
    m = vit_base_clip(img_size=(256, 128), drop_path_rate=0.0).eval()
    assert m.ln_pre is not None
    assert isinstance(m.blocks[0].mlp.act, QuickGELU)
    assert m.norm.eps == 1e-5 and m.blocks[0].norm1.eps == 1e-5
    assert m.clip_proj.shape == (768, 512)
    with torch.no_grad():
        out = m(torch.randn(2, 3, 256, 128))
    assert out.shape == (2, 768) and torch.isfinite(out).all()

    # ... and leaves the ImageNet backbone exactly as it was.
    plain = vit_base_in(img_size=(256, 128), drop_path_rate=0.0)
    assert plain.ln_pre is None and plain.clip_proj is None
    assert isinstance(plain.blocks[0].mlp.act, nn.GELU)
    assert plain.norm.eps == 1e-6


def test_layerwise_residuals_compose_with_clip():
    """The advisor's two designs have to be able to sit on this backbone."""
    for mode in ('indep', 'chain'):
        m = vit_base_clip(img_size=(256, 128), pe_layerwise=mode, drop_path_rate=0.0).eval()
        assert m.pos_delta.shape == (12, 1, 129, 768)
        with torch.no_grad():
            assert torch.isfinite(m(torch.randn(2, 3, 256, 128))).all()


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
