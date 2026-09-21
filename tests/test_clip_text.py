"""Unit tests for the CLIP text tower, the learnable prompts, and the 2x2
supervision wired end to end through build_transformer.

CPU-only.  Run directly::

    python tests/test_clip_text.py

The tokenizer needs ftfy and regex.  Where they are missing the affected tests
skip *loudly* rather than pass quietly -- the cluster has both, and
verify_upload.sh re-runs this there.

What is pinned here, and why each would otherwise fail in silence:

  * an unloaded text tower must not emit zeros.  torch.empty hands back the
    allocation, which in practice is zeros, and a zero text_projection maps
    every sentence to the same zero vector with nothing raising;
  * the tower must be frozen from construction, not from the load path --
    make_optimizer sweeps up anything with requires_grad, and 63M parameters
    joining the optimizer would not announce itself;
  * the attention must be causal, as CLIP trained it;
  * delta_gate=None must stay bit-identical to the code before this feature;
  * f_before must actually differ from f_after, or the whole supervision is
    comparing a tensor with itself.
"""

import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ftfy only matters for non-ASCII repair; our template is ASCII, so a stub keeps
# the structural tests runnable off-cluster.  The real ftfy runs where training
# runs.
try:
    import ftfy  # noqa: F401
except ImportError:
    _stub = types.ModuleType('ftfy')
    _stub.fix_text = lambda s: s
    sys.modules['ftfy'] = _stub

try:
    import regex  # noqa: F401
    HAVE_TOKENIZER = True
except ImportError:
    HAVE_TOKENIZER = False

from model.backbones.clip_text import CLIPTextEncoder, ViewPrompts  # noqa: E402
from model.backbones.vit_pytorch import QuickGELU, TransReID  # noqa: E402
from model.make_model import build_transformer  # noqa: E402
from loss.text_align import AERIAL, GROUND  # noqa: E402

SKIP = 'needs the tokenizer (regex); run on the cluster'


def need_tokenizer():
    if not HAVE_TOKENIZER:
        print('    SKIP  ' + SKIP)
        return False
    return True


# --------------------------------------------------------------------------
# the tower
# --------------------------------------------------------------------------

def test_unloaded_tower_is_not_silently_dead():
    """torch.empty gives zeros; a zero text_projection zeroes every sentence."""
    torch.manual_seed(0)
    enc = CLIPTextEncoder()
    assert enc.loaded is False
    for name in ('positional_embedding', 'text_projection'):
        p = getattr(enc, name)
        assert not torch.all(p == 0), '{} was left at torch.empty'.format(name)
    emb = torch.randn(2, enc.context_length, enc.width)
    out = enc(emb, torch.tensor([5, 5]))
    assert out.shape == (2, 512)
    assert not torch.all(out == 0)
    assert not torch.equal(out[0], out[1])


def test_tower_is_frozen_from_construction():
    """Not from load_clip: make_optimizer takes everything with requires_grad."""
    enc = CLIPTextEncoder()
    assert all(not p.requires_grad for p in enc.parameters())
    n = sum(p.numel() for p in enc.parameters())
    assert 60e6 < n < 70e6, n          # ~63.4M; worth knowing if it ever moves


def test_attention_is_causal():
    """CLIP trained the text tower autoregressively.  Positions after the one
    we read from must not be able to influence it."""
    torch.manual_seed(0)
    enc = CLIPTextEncoder(layers=2)
    emb = torch.randn(1, enc.context_length, enc.width)
    eot = torch.tensor([9])
    with torch.no_grad():
        a = enc(emb, eot)
        emb2 = emb.clone()
        emb2[:, 10:] = torch.randn_like(emb2[:, 10:])     # everything after EOT
        b = enc(emb2, eot)
    assert torch.allclose(a, b, atol=1e-5), 'later positions leaked backwards'


def test_load_clip_rejects_a_checkpoint_without_a_text_tower():
    enc = CLIPTextEncoder()
    try:
        enc.load_clip({'visual.conv1.weight': torch.zeros(1)})
    except RuntimeError as e:
        assert 'missing' in str(e)
    else:
        raise AssertionError('a text-less checkpoint should have been rejected')


# --------------------------------------------------------------------------
# the prompts
# --------------------------------------------------------------------------

def test_prompt_slots_land_where_the_tokenizer_says():
    if not need_tokenizer():
        return
    torch.manual_seed(0)
    p = ViewPrompts(CLIPTextEncoder())
    assert p.view_slot == 5, p.view_slot
    assert p.ctx_slots == [9, 10, 11, 12], p.ctx_slots
    assert p.eot_index == 14, p.eot_index
    assert p.token_ids[p.eot_index].item() == 49407
    assert p.token_ids[0].item() == 49406


def test_only_the_five_slots_are_learnable():
    if not need_tokenizer():
        return
    p = ViewPrompts(CLIPTextEncoder())
    train = {n for n, v in p.named_parameters() if v.requires_grad}
    assert train == {'view_ctx', 'shared_ctx'}, train
    assert sum(v.numel() for v in (p.view_ctx, p.shared_ctx)) == 6 * 512


def test_view_slots_start_from_the_real_word_embeddings():
    """CLIP already places `aerial` and `ground` sensibly; starting there and
    letting the gradient move them beats starting from noise."""
    if not need_tokenizer():
        return
    enc = CLIPTextEncoder()
    p = ViewPrompts(enc)
    table = enc.token_embedding.weight
    assert torch.equal(p.view_ctx[0], table[12440])     # aerial
    assert torch.equal(p.view_ctx[1], table[2461])      # ground


def test_the_two_sentences_differ_only_through_the_view_slot():
    """T_ground - T_aerial has to be a pure viewpoint direction; if the shared
    context leaked a difference the loss would be moving something else too."""
    if not need_tokenizer():
        return
    torch.manual_seed(0)
    p = ViewPrompts(CLIPTextEncoder())
    with torch.no_grad():
        differ = p()
        p.view_ctx[1] = p.view_ctx[0]                  # make the views identical
        same = p()
    assert not torch.allclose(differ[0], differ[1], atol=1e-4)
    assert torch.allclose(same[0], same[1], atol=1e-5), 'something else differs'


def test_gradient_reaches_the_slots_and_stops_at_the_tower():
    if not need_tokenizer():
        return
    p = ViewPrompts(CLIPTextEncoder())
    p().sum().backward()
    assert p.view_ctx.grad.abs().sum() > 0
    assert p.shared_ctx.grad.abs().sum() > 0
    assert all(v.grad is None for n, v in p.named_parameters() if n.startswith('text.'))


def test_template_mismatch_raises_instead_of_misplacing_slots():
    if not need_tokenizer():
        return
    try:
        ViewPrompts(CLIPTextEncoder(), template='a photo of a X person .', n_ctx=4)
    except RuntimeError as e:
        assert 'placeholder' in str(e), str(e)
    else:
        raise AssertionError('too few placeholders should have raised')


# --------------------------------------------------------------------------
# the delta gate
# --------------------------------------------------------------------------

def small_vit(**kw):
    return TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=64,
                     depth=3, num_heads=4, num_classes=0, drop_path_rate=0.0, **kw)


def test_delta_gate_none_is_bit_identical():
    """Every run before this feature passed no gate; that path must not move."""
    torch.manual_seed(0)
    m = small_vit(pe_layerwise='indep').eval()
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)
        x = torch.randn(4, 3, 256, 128)
        assert torch.equal(m(x), m(x, delta_gate=None))
        ones = m(x, delta_gate=torch.ones(4))
    assert torch.allclose(m(x), ones, atol=1e-6)       # gate of ones == no gate


def test_delta_gate_zero_removes_the_increments():
    torch.manual_seed(0)
    m = small_vit(pe_layerwise='indep').eval()
    plain = small_vit(pe_layerwise='none').eval()
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if k != 'pos_delta'})
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)
        x = torch.randn(4, 3, 256, 128)
        assert torch.allclose(m(x, delta_gate=torch.zeros(4)), plain(x), atol=1e-6)


def test_delta_gate_is_per_image():
    """The gate is what makes 'aerial gets corrected, ground does not' possible
    inside one batch."""
    torch.manual_seed(0)
    m = small_vit(pe_layerwise='indep').eval()
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)
        x = torch.randn(4, 3, 256, 128)
        mixed = m(x, delta_gate=torch.tensor([1.0, 0.0, 1.0, 0.0]))
        on = m(x, delta_gate=torch.ones(4))
        off = m(x, delta_gate=torch.zeros(4))
    assert torch.allclose(mixed[0], on[0], atol=1e-6)
    assert torch.allclose(mixed[1], off[1], atol=1e-6)
    assert not torch.allclose(mixed[1], on[1], atol=1e-4)


# --------------------------------------------------------------------------
# the whole path through build_transformer
# --------------------------------------------------------------------------

class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def cfg(text_align, clip_path='',
        template='a photo of a X view person with X X X X .', mod_words=()):
    return Cfg(
        MODEL=Cfg(PRETRAIN_PATH='', PRETRAIN_CHOICE='no', COS_LAYER=False, NECK='bnneck',
                  TRANSFORMER_TYPE='tiny', SIE_CAMERA=False, SIE_VIEW=False, SIE_COE=3.0,
                  STRIDE_SIZE=[16, 16], DROP_PATH=0.0, DROP_OUT=0.0, ATT_DROP_RATE=0.0,
                  PE_TYPE='learnable', PE_KEEP_APE=False, CHART_HIDDEN=64,
                  CHART_THETA_MAX_DEG=30.0, CHART_SCALE_MAX=0.35, CHART_INPUT_NORM=False,
                  ROPE_THETA=10.0, ROPE_FREQ_TRAINABLE=False, ROPE_GATE=False, PE_VIEW_GATE='none', PE_PER_MODALITY=False, PE_ZERO_BASED=False,
                  LAYER_SCALE=False, LAYER_SCALE_INIT=1e-4,
                  PE_LAYERWISE='indep', PE_LAYERS=[], PE_FREEZE_BASE=False,
                  MOD_DELTA=False, MOD_DELTA_REF='RGB',
                  CE_SPLIT_VIEW=False, CE_SPLIT_MODALITY=False,
                  CE_MODALITY_GROUPS=[], TEXT_MODALITY_TARGETS=[],
                  TEXT_ALIGN=text_align, TEXT_TARGET='view',
                  TEXT_SUPERVISION='contrast', TEXT_ALLOW_NO_ACTUATOR=False,
                  TEXT_CLIP_PATH=clip_path, TEXT_N_CTX=4,
                  TEXT_MODALITY='RGB',
                  TEXT_TEMPLATE=template, TEXT_MODALITY_WORDS=list(mod_words)),
        INPUT=Cfg(SIZE_TRAIN=[256, 128]),
        TEST=Cfg(NECK_FEAT='before', MOD_DELTA=True),
        DATASETS=Cfg(MODALITIES=['RGB', 'IR', 'Thermal'], AERIAL_CAMS=[5, 6]),
    )


def tiny(**kw):
    for drop in ('sie_xishu', 'camera', 'view', 'stride_size', 'drop_path_rate',
                 'drop_rate', 'attn_drop_rate', 'img_size'):
        kw.pop(drop, None)
    return TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=768,
                     depth=1, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0,
                     clip_style=True, act_layer=QuickGELU, **kw)


def fake_clip_file():
    """A checkpoint keyed exactly like the real CLIP release, at real widths.

    Random weights are fine -- what is being tested is the plumbing (key names,
    which tensors reach which module, what the aux dict carries), not accuracy.
    """
    import tempfile
    torch.manual_seed(0)
    enc = CLIPTextEncoder()
    sd = {'token_embedding.weight': enc.token_embedding.weight.clone(),
          'positional_embedding': enc.positional_embedding.clone(),
          'ln_final.weight': enc.ln_final.weight.clone(),
          'ln_final.bias': enc.ln_final.bias.clone(),
          'text_projection': enc.text_projection.clone(),
          'logit_scale': torch.tensor(4.6052)}
    for n, blk in enumerate(enc.resblocks):
        p = 'transformer.resblocks.{}.'.format(n)
        sd[p + 'attn.in_proj_weight'] = blk.attn.in_proj_weight.clone()
        sd[p + 'attn.in_proj_bias'] = blk.attn.in_proj_bias.clone()
        sd[p + 'attn.out_proj.weight'] = blk.attn.out_proj.weight.clone()
        sd[p + 'attn.out_proj.bias'] = blk.attn.out_proj.bias.clone()
        sd[p + 'ln_1.weight'] = blk.ln_1.weight.clone()
        sd[p + 'ln_1.bias'] = blk.ln_1.bias.clone()
        sd[p + 'ln_2.weight'] = blk.ln_2.weight.clone()
        sd[p + 'ln_2.bias'] = blk.ln_2.bias.clone()
        sd[p + 'mlp.c_fc.weight'] = blk.mlp[0].weight.clone()
        sd[p + 'mlp.c_fc.bias'] = blk.mlp[0].bias.clone()
        sd[p + 'mlp.c_proj.weight'] = blk.mlp[2].weight.clone()
        sd[p + 'mlp.c_proj.bias'] = blk.mlp[2].bias.clone()
    path = os.path.join(tempfile.mkdtemp(), 'clip.pt')
    torch.save(sd, path)
    return path


def test_text_align_off_changes_nothing():
    torch.manual_seed(0)
    m = build_transformer(10, 0, 0, cfg(False), {'tiny': tiny})
    assert m.prompts is None
    assert not any('prompts' in k for k in m.state_dict())
    imgs = [torch.randn(2, 3, 256, 128) for _ in range(3)]
    cam = [torch.tensor([0, 5]) for _ in range(3)]
    m.eval()
    with torch.no_grad():
        out = m(imgs, torch.tensor([0, 1]), cam)
    assert len(out) == 3, 'the aux dict leaked into a TEXT_ALIGN=False run'


def test_full_path_produces_a_usable_2x2():
    """The one that matters: f_before must differ from f_after, and the four
    groups must be split by view exactly as the camera ids say."""
    if not need_tokenizer():
        return
    path = fake_clip_file()
    m = build_transformer(10, 0, 0, cfg(True, path), {'tiny': tiny}).eval()
    with torch.no_grad():
        m.base.pos_delta.normal_(std=0.1)
        imgs = [torch.randn(4, 3, 256, 128) for _ in range(3)]
        cams = [torch.tensor([0, 5, 2, 6]) for _ in range(3)]   # ground, aerial, ground, aerial
        out = m(imgs, torch.tensor([0, 1, 2, 3]), cams)

    assert len(out) == 4
    aux = out[3]
    assert aux['is_aerial'].tolist() == [False, True, False, True]
    assert aux['feat_after'].shape == (4, 768)
    assert aux['feat_before'].shape == (4, 768)
    # If these matched, the supervision would be comparing a tensor with itself.
    assert not torch.allclose(aux['feat_before'], aux['feat_after'], atol=1e-4)
    assert aux['text'].shape == (2, 512)
    assert abs(float(aux['logit_scale']) - 100.0) < 0.1
    assert aux['proj'].shape == (768, 512)

    from processor.processor import view_align
    loss, stats = view_align(aux)
    assert torch.isfinite(loss)
    for k in ('acc_A_before', 'acc_A_after', 'acc_G_before', 'acc_G_after',
              'push', 'drift', 'n_aerial'):
        assert k in stats, k
    assert stats['n_aerial'] == 2
    # push and drift are differences of margins, so they must be finite reals
    assert abs(stats['push']) < 4 and abs(stats['drift']) < 4


def test_modality_slot_gives_six_anchors_and_supervises_everything():
    """The six-anchor form: one anchor per (modality, view), all three spectra
    supervised, and the margin read against each sample's OWN pair.

    Comparing a thermal feature against the RGB anchors would measure the wrong
    direction entirely, and nothing about that would raise.
    """
    if not need_tokenizer():
        return
    path = fake_clip_file()
    m = build_transformer(10, 0, 0, cfg(True, path,
                                        template='a photo of a X view X person with X X X X .',
                                        mod_words=('rgb', 'infrared', 'thermal')),
                          {'tiny': tiny}).eval()
    assert m.prompts.n_anchor == 6
    assert m.prompts.modality_slot == 7
    assert m.text_modality_index is None, 'all three modalities should be supervised'

    with torch.no_grad():
        m.base.pos_delta.normal_(std=0.1)
        imgs = [torch.randn(4, 3, 256, 128) for _ in range(3)]
        cams = [torch.tensor([0, 5, 2, 6]) for _ in range(3)]
        aux = m(imgs, torch.tensor([0, 1, 2, 3]), cams)[3]

    assert aux['feat_after'].shape == (12, 768), 'all three modalities, not one'
    assert aux['feat_before'].shape == (12, 768)
    assert aux['modality'].tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert aux['is_aerial'].tolist() == [False, True, False, True] * 3
    assert aux['text'].shape == (6, 512)
    assert aux['n_modality'] == 3 and aux['n_view'] == 2

    from processor.processor import view_align
    loss, stats = view_align(aux)
    assert torch.isfinite(loss)
    assert stats['n_aerial'] == 6 and stats['n_total'] == 12
    assert len(stats['push_per_modality']) == 3
    assert all(abs(v) < 4 for v in stats['push_per_modality'])


def test_two_anchor_form_still_reduces_correctly():
    """With no modality words the modality index must be all zeros, so the
    six-anchor code path collapses onto exactly the old behaviour."""
    if not need_tokenizer():
        return
    path = fake_clip_file()
    m = build_transformer(10, 0, 0, cfg(True, path), {'tiny': tiny}).eval()
    assert m.prompts.n_anchor == 2 and m.prompts.modality_slot is None
    with torch.no_grad():
        imgs = [torch.randn(4, 3, 256, 128) for _ in range(3)]
        cams = [torch.tensor([0, 5, 2, 6]) for _ in range(3)]
        aux = m(imgs, torch.tensor([0, 1, 2, 3]), cams)[3]
    assert aux['feat_after'].shape == (4, 768), 'only the one supervised modality'
    assert aux['modality'].tolist() == [0, 0, 0, 0]
    from processor.processor import view_align
    _, stats = view_align(aux)
    assert 'push_per_modality' not in stats


def test_modality_words_must_line_up_with_the_dataset():
    if not need_tokenizer():
        return
    path = fake_clip_file()
    try:
        build_transformer(10, 0, 0, cfg(True, path,
                                        template='a photo of a X view X person with X X X X .',
                                        mod_words=('rgb', 'thermal')), {'tiny': tiny})
    except ValueError as e:
        assert 'MODALITIES' in str(e), str(e)
    else:
        raise AssertionError('a length mismatch should have raised')


def test_text_align_demands_its_own_clip_path():
    """PRETRAIN_PATH points at our CARGO checkpoint on the WHU-MARS legs, which
    has no text tower; silently reusing it would load nothing."""
    if not need_tokenizer():
        return
    try:
        build_transformer(10, 0, 0, cfg(True, ''), {'tiny': tiny})
    except ValueError as e:
        assert 'TEXT_CLIP_PATH' in str(e), str(e)
    else:
        raise AssertionError('a missing TEXT_CLIP_PATH should have raised')


if __name__ == '__main__':
    if not HAVE_TOKENIZER:
        print('NOTE: regex is not installed here -- tokenizer-dependent tests will')
        print('      skip.  The cluster has it; verify_upload.sh re-runs this there.\n')
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
