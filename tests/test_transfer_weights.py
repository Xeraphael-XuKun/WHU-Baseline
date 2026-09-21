"""Unit tests for carrying a trained checkpoint from one dataset to another.

CPU-only and self-contained.  Run directly::

    python tests/test_transfer_weights.py

This is the CARGO -> WHU-MARS handoff: train the HiHR baseline on CARGO, then
start WHU-MARS from those weights instead of from raw CLIP.  The two datasets
have different identity counts, so the classifier cannot transfer and
everything else must.

Both failure modes here are silent -- the run completes and reports a number
either way -- so they are asserted rather than eyeballed:

  * MODEL.PRETRAIN_CHOICE anything but 'imagenet' used to fall through with no
    else branch, training from random init without a word in the log;
  * a checkpoint from this codebase is keyed `base.blocks...` while a
    backbone-only file is keyed `blocks...`, so feeding one to the other's
    loader matched nothing and carried on.
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.vit_pytorch import TransReID, QuickGELU  # noqa: E402
from model.make_model import build_transformer  # noqa: E402


class Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def cfg(pretrain_choice='self', pretrain_path=''):
    return Cfg(
        MODEL=Cfg(PRETRAIN_PATH=pretrain_path, PRETRAIN_CHOICE=pretrain_choice,
                  COS_LAYER=False, NECK='bnneck', TRANSFORMER_TYPE='tiny',
                  SIE_CAMERA=False, SIE_VIEW=False, SIE_COE=3.0,
                  STRIDE_SIZE=[16, 16], DROP_PATH=0.0, DROP_OUT=0.0, ATT_DROP_RATE=0.0,
                  PE_TYPE='learnable', PE_KEEP_APE=False, CHART_HIDDEN=64,
                  CHART_THETA_MAX_DEG=30.0, CHART_SCALE_MAX=0.35, CHART_INPUT_NORM=False,
                  ROPE_THETA=10.0, ROPE_FREQ_TRAINABLE=False, ROPE_GATE=False, PE_VIEW_GATE='none', PE_PER_MODALITY=False, PE_ZERO_BASED=False,
                  LAYER_SCALE=False, LAYER_SCALE_INIT=1e-4,
                  PE_LAYERWISE='none', PE_LAYERS=[], PE_FREEZE_BASE=False,
                  MOD_DELTA=False, MOD_DELTA_REF='RGB',
                  CE_SPLIT_VIEW=False, CE_SPLIT_MODALITY=False,
                  CE_MODALITY_GROUPS=[], TEXT_MODALITY_TARGETS=[],
                  # Mirrors defaults.py rather than relying on getattr fallbacks
                  # in build_transformer -- a fallback there would hide a config
                  # key that a real run genuinely forgot to set.
                  TEXT_ALIGN=False, TEXT_TARGET='view',
                  TEXT_SUPERVISION='contrast', TEXT_ALLOW_NO_ACTUATOR=False,
                  TEXT_CLIP_PATH='', TEXT_N_CTX=4,
                  TEXT_MODALITY='RGB',
                  TEXT_TEMPLATE='a photo of a X view person with X X X X .'),
        DATASETS=Cfg(MODALITIES=['RGB'], AERIAL_CAMS=[]),
        INPUT=Cfg(SIZE_TRAIN=[256, 128]),
        TEST=Cfg(NECK_FEAT='before', MOD_DELTA=True),
    )


def tiny(**kw):
    """vit_base_clip with one block instead of twelve.

    The width stays 768: build_transformer hardcodes `self.in_planes = 768` for
    the classifier and bottleneck, so a narrower backbone would give a model
    whose head does not match its own trunk -- fine for a load test, misleading
    for anyone reading it later.  Depth is what costs time here, not width.
    """
    kw.pop('img_size', None)
    for drop in ('sie_xishu', 'camera', 'view', 'stride_size', 'drop_path_rate',
                 'drop_rate', 'attn_drop_rate'):
        kw.pop(drop, None)
    return TransReID(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=768,
                     depth=1, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.0,
                     clip_style=True, act_layer=QuickGELU, **kw)


FACTORY = {'tiny': tiny}


def build(num_classes, **over):
    return build_transformer(num_classes, 0, 0, cfg(**over), FACTORY)


def save(model):
    path = os.path.join(tempfile.mkdtemp(), 'transformer_60.pth')
    torch.save(model.state_dict(), path)
    return path


# --------------------------------------------------------------------------

def test_backbone_transfers_and_classifier_does_not():
    """CARGO has ~2400 training identities, WHU-MARS ~500: only the head differs."""
    torch.manual_seed(0)
    source = build(2400, pretrain_choice='no')
    with torch.no_grad():
        for p in source.parameters():
            p.normal_(std=0.05)
    path = save(source)

    torch.manual_seed(1)
    target = build(500, pretrain_choice='self', pretrain_path=path)

    src, dst = source.state_dict(), target.state_dict()
    for k in dst:
        if 'classifier' in k:
            continue
        if k in src and src[k].shape == dst[k].shape:
            assert torch.equal(dst[k], src[k]), '{} did not transfer'.format(k)

    assert dst['classifier.weight'].shape == (500, 768)
    assert src['classifier.weight'].shape == (2400, 768)
    # The head must be this model's own fresh init, not anything from the file.
    torch.manual_seed(1)
    reference = build(500, pretrain_choice='no')
    assert torch.equal(dst['classifier.weight'], reference.state_dict()['classifier.weight'])


def test_bottleneck_and_backbone_are_both_covered():
    """A partial load is the quiet failure; count what actually moved."""
    torch.manual_seed(0)
    source = build(2400, pretrain_choice='no')
    with torch.no_grad():
        source.bottleneck.weight.fill_(0.375)
        source.base.cls_token.fill_(0.125)
    path = save(source)

    target = build(500, pretrain_choice='self', pretrain_path=path)
    assert torch.allclose(target.bottleneck.weight, torch.full_like(target.bottleneck.weight, 0.375))
    assert torch.allclose(target.base.cls_token, torch.full_like(target.base.cls_token, 0.125))


def test_backbone_only_checkpoint_is_rejected_with_a_useful_message():
    """`blocks.0...` vs `base.blocks.0...` -- used to match nothing, silently."""
    torch.manual_seed(0)
    backbone_only = tiny(num_classes=0)
    path = os.path.join(tempfile.mkdtemp(), 'backbone.pth')
    torch.save(backbone_only.state_dict(), path)

    try:
        build(500, pretrain_choice='self', pretrain_path=path)
    except RuntimeError as e:
        assert 'nothing loaded' in str(e) and 'imagenet' in str(e), str(e)
    else:
        raise AssertionError('a backbone-only checkpoint should have been rejected')


def test_unknown_pretrain_choice_raises():
    """It used to fall through and train from scratch, looking entirely normal."""
    for choice in ('finetune', 'selfsup', ''):
        try:
            build(500, pretrain_choice=choice)
        except ValueError as e:
            assert 'PRETRAIN_CHOICE' in str(e), str(e)
        else:
            raise AssertionError('{!r} should have raised'.format(choice))


def test_no_means_no_and_says_so():
    m = build(500, pretrain_choice='no')
    assert isinstance(m.classifier, nn.Linear)


def test_shape_mismatch_refuses_to_load_at_all():
    """A checkpoint from a different architecture must not be half-applied.

    It used to be reported and skipped, which left that one tensor at its
    initialisation while every other weight came from the file -- a model nobody
    built on purpose, evaluated under the checkpoint's name.  For `pos_delta`
    that is the whole method silently reverting to the baseline, so the mismatch
    is fatal now and the message names the keys that change tensor shapes.
    """
    torch.manual_seed(0)
    source = build(2400, pretrain_choice='no')
    sd = source.state_dict()
    sd['base.pos_embed'] = torch.zeros(1, 197, 768)     # 224x224 grid, not ours
    path = os.path.join(tempfile.mkdtemp(), 'wrongsize.pth')
    torch.save(sd, path)

    try:
        build(500, pretrain_choice='self', pretrain_path=path)
    except RuntimeError as exc:
        assert 'shape mismatch' in str(exc), exc
        assert 'base.pos_embed' in str(exc), 'the message must name the offending key'
        assert 'PE_PER_MODALITY' in str(exc), 'and point at what changes shapes'
    else:
        raise AssertionError('a mismatched checkpoint loaded without complaint')


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
