"""Twin-anchored modality residuals (MODEL.MOD_DELTA + SOLVER.TWIN_LOSS_WEIGHT).

The whole design rests on claims that are cheap to state and easy to break
silently, so each one gets a test that would fail loudly instead:

  * the banks are additive and zero-initialised, so step 0 is the untouched
    backbone -- if that slips, "the correction did it" stops being provable;
  * the reference modality receives nothing, structurally.  This is the fix for
    what sank the text target, where the anchors were free to move and the
    whole system drifted onto one point;
  * with MOD_DELTA_ONLY the backbone is frozen, which is what makes an
    evaluation with TEST.MOD_DELTA False equal to the baseline bit for bit;
  * the twin target is detached, so no gradient reaches the reference features.

Run:  python tests/test_twin_delta.py
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss.twin_align import twin_align_loss                     # noqa: E402
from model.backbones.vit_pytorch import TransReID               # noqa: E402


def _load(name, *parts):
    """Load a module by path: `datasets/__init__` pulls in make_dataloader,
    which needs timm.  bases.py itself needs nothing beyond torch and PIL, so
    the augmentation tests stay runnable on a machine without the training
    dependencies -- the same trick tests/test_sync_sampler.py uses."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bases = _load('bases_mod', 'datasets', 'bases.py')
ImageDataset, split_erasing = _bases.ImageDataset, _bases.split_erasing

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                                     # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


SMALL = dict(img_size=(256, 128), patch_size=16, stride_size=16, embed_dim=32,
             depth=4, num_heads=4, num_classes=0, drop_path_rate=0.0)
N_MOD = 3          # RGB (reference) + IR + Thermal


def build(seed=0, **kw):
    torch.manual_seed(seed)
    opts = dict(SMALL)
    opts.update(kw)
    return TransReID(**opts).eval()


def batch(n_per=2):
    torch.manual_seed(7)
    return torch.randn(N_MOD * n_per, 3, 256, 128), \
        torch.arange(N_MOD).repeat_interleave(n_per)


# --------------------------------------------------------------------------
# 1. additive and zero-initialised
# --------------------------------------------------------------------------
def test_off_by_default_is_bit_identical():
    torch.manual_seed(0)
    ref = TransReID(**SMALL).eval()
    ref_next = torch.rand(1)
    torch.manual_seed(0)
    new = TransReID(mod_delta_modalities=0, **SMALL).eval()
    new_next = torch.rand(1)

    assert set(ref.state_dict()) == set(new.state_dict()), 'key set changed'
    # A stray draw during construction would decorrelate every later run from
    # the recorded baselines.
    assert torch.equal(ref_next, new_next), 'RNG stream consumed'
    x, _m = batch()
    with torch.no_grad():
        assert torch.equal(ref(x), new(x))


def test_zero_init_reproduces_the_backbone():
    x, modal = batch()
    base = build()
    m = build(mod_delta_modalities=N_MOD - 1)
    assert torch.count_nonzero(m.mod_delta) == 0
    with torch.no_grad():
        assert torch.allclose(base(x), m(x, modal_label=modal), atol=0, rtol=0)


def test_the_only_new_parameter_is_the_bank():
    base = build()
    m = build(mod_delta_modalities=N_MOD - 1)
    assert set(m.state_dict()) - set(base.state_dict()) == {'mod_delta'}
    assert m.mod_delta.shape == (N_MOD - 1, SMALL['depth'],
                                 m.patch_embed.num_patches + 1, SMALL['embed_dim'])


def test_no_label_means_no_correction():
    """`modal_label=None` must leave the model untouched even with a live bank:
    that is the path every existing caller takes."""
    x, modal = batch()
    m = build(mod_delta_modalities=N_MOD - 1)
    with torch.no_grad():
        m.mod_delta.normal_(std=0.1)
        base = build()
        assert torch.allclose(m(x), base(x), atol=0, rtol=0)
        assert not torch.allclose(m(x, modal_label=modal), base(x))


# --------------------------------------------------------------------------
# 2. the reference modality is structurally untouched
# --------------------------------------------------------------------------
def test_reference_rows_never_move():
    """The property the text target could not have.  Whatever the banks hold,
    modality 0 comes out exactly as the frozen backbone would produce it."""
    x, modal = batch(n_per=3)
    m = build(mod_delta_modalities=N_MOD - 1)
    base = build()
    with torch.no_grad():
        m.mod_delta.normal_(std=0.5)
        got = m(x, modal_label=modal)
        want = base(x)
        ref_rows = modal == 0
        assert torch.allclose(got[ref_rows], want[ref_rows], atol=0, rtol=0)
        assert not torch.allclose(got[~ref_rows], want[~ref_rows])


def test_each_bank_moves_only_its_own_modality():
    x, modal = batch(n_per=3)
    m = build(mod_delta_modalities=N_MOD - 1)
    with torch.no_grad():
        before = m(x, modal_label=modal)
        m.mod_delta[0].normal_(std=0.5)          # bank row 0 == modality 1
        after = m(x, modal_label=modal)
        moved = ~torch.isclose(before, after, atol=0, rtol=0).all(dim=1)
        assert torch.equal(moved, modal == 1), (moved.tolist(), modal.tolist())


def test_out_of_range_label_raises():
    x, _m = batch()
    m = build(mod_delta_modalities=N_MOD - 1)
    for bad in (torch.full((x.shape[0],), N_MOD), torch.full((x.shape[0],), -1)):
        try:
            m(x, modal_label=bad)
        except ValueError:
            continue
        raise AssertionError('accepted label %d for %d modalities' % (bad[0], N_MOD))


# --------------------------------------------------------------------------
# 3. the twin loss
# --------------------------------------------------------------------------
def test_zero_when_already_on_the_twin():
    f = torch.randn(4, 8)
    feat = torch.cat([f, f.clone(), f.clone()], dim=0)
    loss, cos = twin_align_loss(feat, N_MOD, 4)
    assert abs(float(loss)) < 1e-6, float(loss)
    assert cos[0] != cos[0], cos                 # nan at the reference
    assert all(abs(c - 1.0) < 1e-6 for c in cos[1:]), cos


def test_scale_free_matches_direction_only():
    """Retrieval normalises before measuring, so a pure norm change must not
    register as misalignment."""
    f = torch.randn(4, 8)
    feat = torch.cat([f, 3.0 * f, 0.2 * f], dim=0)
    loss, _c = twin_align_loss(feat, N_MOD, 4)
    assert abs(float(loss)) < 1e-6, float(loss)


def test_loss_falls_as_the_spectrum_approaches_its_twin():
    torch.manual_seed(0)
    ref = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    other = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    losses = []
    for t in (0.0, 0.5, 0.95):
        moved = torch.nn.functional.normalize((1 - t) * other + t * ref, dim=-1)
        feat = torch.cat([ref, moved, moved.clone()], dim=0)
        losses.append(float(twin_align_loss(feat, N_MOD, 4)[0]))
    assert losses == sorted(losses, reverse=True), losses


def test_target_is_detached():
    """The reference has no bank of its own; a gradient into it would be
    optimising the coordinate frame, which is how the text anchors collapsed."""
    feat = torch.randn(9, 8, requires_grad=True)
    loss, _c = twin_align_loss(feat, N_MOD, 3)
    loss.backward()
    g = feat.grad
    assert torch.count_nonzero(g[:3]) == 0, 'gradient reached the reference rows'
    assert torch.count_nonzero(g[3:]) > 0


def test_shape_mismatch_raises_rather_than_guesses():
    for rows in (8, 10):
        try:
            twin_align_loss(torch.randn(rows, 8), N_MOD, 3)
        except ValueError:
            continue
        raise AssertionError('accepted %d rows for 3x3' % rows)


# --------------------------------------------------------------------------
# 4. freezing, learning rates and the config
# --------------------------------------------------------------------------
class _Cfg(dict):
    __getattr__ = dict.__getitem__


def _solver_cfg(**kw):
    solver = dict(MOD_DELTA_ONLY=False, PLD_ONLY=False, BASE_LR=3.5e-4, WEIGHT_DECAY=1e-4,
                  PRETRAINED_LR=5e-6, BIAS_LR_FACTOR=1, WEIGHT_DECAY_BIAS=1e-4,
                  LARGE_FC_LR=False, PE_DELTA_LR_MULT=1.0, CHART_LR_MULT=1.0,
                  OPTIMIZER_NAME='Adam', CENTER_LR=0.5)
    solver.update(kw)
    return _Cfg(SOLVER=_Cfg(solver), MODEL=_Cfg(MOD_DELTA=True))


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.blocks = nn.Linear(4, 4)
        self.base.mod_delta = nn.Parameter(torch.zeros(2, 3, 5, 4))
        self.base.pos_delta = nn.Parameter(torch.zeros(3, 1, 5, 4))
        self.bottleneck = nn.BatchNorm1d(4)


def test_bank_gets_base_lr_not_the_pretrained_clock():
    from solver.make_optimizer import make_optimizer
    model, center = _Tiny(), nn.Linear(2, 2)
    opt, _oc = make_optimizer(_solver_cfg(), model, center)
    lr = {}
    for group, (name, _p) in zip(opt.param_groups,
                                 [(n, p) for n, p in model.named_parameters()
                                  if p.requires_grad]):
        lr[name] = group['lr']
    assert lr['base.mod_delta'] == 3.5e-4, lr
    assert lr['base.pos_delta'] == 3.5e-4, lr
    assert lr['base.blocks.weight'] == 5e-6, lr


def test_mod_delta_only_freezes_everything_else():
    from solver.make_optimizer import make_optimizer
    model, center = _Tiny(), nn.Linear(2, 2)
    opt, _oc = make_optimizer(_solver_cfg(MOD_DELTA_ONLY=True), model, center)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ['base.mod_delta'], trainable
    assert sum(len(g['params']) for g in opt.param_groups) == 1


def test_mod_delta_only_refuses_a_model_without_banks():
    from solver.make_optimizer import make_optimizer

    class NoBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(4, 4)

    try:
        make_optimizer(_solver_cfg(MOD_DELTA_ONLY=True), NoBank(), nn.Linear(2, 2))
    except ValueError:
        return
    raise AssertionError('froze the whole model and carried on')


# --------------------------------------------------------------------------
# 5. the twin target has to survive the augmentation
# --------------------------------------------------------------------------
def _capture_dataset(tmp):
    """A one-capture, three-modality dataset on disk, shaped like WHU-MARS."""
    from PIL import Image
    import numpy as np
    mods = ['RGB', 'IR', 'Thermal']
    data = {}
    for k, m in enumerate(mods, 1):
        path = os.path.join(tmp, '0005_c1_m%d_f000163.png' % k)
        Image.fromarray((np.random.RandomState(k).rand(40, 20, 3) * 255)
                        .astype('uint8')).save(path)
        data[m] = [(path, 0, 0, k)]
    return data, mods


class _BothRNGs:
    """Reads `random` AND torch's generator, the two the shipped pipeline uses
    (torchvision transforms take the first, timm's RandomErasing the second)."""

    def __call__(self, img):
        import random as _r
        return torch.tensor([_r.random(), float(torch.rand(1))])


def test_augmentation_is_drawn_per_image_by_default():
    """The behaviour every run so far had, pinned so the change is visible."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_BothRNGs(), tie_augmentation=False)
        imgs, _pid, _cams = ds[(0, 0, 0)]
        assert not torch.equal(imgs[0], imgs[1]), 'default should draw per image'


def test_tied_augmentation_gives_one_capture_one_draw():
    """The property the regression target depends on: the three spectra of a
    capture differ only in spectrum, not in flip / crop / erasing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_BothRNGs(), tie_augmentation=True)
        imgs, _pid, _cams = ds[(0, 0, 0)]
        assert torch.equal(imgs[0], imgs[1]) and torch.equal(imgs[0], imgs[2]), imgs


def test_tied_augmentation_still_varies_between_captures():
    """Tying must not freeze the augmentation outright -- that would silently
    remove it from training rather than align it."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_BothRNGs(), tie_augmentation=True)
        # __getitem__ returns (imgs, pid, camids); imgs is a list per modality.
        draws = {tuple(ds[(0, 0, 0)][0][0].tolist()) for _ in range(20)}
        assert len(draws) > 15, 'augmentation collapsed to a constant: %d' % len(draws)


def test_tied_augmentation_holds_for_the_real_pipeline():
    """Same check through the transform the training loader actually builds
    (minus timm's RandomErasing, which is covered by the torch seed above)."""
    import tempfile
    import torchvision.transforms as T
    tf = T.Compose([T.Resize([64, 32], interpolation=3),
                    T.RandomHorizontalFlip(p=0.5), T.Pad(10),
                    T.RandomCrop([64, 32]), T.ToTensor()])
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        loose = ImageDataset(data, mods, transform=tf, tie_augmentation=False)
        tied = ImageDataset(data, mods, transform=tf, tie_augmentation=True)
        # The three modality images differ in content, so compare each spectrum
        # against itself under the two settings is meaningless; what matters is
        # that ONE image put through the pipeline three times agrees.
        data1 = {m: [data['RGB'][0]] for m in mods}
        loose1 = ImageDataset(data1, mods, transform=tf, tie_augmentation=False)
        tied1 = ImageDataset(data1, mods, transform=tf, tie_augmentation=True)
        differed = 0
        for _ in range(40):
            a = loose1[(0, 0, 0)][0]
            differed += int(not (torch.equal(a[0], a[1]) and torch.equal(a[0], a[2])))
            b = tied1[(0, 0, 0)][0]
            assert torch.equal(b[0], b[1]) and torch.equal(b[0], b[2]), 'tied draw split'
        assert differed > 30, 'the loose pipeline barely differed: %d/40' % differed
        del loose, tied


# --------------------------------------------------------------------------
# 6. plumbing
# --------------------------------------------------------------------------
def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def test_processor_refuses_the_twin_loss_without_synchronised_frames():
    """Both twin losses need the frame-synchronised sampler: without it rows i
    and i+per_modality are the same person at unrelated moments, and the loss
    would be calling a pose difference a spectrum difference.

    The guard is pinned by its CONDITION, not by its message.  An earlier
    version matched the wording, so adding the second loss to the same guard
    broke this test while the guard itself was strictly better.
    """
    src = _read('processor', 'processor.py')
    guard = [l for l in src.splitlines() if 'not cfg.DATALOADER.SYNC_FRAMES' in l]
    assert len(guard) == 1, guard
    assert 'twin_weight' in guard[0], guard[0]
    assert 'tnce_weight' in guard[0], guard[0]


class _Cfg2(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def _model_cfg(mod_delta, test_mod_delta):
    """The full cfg build_transformer reads, mirroring defaults.py.  Written out
    rather than getattr-defaulted so a key a real config forgot cannot hide."""
    from model.backbones.vit_pytorch import QuickGELU
    return _Cfg2(
        MODEL=_Cfg2(PRETRAIN_PATH='', PRETRAIN_CHOICE='no', COS_LAYER=False,
                    NECK='bnneck', TRANSFORMER_TYPE='tiny', SIE_CAMERA=False,
                    SIE_VIEW=False, SIE_COE=3.0, STRIDE_SIZE=[16, 16],
                    DROP_PATH=0.0, DROP_OUT=0.0, ATT_DROP_RATE=0.0,
                    PE_TYPE='learnable', PE_KEEP_APE=False, CHART_HIDDEN=64,
                    CHART_THETA_MAX_DEG=30.0, CHART_SCALE_MAX=0.35,
                    CHART_INPUT_NORM=False, ROPE_THETA=10.0,
                    ROPE_FREQ_TRAINABLE=False, ROPE_GATE=False, PE_VIEW_GATE='none', PE_PER_MODALITY=False, PE_ZERO_BASED=False,
                    LAYER_SCALE=False, LAYER_SCALE_INIT=1e-4,
                    PE_LAYERWISE='none', PE_LAYERS=[], PE_FREEZE_BASE=False,
                    MOD_DELTA=mod_delta, MOD_DELTA_REF='RGB',
                    CE_SPLIT_VIEW=False, CE_SPLIT_MODALITY=False,
                    CE_MODALITY_GROUPS=[], TEXT_MODALITY_TARGETS=[],
                    TEXT_ALIGN=False, TEXT_TARGET='view', TEXT_CLIP_PATH='',
                    TEXT_SUPERVISION='contrast', TEXT_ALLOW_NO_ACTUATOR=False,
                    TEXT_N_CTX=4, TEXT_MODALITY='RGB',
                    TEXT_TEMPLATE='a photo of a X view person with X X X X .'),
        DATASETS=_Cfg2(MODALITIES=['RGB', 'IR', 'Thermal'], AERIAL_CAMS=[]),
        INPUT=_Cfg2(SIZE_TRAIN=[256, 128]),
        TEST=_Cfg2(NECK_FEAT='before', MOD_DELTA=test_mod_delta),
    )


def _tiny_factory():
    from model.backbones.vit_pytorch import TransReID as _T, QuickGELU

    def tiny(**kw):
        kw.pop('img_size', None)
        for drop in ('sie_xishu', 'camera', 'view', 'stride_size',
                     'drop_path_rate', 'drop_rate', 'attn_drop_rate'):
            kw.pop(drop, None)
        return _T(img_size=(256, 128), patch_size=16, stride_size=16,
                  embed_dim=768, depth=1, num_heads=12, mlp_ratio=4,
                  qkv_bias=True, drop_path_rate=0.0, clip_style=True,
                  act_layer=QuickGELU, **kw)
    return {'tiny': tiny}


def _built(mod_delta=True, test_mod_delta=True, seed=3):
    from model.make_model import build_transformer
    torch.manual_seed(seed)
    m = build_transformer(7, 0, 0, _model_cfg(mod_delta, test_mod_delta),
                          _tiny_factory()).eval()
    return m


def test_eval_applies_the_bank_of_the_queried_modality():
    """`mode` is the 1-based modality the evaluator iterates with, so mode 2
    must pick bank row 0 and mode 1 (the reference) must pick nothing."""
    x = torch.randn(2, 3, 256, 128)
    m = _built()
    with torch.no_grad():
        m.base.mod_delta.normal_(std=0.3)
        ref = m(x, mode=1)                       # RGB: reference, no bank
        ir = m(x, mode=2)
        tir = m(x, mode=3)
        m.base.mod_delta.zero_()
        plain = m(x, mode=1)
    assert torch.allclose(ref, plain, atol=0, rtol=0), 'reference modality was corrected'
    assert not torch.allclose(ir, plain)
    assert not torch.allclose(tir, plain)
    assert not torch.allclose(ir, tir), 'both spectra got the same bank'


def test_test_mod_delta_false_is_the_uncorrected_model():
    """The theorem, tested rather than grepped: with the banks switched off at
    evaluation the model must return exactly what the untouched tower returns,
    whatever the banks hold.  If this ever fails, the claim that every
    cross-modality change is attributable to the banks fails with it."""
    x = torch.randn(2, 3, 256, 128)
    on = _built(test_mod_delta=True)
    off = _built(test_mod_delta=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        on.base.mod_delta.normal_(std=0.3)
        off.base.mod_delta.copy_(on.base.mod_delta)
        base = _built(mod_delta=False)
        base.load_state_dict({k: v for k, v in on.state_dict().items()
                              if k != 'base.mod_delta'})
        for mode in (1, 2, 3):
            assert torch.allclose(off(x, mode=mode), base(x, mode=mode),
                                  atol=0, rtol=0), mode
            if mode > 1:
                assert not torch.allclose(on(x, mode=mode), base(x, mode=mode)), mode


def test_a_reference_that_is_not_first_is_refused():
    """Bank row k is modality k+1, so a reference anywhere else would attach
    every bank to the wrong spectrum -- silently."""
    from model.make_model import build_transformer
    cfg = _model_cfg(True, True)
    cfg['MODEL']['MOD_DELTA_REF'] = 'IR'
    try:
        build_transformer(7, 0, 0, cfg, _tiny_factory())
    except ValueError:
        return
    raise AssertionError('accepted a reference modality that is not first')


def test_config_differs_from_its_control_only_where_intended():
    import yaml
    a = yaml.safe_load(_read('configs', 'hihr_whu_recipe_hihr.yml'))
    b = yaml.safe_load(_read('configs', 'hihr_whu_twin.yml'))

    def flat(d, prefix=''):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, prefix + k + '.'))
            else:
                out[prefix + k] = v
        return out

    fa, fb = flat(a), flat(b)
    assert fb['MODEL.MOD_DELTA'] is True
    assert fb['SOLVER.MOD_DELTA_ONLY'] is True
    assert fb['DATALOADER.SYNC_FRAMES'] is True
    # Without this the twin target is corrupted by independent flips/crops, and
    # a per-modality constant cannot undo a mirror.
    assert fb['DATALOADER.TIE_AUGMENTATION'] is True
    assert fb['SOLVER.TWIN_LOSS_WEIGHT'] > 0
    assert fb['MODEL.MOD_DELTA_REF'] == fb['DATASETS.MODALITIES'][0]
    # The banks sit on a frozen tower, so stochastic depth and dropout would
    # only add noise to a regression target.
    for k in ('MODEL.DROP_PATH', 'MODEL.DROP_OUT', 'MODEL.ATT_DROP_RATE'):
        assert fb[k] == 0.0, k
    # Everything that defines the recipe has to be inherited untouched.
    for k in ('MODEL.TRANSFORMER_TYPE', 'MODEL.METRIC_LOSS_TYPE', 'MODEL.NO_MARGIN',
              'SOLVER.BASE_LR', 'SOLVER.PRETRAINED_LR', 'SOLVER.IMS_PER_BATCH',
              'SOLVER.WARMUP_ITERS', 'DATALOADER.NUM_INSTANCE', 'TEST.NECK_FEAT'):
        assert fa[k] == fb[k], (k, fa[k], fb[k])
    # v1 starts from the clean baseline, not from CARGO: pe_indep showed the
    # layer-wise deltas are, if anything, slightly negative for cross-thermal.
    assert 'recipe_hihr' in fb['MODEL.PRETRAIN_PATH']
    assert fb.get('MODEL.PE_LAYERWISE', 'none') == 'none'
    # Epoch count has to match the controls: every recorded number, including
    # the 3x3 this run is judged against, is an epoch-60 reading.
    assert fb['SOLVER.MAX_EPOCHS'] == fa['SOLVER.MAX_EPOCHS'] == 60
    assert fb['TEST.WEIGHT'] == 'transformer_60.pth'


def test_launcher_knows_the_run():
    assert 'whu_twin)' in _read('run_hihr.sh')


# --------------------------------------------------------------------------
# untying the erasing alone
# --------------------------------------------------------------------------
class _CountingErase(torch.nn.Module):
    """Stands in for timm's RandomErasing: reads torch's generator and appends
    its draw, so a tied and an untied pipeline are told apart by the tensor.

    Named RandomErasing on purpose -- split_erasing matches on the class name,
    because the pipeline uses timm's and torchvision ships one of its own."""
    def forward(self, x):
        return torch.cat([x, torch.rand(1)])


_CountingErase.__name__ = 'RandomErasing'


def _pipeline():
    import torchvision.transforms as T
    return T.Compose([_BothRNGs(), _CountingErase()])


def test_untying_the_erasing_keeps_the_head_shared():
    """The property that makes this the right fix rather than RE_PROB 0: the
    crop and the flip stay identical across the three spectra, so a twin pair
    still differs only in spectrum, while the erased rectangle is drawn
    independently."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_pipeline(),
                          tie_augmentation=True, tie_erasing=False)
        imgs, _pid, _cams = ds[(0, 0, 0)]
        head = [tuple(i[:2].tolist()) for i in imgs]
        erase = [float(i[2]) for i in imgs]
        assert head[0] == head[1] == head[2], head          # crop / flip shared
        assert len(set(erase)) == 3, erase                  # erasing independent


def test_tying_the_erasing_is_still_the_default():
    """Every run recorded so far had it tied; the flag must not change any of
    them by existing."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_pipeline(), tie_augmentation=True)
        imgs, _pid, _cams = ds[(0, 0, 0)]
        assert torch.equal(imgs[0], imgs[1]) and torch.equal(imgs[0], imgs[2])


def test_untied_erasing_still_varies_between_captures():
    """Untying must not accidentally freeze either half."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        ds = ImageDataset(data, mods, transform=_pipeline(),
                          tie_augmentation=True, tie_erasing=False)
        heads = {tuple(ds[(0, 0, 0)][0][0][:2].tolist()) for _ in range(20)}
        erases = {float(ds[(0, 0, 0)][0][0][2]) for _ in range(20)}
        assert len(heads) > 15, len(heads)
        assert len(erases) > 15, len(erases)


def test_a_pipeline_with_no_erasing_refuses_to_untie_it():
    """Silence here would mean the flag looked set and did nothing at all."""
    import tempfile
    import torchvision.transforms as T
    with tempfile.TemporaryDirectory() as tmp:
        data, mods = _capture_dataset(tmp)
        try:
            ImageDataset(data, mods, transform=T.Compose([_BothRNGs()]),
                         tie_augmentation=True, tie_erasing=False)
        except ValueError as e:
            assert 'RandomErasing' in str(e), e
            return
    raise AssertionError('a pipeline with no erasing was accepted')


def test_split_erasing_only_moves_a_trailing_run():
    import torchvision.transforms as T
    head, tail = split_erasing(T.Compose([_BothRNGs(), _CountingErase(),
                                          _CountingErase()]))
    assert len(head.transforms) == 1 and len(tail.transforms) == 2
    # An erasing op in the MIDDLE means the ordering assumption is wrong;
    # splitting around it would change what the augmentation does.
    head, tail = split_erasing(T.Compose([_CountingErase(), _BothRNGs()]))
    assert tail is None, 'split around a non-trailing erasing op'


def test_the_ablation_config_differs_from_lam25_only_in_the_erasing():
    import yaml
    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out
    load = lambda n: flat(yaml.safe_load(_read('configs', n)))
    a, b = load('hihr_whu_tnce_lam25.yml'), load('hihr_whu_tnce_lam25_re.yml')
    assert set(b) - set(a) == {'DATALOADER.TIE_ERASING'}, set(b) - set(a)
    assert {k for k in a if a[k] != b[k]} == {'INPUT.RE_PROB', 'OUTPUT_DIR'}
    # The point of the run: the recipe's erasing is back, and it is untied.
    assert b['INPUT.RE_PROB'] == 0.5 and a['INPUT.RE_PROB'] == 0.0
    assert b['DATALOADER.TIE_ERASING'] is False
    assert b['DATALOADER.TIE_AUGMENTATION'] is True
    assert b['SOLVER.TNCE_WEIGHT'] == 0.25


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d twin-delta tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
