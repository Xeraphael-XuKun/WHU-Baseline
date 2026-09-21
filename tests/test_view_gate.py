# encoding: utf-8
"""The aerial-gated delta, and the strict-attribution freeze.

Two properties carry these arms, and neither is visible in the mAP:

  * gating on 'aerial' means the delta cannot move ground images at all -- they
    take the delta-free path in BOTH passes.  In eval mode the two are bit
    identical, which is the form pinned below; in training DROP_PATH 0.1 makes
    the two passes different stochastic subnetworks, so the logged `drift` sits
    at a noise floor rather than at zero.  If the bit-identity ever stops
    holding, the arm loses the only thing it was run for.

  * under PLD_ONLY the text loss must have exactly one trainable target it can
    reach.  The loss is a sum of four cross-entropies over one shared forward,
    so detaching cannot separate the delta from the backbone -- only freezing
    can, and the freeze has to cover clip_proj and the prompts as well.

The gate also changes what the model needs at inference, which is a claim in the
paper; the tests below pin that the evaluation path really does supply it.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.vit_pytorch import TransReID  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('PASS  %s' % name)
    else:
        FAIL += 1
        print('FAIL  %s  %s' % (name, detail))


def tiny(**kw):
    torch.manual_seed(0)
    args = dict(img_size=(32, 16), stride_size=16, patch_size=16, embed_dim=32,
                depth=3, num_heads=2, mlp_ratio=1.0, qkv_bias=True,
                clip_style=True, num_classes=0, pe_layerwise='indep')
    args.update(kw)
    return TransReID(**args)


# ----------------------------------------------------------- the gate itself
def test_a_zero_gate_row_is_the_delta_free_forward():
    """The property the whole arm rests on: a row whose gate is 0 comes out
    exactly as if the delta did not exist.

    Asserted in eval mode, which is the only place the equality can be exact --
    with DROP_PATH on, two forwards over different row counts draw different
    stochastic-depth masks.  What the gate guarantees is that the DELTA
    contributes nothing to those rows; the training-time `drift` residue is
    regularisation noise on top of that.
    """
    m = tiny()
    m.pos_delta.data.normal_(std=0.05)          # a delta that actually does something
    m.eval()
    x = torch.randn(4, 3, 32, 16)
    with torch.no_grad():
        off = m(x, None, None, None, delta_gate=torch.zeros(4))
        gate = m(x, None, None, None, delta_gate=torch.tensor([1., 0., 1., 0.]))
    check('gated-off rows equal the delta-free forward, bit for bit',
          torch.equal(gate[1], off[1]) and torch.equal(gate[3], off[3]))
    check('gated-on rows do not', not torch.equal(gate[0], off[0]))


def test_the_gate_is_per_image_not_per_batch():
    """Two images in one batch must be able to disagree; a batch-level switch
    would make the ground rows depend on who they were batched with."""
    m = tiny()
    m.pos_delta.data.normal_(std=0.05)
    m.eval()
    x = torch.randn(2, 3, 32, 16)
    with torch.no_grad():
        both_on = m(x, None, None, None, delta_gate=torch.ones(2))
        mixed = m(x, None, None, None, delta_gate=torch.tensor([1., 0.]))
    check('row 0 is unaffected by row 1 switching off',
          torch.equal(mixed[0], both_on[0]))


def test_an_all_zero_delta_makes_the_gate_a_no_op():
    """Step 0 of any gated run has to reproduce the ungated one, or the arm and
    its control would already differ before training."""
    m = tiny()                                   # pos_delta starts at zeros
    m.eval()
    x = torch.randn(3, 3, 32, 16)
    with torch.no_grad():
        a = m(x, None, None, None)
        b = m(x, None, None, None, delta_gate=torch.tensor([1., 0., 1.]))
    check('at init the gate changes nothing', torch.equal(a, b))


# ------------------------------------------------------------ the wiring
def _model_src():
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'model', 'make_model.py'), encoding='utf-8').read()


def test_the_gate_reaches_both_forward_paths():
    """Training and evaluation must gate identically.  A model trained to
    correct one viewpoint and evaluated correcting both is a different model,
    and the mAP would not say so."""
    src = _model_src()
    check('the gate is built in exactly two places (train + eval)',
          src.count('gate = self._view_gate(') == 2,
          src.count('gate = self._view_gate('))
    check('both forwards pass it to the backbone',
          src.count('delta_gate=gate') == 2, src.count('delta_gate=gate'))


def test_missing_camera_ids_raise_rather_than_default():
    """Without camera ids the gate has no way to tell the viewpoints apart.
    Falling back to "apply everywhere" would produce a run that looks like this
    arm in every log line and is the ungated one."""
    src = _model_src()
    head = src[src.index('def _view_gate'):src.index('def forward(self, x=None')]
    check('camids None raises', 'if camids is None:' in head and 'raise ValueError' in head)
    check('a row-count mismatch raises too', "cams.shape[0] != n_rows" in head)


def test_every_evaluation_call_site_forwards_camera_ids():
    """There are TWO of them -- the periodic eval inside do_train and
    do_inference -- and they have to agree.

    Patching only do_inference is exactly what happened: the gated arm trained
    for ten epochs, hit EVAL_PERIOD, and died inside do_train's evaluation loop.
    Counting the sites rather than checking that one exists is what makes this
    test catch the next one.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'processor', 'processor.py'), encoding='utf-8').read()
    passes = src.count('model(img, mode=mode, camids=camidt)')
    bare = src.count('model(img, mode=mode)')
    check('both evaluation call sites pass camidt', passes == 2, passes)
    check('none is left calling the model without them', bare == 0, bare)


def test_ground_gating_with_text_align_is_refused():
    """PE_VIEW_GATE 'ground' under TEXT_ALIGN makes A_before and A_after one
    tensor with two different targets -- the same contradiction the code already
    refuses for PE_LAYERWISE 'none'.  It has to fail loudly, not produce a
    number that looks like an ablation result."""
    src = _model_src()
    guard = src[src.index("self.pe_view_gate = cfg.MODEL.PE_VIEW_GATE"):
                src.index('# ---- view alignment against CLIP text anchors')]
    check("'ground' + TEXT_ALIGN raises",
          "self.pe_view_gate == 'ground'" in guard and 'self-contradictory' in guard)
    check('a gate without a delta raises',
          "cfg.MODEL.PE_LAYERWISE == 'none'" in guard)
    check('a gate without AERIAL_CAMS raises',
          'DATASETS.AERIAL_CAMS' in guard)
    check('an unknown gate value raises',
          "not in ('none', 'aerial', 'ground')" in guard)


# ------------------------------------------------- the strict-attribution arm
def test_pld_only_freezes_everything_the_text_loss_could_otherwise_reach():
    """The four routes VTC has into trainable weights, and what closes each."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'solver', 'make_optimizer.py'), encoding='utf-8').read()
    blk = src[src.index('if cfg.SOLVER.PLD_ONLY:'):src.index('    params = []')]
    check('pos_delta stays trainable', "'pos_delta' in key" in blk)
    check('the head stays trainable (the classifier starts random here)',
          "key.startswith('bottleneck.')" in blk and "key.startswith('classifier.')" in blk)
    check('everything else is frozen -- including base.clip_proj and the prompts',
          'value.requires_grad_(False)' in blk)
    check('a run with no delta to attribute to raises',
          "cfg.MODEL.PE_LAYERWISE == 'none'" in blk)
    check('the freeze is reported, not silent', 'PLD_ONLY: frozen' in blk)


def test_pld_only_keeps_the_delta_off_the_pretrained_clock():
    """pos_delta lives under `base.`, so the default rule would file it with the
    frozen tower's 5e-6.  It is the only thing learning here; on that clock the
    arm would report "the delta cannot do it" when it never got to move."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'solver', 'make_optimizer.py'), encoding='utf-8').read()
    head = src[src.index('PRETRAINED_LR > 0'):src.index('lr = cfg.SOLVER.PRETRAINED_LR')]
    check('pos_delta is excluded from the pretrained bucket',
          "'pos_delta' not in key" in head)


# --------------------------------------------------------------- the configs
def test_the_three_configs_differ_only_where_intended():
    import yaml

    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    for new, base, added in (
            ('whu_pldaerial_clip', 'whu_text_lam50_clip', {'MODEL.PE_VIEW_GATE'}),
            ('whu_strictpld_clip', 'whu_text_lam50_clip', {'SOLVER.PLD_ONLY'}),
            ('whu_strictpld_ctrl_clip', 'whu_pe_indep_clip', {'SOLVER.PLD_ONLY'})):
        a, b = load(base), load(new)
        check('%s: adds only %s' % (new, sorted(added)[0]),
              set(b) - set(a) == added, sorted(set(b) - set(a)))
        diff = {k for k in a if k in b and a[k] != b[k]}
        check('%s: changes only OUTPUT_DIR' % new, diff == {'OUTPUT_DIR'}, sorted(diff))
        check('%s: keeps the layer-wise delta on' % new,
              b['MODEL.PE_LAYERWISE'] == 'indep', b['MODEL.PE_LAYERWISE'])
        check('%s: still raw CLIP' % new,
              b['MODEL.PRETRAIN_CHOICE'] == 'imagenet' and 'ViT-B-16' in b['MODEL.PRETRAIN_PATH'])

    gated = load('whu_pldaerial_clip')
    check('the gated arm gates on aerial and keeps VTC on',
          gated['MODEL.PE_VIEW_GATE'] == 'aerial' and gated['MODEL.TEXT_ALIGN'] is True)
    check('the gated arm knows which cameras are aerial',
          gated['DATASETS.AERIAL_CAMS'] == [5, 6])

    arm, ctrl = load('whu_strictpld_clip'), load('whu_strictpld_ctrl_clip')
    check('the strict pair differs in the text loss and nothing structural',
          arm['SOLVER.PLD_ONLY'] is True and ctrl['SOLVER.PLD_ONLY'] is True
          and arm['MODEL.TEXT_ALIGN'] is True
          and ctrl.get('MODEL.TEXT_ALIGN', False) is False,
          (arm.get('MODEL.TEXT_ALIGN'), ctrl.get('MODEL.TEXT_ALIGN')))
    check('the strict pair trains the same number of epochs',
          arm['SOLVER.MAX_EPOCHS'] == ctrl['SOLVER.MAX_EPOCHS'])


def test_the_gate_defaults_to_off():
    """Every recorded run has to stay what it was.

    Read off the source rather than by importing config, which needs yacs --
    absent on the machine these tests run on, and irrelevant to the question.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'config', 'defaults.py'), encoding='utf-8').read()
    check("MODEL.PE_VIEW_GATE defaults to 'none'",
          "_C.MODEL.PE_VIEW_GATE = 'none'" in src)
    check('SOLVER.PLD_ONLY defaults to False',
          '_C.SOLVER.PLD_ONLY = False' in src)


def main():
    for fn in (test_a_zero_gate_row_is_the_delta_free_forward,
               test_the_gate_is_per_image_not_per_batch,
               test_an_all_zero_delta_makes_the_gate_a_no_op,
               test_the_gate_reaches_both_forward_paths,
               test_missing_camera_ids_raise_rather_than_default,
               test_every_evaluation_call_site_forwards_camera_ids,
               test_ground_gating_with_text_align_is_refused,
               test_pld_only_freezes_everything_the_text_loss_could_otherwise_reach,
               test_pld_only_keeps_the_delta_off_the_pretrained_clock,
               test_the_three_configs_differ_only_where_intended,
               test_the_gate_defaults_to_off):
        fn()
    print('\n%d/%d passed' % (PASS, PASS + FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
