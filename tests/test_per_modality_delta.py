# encoding: utf-8
"""One independent positional delta per spectrum.

The claim is that RGB, IR and Thermal each read a delta of their own, with the
indep differencing running inside each of them separately.  Three ways that can
be true in the log and false in the tensor:

  * the routing silently collapses -- every row reads the same slice, and the
    arm becomes the shared one under a different name;
  * the differencing runs across the spectrum axis instead of inside it, so one
    spectrum's increments cancel against another's;
  * the evaluation path forgets to route, so a model trained with three deltas
    is measured reading one.

Each has a test below, and each would otherwise produce a number that looks
exactly like a legitimate result.
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


# ------------------------------------------------------------- the routing
def test_each_spectrum_reads_only_its_own_row():
    """Perturb one row; the other spectra must come out bit-identical.

    This is the whole claim.  If it fails, the parameters are three but the
    function is one.
    """
    m = tiny(pe_per_modality=3)
    m.eval()
    x = torch.randn(3, 3, 32, 16)
    modal = torch.tensor([0, 1, 2])
    with torch.no_grad():
        before = m(x, None, modal, None)
        m.pos_delta.data[1].normal_(std=0.1)        # only the IR row
        after = m(x, None, modal, None)
    check('changing the IR row leaves RGB untouched', torch.equal(before[0], after[0]))
    check('...and Thermal untouched', torch.equal(before[2], after[2]))
    check('...while IR itself moves', not torch.equal(before[1], after[1]))


def test_two_images_of_the_same_spectrum_share_a_row():
    """Routing is by spectrum, not by position in the batch."""
    m = tiny(pe_per_modality=3)
    m.pos_delta.data.normal_(std=0.1)
    m.eval()
    x = torch.randn(2, 3, 32, 16)
    with torch.no_grad():
        both_ir = m(x, None, torch.tensor([1, 1]), None)
        mixed = m(x, None, torch.tensor([1, 2]), None)
    check('row 0 is the same whichever spectrum row 1 claims',
          torch.equal(both_ir[0], mixed[0]))
    check('row 1 is not', not torch.equal(both_ir[1], mixed[1]))


# --------------------------------------------------------- the differencing
def test_indep_runs_inside_each_spectrum_not_across_them():
    """`_layerwise_deltas` returns increments; summing them back along DEPTH
    has to return the stored net offsets, per spectrum.

    If the differencing ran along the spectrum axis instead, this cumulative
    sum would not reconstruct the parameter -- and the model would be quietly
    training "IR minus RGB" rather than "IR".
    """
    m = tiny(pe_per_modality=3)
    m.pos_delta.data.normal_(std=0.1)
    d = m._layerwise_deltas()
    check('shape is preserved', d.shape == m.pos_delta.shape, (d.shape, m.pos_delta.shape))
    check('cumulative sum over depth returns the stored offsets',
          torch.allclose(d.cumsum(dim=1), m.pos_delta, atol=1e-6),
          (d.cumsum(dim=1) - m.pos_delta).abs().max().item())
    # The first layer of each spectrum must be its own delta, untouched by the
    # spectrum before it in the tensor.
    check('layer 0 of each spectrum is its own offset',
          torch.equal(d[:, 0], m.pos_delta[:, 0]))


def test_chain_returns_the_parameter_unchanged():
    m = tiny(pe_per_modality=3, pe_layerwise='chain')
    m.pos_delta.data.normal_(std=0.1)
    check('chain does not difference', torch.equal(m._layerwise_deltas(), m.pos_delta))


# ------------------------------------------------------------- the shapes
def test_the_parameter_grows_a_spectrum_axis_and_nothing_else():
    shared = tiny()
    per = tiny(pe_per_modality=3)
    check('shared keeps its shape', shared.pos_delta.dim() == 4, shared.pos_delta.shape)
    check('per-modality adds one leading axis',
          per.pos_delta.shape == (3,) + tuple(shared.pos_delta.shape), per.pos_delta.shape)
    check('exactly three times the parameters',
          per.pos_delta.numel() == 3 * shared.pos_delta.numel())


def test_it_starts_at_zero_like_the_shared_one():
    """Step 0 has to reproduce the baseline, or the arm and its control differ
    before training begins."""
    m = tiny(pe_per_modality=3)
    check('all three rows start at zero', float(m.pos_delta.abs().max()) == 0.0)
    x = torch.randn(3, 3, 32, 16)
    m.eval()
    with torch.no_grad():
        a = m(x, None, torch.tensor([0, 1, 2]), None)
        b = m(x, None, torch.tensor([0, 0, 0]), None)
    check('at init the routing cannot matter', torch.equal(a, b))


# --------------------------------------------------------------- the gate
def test_a_zero_gate_still_means_no_delta():
    """VTC's before pass sets the gate to zero; with a spectrum axis in play
    that must still be the delta-free forward, or before and after would differ
    by more than the delta."""
    m = tiny(pe_per_modality=3)
    m.pos_delta.data.normal_(std=0.1)
    m.eval()
    x = torch.randn(3, 3, 32, 16)
    modal = torch.tensor([0, 1, 2])
    with torch.no_grad():
        gated = m(x, None, modal, None, delta_gate=torch.zeros(3))
        m.pos_delta.data.zero_()
        empty = m(x, None, modal, None)
    check('gate 0 equals a zero delta, bit for bit', torch.equal(gated, empty))


# -------------------------------------------------------------- the guards
def test_a_missing_modality_index_raises():
    """Defaulting to row 0 would make every spectrum read the RGB delta and the
    log would look exactly like a correct run."""
    m = tiny(pe_per_modality=3)
    m.eval()
    try:
        m(torch.randn(2, 3, 32, 16), None, None, None)
        check('no modal_label raises', False, 'it ran')
    except ValueError as e:
        check('no modal_label raises', 'PE_PER_MODALITY needs modal_label' in str(e))


def test_an_out_of_range_index_raises():
    m = tiny(pe_per_modality=3)
    m.eval()
    try:
        m(torch.randn(2, 3, 32, 16), None, torch.tensor([0, 3]), None)
        check('an index past the last spectrum raises', False, 'it ran')
    except ValueError as e:
        check('an index past the last spectrum raises', 'out of range' in str(e))


def test_it_refuses_to_stack_with_the_banks():
    """MOD_DELTA gives each non-reference spectrum a correction too.  Both at
    once is two spellings of one thing, and the ablation could not say which
    produced the number."""
    try:
        tiny(pe_per_modality=3, mod_delta_modalities=2)
        check('PE_PER_MODALITY + MOD_DELTA raises', False, 'it built')
    except ValueError as e:
        check('PE_PER_MODALITY + MOD_DELTA raises', 'one or the other' in str(e))


def test_it_needs_a_layerwise_delta_to_split():
    try:
        tiny(pe_per_modality=3, pe_layerwise='none')
        check('PE_PER_MODALITY without a delta raises', False, 'it built')
    except ValueError as e:
        check('PE_PER_MODALITY without a delta raises', 'no delta to split' in str(e))


# --------------------------------------------------------------- the wiring
def _src(*parts):
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             *parts), encoding='utf-8').read()


def test_all_three_forward_paths_route():
    """Training, VTC's before pass, and evaluation.  Missing any one of them
    produces a model that trains with three deltas and is read with one -- and
    the failure surfaces as a mediocre mAP, not as an error."""
    src = _src('model', 'make_model.py')
    check('training passes the modality index',
          'if (self.use_mod_delta or self.pe_per_modality) else None' in src)
    check('the before pass routes by TRUE spectrum, not the anchor block',
          'modal_label=(rows // per_modality) if self.pe_per_modality else None' in src)
    check('evaluation routes as well',
          'elif self.pe_per_modality:' in src)


def test_the_config_differs_only_where_intended():
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
    base, new = load('whu_text_lam50_clip'), load('whu_pemod_clip')
    check('adds only MODEL.PE_PER_MODALITY',
          set(new) - set(base) == {'MODEL.PE_PER_MODALITY'}, sorted(set(new) - set(base)))
    diff = {k for k in base if k in new and base[k] != new[k]}
    check('changes only OUTPUT_DIR', diff == {'OUTPUT_DIR'}, sorted(diff))
    check('the delta is on and indep', new['MODEL.PE_LAYERWISE'] == 'indep')
    check('three spectra to split between',
          list(new['DATASETS.MODALITIES']) == ['RGB', 'IR', 'Thermal'])
    check('the banks stay off', new.get('MODEL.MOD_DELTA', False) is False)


def test_the_four_arms_form_a_clean_2x2():
    """Per-spectrum deltas and per-spectrum anchors are the two factors.  For
    the interaction to mean anything, each cell has to differ from its
    neighbours in exactly one of them."""
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
    cells = {k: load(v) for k, v in (
        ('shared_2', 'whu_text_lam50_clip'), ('shared_6', 'whu_mtext_lam50_clip'),
        ('per_2', 'whu_pemod_clip'), ('per_6', 'whu_pemod_mtext_clip'))}

    def anchors(c):
        return 6 if c.get('MODEL.TEXT_MODALITY_WORDS') else 2

    def per(c):
        return bool(c.get('MODEL.PE_PER_MODALITY', False))

    for name, want in (('shared_2', (2, False)), ('shared_6', (6, False)),
                       ('per_2', (2, True)), ('per_6', (6, True))):
        check('%s sits where it should in the 2x2' % name,
              (anchors(cells[name]), per(cells[name])) == want,
              (anchors(cells[name]), per(cells[name])))

    # Down a column: only the delta factor moves.
    for a, b in (('shared_2', 'per_2'), ('shared_6', 'per_6')):
        diff = ({k for k in cells[a] if k in cells[b] and cells[a][k] != cells[b][k]}
                | (set(cells[b]) ^ set(cells[a])))
        check('%s -> %s changes only the delta factor' % (a, b),
              diff == {'MODEL.PE_PER_MODALITY', 'OUTPUT_DIR'}, sorted(diff))

    # Across a row: only the anchor factor moves.
    for a, b in (('shared_2', 'shared_6'), ('per_2', 'per_6')):
        diff = ({k for k in cells[a] if k in cells[b] and cells[a][k] != cells[b][k]}
                | (set(cells[b]) ^ set(cells[a])))
        check('%s -> %s changes only the anchor factor' % (a, b),
              diff == {'MODEL.TEXT_MODALITY_WORDS', 'MODEL.TEXT_MODALITY',
                       'MODEL.TEXT_TEMPLATE', 'OUTPUT_DIR'}, sorted(diff))


def test_the_flag_defaults_to_off():
    check("MODEL.PE_PER_MODALITY defaults to False",
          '_C.MODEL.PE_PER_MODALITY = False' in _src('config', 'defaults.py'))


def main():
    for fn in (test_each_spectrum_reads_only_its_own_row,
               test_two_images_of_the_same_spectrum_share_a_row,
               test_indep_runs_inside_each_spectrum_not_across_them,
               test_chain_returns_the_parameter_unchanged,
               test_the_parameter_grows_a_spectrum_axis_and_nothing_else,
               test_it_starts_at_zero_like_the_shared_one,
               test_a_zero_gate_still_means_no_delta,
               test_a_missing_modality_index_raises,
               test_an_out_of_range_index_raises,
               test_it_refuses_to_stack_with_the_banks,
               test_it_needs_a_layerwise_delta_to_split,
               test_all_three_forward_paths_route,
               test_the_config_differs_only_where_intended,
               test_the_four_arms_form_a_clean_2x2,
               test_the_flag_defaults_to_off):
        fn()
    print('\n%d/%d passed' % (PASS, PASS + FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
