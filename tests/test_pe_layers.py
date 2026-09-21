"""Unit tests for MODEL.PE_LAYERS -- which blocks get a positional increment.

CPU-only and self-contained.  Run directly::

    python tests/test_pe_layers.py

THE ABLATION.  The layer-wise delta has always been one row per block, and
nothing has tested whether it needs to be everywhere.  PE_LAYERS names a subset;
the table then holds one row per insertion point, and the stream between two
insertion points carries `pos_embed + delta[j]` and nothing else.

WHAT WOULD GO WRONG QUIETLY.  All of it:

  * an empty or all-blocks list must reproduce the old code BIT FOR BIT, or
    every number recorded before this key stops being comparable;
  * the rows must map to the right blocks -- an off-by-one puts the correction
    one block early and the arm still trains to a plausible mAP;
  * `indep` must cancel against the previous INJECTED row, not the previous
    block, or the stream carries a sum nobody designed at the skipped
    positions;
  * a bad index must raise instead of being dropped, or [0, 12] silently
    becomes [0] and the log still says four insertion points.

The central test builds the same subset two ways -- through PE_LAYERS, and by
hand out of a full-depth model with the unwanted rows zeroed -- and asserts the
outputs are equal.  That checks the placement and the differencing together,
against a construction that does not share the implementation.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.vit_pytorch import TransReID  # noqa: E402

DEPTH = 6
IMG = (32, 16)          # 2x1 patches at stride 16 -> 3 tokens with the CLS


def build(pe_layers=(), layerwise='indep', depth=DEPTH, **kw):
    torch.manual_seed(0)
    m = TransReID(img_size=IMG, patch_size=16, stride_size=16, embed_dim=16,
                  depth=depth, num_heads=2, mlp_ratio=1.0, qkv_bias=True,
                  clip_style=True, num_classes=0, pe_layerwise=layerwise,
                  pe_layers=pe_layers, **kw)
    m.eval()
    return m


def x():
    torch.manual_seed(1)
    return torch.randn(2, 3, *IMG)


def copy_shared(src, dst):
    """Copy every parameter the two models have in common, BY NAME.

    Zipping named_parameters() pairs them positionally, which silently
    misaligns the moment one model has a `pos_delta` and the other does not --
    and the two builds here differ in exactly that.
    """
    have = dict(src.named_parameters())
    with torch.no_grad():
        for name, param in dst.named_parameters():
            if name in have and have[name].shape == param.shape:
                param.copy_(have[name])


# --------------------------------------------------------------------------

def test_empty_list_is_every_block_and_the_shape_is_unchanged():
    a, b = build(()), build(tuple(range(DEPTH)))
    assert a.pos_delta.shape[0] == DEPTH
    assert b.pos_delta.shape[0] == DEPTH
    assert a.pe_layers == list(range(DEPTH))
    assert torch.equal(a.pe_slot, torch.arange(DEPTH))


def test_the_default_path_is_bit_identical_to_before_the_key_existed():
    """A random delta, not a zero one: at zero every arrangement agrees."""
    m = build(())
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)
    ref = m.pos_delta.detach().clone()

    n = build(tuple(range(DEPTH)))
    with torch.no_grad():
        n.pos_delta.copy_(ref)

    with torch.no_grad():
        assert torch.equal(m(x()), n(x()))


def test_the_table_has_one_row_per_insertion_point():
    for layers, rows in (((0,), 1), ((DEPTH - 1,), 1), ((0, 3), 2),
                         ((0, 2, 4), 3)):
        m = build(layers)
        assert m.pos_delta.shape[0] == rows, (layers, m.pos_delta.shape)
        assert m.pe_layers == sorted(layers)


def test_rows_land_on_the_named_blocks_and_nowhere_else():
    m = build((1, 4))
    assert m.pe_slot.tolist() == [-1, 0, -1, -1, 1, -1]


def test_a_subset_matches_a_full_table_with_the_other_rows_zeroed():
    """The independent construction, and the test that matters most.

    Under 'indep' a full-depth table whose rows are held CONSTANT between
    insertion points produces exactly the subset model: the difference
    delta[l] - delta[l-1] is zero wherever the row did not change, so nothing
    is injected there, and it equals the subset's own increment where it did.
    Building the reference that way uses none of the subset code path.
    """
    layers = (0, 2, 4)
    sub = build(layers)
    with torch.no_grad():
        sub.pos_delta.normal_(std=0.1)

    full = build(())
    with torch.no_grad():
        for j, start in enumerate(layers):
            stop = layers[j + 1] if j + 1 < len(layers) else DEPTH
            for l in range(start, stop):
                full.pos_delta[l].copy_(sub.pos_delta[j])

    with torch.no_grad():
        assert torch.allclose(sub(x()), full(x()), atol=1e-6)


def test_chain_mode_also_places_rows_at_the_named_blocks():
    """`chain` adds the row raw, so the check is the injection, not the maths."""
    layers = (1, 3)
    sub = build(layers, layerwise='chain')
    with torch.no_grad():
        sub.pos_delta.normal_(std=0.1)

    full = build((), layerwise='chain')
    with torch.no_grad():
        full.pos_delta.zero_()
        for j, l in enumerate(layers):
            full.pos_delta[l].copy_(sub.pos_delta[j])

    with torch.no_grad():
        assert torch.allclose(sub(x()), full(x()), atol=1e-6)


def test_first_only_injects_exactly_once_at_the_very_front():
    """[0] is the floor of the mechanism: one additive term, nothing layer-wise.

    The independent construction: a full-depth 'indep' table whose rows are ALL
    the same tensor.  Then delta[l] - delta[l-1] is zero for every l > 0, so the
    only surviving injection is delta[0] at block 0 -- built with no reference
    to the subset code path.  The arm's whole reading ("if this matches 14.11
    then layer-wise was never the active ingredient") rests on this being true,
    so it is asserted rather than argued.

    Note it is NOT the same as adding the delta to the pos_embed table: ln_pre
    runs between the table and block 0, so this term lands after that
    normalisation, not before it.
    """
    sub = build((0,))
    with torch.no_grad():
        sub.pos_delta.normal_(std=0.1)

    full = build(())
    with torch.no_grad():
        for l in range(DEPTH):
            full.pos_delta[l].copy_(sub.pos_delta[0])

    with torch.no_grad():
        a, b = sub(x()), full(x())
    assert torch.allclose(a, b, atol=1e-6)

    # ...and a delta of this size genuinely moves the output, or the equality
    # above would hold for a broken implementation just as well.
    plain = build((), layerwise='none')
    copy_shared(sub, plain)
    with torch.no_grad():
        assert not torch.allclose(sub(x()), plain(x()), atol=1e-6)


def test_last_only_leaves_every_earlier_block_untouched():
    """[depth-1] must equal the plain model right up to the final block."""
    m = build((DEPTH - 1,))
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)

    full = build(())
    with torch.no_grad():
        full.pos_delta.zero_()
        # 'indep' cancels against the previous row, so to inject only at the
        # last block the full table must hold zero everywhere before it.
        full.pos_delta[DEPTH - 1].copy_(m.pos_delta[0])

    with torch.no_grad():
        assert torch.allclose(m(x()), full(x()), atol=1e-6)


def test_zero_delta_reproduces_the_baseline_for_every_arrangement():
    """Step 0 of training, which is the property the whole family rests on."""
    plain = build((), layerwise='none')
    for layers in ((), (0,), (DEPTH - 1,), (0, 3), (0, 2, 4)):
        m = build(layers)
        copy_shared(plain, m)
        assert float(m.pos_delta.abs().max()) == 0.0, layers
        with torch.no_grad():
            assert torch.allclose(plain(x()), m(x()), atol=1e-6), layers


def test_per_modality_deltas_shrink_on_the_row_axis_not_the_spectrum_axis():
    m = build((0, 3), pe_per_modality=3)
    assert m.pos_delta.shape[:2] == (3, 2), m.pos_delta.shape
    modal = torch.tensor([0, 2])
    with torch.no_grad():
        m.pos_delta.normal_(std=0.1)
        a = m(x(), modal_label=modal)
        b = m(x(), modal_label=torch.tensor([1, 1]))
    assert not torch.equal(a, b), 'the spectra must still select different rows'


def test_out_of_range_indices_raise_rather_than_being_dropped():
    for bad in ((DEPTH,), (-1,), (0, DEPTH + 3)):
        try:
            build(bad)
        except ValueError as exc:
            assert 'PE_LAYERS' in str(exc) and 'zero-based' in str(exc), exc
        else:
            raise AssertionError('{} should have raised'.format(bad))


def test_duplicates_raise():
    try:
        build((0, 3, 3))
    except ValueError as exc:
        assert 'duplicates' in str(exc), exc
    else:
        raise AssertionError('a repeated block should have raised')


def test_the_list_is_sorted_so_indep_differences_run_forwards():
    """[9, 0, 3] and [0, 3, 9] must be the same model, or the cancellation
    term is subtracting a row from later in the network."""
    m = build((4, 1))
    assert m.pe_layers == [1, 4]
    assert m.pe_slot.tolist() == [-1, 0, -1, -1, 1, -1]


def test_pe_layers_without_pe_layerwise_is_refused_in_make_model():
    """Read the guard out of make_model: the list would be validated, then
    never used, and the run would report the baseline under the arm's name."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'model', 'make_model.py'),
               encoding='utf-8').read()
    assert "cfg.MODEL.PE_LAYERS and cfg.MODEL.PE_LAYERWISE == 'none'" in src
    assert 'pe_layers=list(cfg.MODEL.PE_LAYERS)' in src, \
        'make_model must pass the key through to the backbone'


# ---- config wiring -------------------------------------------------------

ARMS = {'hihr_whu_pe_first_clip.yml': '[0]',
        'hihr_whu_pe_last_clip.yml': '[11]',
        'hihr_whu_pe_every3_clip.yml': '[0, 3, 6, 9]',
        'hihr_whu_pe_every2_clip.yml': '[0, 2, 4, 6, 8, 10]'}


def _keys(path):
    import re
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        m = re.match(r'^(\s*)([A-Z_]+):\s*(.+)$', line)
        if m and m.group(3).strip():
            out[m.group(2)] = m.group(3).strip()
    return out


def test_the_key_exists_in_defaults():
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'config', 'defaults.py'),
               encoding='utf-8').read()
    assert '_C.MODEL.PE_LAYERS = []' in src, \
        'the key is missing from defaults.py, or its default is not empty'


def test_each_arm_differs_from_the_control_in_one_key():
    here = os.path.dirname(os.path.abspath(__file__))
    cfgs = os.path.join(os.path.dirname(here), 'configs')
    control = _keys(os.path.join(cfgs, 'hihr_whu_text_lam50_clip.yml'))
    for name, layers in ARMS.items():
        arm = _keys(os.path.join(cfgs, name))
        differs = {k for k in set(control) | set(arm)
                   if control.get(k) != arm.get(k)}
        assert differs == {'PE_LAYERS', 'OUTPUT_DIR'}, (name, sorted(differs))
        assert arm['PE_LAYERS'] == layers, (name, arm['PE_LAYERS'])
        assert arm['PE_LAYERWISE'] == "'indep'", name


def test_the_four_arms_cover_four_distinct_counts():
    """1, 1, 4, 6 -- two of them matched, which is what isolates placement."""
    counts = [len(eval(v)) for v in ARMS.values()]        # noqa: S307
    assert sorted(counts) == [1, 1, 4, 6], counts
    first = eval(ARMS['hihr_whu_pe_first_clip.yml'])      # noqa: S307
    last = eval(ARMS['hihr_whu_pe_last_clip.yml'])        # noqa: S307
    assert len(first) == len(last) == 1 and first != last
    assert last == [11], 'ViT-B/16 has twelve blocks, so the last one is 11'


def test_the_runner_knows_all_four():
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(os.path.dirname(here), 'run_hihr.sh'),
               encoding='utf-8').read()
    for name in ARMS:
        mode = name[len('hihr_'):-len('.yml')]
        assert '{})'.format(mode) in src, '{} has no branch in run_hihr.sh'.format(mode)


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
