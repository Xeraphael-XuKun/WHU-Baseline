"""The spectrum-target supervision (MODEL.TEXT_TARGET 'modality').

Structurally `view_align` with the axis swapped from viewpoint to spectrum, so
the tests are about the two ways that swap could go wrong silently:

  1. The correction has to be REWARDED.  Every cell is a cross entropy over the
     three spectrum anchors; the "after" column asks every image -- including
     the reference spectrum's own -- to read as the reference.  If that column
     were applied only to the non-reference rows, nothing would hold the
     reference in place and the anchors could drift together underneath it.
  2. The anchors cannot collapse.  If "color", "infrared" and "thermal" drifted
     onto one point, every image would read as color for free.  The three
     "before" cells are the guard -- they score the UNCORRECTED feature, so
     satisfying them forces the backbone to keep the spectra separable while
     the delta makes them read alike.  The tests plant a collapse and check the
     loss notices.

The level hinge this file used to test was deleted on 2026-08-07: it is the
mechanism whu_mod_lam50 collapsed on, and the design is back to the cross
entropy the two successful runs (whu_text_lam50, whu_mtext_lam50) used.  Its
saturation is accepted rather than worked around.

Run:  python tests/test_modality_align.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss.text_align import anchor_cos  # noqa: E402


def _load(name, *parts):
    """Load a module by path -- `processor/__init__` pulls in the trainer."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_proc = _load('proc_mod', 'processor', 'processor.py')
modality_align = _proc.modality_align
feature_geometry = _proc.feature_geometry

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


# --------------------------------------------------------------------------
# fixture: 3 spectra x 4 identities x 2 images, RGB is spectrum 0
# --------------------------------------------------------------------------
DIM, PROJ_DIM, N_SPEC, PER = 16, 8, 3, 12
RGB, IR, TIR = 0, 1, 2


def make_aux(shift=0.0, collapse=False, seed=0, requires_grad=False):
    """A batch where each spectrum starts near its own anchor.

    `shift` moves the non-RGB "after" features towards the RGB anchor: 0.0
    leaves them at their own spectrum, 1.0 puts them exactly on RGB.
    `collapse` drags the three anchors onto one point.
    """
    g = torch.Generator().manual_seed(seed)
    text = torch.zeros(N_SPEC, PROJ_DIM)
    for m in range(N_SPEC):
        text[m, m] = 1.0
    if collapse:
        text[:] = text[RGB]
        text += 1e-3 * torch.randn(N_SPEC, PROJ_DIM, generator=g)

    proj = torch.zeros(DIM, PROJ_DIM)
    proj[:PROJ_DIM, :PROJ_DIM] = torch.eye(PROJ_DIM)      # first 8 dims pass through

    modality = torch.arange(N_SPEC).repeat_interleave(PER // N_SPEC)

    def rows(target_mix):
        f = torch.zeros(PER, DIM)
        for i, m in enumerate(modality.tolist()):
            own = torch.zeros(PROJ_DIM); own[m] = 1.0
            tgt = torch.zeros(PROJ_DIM); tgt[RGB] = 1.0
            mix = own if m == RGB else (1 - target_mix) * own + target_mix * tgt
            f[i, :PROJ_DIM] = mix + 0.02 * torch.randn(PROJ_DIM, generator=g)
        return f

    before, after = rows(0.0), rows(shift)
    if requires_grad:
        before.requires_grad_(True)
        after.requires_grad_(True)
    return {
        'proj': proj, 'text': text, 'logit_scale': torch.tensor(100.0),
        'modality': modality, 'target_row': RGB,
        'feat_before': before, 'feat_after': after,
    }


# --------------------------------------------------------------------------
# 1. the after column is a cross entropy over every row
# --------------------------------------------------------------------------
def test_an_absent_target_list_is_bit_for_bit_the_old_behaviour():
    """The property that keeps whu_mcam's numbers comparable.  Passing the
    explicit list that spells out the old default has to give the same loss and
    the same stats as passing nothing at all."""
    a = make_aux(shift=0.4)
    b = dict(a, target_rows=[RGB] * N_SPEC)
    la, sa = modality_align(a)
    lb, sb = modality_align(b)
    assert torch.equal(la, lb), (float(la), float(lb))
    for k in ('n_total', 'n_moved', 'target_row', 'cos_before', 'cos_after',
              'acc_before', 'acc_after', 'drift'):
        assert sa[k] == sb[k], (k, sa[k], sb[k])


def test_a_per_spectrum_target_only_asks_the_named_row_to_move():
    """whu_ce_mod_g2_mcam_grp: [0, 1, 0] leaves IR reading as itself, so the
    only cell demanding a change is thermal's."""
    aux = dict(make_aux(shift=0.0), target_rows=[0, 1, 0])
    _loss, st = modality_align(aux)
    assert st['target_rows'] == [0, 1, 0]
    # one spectrum of three is asked to move, so a third of the rows
    assert st['n_moved'] == st['n_total'] // N_SPEC, st
    # against the all-to-reference form, which moves two thirds
    _l2, st2 = modality_align(make_aux(shift=0.0))
    assert st2['n_moved'] == 2 * st['n_total'] // N_SPEC, st2


def test_the_grouped_form_costs_less_when_ir_has_not_moved():
    """The mechanism, not just the bookkeeping.  With every feature sitting on
    its own anchor, [0, 1, 0] is satisfied for RGB and IR and only unhappy
    about thermal, so it must score strictly below the form that asks both
    non-reference spectra to read as RGB."""
    plain, _ = modality_align(make_aux(shift=0.0))
    grouped, _ = modality_align(dict(make_aux(shift=0.0), target_rows=[0, 1, 0]))
    assert float(grouped) < float(plain), (float(grouped), float(plain))


def _move_one_spectrum(aux, m, towards, amount):
    """Interpolate ONLY spectrum `m`'s after-features towards anchor `towards`.

    make_aux's `shift` moves every non-reference spectrum at once, so under
    [0, 1, 0] it improves thermal and worsens IR in the same step and the net
    can go either way -- measured: 33.3 / 17.6 / 9.4 / 33.1 at shift
    0.0 / 0.3 / 0.6 / 1.0, falling before it rises.  Isolating one spectrum is
    the only way to see the sign this arm is built on.
    """
    aux = dict(aux)
    after = aux['feat_after'].clone()
    sel = aux['modality'] == m
    own = torch.zeros(after.shape[1]); own[m] = 1.0
    tgt = torch.zeros(after.shape[1]); tgt[towards] = 1.0
    after[sel] = (1 - amount) * own + amount * tgt
    aux['feat_after'] = after
    return aux


def test_moving_ir_away_from_itself_is_punished_under_the_grouped_form():
    """The sign that the whole arm rests on.  Dragging IR towards RGB is a
    GAIN when every spectrum is corrected to the reference, and a LOSS under
    [0, 1, 0] where IR is told to stay where it is.  Same features, opposite
    sign."""
    base = make_aux(shift=0.0)
    still = _move_one_spectrum(base, m=1, towards=RGB, amount=0.0)
    moved = _move_one_spectrum(base, m=1, towards=RGB, amount=0.7)

    grouped = lambda a: float(modality_align(dict(a, target_rows=[0, 1, 0]))[0])
    plain = lambda a: float(modality_align(a)[0])

    assert grouped(moved) > grouped(still), (grouped(moved), grouped(still))
    assert plain(moved) < plain(still), (plain(moved), plain(still))


def test_the_before_guard_is_untouched_by_the_grouping():
    """The anti-collapse guard still targets each spectrum's own anchor, so
    collapsed anchors are punished under either form."""
    for rows in (None, [0, 1, 0]):
        aux = make_aux(collapse=True)
        if rows is not None:
            aux['target_rows'] = rows
        _l, st = modality_align(aux)
        assert st['acc_before'][1] < 0.9, (rows, st['acc_before'])


def test_the_drift_guard_still_reads_the_reference(  ):
    """drift is the reference spectrum moving against its own anchor.  It has
    to keep that meaning under a per-spectrum target, or the two arms' logs
    stop being comparable."""
    for rows in (None, [0, 1, 0]):
        aux = make_aux(shift=0.5)
        if rows is not None:
            aux['target_rows'] = rows
        _l, st = modality_align(aux)
        # RGB targets itself in both forms, so its drift is the same number
        assert abs(st['drift']) < 0.05, (rows, st['drift'])


def test_the_after_column_covers_the_reference_spectrum_too():
    """The property that replaced the detached level.

    The reference rows are inside the same cross entropy as everyone else, so
    dragging them off their anchor costs.  That is what pins the coordinate
    frame; with the level hinge it was a separate term whose target was read
    off the reference's own current cosine, which is exactly how that version
    escalated until every spectrum sat on one point.
    """
    base, _ = modality_align(make_aux(shift=1.0))
    moved = make_aux(shift=1.0)
    sel = moved['modality'] == RGB
    moved['feat_after'] = moved['feat_after'].clone()
    moved['feat_after'][sel, :PROJ_DIM] = 0.0
    moved['feat_after'][sel, IR] = 1.0          # reference dragged onto infrared
    after, _ = modality_align(moved)
    assert float(after) > float(base) + 0.5, (float(base), float(after))


def test_both_forward_passes_receive_gradient():
    """before and after are separate passes over the same images; a wiring
    mistake that scored one of them twice would leave the other with no
    gradient and would not show up in the loss value."""
    aux = make_aux(shift=0.5, requires_grad=True)
    loss, _ = modality_align(aux)
    loss.backward()
    assert float(aux['feat_before'].grad.abs().sum()) > 0
    assert float(aux['feat_after'].grad.abs().sum()) > 0


def test_no_hinge_leftovers_in_the_stats():
    """Guards the deletion: a half-removed hinge would leave these keys behind
    and the log would print a level that nothing computes any more."""
    _l, s = modality_align(make_aux(shift=0.5))
    for gone in ('ref', 'slack', 'hinge_margin'):
        assert gone not in s, gone


def test_modality_align_takes_no_margin_argument():
    import inspect
    params = list(inspect.signature(modality_align).parameters)
    assert params == ['aux'], params


# --------------------------------------------------------------------------
# 3. anchors cannot collapse
# --------------------------------------------------------------------------
def test_collapsed_anchors_are_punished_by_the_before_cells():
    healthy, _ = modality_align(make_aux(shift=0.5))
    collapsed, _ = modality_align(make_aux(shift=0.5, collapse=True))
    assert collapsed > healthy + 0.5, (float(collapsed), float(healthy))


def test_accuracy_of_the_before_cells_reports_the_collapse():
    _l, ok = modality_align(make_aux(shift=0.5))
    _l, bad = modality_align(make_aux(shift=0.5, collapse=True))
    assert min(ok['acc_before']) > 0.99, ok['acc_before']
    assert min(bad['acc_before']) < 0.6, bad['acc_before']


# --------------------------------------------------------------------------
# 4. the loss responds in the right direction
# --------------------------------------------------------------------------
def test_moving_the_other_spectra_towards_rgb_lowers_the_loss():
    losses = [float(modality_align(make_aux(shift=s))[0])
              for s in (0.0, 0.3, 0.6, 0.9)]
    assert losses == sorted(losses, reverse=True), losses


def test_acc_after_reports_the_correction_landing():
    """Both accuracies saturate -- that is accepted here -- so this only
    checks they point the right way at the two extremes.  The cosines are what
    stays readable in between."""
    _l, none = modality_align(make_aux(shift=0.0))
    _l, done = modality_align(make_aux(shift=0.95))
    assert none['acc_after'][IR] < 0.01, none['acc_after']
    assert done['acc_after'][IR] > 0.99, done['acc_after']
    assert done['acc_after'][RGB] > 0.99, done['acc_after']   # never asked to move


def test_stats_have_one_entry_per_spectrum():
    _l, s = modality_align(make_aux(shift=0.5))
    for k in ('cos_before', 'cos_after', 'acc_before', 'acc_after'):
        assert len(s[k]) == N_SPEC, (k, s[k])
        assert all(np.isfinite(v) for v in s[k]), (k, s[k])
    assert s['n_moved'] == PER - PER // N_SPEC, s['n_moved']


def test_drift_is_the_target_spectrum_moving():
    _l, s = modality_align(make_aux(shift=0.5))
    assert abs(s['drift'] - (s['cos_after'][RGB] - s['cos_before'][RGB])) < 1e-9


# --------------------------------------------------------------------------
# 5. the 768-d readout
# --------------------------------------------------------------------------
def _geom_batch(cross_shift, n_mod=3, n_pid=4, k=4, dim=32, seed=0):
    """Modality-major, identity-contiguous in groups of k -- PKMSampler's layout."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for m in range(n_mod):
        off = torch.zeros(dim)
        off[dim - 1 - m] = cross_shift
        for p in range(n_pid):
            base = torch.zeros(dim)
            base[p] = 1.0
            for _ in range(k):
                rows.append(base + off + 0.01 * torch.randn(dim, generator=g))
    return torch.stack(rows)


def test_geometry_orders_same_cross_and_diffpid():
    """With a modality shift planted, the three quantities must come out in the
    order the real data shows: d_same < d_cross < d_diffpid."""
    g = feature_geometry(_geom_batch(0.3), 3, 4)
    assert g is not None
    assert g['d_same'] < min(g['d_cross'].values()), g
    assert max(g['d_cross'].values()) < g['d_diffpid'], g


def test_geometry_reports_no_gap_where_none_was_planted():
    """The other half of the previous test, and the one that matters more: if
    the three spectra differ only by noise, d_cross has to land on d_same
    rather than above it.  A readout that always shows a gap would have us
    chasing one that is not there."""
    g = feature_geometry(_geom_batch(0.0), 3, 4)
    for pair, v in g['d_cross'].items():
        assert abs(v - g['d_same']) < 0.1 * g['d_same'], (pair, v, g['d_same'])


def test_geometry_sees_a_planted_modality_shift():
    clean = feature_geometry(_geom_batch(0.0), 3, 4)
    shifted = feature_geometry(_geom_batch(1.5), 3, 4)
    ratio = lambda g, k: g['d_cross'][k] / g['d_same']
    assert ratio(shifted, (0, 2)) > ratio(clean, (0, 2)) + 0.5, (clean, shifted)
    # The plant is cross-modality only, so the within-spectrum spread -- which
    # is what d_diffpid measures -- must be essentially unchanged.  Compared
    # relatively: these ratios are ~17, so an absolute tolerance would be
    # meaningless.
    a = clean['d_diffpid'] / clean['d_same']
    b = shifted['d_diffpid'] / shifted['d_same']
    assert abs(b - a) / a < 0.05, (a, b)


def test_geometry_refuses_a_batch_it_cannot_label():
    """Identity comes from position, so a batch that does not divide evenly is
    reported as unavailable rather than silently mislabelled."""
    assert feature_geometry(torch.randn(50, 8), 3, 4) is None
    assert feature_geometry(torch.randn(18, 8), 3, 4) is None      # 6 per mod, 1 pid


# --------------------------------------------------------------------------
# 6. plumbing
# --------------------------------------------------------------------------
def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def test_default_target_is_view_so_existing_runs_are_untouched():
    src = _read('config', 'defaults.py')
    assert "_C.MODEL.TEXT_TARGET = 'view'" in src


def _flat(node, prefix=()):
    """Every leaf of a parsed yaml as {dotted.key: value}.

    An exhaustive diff rather than a hand-written list of keys to compare:
    hihr_whu_mcam is meant to differ from its control in four places and
    nowhere else, and a test that checks a list cannot see a key someone adds
    later.  Flattening both files and differencing the dicts can.
    """
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flat(v, prefix + (str(k),)))
    else:
        out['.'.join(prefix)] = node
    return out


def test_mcam_differs_from_text_lam50_only_where_intended():
    import yaml
    a = _flat(yaml.safe_load(_read('configs', 'hihr_whu_text_lam50.yml')))
    b = _flat(yaml.safe_load(_read('configs', 'hihr_whu_mcam.yml')))
    assert set(b) - set(a) == {'MODEL.TEXT_TARGET',
                               'MODEL.TEXT_MODALITY_WORDS'}, set(b) - set(a)
    assert set(a) - set(b) == set(), set(a) - set(b)
    moved = {k for k in a if a[k] != b[k]}
    assert moved == {'MODEL.TEXT_TEMPLATE', 'OUTPUT_DIR'}, sorted(moved)


def test_mcam_has_seven_learnable_vectors():
    """The whole change, stated as the number it produces: the view form learns
    2 words + 4 shared = 6 vectors, this one learns 3 + 4 = 7."""
    import yaml
    a = yaml.safe_load(_read('configs', 'hihr_whu_text_lam50.yml'))['MODEL']
    b = yaml.safe_load(_read('configs', 'hihr_whu_mcam.yml'))['MODEL']
    assert 2 + a['TEXT_N_CTX'] == 6                      # aerial, ground
    assert len(b['TEXT_MODALITY_WORDS']) + b['TEXT_N_CTX'] == 7, b


def test_mcam_prompt_is_the_intended_one():
    import yaml
    m = yaml.safe_load(_read('configs', 'hihr_whu_mcam.yml'))['MODEL']
    assert m['TEXT_TARGET'] == 'modality'
    assert m['TEXT_MODALITY_WORDS'] == ['color', 'infrared', 'thermal']
    assert m['TEXT_TEMPLATE'] ==         'a photo of a person taken by a X camera with X X X X .'


def test_mcam_template_has_one_spectrum_slot_plus_n_ctx():
    """Under TEXT_TARGET 'modality' ViewPrompts wants 1 + n_ctx placeholders
    and no modality slot; one X too many or too few raises at startup, but a
    template that still says 'view' would parse fine and quietly describe the
    wrong axis."""
    import yaml
    m = yaml.safe_load(_read('configs', 'hihr_whu_mcam.yml'))['MODEL']
    tmpl = m['TEXT_TEMPLATE']
    assert tmpl.split().count('X') == 1 + m['TEXT_N_CTX'], tmpl
    assert 'view' not in tmpl.split(), tmpl
    assert 'aerial' not in tmpl and 'ground' not in tmpl, tmpl


def test_mcam_reference_word_and_reference_directory_are_kept_apart():
    """TEXT_MODALITY indexes DATASETS.MODALITIES (directory names) while
    TEXT_MODALITY_WORDS holds CLIP vocabulary.  'RGB' and 'color' name the same
    spectrum in two different namespaces, and make_model would raise if the
    first were replaced by the second."""
    import yaml
    c = yaml.safe_load(_read('configs', 'hihr_whu_mcam.yml'))
    mods = c['DATASETS']['MODALITIES']
    assert c['MODEL']['TEXT_MODALITY'] in mods, (c['MODEL']['TEXT_MODALITY'], mods)
    assert len(c['MODEL']['TEXT_MODALITY_WORDS']) == len(mods)


def test_launcher_knows_the_run():
    src = _read('run_hihr.sh')
    assert 'whu_mcam)' in src
    assert 'whu_mcam|' in src                      # and named in the usage line


def test_the_deleted_hinge_is_gone_everywhere():
    """The mechanism whu_mod_lam50 collapsed on.  Deleted rather than left
    switched off, so that no future config can reach it by setting a key."""
    assert 'level_hinge' not in _read('loss', 'text_align.py')
    assert 'TEXT_HINGE_MARGIN' not in _read('config', 'defaults.py')
    proc = _read('processor', 'processor.py')
    for gone in ('level_hinge', 'hinge_margin'):
        assert gone not in proc, gone


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d modality-target tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
