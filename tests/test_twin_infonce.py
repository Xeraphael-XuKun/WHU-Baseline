"""Unit tests for the twin contrastive loss.

The design rests on four claims, and each is a way it could go wrong while the
training curve still looked healthy:

  1. Collapse is the WORST answer, not the best.  This is the whole reason the
     loss changed shape when the backbone was unfrozen -- the cosine
     regression whu_twin used is minimised by putting every feature on one
     point, which is safe only while nothing can move.
  2. The other captures of the same identity are excluded from the
     denominator.  Counting them as negatives would push a person's own frames
     apart, against the ID loss, and nothing about the loss value would say so.
  3. The reference features carry no gradient, so the reference spectrum stays
     a coordinate frame and "the RGB diagonal moved" remains a failure signal.
  4. rank1 means what the log says it means.  It is the number the run is
     judged on -- the probe measured that centring moves this loss by 57% and
     rank1 by 0.1 points -- so a wrong rank1 would misread the whole run.

Run:  python tests/test_twin_infonce.py
"""
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss.twin_infonce import twin_infonce_loss  # noqa: E402

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
# fixture: 3 spectra x 4 identities x 4 captures, modality-major
# --------------------------------------------------------------------------
DIM, N_MOD, N_PID, N_INST = 12, 3, 4, 4
PER = N_PID * N_INST
TAU = 0.05


def pids():
    return torch.arange(PER) // N_INST


def batch(shift=0.0, seed=0, requires_grad=False):
    """Row i of block m is capture i in spectrum m.

    `shift` moves the non-reference spectra away from their twin along a
    per-capture random direction: 0.0 leaves every twin exactly on top of its
    reference, 1.0 puts it a long way off.
    """
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(PER, DIM, generator=g), dim=-1)
    blocks = [base]
    for _m in range(1, N_MOD):
        off = F.normalize(torch.randn(PER, DIM, generator=g), dim=-1)
        blocks.append(base + shift * off)
    feat = torch.cat(blocks, dim=0)
    if requires_grad:
        feat.requires_grad_(True)
    return feat


# --------------------------------------------------------------------------
# 1. collapse is the worst answer
# --------------------------------------------------------------------------
def test_total_collapse_reaches_the_ceiling_not_zero():
    """The property the whole switch away from cosine regression buys.

    Every feature identical: the cosine regression whu_twin used would report
    0, its best possible value.  Here every candidate is equally likely, so the
    loss must sit at log(n_candidates) -- its worst.  n_candidates is the twin
    plus the strangers, i.e. PER minus the identity's other N_INST-1 captures.
    """
    feat = torch.ones(N_MOD * PER, DIM)
    loss, _stats = twin_infonce_loss(feat, pids(), N_MOD, PER, TAU)
    n_cand = PER - (N_INST - 1)
    assert abs(float(loss) - float(torch.tensor(float(n_cand)).log())) < 1e-4, \
        (float(loss), n_cand)


def test_a_perfect_match_gives_almost_zero():
    loss, stats = twin_infonce_loss(batch(shift=0.0), pids(), N_MOD, PER, TAU)
    assert float(loss) < 1e-3, float(loss)
    for m in (1, 2):
        assert stats['rank1'][m] == 1.0, stats['rank1']


def test_the_loss_rises_as_the_twins_drift_apart():
    losses = [float(twin_infonce_loss(batch(shift=s), pids(), N_MOD, PER, TAU)[0])
              for s in (0.0, 0.5, 1.0, 2.0)]
    assert losses == sorted(losses), losses


# --------------------------------------------------------------------------
# 2. the denominator
# --------------------------------------------------------------------------
def test_same_identity_non_twins_are_not_negatives():
    """Planted so the two readings must differ: the identity's other captures
    are placed right on top of the query.  If they counted as negatives the
    loss would be large; excluded, the twin is unopposed and it is ~0.

    A version that forgot the mask would still produce a smooth, plausible
    curve -- it would just be fighting the ID loss the whole time.
    """
    base = F.normalize(torch.randn(PER, DIM, generator=torch.Generator().manual_seed(1)),
                       dim=-1)
    # every capture of an identity shares one direction
    ident = base.view(N_PID, N_INST, DIM)[:, :1].expand(N_PID, N_INST, DIM)
    ref = ident.reshape(PER, DIM).clone()
    feat = torch.cat([ref] * N_MOD, dim=0)
    loss, stats = twin_infonce_loss(feat, pids(), N_MOD, PER, TAU)
    assert float(loss) < 1e-3, float(loss)
    assert stats['rank1'][1] == 1.0, stats['rank1']


def test_a_stranger_placed_on_the_query_destroys_rank1():
    """The mirror of the test above: a DIFFERENT identity on top of the query
    must count, or the loss is not contrastive at all."""
    feat = batch(shift=0.5)
    f = feat.clone()
    # put identity 3's reference row exactly on identity 0's thermal row
    f[0] = f[2 * PER]                     # ref row 0 <- thermal row 0's own value
    f[N_INST] = f[2 * PER]                # a stranger's ref row, same value
    _loss, stats = twin_infonce_loss(f, pids(), N_MOD, PER, TAU)
    assert stats['rank1'][2] < 1.0, stats['rank1']


# --------------------------------------------------------------------------
# 3. the reference is a frame, not a participant
# --------------------------------------------------------------------------
def test_no_gradient_reaches_the_reference_block():
    feat = batch(shift=0.5, requires_grad=True)
    loss, _stats = twin_infonce_loss(feat, pids(), N_MOD, PER, TAU)
    loss.backward()
    g = feat.grad
    assert float(g[:PER].abs().sum()) == 0.0, float(g[:PER].abs().sum())
    assert float(g[PER:].abs().sum()) > 0.0


def test_the_reference_block_is_not_supervised_against_itself():
    _l, stats = twin_infonce_loss(batch(shift=0.5), pids(), N_MOD, PER, TAU)
    for k in ('rank1', 'cos', 'margin'):
        assert len(stats[k]) == N_MOD, (k, stats[k])
        assert stats[k][0] != stats[k][0], (k, stats[k])       # nan at the ref
        assert all(v == v for v in stats[k][1:]), (k, stats[k])


# --------------------------------------------------------------------------
# 4. rank1 and the reported statistics
# --------------------------------------------------------------------------
def test_rank1_counts_twins_that_beat_every_stranger():
    feat = batch(shift=0.6, seed=3)
    _l, stats = twin_infonce_loss(feat, pids(), N_MOD, PER, TAU)
    f = F.normalize(feat.float(), dim=-1).view(N_MOD, PER, -1)
    p = pids()
    for m in (1, 2):
        cos = f[m] @ f[0].t()
        wins = 0
        for i in range(PER):
            strangers = [float(cos[i, j]) for j in range(PER) if p[j] != p[i]]
            wins += float(cos[i, i]) > max(strangers)
        assert abs(stats['rank1'][m] - wins / PER) < 1e-6, (m, stats['rank1'][m],
                                                            wins / PER)


def test_margin_and_cos_are_in_cosine_units():
    """Both are printed next to cosines measured elsewhere, so neither may
    carry the temperature."""
    a = twin_infonce_loss(batch(shift=0.5), pids(), N_MOD, PER, 0.01)[1]
    b = twin_infonce_loss(batch(shift=0.5), pids(), N_MOD, PER, 0.5)[1]
    for k in ('cos', 'margin'):
        for m in (1, 2):
            assert abs(a[k][m] - b[k][m]) < 1e-6, (k, m, a[k][m], b[k][m])


def test_temperature_scales_the_loss_but_not_the_ranking():
    la, sa = twin_infonce_loss(batch(shift=0.6), pids(), N_MOD, PER, 0.01)
    lb, sb = twin_infonce_loss(batch(shift=0.6), pids(), N_MOD, PER, 0.5)
    assert abs(float(la) - float(lb)) > 1e-3, (float(la), float(lb))
    # slot 0 is nan (the reference is not supervised) and nan != nan, so the
    # comparison has to skip it rather than fail for the wrong reason.
    assert sa['rank1'][1:] == sb['rank1'][1:], (sa['rank1'], sb['rank1'])


# --------------------------------------------------------------------------
# 5. guards
# --------------------------------------------------------------------------
def _raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError('expected a ValueError mentioning %r' % needle)


def test_a_batch_of_the_wrong_length_raises():
    _raises(lambda: twin_infonce_loss(batch()[:-1], pids(), N_MOD, PER, TAU),
            'rows')


def test_a_pid_vector_of_the_wrong_length_raises():
    _raises(lambda: twin_infonce_loss(batch(), pids()[:-1], N_MOD, PER, TAU),
            'pids')


def test_a_non_positive_temperature_raises():
    """0.0 would divide by zero and produce nan, which shows up epochs later
    as a dead run rather than as an error."""
    _raises(lambda: twin_infonce_loss(batch(), pids(), N_MOD, PER, 0.0), 'tau')


def test_the_processor_refuses_the_loss_without_frame_synchronisation():
    """Read from the source: without SYNC_FRAMES rows i and i+PER are the same
    person at unrelated moments, and the loss would be calling a pose
    difference a spectrum difference."""
    src = open(os.path.join(ROOT, 'processor', 'processor.py'),
               encoding='utf-8').read()
    assert 'tnce_weight > 0) and not cfg.DATALOADER.SYNC_FRAMES' in src, \
        'the SYNC_FRAMES guard does not cover TNCE_WEIGHT'


def test_the_log_prints_rank1_before_the_loss():
    """Not cosmetic.  The probe measured that plain centring drops this loss by
    56.8% while moving rank1 by 0.1 points, so a reader who sees the loss first
    concludes the method works when nothing has happened."""
    src = open(os.path.join(ROOT, 'processor', 'processor.py'),
               encoding='utf-8').read()
    i = src.index("Twin-NCE: rank1")
    j = src.index("loss={:.4f}", i)
    assert i < j, 'the loss is printed before rank1'


def test_the_config_is_the_baseline_plus_the_intended_keys_only():
    import yaml
    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out
    here = os.path.join(ROOT, 'configs')
    a = flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_recipe_hihr.yml'),
                                 encoding='utf-8')))
    b = flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_tnce.yml'),
                                 encoding='utf-8')))
    assert set(b) - set(a) == {'DATALOADER.SYNC_FRAMES', 'DATALOADER.TIE_AUGMENTATION',
                               'SOLVER.TNCE_WEIGHT', 'SOLVER.TNCE_TAU'}, set(b) - set(a)
    assert set(a) - set(b) == set(), set(a) - set(b)
    moved = {k for k in a if a[k] != b[k]}
    assert moved == {'INPUT.RE_PROB', 'SOLVER.EVAL_PERIOD', 'OUTPUT_DIR'}, sorted(moved)
    assert b['SOLVER.TNCE_TAU'] == 0.003 and b['SOLVER.TNCE_WEIGHT'] == 1.0
    assert b['INPUT.RE_PROB'] == 0.0 and b['DATALOADER.SYNC_FRAMES'] is True


def test_the_launcher_knows_the_run():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    assert 'whu_tnce)' in src, 'no case branch'
    # the usage block closes the list with `whu_tnce}`; earlier entries end
    # in `|`.  Either way the name has to appear twice, once as a branch and
    # once in the message a wrong mode name prints.
    assert src.count('whu_tnce') >= 2, 'not named in the usage message'


# --------------------------------------------------------------------------
# 6. the weight ramp
# --------------------------------------------------------------------------
def _ramp(epoch, weight, warmup):
    """The expression processor.py evaluates once per epoch, restated here so
    the arithmetic is pinned independently of the training loop."""
    return weight if warmup <= 0 else weight * min(1.0, (epoch - 1) / float(warmup))


def test_the_ramp_starts_at_zero_and_reaches_the_weight():
    assert _ramp(1, 1.0, 10) == 0.0
    assert _ramp(6, 1.0, 10) == 0.5
    assert _ramp(11, 1.0, 10) == 1.0
    assert _ramp(60, 1.0, 10) == 1.0


def test_a_zero_warmup_is_the_old_flat_behaviour():
    for e in (1, 5, 60):
        assert _ramp(e, 0.7, 0) == 0.7


def test_the_processor_gates_on_the_ramped_weight_not_the_configured_one():
    """The bug this would be: computing the ramp and then testing
    `tnce_weight > 0`, so epoch 1 still pays for a forward pass whose loss is
    multiplied by zero -- and worse, `if tnce_now > 0` is what makes epoch 1
    reproduce the baseline exactly."""
    src = open(os.path.join(ROOT, 'processor', 'processor.py'),
               encoding='utf-8').read()
    assert 'if tnce_now > 0:' in src, 'the guard still reads the unramped weight'
    assert 'loss + tnce_now * loss_tnce' in src, 'the loss uses the unramped weight'


def test_the_two_follow_up_configs_differ_from_whu_tnce_in_one_key_each():
    import yaml
    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out
    here = os.path.join(ROOT, 'configs')
    a = flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_tnce.yml'),
                                 encoding='utf-8')))
    warm = flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_tnce_warm.yml'),
                                    encoding='utf-8')))
    lam = flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_tnce_lam25.yml'),
                                   encoding='utf-8')))
    assert set(warm) - set(a) == {'SOLVER.TNCE_WARMUP_EPOCHS'}, set(warm) - set(a)
    assert {k for k in a if a[k] != warm[k]} == {'OUTPUT_DIR'}
    assert warm['SOLVER.TNCE_WARMUP_EPOCHS'] == 10
    assert warm['SOLVER.TNCE_WEIGHT'] == 1.0

    assert set(lam) - set(a) == set(), set(lam) - set(a)
    assert {k for k in a if a[k] != lam[k]} == {'SOLVER.TNCE_WEIGHT', 'OUTPUT_DIR'}
    assert lam['SOLVER.TNCE_WEIGHT'] == 0.25

    # The two must differ from each other in exactly the way the pair is meant
    # to test -- magnitude against timing -- or they answer the same question.
    assert warm['SOLVER.TNCE_WEIGHT'] != lam['SOLVER.TNCE_WEIGHT']
    assert lam.get('SOLVER.TNCE_WARMUP_EPOCHS', 0) == 0


def test_the_launcher_knows_both_follow_ups():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    for mode in ('whu_tnce_warm', 'whu_tnce_lam25'):
        assert mode + ')' in src, mode
        assert src.count(mode) >= 2, mode


# --------------------------------------------------------------------------
# 7. the blank control
# --------------------------------------------------------------------------
def test_the_control_differs_from_whu_tnce_only_in_the_weight():
    """It has to be a control in both directions at once: one key from the
    twin runs (so it isolates the loss) and the loss switched off (so it
    isolates the changed input pipeline against whu_recipe_hihr).  A second
    difference in either direction destroys both readings."""
    import yaml
    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out
    here = os.path.join(ROOT, 'configs')
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, n), encoding='utf-8')))
    tnce, ctrl = load('hihr_whu_tnce.yml'), load('hihr_whu_tnce_ctrl.yml')
    assert set(ctrl) == set(tnce), set(ctrl) ^ set(tnce)
    assert {k for k in tnce if tnce[k] != ctrl[k]} == {'SOLVER.TNCE_WEIGHT',
                                                       'OUTPUT_DIR'}
    assert ctrl['SOLVER.TNCE_WEIGHT'] == 0.0

    base = load('hihr_whu_recipe_hihr.yml')
    moved = {k for k in base if k in ctrl and base[k] != ctrl[k]}
    assert moved == {'INPUT.RE_PROB', 'SOLVER.EVAL_PERIOD', 'OUTPUT_DIR'}, sorted(moved)
    # The pipeline change under suspicion, pinned so the control cannot quietly
    # stop testing it.
    assert ctrl['INPUT.RE_PROB'] == 0.0 and base['INPUT.RE_PROB'] == 0.5
    assert ctrl['DATALOADER.SYNC_FRAMES'] is True


def test_a_zero_weight_skips_the_twin_forward_pass_entirely():
    """With TNCE_WEIGHT 0 the guard must skip the block rather than compute a
    loss and multiply it by zero: the absence of the Twin-NCE lines from the
    log is how the control is recognised at a glance, and a wasted forward pass
    would also make the control slower than the runs it controls."""
    src = open(os.path.join(ROOT, 'processor', 'processor.py'),
               encoding='utf-8').read()
    i = src.index('if tnce_now > 0:')
    j = src.index('twin_infonce_loss(', i)
    assert i < j, 'the loss is computed outside the guard'


def test_the_launcher_knows_the_control():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    assert 'whu_tnce_ctrl)' in src
    assert src.count('whu_tnce_ctrl') >= 2


# --------------------------------------------------------------------------
# 8. the 2x2 as a whole
# --------------------------------------------------------------------------
def _flat_cfg(name):
    import yaml
    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out
    return flat(yaml.safe_load(open(os.path.join(ROOT, 'configs', name),
                                    encoding='utf-8')))


def test_the_ablation_is_a_clean_two_by_two():
    """Every edge must be ONE factor, or the interaction gets attributed to the
    wrong one.  Checked as the four configs rather than as a description of
    them, because that is the thing that can drift.

    Down a column: the twin loss at fixed augmentation.
    Across a row:  the augmentation at fixed loss (RE_PROB and TIE_ERASING move
                   together on purpose -- putting the erasing back is only
                   meaningful if it is untied, so they are one factor).
    """
    cell = {k: _flat_cfg('hihr_whu_tnce_%s.yml' % k)
            for k in ('ctrl', 'lam25', 'ctrl_re', 'lam25_re')}
    loss_edge = ({'SOLVER.TNCE_WEIGHT'}, )
    aug_edge = ({'INPUT.RE_PROB', 'DATALOADER.TIE_ERASING'}, )

    def diff(a, b):
        return {k for k in set(cell[a]) | set(cell[b])
                if cell[a].get(k) != cell[b].get(k)} - {'OUTPUT_DIR'}

    assert diff('ctrl', 'lam25') == loss_edge[0], diff('ctrl', 'lam25')
    assert diff('ctrl_re', 'lam25_re') == loss_edge[0], diff('ctrl_re', 'lam25_re')
    assert diff('ctrl', 'ctrl_re') == aug_edge[0], diff('ctrl', 'ctrl_re')
    assert diff('lam25', 'lam25_re') == aug_edge[0], diff('lam25', 'lam25_re')

    # And the corners really are the four combinations they claim to be.
    for k in ('ctrl', 'ctrl_re'):
        assert cell[k]['SOLVER.TNCE_WEIGHT'] == 0.0, k
    for k in ('lam25', 'lam25_re'):
        assert cell[k]['SOLVER.TNCE_WEIGHT'] == 0.25, k
    for k in ('ctrl', 'lam25'):
        assert cell[k]['INPUT.RE_PROB'] == 0.0, k
        assert cell[k].get('DATALOADER.TIE_ERASING', True) is True, k
    for k in ('ctrl_re', 'lam25_re'):
        assert cell[k]['INPUT.RE_PROB'] == 0.5, k
        assert cell[k]['DATALOADER.TIE_ERASING'] is False, k


def test_the_launcher_knows_every_corner():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    for mode in ('whu_tnce_ctrl', 'whu_tnce_lam25', 'whu_tnce_ctrl_re',
                 'whu_tnce_lam25_re'):
        assert mode + ')' in src, mode
        assert src.count(mode) >= 2, mode


# --------------------------------------------------------------------------
# 9. the lambda sweep on the untied-erasing arm
# --------------------------------------------------------------------------
SWEEP = {'ctrl_re': 0.0, 'lam25_re': 0.25, 'lam50_re': 0.5,
         'lam100_re': 1.0, 'lam200_re': 2.0}


def test_the_sweep_varies_lambda_and_nothing_else():
    """Five configs on one axis.  A second key drifting into any of them turns
    a dose-response curve into five unrelated runs, and the curve is the whole
    point -- 0.25 already buys +6.2 pct on cross-thermal for -0.5 pct pooled,
    so what is being read off these is where that trade stops paying.
    """
    ref = _flat_cfg('hihr_whu_tnce_lam25_re.yml')
    for tag, want in SWEEP.items():
        c = _flat_cfg('hihr_whu_tnce_%s.yml' % tag)
        assert set(c) == set(ref), (tag, set(c) ^ set(ref))
        moved = {k for k in ref if ref[k] != c[k]} - {'OUTPUT_DIR'}
        assert moved <= {'SOLVER.TNCE_WEIGHT'}, (tag, sorted(moved))
        assert c['SOLVER.TNCE_WEIGHT'] == want, (tag, c['SOLVER.TNCE_WEIGHT'])


def test_every_sweep_config_keeps_the_untied_erasing():
    """The arm is defined by these two, and they are exactly what the earlier
    runs got wrong: with erasing tied the twin pair shares one erased rectangle
    and lambda moved cross-thermal by nothing at all (4.12 / 4.20 / 4.10)."""
    for tag in SWEEP:
        c = _flat_cfg('hihr_whu_tnce_%s.yml' % tag)
        assert c['INPUT.RE_PROB'] == 0.5, tag
        assert c['DATALOADER.TIE_ERASING'] is False, tag
        assert c['DATALOADER.TIE_AUGMENTATION'] is True, tag
        assert c['DATALOADER.SYNC_FRAMES'] is True, tag


def test_the_launcher_knows_every_sweep_point():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    for tag in SWEEP:
        mode = 'whu_tnce_' + tag
        assert mode + ')' in src, mode
        assert src.count(mode) >= 2, mode


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d twin-InfoNCE tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
