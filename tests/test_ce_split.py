"""Unit tests for splitting the cross-entropy label space.

The change is small and every way it can go wrong is silent:

  * the classifier and the loss can disagree about how many classes there are,
    which is an index error at best and a wrong target at worst;
  * the triplet can be handed the split label by accident, which would remove
    the only thing still aligning viewpoints and spectra -- and the loss curve
    would look perfectly healthy;
  * the view split can quietly do nothing, because DATASETS.AERIAL_CAMS ships
    empty in every WHU config and then every image counts as ground.  A run
    like that is indistinguishable from the baseline in every line of the log.

Run:  python tests/test_ce_split.py
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss.make_loss import ce_modality_groups, ce_slot_count  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


def _load(name, *parts):
    """`processor/__init__` pulls in the trainer; load the file by path."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce_split_target = _load('proc_ce', 'processor', 'processor.py').ce_split_target


def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


class _Cfg(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def cfg(view=False, mod=False, mods=('RGB', 'IR', 'Thermal'), groups=()):
    # Every key the real config has.  A stub that lags behind defaults.py turns
    # a missing-key bug into a test suite that passes -- this has happened four
    # times on this project, so the stub is extended, never worked around with
    # getattr fallbacks in the code under test.
    return _Cfg(MODEL=_Cfg(CE_SPLIT_VIEW=view, CE_SPLIT_MODALITY=mod,
                           CE_MODALITY_GROUPS=list(groups)),
                DATASETS=_Cfg(MODALITIES=list(mods)))


# --------------------------------------------------------------------------
# fixture: 3 spectra x 2 identities x 2 captures, modality-major
# --------------------------------------------------------------------------
N_MOD, PER = 3, 4
AERIAL = torch.tensor([5, 6])
# rows 0..3 of each spectrum block: pid 0, 0, 1, 1 with cameras 5(aerial),
# 1(ground), 6(aerial), 2(ground)
PIDS = torch.tensor([0, 0, 1, 1])
CAMS = torch.tensor([5, 1, 6, 2])


def batch():
    target_rep = PIDS.repeat(N_MOD)
    camids = [CAMS.clone() for _ in range(N_MOD)]
    return target_rep, camids


# --------------------------------------------------------------------------
# 1. the slot count is defined once
# --------------------------------------------------------------------------
def test_slot_count_covers_the_four_combinations():
    assert ce_slot_count(cfg()) == 1
    assert ce_slot_count(cfg(view=True)) == 2
    assert ce_slot_count(cfg(mod=True)) == 3
    assert ce_slot_count(cfg(view=True, mod=True)) == 6


def test_an_empty_grouping_is_one_slot_per_modality():
    """The property that keeps whu_ce_mod's 10.49 / 34.77 comparable: adding
    the key must not change what any earlier config did."""
    assert ce_modality_groups(cfg(mod=True)) == [0, 1, 2]
    assert ce_slot_count(cfg(mod=True)) == 3


def test_merging_rgb_and_ir_halves_the_label_space():
    """whu_ce_mod_g2: Thermal keeps its own class, RGB and IR share one."""
    assert ce_modality_groups(cfg(mod=True, groups=(0, 0, 1))) == [0, 0, 1]
    assert ce_slot_count(cfg(mod=True, groups=(0, 0, 1))) == 2
    # and it still composes with the view split
    assert ce_slot_count(cfg(view=True, mod=True, groups=(0, 0, 1))) == 4


def test_a_grouping_the_wrong_length_is_refused():
    """It is parallel to MODALITIES; a short list would index out of range on
    some batches and not others, which is the worst kind."""
    for bad in ((0, 0), (0, 0, 1, 1)):
        try:
            ce_slot_count(cfg(mod=True, groups=bad))
        except ValueError as exc:
            assert 'parallel to DATASETS.MODALITIES' in str(exc), exc
        else:
            raise AssertionError('accepted %r' % (bad,))


def test_a_grouping_with_a_gap_is_refused_unless_asked_for():
    """[0, 0, 2] sizes the head for three slots while only two are ever used:
    500 dead columns and a label space that does not match what the config
    appears to describe.  Almost always a typo, so it stays an error -- but
    whu_ce_sparse_clip wants exactly that, and asks for it by name."""
    try:
        ce_slot_count(cfg(mod=True, groups=(0, 0, 2)))
    except ValueError as exc:
        assert 'every index from 0 upwards' in str(exc), exc
        assert 'CE_ALLOW_SPARSE_SLOTS' in str(exc), 'the message must name the way out'
    else:
        raise AssertionError('a gapped grouping was accepted without the flag')


def test_the_opt_in_allocates_the_wider_head_and_leaves_the_slot_empty():
    """With the flag on, three slots are allocated and slot 1 is never a target.
    Both halves matter: the width is what makes this a control for whu_ce_mod,
    and the emptiness is what makes those columns pure negatives."""
    c = cfg(mod=True, groups=(0, 0, 2))
    c['MODEL']['CE_ALLOW_SPARSE_SLOTS'] = True
    assert ce_modality_groups(c) == [0, 0, 2]
    assert ce_slot_count(c) == 3, 'the head has to be sized for three, not two'

    t, cams = batch()
    out, slots = ce_split_target(t, cams, N_MOD, AERIAL, False, True, [0, 0, 2])
    assert slots == 3
    used = sorted(set(int(x) for x in out))
    # label = pid * 3 + slot, slots in {0, 2}: pid 0 -> {0, 2}, pid 1 -> {3, 5}
    assert used == [0, 2, 3, 5], used
    per_identity = sorted(x % slots for x in used)
    assert per_identity == [0, 0, 2, 2], per_identity
    assert 1 not in set(x % slots for x in used), 'slot 1 must stay empty'
    # RGB and IR still share a slot -- the grouping's meaning is unchanged
    assert int(out[0]) == int(out[PER]), 'RGB and IR must still collide'
    assert int(out[0]) != int(out[2 * PER]), 'Thermal must still be apart'


def test_a_negative_group_index_is_refused_even_with_the_flag():
    c = cfg(mod=True, groups=(0, 0, -1))
    c['MODEL']['CE_ALLOW_SPARSE_SLOTS'] = True
    try:
        ce_modality_groups(c)
    except ValueError as exc:
        assert 'must be >= 0' in str(exc), exc
    else:
        raise AssertionError('a negative slot index was accepted')


def test_a_grouping_without_the_modality_split_is_refused():
    """The mtext lesson.  TEXT_MODALITY was set in that config and silently
    ignored, and nothing in the log said so -- the run looked like the intended
    one from start to finish.  A grouping with CE_SPLIT_MODALITY off is the
    same trap, so it fails at build time instead."""
    try:
        ce_slot_count(cfg(mod=False, groups=(0, 0, 1)))
    except ValueError as exc:
        assert 'silently ignored' in str(exc), exc
    else:
        raise AssertionError('an ignored grouping was accepted')


def test_the_classifier_and_the_loss_read_the_same_definition():
    """Two copies of this expression would be two chances for the head width
    and the label space to disagree, which is an index error at best."""
    head = _read('model', 'make_model.py')
    assert 'from loss.make_loss import' in head and 'ce_slot_count' in head, \
        'make_model no longer imports the shared slot count'
    assert 'ce_slot_count(cfg)' in _read('loss', 'make_loss.py')
    assert 'self.num_classes * self.ce_slots' in _read('model', 'make_model.py')


# --------------------------------------------------------------------------
# 2. the remap
# --------------------------------------------------------------------------
def test_no_split_returns_the_original_labels():
    """The property that keeps every recorded run valid."""
    t, c = batch()
    out, slots = ce_split_target(t, c, N_MOD, AERIAL, False, False)
    assert slots == 1
    assert torch.equal(out, t)


def test_the_modality_split_separates_the_three_blocks():
    t, c = batch()
    out, slots = ce_split_target(t, c, N_MOD, AERIAL, False, True)
    assert slots == 3
    # pid 0 appears in all three blocks and must land on three different labels
    got = [int(out[m * PER + 0]) for m in range(N_MOD)]
    assert len(set(got)) == 3, got
    # ... and pid 1's labels must not collide with pid 0's
    other = [int(out[m * PER + 2]) for m in range(N_MOD)]
    assert not (set(got) & set(other)), (got, other)
    assert int(out.max()) < 2 * slots


def test_the_grouping_gives_rgb_and_ir_one_label_and_thermal_another():
    """The whole point of whu_ce_mod_g2, checked on labels rather than on a
    slot count: cross entropy must go on merging RGB with IR -- that pair cost
    whu_ce_mod 78% of its lost cross-modal mAP -- while Thermal stays apart."""
    t, c = batch()
    out, slots = ce_split_target(t, c, N_MOD, AERIAL, False, True, [0, 0, 1])
    assert slots == 2
    rgb, ir, th = (out[m * PER + 0] for m in range(N_MOD))   # pid 0, each block
    assert int(rgb) == int(ir), 'RGB and IR were separated after all'
    assert int(th) != int(rgb), 'Thermal was merged with RGB'
    # 2 identities x 2 slots, nothing else
    assert len(set(out.tolist())) == 4, sorted(set(out.tolist()))


def test_the_identity_grouping_reproduces_the_ungrouped_call():
    """[0, 1, 2] is what None means; if the two paths ever disagreed, the
    grouped runs and whu_ce_mod would stop being comparable."""
    t, c = batch()
    a, sa = ce_split_target(t, c, N_MOD, AERIAL, False, True)
    b, sb = ce_split_target(t, c, N_MOD, AERIAL, False, True, [0, 1, 2])
    assert sa == sb and torch.equal(a, b)


def test_the_grouped_labels_stay_inside_the_head():
    t, c = batch()
    n_pid = int(t.max()) + 1
    for view in (False, True):
        out, slots = ce_split_target(t, c, N_MOD, AERIAL, view, True, [0, 0, 1])
        assert int(out.min()) >= 0
        assert int(out.max()) < n_pid * slots, (view, int(out.max()))


def test_the_processor_passes_the_grouping_through():
    """A config key that reaches neither ce_slot_count nor the label remap
    would produce a head sized for two slots and labels using three."""
    src = _read('processor', 'processor.py')
    assert 'ce_groups = list(cfg.MODEL.CE_MODALITY_GROUPS) or None' in src
    assert 'ce_split_view, ce_split_mod, ce_groups)' in src


def test_the_view_split_separates_aerial_from_ground():
    t, c = batch()
    out, slots = ce_split_target(t, c, N_MOD, AERIAL, True, False)
    assert slots == 2
    # rows 0 and 1 are the same identity, aerial and ground
    assert int(out[0]) != int(out[1])
    # rows 1 and 3 are different identities, both ground
    assert int(out[1]) != int(out[3])
    # the same (pid, view) in a different spectrum keeps ONE label -- the view
    # split must not accidentally split by modality as well
    assert int(out[0]) == int(out[PER]) == int(out[2 * PER])


def test_both_splits_compose_without_collision():
    t, c = batch()
    out, slots = ce_split_target(t, c, N_MOD, AERIAL, True, True)
    assert slots == 6
    # 2 identities x 2 views x 3 spectra = 12 distinct labels, all in range
    assert len(set(out.tolist())) == 12, sorted(set(out.tolist()))
    assert int(out.max()) < 2 * slots


def test_every_label_stays_inside_the_classifier_width():
    """`num_classes * slots` is what make_model allocates; a label at or above
    it is an index error inside cross_entropy, which surfaces as a CUDA assert
    far from the cause."""
    t, c = batch()
    n_pid = int(t.max()) + 1
    for view, mod in ((True, False), (False, True), (True, True)):
        out, slots = ce_split_target(t, c, N_MOD, AERIAL, view, mod)
        assert int(out.min()) >= 0
        assert int(out.max()) < n_pid * slots, (view, mod, int(out.max()), n_pid * slots)


def test_an_empty_aerial_list_makes_the_view_split_a_no_op():
    """Documents exactly why do_train refuses the combination: nothing here
    raises, the labels just silently stop encoding the view."""
    t, c = batch()
    out, _s = ce_split_target(t, c, N_MOD, torch.tensor([], dtype=torch.long),
                              True, False)
    assert int(out[0]) == int(out[1]), 'aerial and ground got different labels'


# --------------------------------------------------------------------------
# 3. the triplet must keep the raw pid
# --------------------------------------------------------------------------
def test_the_triplet_is_given_the_unsplit_target():
    """The division of labour the whole experiment rests on: CE stops merging
    a person's conditions, the metric loss goes on merging them.  Handing the
    split label to the triplet as well would remove the only thing still
    aligning viewpoints and spectra, and the loss curve would look fine."""
    src = _read('loss', 'make_loss.py')
    assert 'TRI_LOSS = triplet(feat, target)[0]' in src, \
        'the triplet is no longer using the raw pid'
    assert 'ce_target = target if target_ce is None else target_ce' in src
    assert 'def loss_func(score, feat, target, target_ce=None):' in src


def test_the_default_call_reproduces_the_old_behaviour():
    """`target_ce=None` has to be the old path exactly, or every recorded run
    stops being comparable."""
    src = _read('loss', 'make_loss.py')
    i = src.index('ce_target = target if target_ce is None else target_ce')
    j = src.index('TRI_LOSS', i)
    body = src[i:j]
    assert 'ce_target' in body and 'xent(score, ce_target)' in body


# --------------------------------------------------------------------------
# 4. the guard and the configs
# --------------------------------------------------------------------------
def test_the_processor_refuses_a_view_split_without_aerial_cams():
    src = _read('processor', 'processor.py')
    guard = [l for l in src.splitlines() if 'ce_aerial.numel() == 0' in l]
    assert len(guard) == 1, guard
    assert 'ce_split_view' in guard[0], guard[0]


def test_the_three_configs_differ_from_the_baseline_only_as_intended():
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
    base = load('hihr_whu_recipe_hihr.yml')
    want = {'view': (True, False), 'mod': (False, True), 'both': (True, True)}
    for tag, (view, mod) in want.items():
        c = load('hihr_whu_ce_%s.yml' % tag)
        added = set(c) - set(base)
        expect = {'MODEL.CE_SPLIT_VIEW', 'MODEL.CE_SPLIT_MODALITY'}
        if view:
            expect.add('DATASETS.AERIAL_CAMS')
        assert added == expect, (tag, sorted(added))
        assert {k for k in base if base[k] != c[k]} == {'OUTPUT_DIR'}, tag
        assert c['MODEL.CE_SPLIT_VIEW'] is view, tag
        assert c['MODEL.CE_SPLIT_MODALITY'] is mod, tag
        if view:
            # Without this the split is a no-op; the guard raises, but the
            # config must not rely on the guard to be correct.
            assert c['DATASETS.AERIAL_CAMS'] == [5, 6], tag


def test_the_grouped_config_differs_from_whu_ce_mod_by_one_key():
    """One factor wide, or the comparison with 10.49 / 34.77 says nothing."""
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
    ref = load('hihr_whu_ce_mod.yml')
    g2 = load('hihr_whu_ce_mod_g2.yml')
    assert set(g2) - set(ref) == {'MODEL.CE_MODALITY_GROUPS'}, sorted(set(g2) - set(ref))
    assert set(ref) - set(g2) == set()
    assert {k for k in ref if ref[k] != g2[k]} == {'OUTPUT_DIR'}
    assert g2['MODEL.CE_MODALITY_GROUPS'] == [0, 0, 1]
    # The grouping is positional, so it only means "Thermal alone" for this
    # order.  A reordered MODALITIES would silently group a different pair.
    assert g2['DATASETS.MODALITIES'] == ['RGB', 'IR', 'Thermal']
    assert g2['MODEL.CE_SPLIT_MODALITY'] is True
    assert g2['MODEL.CE_SPLIT_VIEW'] is False


def test_the_four_tuning_arms_each_change_one_thing():
    """A hyper-parameter sweep is only readable if the arms are one factor
    wide, and these were generated from the g2 config rather than copied so
    that stays true.  The two epoch arms carry two extra keys each; both are
    forced by MAX_EPOCHS rather than chosen -- CHECKPOINT_PERIOD decides
    whether the final epoch is saved at all, and TEST.WEIGHT names the file
    reeval.sh and dump_all.sh look for."""
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
    g2 = load('hihr_whu_ce_mod_g2.yml')
    arms = {'re70': ({'INPUT.RE_PROB': 0.7}, set())}
    for n in (40, 50, 90, 120):
        arms['e%d' % n] = ({'SOLVER.MAX_EPOCHS': n,
                            'SOLVER.CHECKPOINT_PERIOD': n,
                            'TEST.WEIGHT': 'transformer_%d.pth' % n}, set())
    weights = {'tri2': 2.0, 'tri23': 2.3, 'tri26': 2.6, 'tri3': 3.0, 'tri4': 4.0}
    for tag in weights:
        arms[tag] = ({}, {'MODEL.TRIPLET_LOSS_WEIGHT'})
    # Regularisation.  DROP_PATH / ATT_DROP_RATE are absent from the g2 config
    # -- they take defaults.py's 0.1 and 0.0 -- so those arms ADD a key, while
    # WEIGHT_DECAY and IF_LABELSMOOTH are set there and get edited.
    arms['dp20'] = ({}, {'MODEL.DROP_PATH'})
    arms['dp30'] = ({}, {'MODEL.DROP_PATH'})
    arms['attn10'] = ({}, {'MODEL.ATT_DROP_RATE'})
    arms['wd5e4'] = ({'SOLVER.WEIGHT_DECAY': '5e-4'}, set())
    arms['ls'] = ({'MODEL.IF_LABELSMOOTH': 'on'}, set())
    # The one arm that is deliberately NOT one factor wide: it asks whether the
    # axis exists at all, in one run instead of six.
    arms['regall'] = ({'SOLVER.WEIGHT_DECAY': '5e-4',
                       'MODEL.IF_LABELSMOOTH': 'on'},
                      {'MODEL.DROP_PATH', 'MODEL.ATT_DROP_RATE'})
    drops = {'dp20': 0.2, 'dp30': 0.3, 'regall': 0.2}
    # The text line.  PE_LAYERWISE is not a second idea being added: the loss
    # contrasts a delta-on pass with a delta-off one, so with no delta the two
    # are the same tensor and the before/after cells contradict each other.
    text = {'MODEL.PE_LAYERWISE', 'MODEL.PE_FREEZE_BASE', 'MODEL.TEXT_ALIGN',
            'MODEL.TEXT_TARGET', 'MODEL.TEXT_TEMPLATE', 'MODEL.TEXT_N_CTX',
            'MODEL.TEXT_MODALITY', 'MODEL.TEXT_MODALITY_WORDS',
            'SOLVER.TEXT_LOSS_WEIGHT'}
    arms['pe'] = ({}, text)
    arms['mcam'] = ({}, text)
    arms['mcam_grp'] = ({}, text | {'MODEL.TEXT_MODALITY_TARGETS'})
    # The stacking arm.  Two keys on purpose -- it exists to decide two
    # individually in-band results in one run.  Registered after the loop
    # above, which would otherwise mark it as triplet-weight-only.
    arms['ls_tri23'] = ({'MODEL.IF_LABELSMOOTH': 'on'},
                        {'MODEL.TRIPLET_LOSS_WEIGHT'})
    weights['ls_tri23'] = 2.3
    # The initialisation arm.  Both keys move together by necessity: 'imagenet'
    # names a backbone-only archive and 'self' names a checkpoint from this
    # codebase, so pointing one at the other's file makes load_param raise.
    arms['clip'] = ({'MODEL.PRETRAIN_CHOICE': 'imagenet',
                     'MODEL.PRETRAIN_PATH': 'ViT-B-16.pt'}, set())
    # The stacking arms.  Each carries the raw-CLIP initialisation plus exactly
    # the keys one measured arm adds to whu_recipe_clip, so the increment they
    # test is that arm and nothing else.  Checked as set identity below.
    init = {'MODEL.PRETRAIN_CHOICE': 'imagenet',
            'MODEL.PRETRAIN_PATH': 'ViT-B-16.pt'}
    pe_keys = {'MODEL.PE_LAYERWISE', 'MODEL.PE_FREEZE_BASE',
               'SOLVER.PE_DELTA_LR_MULT'}
    text_keys = pe_keys | {
        'MODEL.TEXT_ALIGN', 'MODEL.TEXT_CLIP_PATH', 'MODEL.TEXT_MODALITY_WORDS',
        'MODEL.TEXT_N_CTX', 'MODEL.TEXT_TEMPLATE', 'SOLVER.TEXT_LOSS_WEIGHT',
        'DATASETS.AERIAL_CAMS'}
    arms['pe_clip'] = (dict(init), pe_keys)
    arms['mtext_clip'] = (dict(init), text_keys)
    # Same epoch-budget question as e90 on the cargo tower, asked again because
    # the raw tower has had 60 fewer epochs of training in total.
    arms['e90_clip'] = (dict(init, **{
        'SOLVER.MAX_EPOCHS': 90, 'SOLVER.CHECKPOINT_PERIOD': 90,
        'TEST.WEIGHT': 'transformer_90.pth'}), set())

    # Every hihr_whu_ce_mod_g2_*.yml on disk has to appear above.  Without this
    # a new arm can be added, launched, and never checked -- and the failure it
    # would have caught is a config that differs by more than it claims.
    import glob
    on_disk = {os.path.basename(p)[len('hihr_whu_ce_mod_g2_'):-len('.yml')]
               for p in glob.glob(os.path.join(here, 'hihr_whu_ce_mod_g2_*.yml'))}
    assert on_disk == set(arms), (sorted(on_disk ^ set(arms)))
    for tag, (changed, added) in arms.items():
        c = load('hihr_whu_ce_mod_g2_%s.yml' % tag)
        assert set(c) - set(g2) == added, (tag, sorted(set(c) - set(g2)))
        assert set(g2) - set(c) == set(), tag
        moved = {k for k in g2 if g2[k] != c[k]}
        assert moved == set(changed) | {'OUTPUT_DIR'}, (tag, sorted(moved))
        for k, v in changed.items():
            assert c[k] == v, (tag, k, c[k])
        if tag in weights:
            assert c['MODEL.TRIPLET_LOSS_WEIGHT'] == weights[tag], tag
        if tag in drops:
            assert c['MODEL.DROP_PATH'] == drops[tag], tag
        if tag in ('attn10', 'regall'):
            assert c['MODEL.ATT_DROP_RATE'] == 0.1, tag
        # WEIGHT_DECAY_BIAS stays put on purpose: moving it too would make the
        # weight-decay arm two factors, and biases are usually excluded from
        # decay altogether.
        assert c['SOLVER.WEIGHT_DECAY_BIAS'] == g2['SOLVER.WEIGHT_DECAY_BIAS'], tag
        # the thing under test in every arm is still g2's grouping
        assert c['MODEL.CE_MODALITY_GROUPS'] == [0, 0, 1], tag
        # ... and every arm but the initialisation one starts from the same
        # weights, or a difference in the result has two possible causes
        if not tag.endswith('clip'):
            assert c['MODEL.PRETRAIN_CHOICE'] == 'self', tag
            assert 'cargo_base' in c['MODEL.PRETRAIN_PATH'], tag

    # The stacking arm has to BE its two parents composed, not a third recipe
    # that happens to sit near them.  Checked against each parent separately,
    # because a run that differs from both in some other key would answer a
    # question nobody asked.
    stack = load('hihr_whu_ce_mod_g2_ls_tri23.yml')
    ls_p = load('hihr_whu_ce_mod_g2_ls.yml')
    tri_p = load('hihr_whu_ce_mod_g2_tri23.yml')
    assert set(stack) - set(ls_p) == {'MODEL.TRIPLET_LOSS_WEIGHT'}
    assert {k for k in ls_p if ls_p[k] != stack[k]} == {'OUTPUT_DIR'}
    assert set(stack) - set(tri_p) == set()
    assert ({k for k in tri_p if tri_p[k] != stack[k]}
            == {'MODEL.IF_LABELSMOOTH', 'OUTPUT_DIR'})
    assert stack['MODEL.IF_LABELSMOOTH'] == 'on'
    assert stack['MODEL.TRIPLET_LOSS_WEIGHT'] == 2.3

    # The text arms have to differ from EACH OTHER by one thing as well, or the
    # three-point comparison they exist for says nothing.  pe is the control:
    # same second forward pass, same prompts, loss weight zero.
    pe, mc, gr = (load('hihr_whu_ce_mod_g2_%s.yml' % t) for t in
                  ('pe', 'mcam', 'mcam_grp'))
    assert {k for k in mc if mc[k] != pe[k]} == {
        'SOLVER.TEXT_LOSS_WEIGHT', 'OUTPUT_DIR'}, sorted(
            k for k in mc if mc[k] != pe[k])
    assert pe['SOLVER.TEXT_LOSS_WEIGHT'] == 0.0
    assert mc['SOLVER.TEXT_LOSS_WEIGHT'] == 5.0
    assert set(gr) - set(mc) == {'MODEL.TEXT_MODALITY_TARGETS'}
    assert {k for k in mc if mc[k] != gr[k]} == {'OUTPUT_DIR'}
    assert gr['MODEL.TEXT_MODALITY_TARGETS'] == [0, 1, 0]
    # positional, so it only means "Thermal alone" for this spectrum order
    assert gr['DATASETS.MODALITIES'] == ['RGB', 'IR', 'Thermal']
    assert gr['MODEL.TEXT_MODALITY'] == 'RGB'
    # and the prompt is whu_mcam's, not a retyped variant of it
    mcam = load('hihr_whu_mcam.yml')
    for k in ('MODEL.TEXT_TEMPLATE', 'MODEL.TEXT_N_CTX',
              'MODEL.TEXT_MODALITY_WORDS', 'MODEL.TEXT_MODALITY'):
        assert gr[k] == mcam[k], (k, gr[k], mcam[k])


def test_the_regularisation_keys_are_live():
    """Every one of them is a config key this project has never set, and this
    repo has a history of keys that yacs accepts and nothing reads --
    MODEL.ID_LOSS_TYPE, TEXT_MODALITY in the mtext runs, TEST.FEAT_NORM until
    it was fixed on 2026-08-13.  Traced to where each actually does something
    before six runs are built on them."""
    mm = _read('model', 'make_model.py')
    for key in ('DROP_PATH', 'DROP_OUT', 'ATT_DROP_RATE'):
        assert 'cfg.MODEL.%s' % key in mm, key

    vit = _read('model', 'backbones', 'vit_pytorch.py')
    # drop_path_rate -> per-block rate -> an actual DropPath module
    assert 'torch.linspace(0, drop_path_rate, depth)' in vit
    assert 'DropPath(drop_path) if drop_path > 0. else nn.Identity()' in vit
    # attn_drop_rate -> nn.Dropout applied to the attention matrix
    assert 'self.attn_drop = nn.Dropout(attn_drop)' in vit
    assert 'attn = self.attn_drop(attn)' in vit

    # label smoothing: a string comparison, not the truthiness test that made
    # TEST.FEAT_NORM a no-op, and sized over the SPLIT label space
    ml = _read('loss', 'make_loss.py')
    assert "cfg.MODEL.IF_LABELSMOOTH == 'on'" in ml
    assert 'CrossEntropyLabelSmooth(num_classes=num_classes * ce_slot_count(cfg))' in ml

    # weight decay reaches the parameter groups
    assert 'weight_decay = cfg.SOLVER.WEIGHT_DECAY' in _read('solver', 'make_optimizer.py')


def test_the_triplet_weight_is_a_live_key():
    """MODEL.ID_LOSS_TYPE is set in configs and read by nothing -- dead code
    that looks configured.  Before an arm is built on TRIPLET_LOSS_WEIGHT, the
    multiplication has to be there."""
    src = _read('loss', 'make_loss.py')
    assert 'cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS' in src


def test_the_backbone_by_grouping_square_is_square():
    """Four runs, two factors, and each edge has to differ by exactly one of
    them or the 2x2 says nothing.

                          from cargo_base      from raw CLIP
        unsplit CE          recipe_hihr         recipe_clip
        CE grouping         ce_mod_g2           ce_mod_g2_clip

    Dropping the CARGO leg was measured at +1.75 on the grouped row, which is
    larger than the grouping's own effect on the cargo row -- so the grouping's
    contribution has to be re-measured on the backbone actually being used, and
    that only means something if the square is square.
    """
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
    cells = {k: load('hihr_whu_%s.yml' % k) for k in
             ('recipe_hihr', 'recipe_clip', 'ce_mod_g2', 'ce_mod_g2_clip')}

    ce_keys = {'MODEL.CE_SPLIT_VIEW', 'MODEL.CE_SPLIT_MODALITY',
               'MODEL.CE_MODALITY_GROUPS'}
    init_keys = {'MODEL.PRETRAIN_CHOICE', 'MODEL.PRETRAIN_PATH'}

    # horizontal edges: only the initialisation moves
    for a, b in (('recipe_hihr', 'recipe_clip'), ('ce_mod_g2', 'ce_mod_g2_clip')):
        assert set(cells[a]) == set(cells[b]), (a, b)
        moved = {k for k in cells[a] if cells[a][k] != cells[b][k]}
        assert moved == init_keys | {'OUTPUT_DIR'}, (a, b, sorted(moved))
        assert cells[b]['MODEL.PRETRAIN_CHOICE'] == 'imagenet'
        assert cells[b]['MODEL.PRETRAIN_PATH'] == 'ViT-B-16.pt'

    # vertical edges: only the grouping moves
    for a, b in (('recipe_hihr', 'ce_mod_g2'), ('recipe_clip', 'ce_mod_g2_clip')):
        assert set(cells[b]) - set(cells[a]) == ce_keys, (a, b)
        assert set(cells[a]) - set(cells[b]) == set(), (a, b)
        moved = {k for k in cells[a] if cells[a][k] != cells[b][k]}
        assert moved == {'OUTPUT_DIR'}, (a, b, sorted(moved))
        assert cells[b]['MODEL.CE_MODALITY_GROUPS'] == [0, 0, 1]


def test_the_five_reruns_carry_the_same_keys_as_the_runs_they_redo():
    """A re-run is only a re-run if it changes the initialisation and nothing
    else.  Checked as a set identity rather than by eye:

        keys(X_clip) - keys(recipe_clip)  ==  keys(X) - keys(recipe_hihr)

    Each source config only ADDS keys to recipe_hihr -- none changes or removes
    one -- so this is the whole difference, not a summary of it.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_%s.yml' % n),
                                              encoding='utf-8')))
    old_base, new_base = load('recipe_hihr'), load('recipe_clip')
    for src in ('pe_indep', 'pe_chain', 'text_lam50', 'mtext_lam50', 'mcam'):
        new = load(src + '_clip')
        assert set(new) - set(new_base) == set(load(src)) - set(old_base), src
        assert set(new_base) - set(new) == set(), src
        moved = {k for k in new_base if k in new and new_base[k] != new[k]}
        assert moved == {'OUTPUT_DIR'}, (src, sorted(moved))
        assert new['MODEL.PRETRAIN_CHOICE'] == 'imagenet', src
        assert new['MODEL.PRETRAIN_PATH'] == 'ViT-B-16.pt', src
    # the spectrum-axis one is the only one aimed at the remaining deficit
    assert load('mcam_clip')['MODEL.TEXT_TARGET'] == 'modality'
    for other in ('text_lam50_clip', 'mtext_lam50_clip'):
        assert 'MODEL.TEXT_TARGET' not in load(other), other

    # The grouped spectrum arm differs from the ungrouped one by that key and
    # nothing else.  Measured 2026-08-13 on the g2 tower, where the same pair
    # shared a control at 11.75: ungrouped 10.69, grouped 11.63.  Asking the
    # text loss to drag RGB<->IR together costs 0.94 mAP, five times the band --
    # the second independent confirmation that the pair is already aligned and
    # every mechanism pointed at it loses money.
    grp, plain = load('mcam_grp_clip'), load('mcam_clip')
    assert set(grp) - set(plain) == {'MODEL.TEXT_MODALITY_TARGETS'}
    assert {k for k in plain if plain[k] != grp[k]} == {'OUTPUT_DIR'}
    assert grp['MODEL.TEXT_MODALITY_TARGETS'] == [0, 1, 0]
    assert grp['DATASETS.MODALITIES'] == ['RGB', 'IR', 'Thermal']


def test_the_cargo_transfer_arm_uses_cargos_own_camera_ids():
    """The one way this arm fails silently.

    WHU-MARS ships DATASETS.AERIAL_CAMS [5, 6].  CARGO parses Cam1..Cam13
    one-based and datasets/cargo.py's `_camid` returns `camid - 1` under
    PROTOCOL 'ALL', so its five drones are ZERO-BASED 0..4 -- and 5 and 6 there
    are two GROUND cameras.  With the wrong list `_text_subset` marks every
    image ground, the aerial->ground term never fires, and the run finishes
    looking entirely normal with a text loss that trained nothing.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    c = load('cargo_pe_text')
    assert c['DATASETS.AERIAL_CAMS'] == [0, 1, 2, 3, 4], c['DATASETS.AERIAL_CAMS']
    # every config in this repo writes NAMES as ('CARGO'), which yaml reads as
    # the literal string "('CARGO')" -- not a tuple
    assert 'CARGO' in c['DATASETS.NAMES'], c['DATASETS.NAMES']
    assert c['DATASETS.PROTOCOL'] == 'ALL', 'the zero-basing above assumes ALL'
    # the source of that mapping, pinned so a refactor there fails here
    src = _read('datasets', 'cargo.py')
    assert 'aerial_max_cam = 5' in src
    assert 'return camid - 1' in src

    # it differs from the run it is compared against by the transferable keys
    # only -- and NOT by the two that CARGO cannot carry
    base = load('cargo_base')
    added = set(c) - set(base)
    assert added == {
        'MODEL.PE_LAYERWISE', 'MODEL.PE_FREEZE_BASE', 'MODEL.TEXT_ALIGN',
        'MODEL.TEXT_CLIP_PATH', 'MODEL.TEXT_TEMPLATE', 'MODEL.TEXT_N_CTX',
        'MODEL.TEXT_MODALITY', 'SOLVER.PE_DELTA_LR_MULT',
        'SOLVER.TEXT_LOSS_WEIGHT', 'DATASETS.AERIAL_CAMS'}, sorted(added)
    assert {k for k in base if base[k] != c[k]} == {'OUTPUT_DIR'}
    for absent in ('MODEL.CE_SPLIT_MODALITY', 'MODEL.CE_MODALITY_GROUPS',
                   'MODEL.TEXT_MODALITY_WORDS', 'MODEL.TEXT_TARGET'):
        assert absent not in c, absent
    # single pseudo-modality, and TEXT_MODALITY has to name it
    assert c['DATASETS.MODALITIES'] == ['RGB']
    assert c['MODEL.TEXT_MODALITY'] == 'RGB'


def test_the_view_split_arm_differs_by_one_key():
    """The viewpoint split, re-measured on the raw CLIP tower on top of the main
    method.  Its one prior measurement (9.35 / 23.65, against a 10.23 baseline)
    was taken on the collapsed CARGO tower, where the delta read +0.13 and the
    text +0.24 -- both of which turned out to be +1.30 and +2.22 once that leg
    was dropped.  A single key has to be the only difference, or the re-measure
    answers a different question than the one it is compared against.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    base, new = load('whu_text_lam50_clip'), load('whu_ce_view_clip')
    assert set(new) - set(base) == {'MODEL.CE_SPLIT_VIEW'}, sorted(set(new) - set(base))
    diff = {k for k in base if k in new and base[k] != new[k]}
    assert diff == {'OUTPUT_DIR'}, sorted(diff)
    assert new['MODEL.CE_SPLIT_VIEW'] is True
    # The spectrum split must stay off: both at once is whu_ce_both, 3000
    # classes, which was the worst arm of that group by a wide margin.
    assert new.get('MODEL.CE_SPLIT_MODALITY', False) is False
    # And the split is meaningless without the camera list -- do_train refuses
    # that pairing rather than mapping every image to "ground".
    assert new['DATASETS.AERIAL_CAMS'] == [5, 6]
    # Still the main method underneath, or 14.11 is not its control.
    assert new['MODEL.PE_LAYERWISE'] == 'indep' and new['MODEL.TEXT_ALIGN'] is True


def test_the_cargo_arm_has_a_control():
    """cargo_pe_text bundles the delta and the text loss, so its -0.66 against
    cargo_base could be either.  The control isolates it, exactly as
    whu_ce_mod_g2_pe did for the WHU text arms -- without that one, mcam's
    +1.34 would have been credited to the text loss when all of it was the
    delta.

    Text weight 0 rather than the text keys removed: the tower is still built
    and the second delta-free forward still runs, so the two arms cost the same
    and differ in one number.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    both, ctrl = load('cargo_pe_text'), load('cargo_pe')
    assert set(both) == set(ctrl)
    assert {k for k in both if both[k] != ctrl[k]} == {
        'SOLVER.TEXT_LOSS_WEIGHT', 'OUTPUT_DIR'}, sorted(
            k for k in both if both[k] != ctrl[k])
    assert both['SOLVER.TEXT_LOSS_WEIGHT'] == 5.0
    assert ctrl['SOLVER.TEXT_LOSS_WEIGHT'] == 0.0
    # the tower is still constructed, so the two runs cost the same
    assert ctrl['MODEL.TEXT_ALIGN'] is True
    assert ctrl['MODEL.PE_LAYERWISE'] == 'indep'
    # and it still carries CARGO's own camera ids, not WHU's
    assert ctrl['DATASETS.AERIAL_CAMS'] == [0, 1, 2, 3, 4]


def test_the_epoch_budget_is_checked_at_both_ends():
    """One arm's 90-epoch check is a weaker claim than two at opposite ends of
    the method spectrum -- and the baseline is the end that matters, because
    every increment this project reports is measured against it."""
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    for src, e90 in (('whu_recipe_clip', 'whu_recipe_clip_e90'),
                     ('whu_text_lam50_clip', 'whu_text_lam50_e90_clip'),
                     ('whu_ce_mod_g2_clip', 'whu_ce_mod_g2_e90_clip')):
        a, b = load(src), load(e90)
        assert set(a) == set(b), (src, e90)
        assert {k for k in a if a[k] != b[k]} == {
            'SOLVER.MAX_EPOCHS', 'SOLVER.CHECKPOINT_PERIOD', 'TEST.WEIGHT',
            'OUTPUT_DIR'}, (e90, sorted(k for k in a if a[k] != b[k]))
        assert b['SOLVER.MAX_EPOCHS'] == 90
        assert b['TEST.WEIGHT'] == 'transformer_90.pth'


def test_the_delta_learning_rate_is_finally_swept():
    """pos_delta is excluded from the PRETRAINED_LR tier (make_optimizer.py) so
    it takes BASE_LR, then SOLVER.PE_DELTA_LR_MULT on top -- which has been 1.0
    in every run this project has ever made, while the delta grew into its
    single largest contribution."""
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    ref = load('whu_text_lam50_clip')
    assert ref['SOLVER.PE_DELTA_LR_MULT'] == 1.0
    for tag, want in (('05', 0.5), ('20', 2.0)):
        c = load('whu_text_lam50_pedlr%s_clip' % tag)
        assert set(c) == set(ref), tag
        assert {k for k in ref if ref[k] != c[k]} == {
            'SOLVER.PE_DELTA_LR_MULT', 'OUTPUT_DIR'}, tag
        assert c['SOLVER.PE_DELTA_LR_MULT'] == want, tag
    # and the multiplier really is applied to pos_delta, not decoration
    opt = _read('solver', 'make_optimizer.py')
    assert 'lr = lr * cfg.SOLVER.PE_DELTA_LR_MULT' in opt
    assert "'pos_delta' not in key" in opt, 'the delta is no longer on BASE_LR'


def test_the_two_cargo_era_justifications_get_remeasured():
    """Two design choices whose evidence came from the tower that suppressed
    everything, and which are therefore re-taken on the raw one:

      * the GROUPING.  "merge RGB+IR, split Thermal off" was chosen from
        whu_ce_mod 10.49 against whu_ce_mod_g2 11.58 -- both cargo-era, as was
        the section [2b] measurement behind them.  whu_ce_mod_clip is the
        three-way form on the raw tower.
      * the WEIGHT.  Every lambda tried (0.1 / 1.0 / 5.0 / 10.0) was tried
        where the text loss was worth +0.24 at best, i.e. inside the band; 5.0
        came out of choosing among readings that did not differ, and was
        carried over to a tower where the same loss is worth +0.92.

    This project has already paid once for inheriting a cargo-era magnitude:
    the positional delta was closed as noise at +0.13 and is worth +1.30 here.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_%s.yml' % n),
                                              encoding='utf-8')))

    # the three-way split is the grouped one with the grouping REMOVED
    g2c, three = load('ce_mod_g2_clip'), load('ce_mod_clip')
    assert set(g2c) - set(three) == {'MODEL.CE_MODALITY_GROUPS'}
    assert set(three) - set(g2c) == set()
    assert {k for k in three if three[k] != g2c[k]} == {'OUTPUT_DIR'}
    assert three['MODEL.CE_SPLIT_MODALITY'] is True, 'it must still split'
    assert three['MODEL.PRETRAIN_CHOICE'] == 'imagenet'

    # the sweep is one key off the measured point, on the 2-anchor variant
    five = load('text_lam50_clip')
    for tag, lam in (('lam10', 1.0), ('lam100', 10.0)):
        c = load('text_%s_clip' % tag)
        assert set(c) == set(five), tag
        assert {k for k in five if five[k] != c[k]} == {
            'SOLVER.TEXT_LOSS_WEIGHT', 'OUTPUT_DIR'}, tag
        assert c['SOLVER.TEXT_LOSS_WEIGHT'] == lam, tag
        # 2 anchors, not 6: the sweep is on the form the paper would report
        assert 'MODEL.TEXT_MODALITY_WORDS' not in c, tag
        assert c['MODEL.TEXT_MODALITY'] == 'RGB', tag
    # and lambda 0 already exists as a run, so the axis has a real zero
    assert 'SOLVER.TEXT_LOSS_WEIGHT' not in load('pe_indep_clip')


def test_the_stacking_arms_are_their_parts_composed():
    """A stack is only readable if each of its layers is exactly one measured
    arm.  Checked against the arms whose numbers it will be compared with:

        recipe_clip      11.99      no grouping, no delta, no text
        ce_mod_g2_clip   13.33      + grouping
        ce_mod_g2_pe_clip           + delta   <- this run
        ce_mod_g2_mtext_clip        + text    <- and this one

    so that ce_mod_g2_pe_clip minus ce_mod_g2_clip is the delta's contribution
    on a grouped backbone, and the three-way minus the two-way is the text
    loss's.  A key that crept in anywhere would make one of those differences
    mean two things.
    """
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
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_whu_%s.yml' % n),
                                              encoding='utf-8')))
    g2c, rec = load('ce_mod_g2_clip'), load('recipe_clip')
    for stack, measured in (('ce_mod_g2_pe_clip', 'pe_indep_clip'),
                            ('ce_mod_g2_mtext_clip', 'mtext_lam50_clip')):
        s = load(stack)
        # exactly what the measured arm adds to the unsplit baseline
        assert set(s) - set(g2c) == set(load(measured)) - set(rec), stack
        assert set(g2c) - set(s) == set(), stack
        moved = {k for k in g2c if k in s and g2c[k] != s[k]}
        assert moved == {'OUTPUT_DIR'}, (stack, sorted(moved))
        # and the grouping it is stacked onto is still there
        assert s['MODEL.CE_MODALITY_GROUPS'] == [0, 0, 1], stack
        assert s['MODEL.PRETRAIN_CHOICE'] == 'imagenet', stack

    # the three-way is the two-way plus the text keys, nothing else
    two, three = load('ce_mod_g2_pe_clip'), load('ce_mod_g2_mtext_clip')
    assert set(three) - set(two) == {
        'MODEL.TEXT_ALIGN', 'MODEL.TEXT_CLIP_PATH', 'MODEL.TEXT_MODALITY_WORDS',
        'MODEL.TEXT_N_CTX', 'MODEL.TEXT_TEMPLATE', 'SOLVER.TEXT_LOSS_WEIGHT',
        'DATASETS.AERIAL_CAMS'}, sorted(set(three) - set(two))
    assert {k for k in two if k in three and two[k] != three[k]} == {'OUTPUT_DIR'}

    # SOLVER.PE_DELTA_LR_MULT is read by make_optimizer, not by the model; a
    # copy that landed under MODEL would be rejected by yacs at merge time.
    for n in ('ce_mod_g2_pe_clip', 'ce_mod_g2_mtext_clip'):
        raw = yaml.safe_load(open(os.path.join(here, 'hihr_whu_%s.yml' % n),
                                  encoding='utf-8'))
        assert 'PE_DELTA_LR_MULT' in raw['SOLVER'], n
        assert 'PE_DELTA_LR_MULT' not in raw['MODEL'], n

    # the text loss constrains the DELTA, so it cannot ship without one
    assert three['MODEL.PE_LAYERWISE'] == 'indep'
    assert three['SOLVER.TEXT_LOSS_WEIGHT'] == 5.0


def test_direction_c_is_not_among_the_reruns():
    """Closed on the evaluation protocol -- the sysu junk rule drops the
    query's whole camera, so the twin frame is never scored -- and no backbone
    changes that.  A `_clip` twin of it would be GPU spent on a settled
    question, so its absence is pinned rather than left to memory."""
    import glob
    here = os.path.join(ROOT, 'configs')
    for p in glob.glob(os.path.join(here, '*_clip.yml')):
        n = os.path.basename(p)
        assert 'tnce' not in n and 'twin' not in n and 'sync' not in n, n


def test_the_launcher_knows_all_three():
    src = _read('run_hihr.sh')
    for tag in ('mod_g2_pe', 'mod_g2_mcam', 'mod_g2_mcam_grp',
                'view', 'mod', 'both', 'mod_g2', 'mod_g2_re70',
                'mod_g2_e40', 'mod_g2_e50', 'mod_g2_e90', 'mod_g2_e120',
                'mod_g2_tri2', 'mod_g2_tri23', 'mod_g2_tri26',
                'mod_g2_tri3', 'mod_g2_tri4',
                'mod_g2_regall', 'mod_g2_dp20', 'mod_g2_dp30',
                'mod_g2_attn10', 'mod_g2_wd5e4', 'mod_g2_ls'):
        mode = 'whu_ce_' + tag
        assert mode + ')' in src, mode
        assert src.count(mode) >= 2, mode
        assert os.path.exists(os.path.join(ROOT, 'configs', 'hihr_%s.yml' % mode)), mode


def test_the_logged_accuracy_uses_the_ce_label_space():
    """Measured, not hypothetical: whu_ce_mod printed `Acc: 0.001` at epoch 6
    while its cross entropy had already fallen from ln(1500)=7.31 to 2.36.  The
    classifier predicts `pid * slots + slot` and the accuracy was comparing
    that against the raw pid, so it read ~0 however well the head was doing --
    a diagnostic line that says the opposite of the truth is worse than none.
    """
    src = _read('processor', 'processor.py')
    assert 'target_rep if target_ce is None else target_ce)).float().mean()' in src,         'Acc is not measured against the CE label space'
    assert '(cls_score.max(1)[1] == target_rep).float().mean()' not in src


def test_the_batch_geometry_is_printed_on_every_run():
    """It used to live inside the text and twin branches only, so a plain run
    printed nothing -- and the first read on whether a change moved the feature
    geometry had to wait for the run to end, the checkpoint to survive the copy
    and a feature dump to be scheduled.  Two of those three failed this week.

    Computed and printed exactly once: three branches each doing it would put a
    duplicate line under precisely the configs whose diagnostics matter most.
    """
    src = _read('processor', 'processor.py')
    assert src.count("768d d_same") == 1, 'the geometry line is printed twice'
    assert src.count('feature_geometry(') == 2,         'feature_geometry is defined once and called once'
    i = src.index('geom = feature_geometry(global_feat, num_modalities, group_size)')
    j = src.index("if (n_iter + 1) % log_period == 0:")
    assert j < i, 'the geometry is computed outside the LOG_PERIOD block'


def test_the_geometry_line_reports_ratios_to_d_same():
    """Absolute distances drift with training progress; the ratios are what
    can be compared across runs, and what diag/analyse_modality.py quotes."""
    src = _read('processor', 'processor.py')
    i = src.index("768d d_same")
    body = src[i:i + 600]
    assert 'v / d' in body and "geom['d_diffpid'] / d" in body


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d CE-split tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
