"""Unit tests for the HiHR baseline recipe (arXiv:2607.09186v1, sections 4.2/4.4).

CPU-only and self-contained: no dataset, no CLIP download, no yacs.
Run directly::

    python tests/test_hihr_recipe.py

A training recipe fails silently.  Set the pretrained tower to the head's
learning rate and the run still completes, still reports a number, and is just
wrong -- so each knob gets pinned here rather than eyeballed in a log.

Covers the four places our repo differed from the paper:
  * Adam with a two-tier lr (5e-6 pretrained / 3.5e-4 fresh)
  * warmup counted in iterations, not epochs
  * cosine floor proportional to each group's own peak
  * plain triplet rather than WRT
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.make_optimizer import make_optimizer  # noqa: E402
from solver.scheduler_factory import create_scheduler  # noqa: E402


class Cfg(dict):
    """Minimal stand-in for a yacs node: attribute access over a dict."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def solver_cfg(**over):
    s = Cfg(BASE_LR=0.00035, PRETRAINED_LR=0.000005, BIAS_LR_FACTOR=1,
            WEIGHT_DECAY=1e-4, WEIGHT_DECAY_BIAS=1e-4, LARGE_FC_LR=False,
            CHART_LR_MULT=1.0, PE_DELTA_LR_MULT=1.0, MOMENTUM=0.9,
            MOD_DELTA_ONLY=False, PLD_ONLY=False,
            CENTER_LR=0.5, OPTIMIZER_NAME='Adam', MAX_EPOCHS=60,
            WARMUP_ITERS=100, WARMUP_EPOCHS=5,
            LR_MIN_FACTOR=0.002, WARMUP_LR_FACTOR=0.01)
    s.update(over)
    return Cfg(SOLVER=s)


class Toy(nn.Module):
    """Same shape of name tree as build_transformer: base.* + fresh heads."""

    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.blocks = nn.Linear(4, 4)
        self.base.pos_delta = nn.Parameter(torch.zeros(2, 4))
        self.classifier = nn.Linear(4, 3, bias=False)
        self.bottleneck = nn.BatchNorm1d(4)


class NoCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


def groups_by_name(cfg, model):
    opt, _ = make_optimizer(cfg, model, NoCenter())
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(names) == len(opt.param_groups)
    return dict(zip(names, opt.param_groups))


def test_two_tier_learning_rate():
    """4.2: 3.5e-4 for randomly initialized modules, 5e-6 for pretrained."""
    g = groups_by_name(solver_cfg(), Toy())
    assert g['base.blocks.weight']['lr'] == 5e-6
    assert g['base.blocks.bias']['lr'] == 5e-6
    assert g['classifier.weight']['lr'] == 3.5e-4
    assert g['bottleneck.weight']['lr'] == 3.5e-4
    # pos_delta lives under base.* but starts at zero -- it is a new module.
    assert g['base.pos_delta']['lr'] == 3.5e-4


def test_split_is_off_by_default():
    """PRETRAINED_LR = 0 must reproduce the old single-rate behaviour."""
    g = groups_by_name(solver_cfg(PRETRAINED_LR=0.0), Toy())
    assert {grp['lr'] for grp in g.values()} == {3.5e-4}


def test_bias_factor_multiplies_the_chosen_rate():
    """BIAS_LR_FACTOR used to overwrite the lr, which would erase the split."""
    g = groups_by_name(solver_cfg(BIAS_LR_FACTOR=2), Toy())
    assert g['base.blocks.bias']['lr'] == 2 * 5e-6
    assert g['base.blocks.weight']['lr'] == 5e-6
    # ... and it still means what it always meant when the split is off.
    g2 = groups_by_name(solver_cfg(PRETRAINED_LR=0.0, BIAS_LR_FACTOR=2), Toy())
    assert g2['base.blocks.bias']['lr'] == 2 * 3.5e-4


def test_optimizer_is_adam():
    opt, _ = make_optimizer(solver_cfg(), Toy(), NoCenter())
    assert isinstance(opt, torch.optim.Adam) and not isinstance(opt, torch.optim.AdamW)


def test_warmup_is_counted_in_iterations():
    """4.2: 100 iterations of warmup, then cosine -- not 100 epochs, not 5."""
    cfg = solver_cfg()
    model = Toy()
    opt, _ = make_optimizer(cfg, model, NoCenter())
    sched = create_scheduler(cfg, opt, iters_per_epoch=1000)

    assert sched.t_in_epochs is False
    assert sched.warmup_t == 100
    assert sched.t_initial == 60 * 1000

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    head = names.index('classifier.weight')
    peak = 3.5e-4
    # Relative tolerances: the ramp accumulates 100 float additions, so it
    # lands on the peak to ~1e-5 relative, not to the last bit.
    assert abs(sched._get_lr(0)[head] / (0.01 * peak) - 1) < 1e-9      # warmup start
    assert abs(sched._get_lr(100)[head] / peak - 1) < 1e-4            # warmup end == peak
    assert sched._get_lr(50)[head] < sched._get_lr(100)[head]         # monotone ramp
    assert sched._get_lr(30000)[head] < peak                          # cosine decays after


def test_epoch_mode_still_works_when_iters_are_off():
    cfg = solver_cfg(WARMUP_ITERS=0)
    opt, _ = make_optimizer(cfg, Toy(), NoCenter())
    sched = create_scheduler(cfg, opt, iters_per_epoch=1000)
    assert sched.t_in_epochs is True
    assert sched.warmup_t == 5 and sched.t_initial == 60


def test_iteration_mode_demands_the_iteration_count():
    cfg = solver_cfg()
    opt, _ = make_optimizer(cfg, Toy(), NoCenter())
    try:
        create_scheduler(cfg, opt)
    except ValueError as e:
        assert 'iters_per_epoch' in str(e), str(e)
    else:
        raise AssertionError('missing iters_per_epoch should have raised')


def test_cosine_floor_is_proportional_per_group():
    """One shared absolute floor would bend the two tiers differently."""
    cfg = solver_cfg()
    model = Toy()
    opt, _ = make_optimizer(cfg, model, NoCenter())
    sched = create_scheduler(cfg, opt, iters_per_epoch=1000)

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    head = names.index('classifier.weight')
    tower = names.index('base.blocks.weight')

    end = sched._get_lr(sched.t_initial)
    assert abs(end[head] / 3.5e-4 - 0.002) < 1e-6
    assert abs(end[tower] / 5e-6 - 0.002) < 1e-6
    # The same fraction of two different peaks -- that is the whole point.
    assert abs(end[head] / end[tower] - 3.5e-4 / 5e-6) < 1e-3


def test_scalar_lr_min_still_broadcasts():
    """The upstream scalar API must keep working for any older caller."""
    from solver.cosine_lr import CosineLRScheduler
    opt = torch.optim.SGD([{'params': [nn.Parameter(torch.zeros(1))], 'lr': 0.1},
                           {'params': [nn.Parameter(torch.zeros(1))], 'lr': 0.01}], lr=0.1)
    sched = CosineLRScheduler(opt, t_initial=10, lr_min=1e-5, warmup_t=2,
                              warmup_lr_init=1e-4, cycle_limit=1)
    assert sched.lr_min == [1e-5, 1e-5]
    assert sched.warmup_lr_init == [1e-4, 1e-4]
    assert sched._get_lr(0) == [1e-4, 1e-4]
    assert len(sched._get_lr(5)) == 2


def test_step_update_and_step_are_safe_in_either_mode():
    """processor.py calls both unconditionally; neither may corrupt the other."""
    for iters in (100, 0):
        cfg = solver_cfg(WARMUP_ITERS=iters)
        opt, _ = make_optimizer(cfg, Toy(), NoCenter())
        sched = create_scheduler(cfg, opt, iters_per_epoch=1000)
        before = [g['lr'] for g in opt.param_groups]
        inert = sched.step if iters else sched.step_update
        inert(3)
        assert [g['lr'] for g in opt.param_groups] == before, \
            'the wrong clock moved the learning rate (WARMUP_ITERS={})'.format(iters)
        active = sched.step_update if iters else sched.step
        active(3)
        assert [g['lr'] for g in opt.param_groups] != before


def test_config_matches_the_paper():
    """Read the shipped yaml and check it against section 4.2 line by line."""
    import re
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'configs', 'hihr_cargo_base.yml')
    with open(path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    def decoded(section, key):
        """What yacs will hand the code, not what YAML literally parsed.

        This repo writes `NAMES: ('CARGO')`.  Parentheses mean nothing in YAML,
        so safe_load returns the seven-character string "('CARGO')" -- but
        yacs runs literal_eval over every merged value, so the training run
        actually sees 'CARGO'.  Comparing against the raw parse would test the
        wrong thing and, worse, would pass if someone "fixed" the quoting in a
        way that changed the value.
        """
        from ast import literal_eval
        v = cfg[section][key]
        if isinstance(v, str):
            try:
                return literal_eval(v)
            except (ValueError, SyntaxError):
                return v
        return v

    assert cfg['MODEL']['TRANSFORMER_TYPE'] == 'vit_base_clip'   # CLIP-ViT-B/16
    assert cfg['MODEL']['METRIC_LOSS_TYPE'] == 'triplet'         # not wrt
    assert cfg['MODEL']['IF_LABELSMOOTH'] == 'off'
    # CARGO scores with the Market-1501 junk rule; the repo default 'sysu'
    # would read several points high and match nothing in the paper.
    assert decoded('DATASETS', 'NAMES') == 'CARGO'
    assert cfg['DATASETS']['PROTOCOL'] == 'ALL'                  # table 4 reports ALL
    assert cfg['TEST']['METRIC'] == 'market'
    assert cfg['INPUT']['SIZE_TRAIN'] == [256, 128]
    assert cfg['DATALOADER']['NUM_INSTANCE'] == 4
    assert cfg['SOLVER']['OPTIMIZER_NAME'] == 'Adam'
    assert cfg['SOLVER']['MAX_EPOCHS'] == 60
    assert cfg['SOLVER']['IMS_PER_BATCH'] == 64
    assert cfg['SOLVER']['BASE_LR'] == 0.00035
    assert cfg['SOLVER']['PRETRAINED_LR'] == 0.000005
    assert cfg['SOLVER']['WARMUP_ITERS'] == 100
    assert cfg['SOLVER']['LOSS_TYPE'] == 'base'
    # The checkpoint the test stage loads has to be the one training writes.
    assert cfg['TEST']['WEIGHT'] == 'transformer_{}.pth'.format(cfg['SOLVER']['MAX_EPOCHS'])
    assert cfg['SOLVER']['CHECKPOINT_PERIOD'] == cfg['SOLVER']['MAX_EPOCHS']

    # Every key must exist in defaults.py or yacs rejects the merge.
    with open(os.path.join(os.path.dirname(path), '..', 'config', 'defaults.py'),
              encoding='utf-8') as fh:
        defaults = fh.read()
    for section, body in cfg.items():
        if isinstance(body, dict):
            for k in body:
                assert re.search(r'^_C\.{}\.{}\s*='.format(section, k), defaults, re.M), \
                    '{}.{} is not in defaults.py'.format(section, k)


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
