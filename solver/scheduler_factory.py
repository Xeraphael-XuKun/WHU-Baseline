""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer, iters_per_epoch=None):
    """Cosine schedule with warmup, in either epoch or iteration units.

    SOLVER.WARMUP_ITERS > 0 switches the clock to optimizer updates, which is
    the only way to express HiHR's "warm-up strategy with 100 iterations"
    (section 4.2): at roughly 1.9k iterations per epoch, epoch granularity can
    only round that to 0 or to 19x too long.

    The floor and the warmup start are fractions of each parameter group's own
    peak lr rather than one shared number, so a two-tier setup (pretrained
    tower slow, fresh head fast) gets two identically shaped curves.
    """
    peaks = [g['lr'] for g in optimizer.param_groups]
    lr_min = [v * cfg.SOLVER.LR_MIN_FACTOR for v in peaks]
    warmup_lr_init = [v * cfg.SOLVER.WARMUP_LR_FACTOR for v in peaks]

    if cfg.SOLVER.WARMUP_ITERS > 0:
        if not iters_per_epoch:
            raise ValueError('SOLVER.WARMUP_ITERS needs iters_per_epoch; pass '
                             'len(train_loader) into create_scheduler')
        t_initial = cfg.SOLVER.MAX_EPOCHS * iters_per_epoch
        warmup_t = cfg.SOLVER.WARMUP_ITERS
        t_in_epochs = False
    else:
        t_initial = cfg.SOLVER.MAX_EPOCHS
        warmup_t = cfg.SOLVER.WARMUP_EPOCHS
        t_in_epochs = True

    return CosineLRScheduler(
        optimizer,
        t_initial=t_initial,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=t_in_epochs,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )
