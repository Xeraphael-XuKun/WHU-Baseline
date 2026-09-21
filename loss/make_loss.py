# encoding: utf-8

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .wrt_loss import TripletLoss_WRT
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def ce_modality_groups(cfg):
    """Which CE slot each modality falls into, as a list of ints.

    Empty MODEL.CE_MODALITY_GROUPS means one slot per modality -- what
    whu_ce_mod ran.  [0, 0, 1] merges RGB and IR and leaves Thermal on its own,
    which is whu_ce_mod_g2; see config/defaults.py for the measurement that
    picked that grouping.

    Validated here rather than at either call site: make_model sizes the
    classifier from this and make_loss sizes the label smoothing, so the two
    reading it differently is an index error inside cross_entropy, and CUDA
    reports those far from the cause.
    """
    mods = list(cfg.DATASETS.MODALITIES)
    groups = [int(g) for g in cfg.MODEL.CE_MODALITY_GROUPS]
    if not groups:
        return list(range(len(mods)))
    if len(groups) != len(mods):
        raise ValueError(
            'MODEL.CE_MODALITY_GROUPS has {} entries for the {} modalities {} '
            '-- it is parallel to DATASETS.MODALITIES'.format(
                len(groups), len(mods), mods))
    if min(groups) < 0:
        raise ValueError('MODEL.CE_MODALITY_GROUPS entries must be >= 0, got {}'
                         .format(groups))
    if sorted(set(groups)) != list(range(max(groups) + 1)):
        # A gap leaves classifier columns no label ever selects: the head comes
        # out wider than the label space.  Usually a typo, and one that wastes a
        # whole run silently -- so it has to be asked for by name.
        if not getattr(cfg.MODEL, 'CE_ALLOW_SPARSE_SLOTS', False):
            raise ValueError(
                'MODEL.CE_MODALITY_GROUPS must use every index from 0 upwards, '
                'got {}.  Set MODEL.CE_ALLOW_SPARSE_SLOTS True if the unused '
                'slots are deliberate.'.format(groups))
        missing = sorted(set(range(max(groups) + 1)) - set(groups))
        print('CE sparse slots: groups {} allocate {} slots and use {}; slots {} '
              'stay empty.  With IF_LABELSMOOTH off those columns are pure '
              'negatives -- never a target, only ever pushed away from.'
              .format(groups, max(groups) + 1, sorted(set(groups)), missing))
    return groups


def ce_slot_count(cfg):
    """How many classes each identity is split into for the CROSS ENTROPY.

    One definition, imported by both make_loss (which sizes label smoothing)
    and make_model (which sizes the classifier).  Two copies of this expression
    would be two chances for the head and the loss to disagree about the label
    space, and that disagreement is an index error at best and a silently
    wrong target at worst.
    """
    slots = 1
    groups = ce_modality_groups(cfg)               # validates on every build
    if cfg.MODEL.CE_SPLIT_MODALITY:
        slots *= max(groups) + 1
    elif list(cfg.MODEL.CE_MODALITY_GROUPS):
        # The lesson from mtext, where TEXT_MODALITY was set and then ignored:
        # a key that has no effect still appears in the config dump, so the run
        # looks like the one that was intended in every line of the log.
        raise ValueError(
            'MODEL.CE_MODALITY_GROUPS is set but MODEL.CE_SPLIT_MODALITY is '
            'False -- the grouping would be silently ignored')
    if cfg.MODEL.CE_SPLIT_VIEW:
        slots *= 2
    return slots


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss()
        print("using soft triplet loss for training")
    elif 'wrt' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss_WRT()
        print("using WRT loss for training")
    else:
        raise ValueError('expected METRIC_LOSS_TYPE should be triplet or wrt but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        # The CE label space, not the identity count: with a split they differ.
        xent = CrossEntropyLabelSmooth(num_classes=num_classes * ce_slot_count(cfg))
        print("label smooth on, numclasses:", num_classes * ce_slot_count(cfg))

    sampler_name = sampler.upper()
    if sampler_name not in ('PKM', 'PKM_VIEW'):
        raise ValueError('expected PKM or PKM_VIEW sampler, but got {}'.format(sampler))

    def loss_func(score, feat, target, target_ce=None):
        # `target_ce` is the split label; the triplet always keeps the raw pid,
        # so it goes on pulling a person's spectra and viewpoints together while
        # the classifier stops being asked to.  None reproduces the old
        # behaviour exactly, which is what keeps every recorded run valid.
        ce_target = target if target_ce is None else target_ce
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            ID_LOSS = xent(score, ce_target)
        else:
            ID_LOSS = F.cross_entropy(score, ce_target)

        TRI_LOSS = triplet(feat, target)[0]
        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS, ID_LOSS, TRI_LOSS

    return loss_func, center_criterion
