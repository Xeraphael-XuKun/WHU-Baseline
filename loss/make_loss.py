import torch.nn.functional as F
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):
    if cfg.MODEL.METRIC_LOSS_TYPE != 'triplet' or cfg.MODEL.IF_LABELSMOOTH != 'off':
        raise ValueError('A baseline uses soft-margin Triplet and unsmoothed CE')
    if cfg.SOLVER.LOSS_TYPE != 'base':
        raise ValueError('Only the A baseline loss is implemented on this branch')
    # Compatibility only: retain historical CUDA RNG consumption. Never added
    # to loss and never stepped; remove only with a newly trained matched A.
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=2048, use_gpu=True)
    triplet = TripletLoss()

    def loss_func(score, feat, target):
        id_loss = F.cross_entropy(score, target)
        tri_loss = triplet(feat, target)[0]
        return (cfg.MODEL.ID_LOSS_WEIGHT * id_loss +
                cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss, id_loss, tri_loss)

    return loss_func, center_criterion
