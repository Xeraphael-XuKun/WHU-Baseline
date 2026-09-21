from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train

import argparse
import os
import random

import numpy as np
import torch

from config import cfg


def setup_cuda_visible_devices(config):
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return
    if config.MODEL.DEVICE_ID:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.MODEL.DEVICE_ID


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WHU-MARS baseline training')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank',
                        default=-1, type=int)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setup_cuda_visible_devices(cfg)
    local_rank = int(os.environ.get('LOCAL_RANK',
                                    args.local_rank if args.local_rank >= 0 else 0))
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=True)
    logger.info('Saving model in the path :{}'.format(cfg.OUTPUT_DIR))
    logger.info(args)
    if args.config_file:
        logger.info('Loaded configuration file {}'.format(args.config_file))
    if not cfg.MODEL.DIST_TRAIN or local_rank == 0:
        logger.info('Running with config:\n{}'.format(cfg))

    (train_loader, _, val_loaders, num_querys, num_classes,
     camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(
        cfg, optimizer, iters_per_epoch=len(train_loader))

    do_train(
        cfg, model, center_criterion, train_loader, val_loaders,
        optimizer, optimizer_center, scheduler, loss_func,
        num_querys, local_rank)
