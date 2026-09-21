from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from processor import do_inference

import argparse
import os

from config import cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WHU-MARS baseline evaluation')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.environ.get('CUDA_VISIBLE_DEVICES') and cfg.MODEL.DEVICE_ID:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    (_, _, val_loaders, num_querys, num_classes,
     camera_num, view_num) = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes,
                       camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    do_inference(cfg, model, val_loaders, num_querys)
