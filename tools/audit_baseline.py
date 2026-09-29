"""A 配置、真实数据与模型/loss 构建预检；不是正式训练。"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrain', required=True)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/A_baseline_candidate.yml'))
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.DATASETS.ROOT_DIR = args.data_root
    cfg.MODEL.PRETRAIN_PATH = args.pretrain
    cfg.freeze()
    torch.manual_seed(cfg.SOLVER.SEED)
    loader, _, _, _, classes, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, classes, cameras, views)
    _, center = make_loss(cfg, classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    create_scheduler(cfg, optimizer, len(loader))
    print('MODEL_BUILD_PREFLIGHT_OK: model, loss, optimizer, scheduler and data constructed; trajectory={}'.format(cfg.MODEL.TRAJECTORY.ENABLED))
    print('尚未开始正式训练；启动后需确认首个正常 iteration。')

if __name__ == '__main__':
    main()
