"""完整 CLIP 的合成小 batch 检查；不代表真实数据训练或正式效果。"""
import argparse
import gc
import json
import logging
from pathlib import Path
import random
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import cfg
from model import make_model
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from processor import do_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrain', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--amp-init-scale', type=float, default=65536.0)
    p.add_argument('--gain-bound', type=float, default=0.1)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'smoke_checkpoint.pth').exists():
        raise RuntimeError('请使用新的 smoke 输出目录')
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1]/'configs/A_trajectory.yml'))
    cfg.MODEL.PRETRAIN_PATH = args.pretrain
    cfg.MODEL.TRAJECTORY.GAIN_BOUND = args.gain_bound
    def seed():
        random.seed(1234); np.random.seed(1234)
        torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
    g = torch.Generator().manual_seed(731)
    images = [torch.randn(4,3,256,128,generator=g) for _ in range(3)]
    probe = images[0][:2].cuda()
    cfg.MODEL.TRAJECTORY.ENABLED = False
    seed(); baseline = make_model(cfg,500,5,3)
    baseline_init = {k:v.clone() for k,v in baseline.state_dict().items()}
    rng = torch.get_rng_state()
    baseline.cuda().eval()
    with torch.no_grad(): expected = baseline(probe,mode=1).cpu()
    del baseline; gc.collect(); torch.cuda.empty_cache()
    cfg.MODEL.TRAJECTORY.ENABLED = True
    seed(); model = make_model(cfg,500,5,3)
    assert torch.equal(rng,torch.get_rng_state()), 'RNG changed'
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline_init.items())
    del baseline_init
    gain = model.base.trajectory.gain
    assert tuple(gain.shape) == (11,1,129,768) and gain.count_nonzero()==0
    model.cuda().eval()
    with torch.no_grad():
        assert torch.equal(expected,model(probe,mode=1).cpu())
        assert torch.equal(expected,model(probe,mode=1,trajectory_gate=0).cpu())
    loss_fn, center = make_loss(cfg,500)
    optimizer, optimizer_center = make_optimizer(cfg,model,center)
    scheduler = create_scheduler(cfg,optimizer,469)
    original_scaler = torch.cuda.amp.GradScaler
    torch.cuda.amp.GradScaler = lambda: original_scaler(init_scale=args.amp_init_scale)
    class TwoBatches:
        batch_size=4
        def __len__(self): return 2
        def __iter__(self):
            return iter([(images,torch.tensor([0,0,1,1]),
                          [torch.zeros(4,dtype=torch.long) for _ in range(3)])]*2)
    cfg.SOLVER.MAX_EPOCHS=1
    cfg.SOLVER.LOG_PERIOD=1
    cfg.OUTPUT_DIR=str(output)
    torch.cuda.reset_peak_memory_stats()
    try:
        do_train(cfg,model,center,TwoBatches(),[],optimizer,optimizer_center,
                 scheduler,loss_fn,[],0)
    finally:
        torch.cuda.amp.GradScaler=original_scaler
    assert gain.grad is not None and torch.isfinite(gain.grad).all()
    assert gain.count_nonzero()>0 and optimizer.state[gain]['step'].item()==2
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval(); model.base.collect_trajectory_stats=False
    with torch.no_grad():
        after=model(probe,mode=1).cpu()
        disabled=model(probe,mode=1,trajectory_gate=0).cpu()
    assert not torch.equal(after,disabled)
    checkpoint=output/'smoke_checkpoint.pth'
    torch.save(model.state_dict(),checkpoint)
    gain.data.zero_()
    model.load_param(str(checkpoint))
    with torch.no_grad(): assert torch.equal(after,model(probe,mode=1).cpu())
    result=dict(status='TRAJECTORY_SMOKE_OK',device=torch.cuda.get_device_name(0),
                torch=torch.__version__,amp_init_scale=args.amp_init_scale,gain_bound=args.gain_bound,
                synthetic_images=12,optimizer_steps=2,gain_parameters=gain.numel(),
                gain_abs_mean=gain.detach().abs().mean().item(),
                gate_feature_max_diff=(after-disabled).abs().max().item(),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                zero_init_exact=True,checkpoint_reload_exact=True)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
