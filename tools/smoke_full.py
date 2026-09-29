"""真实 CLIP、合成小 batch 的 full 训练/保存/复评检查；不等于正式训练。"""
import argparse
import gc
import json
import logging
from pathlib import Path
import sys
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
    p.add_argument('--gain-bound', type=float, default=0.0)
    p.add_argument('--amp-init-scale', type=float, default=65536.0)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'smoke_checkpoint.pth'
    if checkpoint.exists():
        raise RuntimeError('请使用新的 smoke 输出目录')
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1]/'configs/A_trajectory_full.yml'))
    cfg.MODEL.PRETRAIN_PATH = args.pretrain
    cfg.MODEL.FULL.TEXT_CLIP_PATH = args.pretrain
    cfg.MODEL.TRAJECTORY.GAIN_BOUND = args.gain_bound
    cfg.SOLVER.MAX_EPOCHS = 1
    cfg.SOLVER.LOG_PERIOD = 1
    cfg.OUTPUT_DIR = str(output)
    model = make_model(cfg, 500, 7, 3).cuda()
    loss_fn, center = make_loss(cfg, 500)
    optimizer, optimizer_center = make_optimizer(cfg, model, center)
    scheduler = create_scheduler(cfg, optimizer, 469)
    initial_prompt = model.prompts.shared_ctx.detach().clone()
    initial_projection = model.base.clip_proj.detach().clone()
    images = [torch.randn(4,3,256,128) for _ in range(3)]
    cams = [torch.tensor([5,0,6,1]) for _ in range(3)]
    class Batches:
        batch_size = 4
        def __len__(self): return 16
        def __iter__(self):
            return iter([(images,torch.tensor([0,0,1,1]),cams)]*16)
    original_scaler = torch.cuda.amp.GradScaler
    torch.cuda.amp.GradScaler = lambda: original_scaler(init_scale=args.amp_init_scale)
    torch.cuda.reset_peak_memory_stats()
    try:
        do_train(cfg,model,center,Batches(),[],optimizer,optimizer_center,scheduler,loss_fn,[],0)
    finally:
        torch.cuda.amp.GradScaler = original_scaler
    gain = model.base.trajectory.gain
    steps = int(optimizer.state[gain]['step'].item())
    assert steps >= 2, '需要至少两次有效更新'
    for param in [gain,model.prompts.view_ctx,model.prompts.shared_ctx,model.base.clip_proj]:
        assert param.grad is not None and torch.isfinite(param.grad).all()
    assert gain.count_nonzero() > 0
    assert not torch.equal(initial_prompt,model.prompts.shared_ctx)
    assert not torch.equal(initial_projection,model.base.clip_proj)
    assert all(not p.requires_grad and p.grad is None for p in model.prompts.text.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model.eval(); model.base.collect_trajectory_stats = False
    probe = images[0][:2].cuda()
    expected = {}
    with torch.no_grad():
        for neck in ['after','before']:
            model.neck_feat = neck
            expected[neck] = model(probe,mode=1).cpu()
    torch.save(model.state_dict(), checkpoint)
    memory = torch.cuda.max_memory_allocated()/1024**2
    del optimizer,optimizer_center,scheduler,model,center,loss_fn,gain
    gc.collect();torch.cuda.empty_cache()
    # Fresh model and on-disk load, not a reload into the same instance.
    restored = make_model(cfg,500,7,3)
    restored.load_param(str(checkpoint))
    restored.cuda().eval()
    with torch.no_grad():
        for neck in ['after','before']:
            restored.neck_feat = neck
            assert torch.equal(expected[neck],restored(probe,mode=1).cpu())
    result=dict(status='FULL_SMOKE_OK',gain_bound=args.gain_bound,
                amp_init_scale=args.amp_init_scale,synthetic_images=12,batches=16,
                effective_updates=steps,peak_allocated_MiB=memory,
                prompt_updated=True,projection_updated=True,text_frozen=True,
                checkpoint_pre_post_exact=True)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
