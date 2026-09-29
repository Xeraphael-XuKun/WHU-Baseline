"""Mechanism gates: formula, actual block input, zero-init, RNG and gradients."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.backbones.trajectory import TokenTrajectory
from model.backbones.vit_pytorch import TransReID
from config import cfg
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from model.make_model import build_transformer
from unittest.mock import patch


def main():
    torch.set_num_threads(4)
    t = TokenTrajectory(3, 5, 8, gain_bound=0)
    t.gain.data.fill_(0.2)
    v, old = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    assert t.correction(0, None, None) is None
    torch.testing.assert_close(t.correction(1, v, None), 0.2 * F.layer_norm(v, (8,)))
    torch.testing.assert_close(t.correction(2, v, old), 0.2 * F.layer_norm(2*v-old, (8,)))
    t.acceleration_mix = 0
    torch.testing.assert_close(t.correction(2, v, old), t.correction(1, v, None))
    t.acceleration_mix = 1
    # An independent three-block recurrence detects using the uncorrected input.
    blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
    x = torch.randn(2, 5, 8)
    h0 = blocks[0](x); v0 = h0-x
    in1 = h0 + 0.2 * F.layer_norm(v0, (8,))
    h1 = blocks[1](in1); v1 = h1-in1
    in2 = h1 + 0.2 * F.layer_norm(2*v1-v0, (8,))
    torch.testing.assert_close(t(x, blocks), blocks[2](in2))
    torch.testing.assert_close(t(x, blocks), t(x, blocks))
    bounded = TokenTrajectory(3,5,8,gain_bound=0.1)
    bounded.effective_gain().sum().backward()
    assert torch.equal(bounded.gain.grad,torch.ones_like(bounded.gain))
    bounded.gain.data.fill_(10)
    assert bounded.effective_gain().abs().max() <= 0.1
    args = dict(img_size=(32,32), patch_size=16, stride_size=16, embed_dim=32,
                depth=3, num_heads=4, drop_path_rate=0.1)
    torch.manual_seed(17); base = TransReID(**args); rng = torch.get_rng_state()
    torch.manual_seed(17); traj = TransReID(**args, trajectory_enabled=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, traj.state_dict()[k]) for k,v in base.state_dict().items())
    x = torch.randn(2,3,32,32)
    for m in (base, traj): m.train()
    torch.manual_seed(19); a = base(x); a.square().mean().backward()
    torch.manual_seed(19); b = traj(x); b.square().mean().backward()
    assert torch.equal(a,b)
    for (name,p) in base.named_parameters():
        q = dict(traj.named_parameters())[name]
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert torch.isfinite(traj.trajectory.gain.grad).all()
    assert traj.trajectory.gain.grad.abs().sum() > 0
    traj.trajectory.gain.data.fill_(0.03)
    base.eval(); traj.eval()
    assert torch.equal(base(x),traj(x,trajectory_gate=0))
    assert not torch.equal(traj(x),base(x))
    # Nonzero gains must propagate through velocity into earlier blocks.
    probe = TokenTrajectory(3,5,8); probe.gain.data.fill_(0.1)
    v.requires_grad_(); old.requires_grad_()
    probe.correction(2,v,old).square().sum().backward()
    assert v.grad.abs().sum() > 0 and old.grad.abs().sum() > 0
    cfg.merge_from_file(str(Path(__file__).resolve().parents[1]/'configs/A_trajectory.yml'))
    holder = nn.Module(); holder.base = traj
    opt, _ = make_optimizer(cfg,holder,nn.Linear(1,1))
    group = next(g for g in opt.param_groups if g['params'][0] is traj.trajectory.gain)
    assert group['lr'] == 0.00035 and group['weight_decay'] == 0.0001
    sched = create_scheduler(cfg,opt,469)
    assert abs(group['lr']-0.0000035) < 1e-12
    sched.step_update(100)
    backbone_group = next(g for g in opt.param_groups if g['params'][0] is traj.cls_token)
    assert 0.00034 < group['lr'] <= 0.00035
    assert abs(group['lr']/backbone_group['lr']-70) < 1e-10
    # Reject an evaluation config that silently disables or changes the method.
    wrapper = build_transformer.__new__(build_transformer)
    nn.Module.__init__(wrapper); wrapper.base = traj
    valid = wrapper.state_dict()
    for bad in ({k:v for k,v in valid.items() if 'trajectory.' not in k},
                dict(valid, **{'base.trajectory.spec': torch.tensor([1.,0.],dtype=torch.float64)})):
        with patch('torch.load', return_value=bad):
            try: wrapper.load_param('synthetic-checkpoint')
            except RuntimeError: pass
            else: raise AssertionError('Accepted incompatible checkpoint')
    print('TRAJECTORY_MECHANISM_OK: formula, recurrence, zero-init, RNG, gradients, gate, optimizer, scheduler')


if __name__ == '__main__':
    main()
