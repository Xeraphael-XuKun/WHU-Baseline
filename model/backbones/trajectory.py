"""Trajectory-v1: channel-normalized cross-block velocity extrapolation."""
import torch
from torch import nn
from torch.nn import functional as F


class TokenTrajectory(nn.Module):
    def __init__(self, depth, num_tokens, embed_dim, acceleration_mix=1.0, gain_bound=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.acceleration_mix = acceleration_mix
        self.gain_bound = gain_bound
        if gain_bound < 0:
            raise ValueError('gain_bound must be nonnegative; 0 means direct gain')
        # Persist method semantics, so evaluation cannot silently change beta/bound.
        self.register_buffer('spec', torch.tensor([acceleration_mix, gain_bound], dtype=torch.float64))
        self.gain = nn.Parameter(torch.zeros(depth - 1, 1, num_tokens, embed_dim))

    def effective_gain(self, layer=None):
        gain = self.gain if layer is None else self.gain[layer]
        if self.gain_bound == 0:
            return gain
        return self.gain_bound * torch.tanh(gain / self.gain_bound)

    def correction(self, layer, prev_v, prev_prev_v, gate=1.0):
        if layer == 0 or prev_v is None:
            return None
        # Differences and normalization in FP32; preserve the gradient history.
        direction = prev_v.float()
        if prev_prev_v is not None:
            direction = direction + self.acceleration_mix * (direction - prev_prev_v.float())
        direction = F.layer_norm(direction, (self.embed_dim,)).to(prev_v.dtype)
        return gate * self.effective_gain(layer - 1) * direction

    def forward(self, x, blocks, gate=1.0, collect_stats=False):
        prev_v = prev_prev_v = None
        stats = []
        for layer, block in enumerate(blocks):
            correction = self.correction(layer, prev_v, prev_prev_v, gate)
            if collect_stats and correction is not None:
                with torch.no_grad():
                    c = correction.detach().float().norm(dim=-1).mean()
                    h = x.detach().float().norm(dim=-1).mean()
                    v = prev_v.detach().float().norm(dim=-1).mean()
                    a = (prev_v - prev_prev_v).detach().float().norm(dim=-1).mean() if prev_prev_v is not None else v.new_zeros(())
                    stats.append(torch.stack((v, a, c, c / h.clamp_min(1e-12))))
            if correction is not None:
                x = x + correction
            block_input = x
            x = block(x)
            prev_prev_v, prev_v = prev_v, x - block_input
        # Diagnostics retain no graph; histories themselves are local to this call.
        self.last_stats = torch.stack(stats).detach() if stats else None
        return x
