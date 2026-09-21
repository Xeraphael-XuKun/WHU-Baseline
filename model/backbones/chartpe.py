"""ChartPE: identity-consistent coordinate-chart positional encoding.

Two building blocks live here:

* ``TopologyPreservingChart`` (TPCG) -- predicts, for the current image, a
  continuous / monotonic / fold-free 2D coordinate for every patch.  It never
  moves pixels or features; it only produces coordinates.
* ``ChartRotaryEmbedding`` -- writes those coordinates into attention by
  rotating Q and K (mixed 2D RoPE).  V is left untouched.

Frequencies follow the official RoPE-ViT "mixed" setting (naver-ai/rope-vit).
Coordinates are expressed in *patch-grid units* so that a zero-initialised
TPCG reproduces exactly the centered integer grid that fixed RoPE uses -- the
two ablation arms therefore start from an identical state and differ only in
whether the coordinates are allowed to adapt.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_2d_freqs(head_dim, num_heads, theta=10.0, rotate=True):
    """Mixed 2D RoPE frequencies, ported verbatim from naver-ai/rope-vit.

    Returns a tensor of shape ``[2, num_heads, head_dim // 2]`` where index 0
    along the first axis holds the x-components and index 1 the y-components.
    Each head gets its own random base orientation, so different heads look at
    the plane from different directions.
    """
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * math.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(math.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(math.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    return torch.stack([freqs_x, freqs_y], dim=0)


def rotate_half(x):
    """Treat adjacent channel pairs as 2D vectors and rotate each by 90 deg.

    ``(a, b) -> (-b, a)``.  Note this is the *adjacent-pair* convention (which
    matches RoPE-ViT's ``reshape(..., -1, 2)``), not the half-split convention
    used by some LLM implementations.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_centered_grid(num_y, num_x, device=None, dtype=torch.float32,
                        zero_based=False):
    """Fixed patch grid in grid units.

    Returns ``[num_y * num_x, 2]`` with ``(x, y)`` in the last dimension and
    row-major ordering (y outer, x inner) so that it matches the token order
    produced by ``PatchEmbed`` (``flatten(2).transpose(1, 2)``).

    For a 16x8 grid this yields x in {-3.5 .. 3.5} and y in {-7.5 .. 7.5},
    spacing exactly 1 -- i.e. RoPE-ViT's integer grid up to a constant shift.
    The shift is immaterial for patch-patch interactions (only phase
    *differences* reach attention) but not for CLS, whose phase is pinned at 0;
    ``zero_based`` reproduces RoPE-ViT's own ``t_x = (t % end_x)`` exactly, which
    is what its pretrained weights were trained against.
    """
    xs = torch.arange(num_x, device=device, dtype=dtype)
    ys = torch.arange(num_y, device=device, dtype=dtype)
    if not zero_based:
        xs = xs - (num_x - 1) / 2.0
        ys = ys - (num_y - 1) / 2.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


class ChartRotaryEmbedding(nn.Module):
    """Rotate Q/K by a per-token phase derived from chart coordinates.

    ``phase = omega_x * q_x + omega_y * q_y`` per (head, channel-pair); the
    pair is then rotated by that phase.  Because a rotation only survives the
    Q-K inner product as a *difference* of phases, attention sees relative
    geometry rather than absolute position.
    """

    def __init__(self, head_dim, num_heads, theta=10.0, trainable=False, gate=False):
        super().__init__()
        assert head_dim % 2 == 0, 'head_dim must be even for rotary embedding'
        freqs = init_2d_freqs(head_dim, num_heads, theta=theta, rotate=True)
        # [2, heads, head_dim//2] -> [heads, head_dim//2, 2] for the einsum below.
        self.freqs = nn.Parameter(freqs.permute(1, 2, 0).contiguous(), requires_grad=trainable)
        # Zero-initialised gain on the phase -- the rotary counterpart of
        # `pos_delta`.  At alpha = 0 the phase is identically zero, so cos = 1
        # and sin = 0 and BOTH q and k come out bit-exactly unchanged: the whole
        # model is then the CLIP baseline, and every later difference is
        # attributable to the rotation alone.  Scaling the *angle* (rather than
        # blending the rotated vector back in) keeps the operator a rotation, so
        # |q| never depends on where the patch sits; a side branch
        # `q + g * rope(q)` would give position a first-order grip on the norm
        # of q and therefore on the attention logits, which is not what rotary
        # position encoding is for.
        #
        # One scalar per layer, because each Attention owns its own instance.
        self.alpha = nn.Parameter(torch.zeros(1)) if gate else None

    def forward(self, q, k, chart):
        """q, k: ``[B, heads, N, head_dim]``.  chart: ``[B, N, 2]`` as (x, y).

        The chart is expected to already contain the CLS row (all zeros), so
        no slicing is needed here: a zero coordinate gives cos=1 / sin=0 and
        leaves the CLS token bit-exactly unchanged.
        """
        input_dtype = q.dtype
        # einsum is on autocast's fp16 list, so .float() alone is not enough --
        # the whole phase computation must run with autocast disabled.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            phase = torch.einsum('bnd,hcd->bhnc', chart.float(), self.freqs.float())
            if self.alpha is not None:
                phase = phase * self.alpha.float()
            phase = torch.repeat_interleave(phase, 2, dim=-1)
            cos, sin = phase.cos(), phase.sin()
            q_f, k_f = q.float(), k.float()
            q_out = q_f * cos + rotate_half(q_f) * sin
            k_out = k_f * cos + rotate_half(k_f) * sin
        return q_out.to(input_dtype), k_out.to(input_dtype)


class TopologyPreservingChart(nn.Module):
    """TPCG -- generate a continuous, monotonic, fold-free per-image chart.

    Degrees of freedom are deliberately tiny: ``H + W + 2`` (one positive
    spacing per row, one per column, plus a global restricted rotation and an
    area-preserving anisotropic scale).  Positive spacings make row/column
    order impossible to invert by construction, and det(A) = 1 forbids the
    whole chart from collapsing or exploding.

    ``grid_h`` / ``grid_w`` are *training-time* constants used to convert the
    normalised axes into patch-grid units.  They are intentionally not
    recomputed from the runtime shape: at a higher test resolution the chart's
    span stays the same (person-normalised) while sampling gets denser.
    """

    def __init__(self, dim, hidden=64, grid_h=16, grid_w=8,
                 theta_max=math.pi / 6, s_max=0.35, eps=1e-4, input_norm=False,
                 zero_based=False):
        super().__init__()
        hidden = hidden or max(dim // 4, 32)
        # Standardise every (image, channel) over the patch grid before the chart
        # head sees it.  The chart is meant to describe geometry, but the raw
        # patch features are dominated by which sensor took the picture: measured
        # at TPCG's input, the modality signal of the Thermal pairs is 1.11 in
        # units of the within-modality jitter, against 0.11 for the RGB<->IR pair
        # that has no modality gap.  Instance norm removes the per-image,
        # per-channel level -- which is the modality -- and keeps the spatial
        # relations -- which is the content; it cuts that 1.11 to 0.20 while
        # leaving the RGB<->IR control at 0.11.  No affine parameters: a shared
        # scale/shift cannot carry per-image modality information anyway, and
        # this keeps the module identical to what was measured.
        self.input_norm = input_norm
        self.local = nn.Sequential(
            # Depthwise 3x3 -- cheap, lets each patch see its local neighbourhood.
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            # 1x1 bottleneck -- kept small on purpose to limit identity leakage.
            nn.Conv2d(dim, hidden, 1),
        )
        self.row_head = nn.Linear(hidden, 1)     # one positive spacing per row
        self.col_head = nn.Linear(hidden, 1)     # one positive spacing per column
        self.global_head = nn.Linear(hidden, 2)  # theta and s for the whole image

        self.grid_h = grid_h
        self.grid_w = grid_w
        self.zero_based = zero_based
        self.theta_max = theta_max
        self.s_max = s_max
        self.eps = eps

        # Identity initialisation: zero logits -> equal spacings, theta = s = 0,
        # so the very first forward reproduces the regular grid exactly and the
        # pretrained backbone is not disturbed.  (Must stay zero: TransReID's
        # trailing self.apply(_init_weights) would clobber these, which is why
        # this module is constructed after that call.)
        for head in (self.row_head, self.col_head, self.global_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def positive_axis(self, logits):
        """logits ``[B, L]`` -> (axis ``[B, L]`` in [-1, 1], delta ``[B, L]`` > 0).

        softplus makes every spacing positive, the cumulative sum turns
        spacings into monotone tick positions, and the normalisation removes
        any overall scale (only the *ratio* of spacings can be learned).
        """
        delta = F.softplus(logits) + self.eps
        center = torch.cumsum(delta, dim=1) - 0.5 * delta
        axis = 2.0 * center / delta.sum(dim=1, keepdim=True) - 1.0
        return axis, delta

    def forward(self, x):
        """x: ``[B, H, W, D]`` patch map (no CLS).  Returns (q, stats)."""
        x = x.permute(0, 3, 1, 2)                # [B, D, H, W]
        if self.input_norm:
            x = F.instance_norm(x)
        f = self.local(x)                        # [B, C, H, W]
        row = f.mean(dim=3).transpose(1, 2)      # [B, H, C]  -- pool over columns
        col = f.mean(dim=2).transpose(1, 2)      # [B, W, C]  -- pool over rows

        v, dy = self.positive_axis(self.row_head(row).squeeze(-1))   # [B, H]
        u, dx = self.positive_axis(self.col_head(col).squeeze(-1))   # [B, W]

        # Grid-unit calibration.  This happens BEFORE A is applied: rotation and
        # anisotropic scale do not commute, and applying A in grid units is what
        # keeps the identity-init chart equal to build_centered_grid().
        u = u * (self.grid_w / 2.0)
        v = v * (self.grid_h / 2.0)

        yy = v[:, :, None].expand(-1, -1, u.size(1))
        xx = u[:, None, :].expand(-1, v.size(1), -1)
        base = torch.stack([xx, yy], dim=-1)     # [B, H, W, 2] as (x, y)

        raw = self.global_head(f.mean(dim=(2, 3)))                   # [B, 2]
        theta = self.theta_max * torch.tanh(raw[:, 0])
        s = self.s_max * torch.tanh(raw[:, 1])
        ct, st = theta.cos(), theta.sin()
        ex, ey = s.exp(), (-s).exp()             # ex * ey == 1  ->  det(A) == 1
        A = torch.stack([ct * ex, -st * ey, st * ex, ct * ey], dim=-1).view(-1, 2, 2)

        q = torch.einsum('bij,bhwj->bhwi', A, base)
        if self.zero_based:
            # After A, never before: the rotation has to be about the centre of
            # the image, so the shift to a 0-based origin comes last.  Identity
            # init then lands on {0 .. W-1} x {0 .. H-1}, matching RoPE-ViT.
            q = q + q.new_tensor([(self.grid_w - 1) / 2.0, (self.grid_h - 1) / 2.0])
        stats = {
            'dx': dx.detach(), 'dy': dy.detach(),
            'theta': theta.detach(), 'scale': s.detach(), 'A': A.detach(),
        }
        return q, stats
