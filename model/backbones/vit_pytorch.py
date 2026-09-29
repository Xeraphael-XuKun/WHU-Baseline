""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from .trajectory import TokenTrajectory



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class QuickGELU(nn.Module):
    """CLIP's activation.  Not interchangeable with nn.GELU.

    OpenAI trained every CLIP ViT with ``x * sigmoid(1.702 * x)``.  It tracks
    the true GELU closely enough that a model built with nn.GELU still runs and
    still trains -- it just quietly starts from weights that were fitted to a
    different function.  That is the kind of bug that costs a mAP point and
    never announces itself, so the CLIP factory pins this explicitly.
    """

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        a = self.attn(self.norm1(x))
        m_in = x + self.drop_path(a)
        m = self.mlp(self.norm2(m_in))
        return m_in + self.drop_path(m)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x

class TransReID(nn.Module):
    """A baseline CLIP image tower; optional methods belong in separate modules."""
    def __init__(self, img_size=(256, 128), patch_size=16, stride_size=16,
                 in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, hw_ratio=1, act_layer=QuickGELU,
                 trajectory_enabled=False, acceleration_mix=1.0, gain_bound=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      stride_size=stride_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.hw_ratio = hw_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.ln_pre = norm_layer(embed_dim)
        # Retained for existing A checkpoints and future optional VTC.
        self.clip_proj = nn.Parameter(torch.zeros(embed_dim, 512))
        self.norm = norm_layer(embed_dim)
        # Retain this unused head: its initialization consumes A's CPU RNG.
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        # Create after the baseline initialization; zeros consume no random draws.
        self.trajectory = (TokenTrajectory(depth, num_patches + 1, embed_dim,
                                          acceleration_mix, gain_bound) if trajectory_enabled else None)
        self.collect_trajectory_stats = False

    def forward_features(self, x, trajectory_gate=1.0):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.ln_pre(x)
        if self.trajectory is None or trajectory_gate == 0:
            for blk in self.blocks:
                x = blk(x)
        else:
            x = self.trajectory(x, self.blocks, trajectory_gate,
                                self.collect_trajectory_stats)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, trajectory_gate=1.0):
        return self.forward_features(x, trajectory_gate)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_clip_visual(self, param_dict):
        """Load OpenAI CLIP's image tower into this ViT.

        Verified against the real ViT-B-16.pt on the cluster (305 tensors, 152
        of them under `visual.`, 12 resblocks).  Every name below was read off
        that file, not inferred.

        Three things are worth stating because they are invisible at load time:

        * `in_proj_weight` is [3*dim, dim] stacked q, k, v -- exactly the layout
          our fused `qkv` Linear expects, so it is a straight copy.
        * CLIP's patch conv has no bias.  Ours does; zeroing it makes the two
          numerically identical while leaving the parameter free to move.
        * the position table is 197 rows for a 14x14 grid, ours is 129 for
          16x8, so it goes through the same interpolation as every other
          checkpoint (resize_pos_embed, below).
        """
        if self.ln_pre is None:
            raise RuntimeError(
                'this is a CLIP checkpoint but the model was built without '
                'ln_pre / QuickGELU; use TRANSFORMER_TYPE vit_base_clip '
                '(loading CLIP weights into a plain ViT trains to garbage '
                'without ever raising)')

        pairs = [('visual.conv1.weight', 'patch_embed.proj.weight'),
                 ('visual.ln_pre.weight', 'ln_pre.weight'),
                 ('visual.ln_pre.bias', 'ln_pre.bias'),
                 ('visual.ln_post.weight', 'norm.weight'),
                 ('visual.ln_post.bias', 'norm.bias'),
                 ('visual.proj', 'clip_proj')]
        for n in range(len(self.blocks)):
            src, dst = 'visual.transformer.resblocks.{}.'.format(n), 'blocks.{}.'.format(n)
            pairs += [(src + 'ln_1.weight', dst + 'norm1.weight'),
                      (src + 'ln_1.bias', dst + 'norm1.bias'),
                      (src + 'attn.in_proj_weight', dst + 'attn.qkv.weight'),
                      (src + 'attn.in_proj_bias', dst + 'attn.qkv.bias'),
                      (src + 'attn.out_proj.weight', dst + 'attn.proj.weight'),
                      (src + 'attn.out_proj.bias', dst + 'attn.proj.bias'),
                      (src + 'ln_2.weight', dst + 'norm2.weight'),
                      (src + 'ln_2.bias', dst + 'norm2.bias'),
                      (src + 'mlp.c_fc.weight', dst + 'mlp.fc1.weight'),
                      (src + 'mlp.c_fc.bias', dst + 'mlp.fc1.bias'),
                      (src + 'mlp.c_proj.weight', dst + 'mlp.fc2.weight'),
                      (src + 'mlp.c_proj.bias', dst + 'mlp.fc2.bias')]

        own = self.state_dict()
        count = 0
        for src, dst in pairs:
            if src not in param_dict:
                raise RuntimeError('CLIP checkpoint is missing {}'.format(src))
            v = param_dict[src]
            if own[dst].shape != v.shape:
                raise RuntimeError('shape mismatch {} {} -> {} {}'.format(
                    src, tuple(v.shape), dst, tuple(own[dst].shape)))
            own[dst].copy_(v)
            count += 1

        own['cls_token'].copy_(param_dict['visual.class_embedding'].reshape(1, 1, -1))
        own['patch_embed.proj.bias'].zero_()
        count += 2

        if self.pos_embed is not None:
            pe = param_dict['visual.positional_embedding'].unsqueeze(0)   # [1, 197, D]
            if pe.shape != self.pos_embed.shape:
                pe = resize_pos_embed(pe, self.pos_embed, self.patch_embed.num_y,
                                      self.patch_embed.num_x, self.hw_ratio)
            own['pos_embed'].copy_(pe)
            count += 1

        filled = {dst for _, dst in pairs}
        filled.update(('cls_token', 'pos_embed', 'patch_embed.proj.bias'))
        skipped = sorted(k for k in own if k not in filled)
        print('Loaded %d CLIP visual tensors; %d model tensors left at init: %s'
              % (count, len(skipped), skipped if len(skipped) <= 8 else skipped[:8] + ['...']))
        return count

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(param_dict, nn.Module):
            param_dict = param_dict.state_dict()
        if 'visual.conv1.weight' not in param_dict:
            raise ValueError('A baseline requires the original CLIP ViT-B/16 weights')
        self._load_clip_visual(param_dict)


def resize_pos_embed(posemb, posemb_new, hight, width, hw_ratio=1):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old_h = int(math.sqrt(len(posemb_grid)*hw_ratio))
    gs_old_w = gs_old_h // hw_ratio
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

def vit_base_clip(img_size=(256, 128), stride_size=16, drop_rate=0.0,
                  attn_drop_rate=0.0, drop_path_rate=0.1,
                  trajectory_enabled=False, acceleration_mix=1.0, gain_bound=0.1):
    return TransReID(img_size=img_size, patch_size=16, stride_size=stride_size,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                    qkv_bias=True, drop_path_rate=drop_path_rate,
                    drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-5),
                    hw_ratio=1, act_layer=QuickGELU,
                    trajectory_enabled=trajectory_enabled, acceleration_mix=acceleration_mix,
                    gain_bound=gain_bound)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
