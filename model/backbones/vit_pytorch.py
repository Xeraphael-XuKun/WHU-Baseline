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

from .chartpe import ChartRotaryEmbedding, TopologyPreservingChart, build_centered_grid


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

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rope=False, rope_theta=10.0, rope_freq_trainable=False,
                 rope_gate=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Each layer owns its rotary module so every layer draws its own head
        # orientations; the draws come from the global RNG seeded in train.py.
        self.rope = ChartRotaryEmbedding(
            head_dim, num_heads, theta=rope_theta, trainable=rope_freq_trainable,
            gate=rope_gate) if rope else None

    def forward(self, x, chart=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if chart is not None and self.rope is not None:
            # Position enters here and only here: Q/K get rotated, V is untouched.
            q, k = self.rope(q, k, chart)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rope=False, rope_theta=10.0, rope_freq_trainable=False,
                 rope_gate=False, layer_scale=False, layer_scale_init=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            rope=rope, rope_theta=rope_theta, rope_freq_trainable=rope_freq_trainable,
            rope_gate=rope_gate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # LayerScale (CaiT / DeiT-III).  Off by default so that every existing
        # checkpoint keeps an identical state_dict; on, it adds the two vectors
        # the *_LS pretrained weights carry and cannot be loaded without.
        if layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x, chart=None):
        a = self.attn(self.norm1(x), chart=chart)
        m_in = x + self.drop_path(a if self.gamma_1 is None else self.gamma_1 * a)
        m = self.mlp(self.norm2(m_in))
        return m_in + self.drop_path(m if self.gamma_2 is None else self.gamma_2 * m)


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

class PatchEmbed_ICS(nn.Module):
    """ Image to Patch Embedding with ICS
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size


        hidden_dim = 64
        stem_stride = 2
        stride_size = patch_size = patch_size[0] // stem_stride
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3,bias=False),
            IBN(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
            IBN(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        in_chans = hidden_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., ics_embedding=False, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0, hw_ratio=1,
                 pe_type='learnable', pe_keep_ape=False, chart_hidden=64, chart_theta_max_deg=30.0,
                 chart_scale_max=0.35, chart_input_norm=False, rope_theta=10.0,
                 rope_freq_trainable=False, rope_gate=False,
                 pe_zero_based=False, layer_scale=False,
                 layer_scale_init=1e-4, pe_layerwise='none', pe_freeze_base=False,
                 mod_delta_modalities=0, pe_per_modality=0, pe_layers=(),
                 clip_style=False, act_layer=nn.GELU):
        super().__init__()
        if pe_type not in ('learnable', 'rope_fixed', 'chartpe'):
            raise ValueError("pe_type must be 'learnable', 'rope_fixed' or 'chartpe', got {}".format(pe_type))
        if pe_layerwise not in ('none', 'indep', 'chain'):
            raise ValueError("pe_layerwise must be 'none', 'indep' or 'chain', got {}".format(pe_layerwise))
        if pe_type != 'learnable':
            # SIE adds absolute per-camera/view offsets to the token features; it has
            # no meaning once position enters through Q/K rotation instead.
            assert camera == 0 and view == 0, 'SIE embedding is not supported with rotary PE'
        self.pe_type = pe_type
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        if ics_embedding:
            self.patch_embed = PatchEmbed_ICS(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        # Training-time grid, frozen at build time: the chart span stays fixed
        # when the test resolution changes (only the sampling gets denser).
        self.grid_h = self.patch_embed.num_y
        self.grid_w = self.patch_embed.num_x

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Rotary modes drop the additive position table unless pe_keep_ape asks
        # to keep it (hybrid: pretrained absolute position + rotary relative).
        keep_ape = (pe_type == 'learnable') or pe_keep_ape
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if keep_ape else None
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        self.hw_ratio = hw_ratio
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,
                rope=(pe_type != 'learnable'), rope_theta=rope_theta, rope_freq_trainable=rope_freq_trainable,
                rope_gate=rope_gate,
                layer_scale=layer_scale, layer_scale_init=layer_scale_init)
            for i in range(depth)])

        # CLIP's image tower normalises once more between the position embedding
        # and the first block (`ln_pre`).  Nothing else we load has it, and
        # leaving it out silently feeds every CLIP weight the wrong input scale.
        self.clip_style = clip_style
        self.ln_pre = norm_layer(embed_dim) if clip_style else None
        # `visual.proj`: 768 -> 512, the map into CLIP's shared image/text space.
        # Unused by the ReID path, but it is the only route from our features to
        # a text prompt, so it is loaded now rather than in a second pass.
        self.clip_proj = nn.Parameter(torch.zeros(embed_dim, 512)) if clip_style else None

        # Layer-wise positional residuals.  The pretrained table stays the
        # anchor and every block gets its own increment on top of it; the
        # increments are the only new parameters, and they start at zero, so
        # step 0 reproduces the single-injection baseline bit for bit.  Nothing
        # here draws from the RNG, which keeps the 'none' path reproducible.
        self.pe_layerwise = pe_layerwise
        self.pe_per_modality = int(pe_per_modality)
        if self.pe_per_modality and mod_delta_modalities:
            # Both add a per-spectrum positional term.  Together they would be
            # two ways of expressing one thing, and the ablation could not say
            # which produced the number.
            raise ValueError(
                'MODEL.PE_PER_MODALITY and MODEL.MOD_DELTA both give each spectrum '
                'its own positional correction; enable one or the other')
        if self.pe_per_modality and pe_layerwise == 'none':
            raise ValueError('MODEL.PE_PER_MODALITY needs MODEL.PE_LAYERWISE '
                             'indep or chain -- there is no delta to split')
        if pe_layerwise != 'none':
            if self.pos_embed is None:
                raise ValueError(
                    'pe_layerwise needs a pos_embed table to build on (the whole '
                    'point is to inherit the pretrained one); set MODEL.PE_KEEP_APE '
                    'True if you want this alongside rotary PE')
            # WHERE the increments are injected.  Empty means every block, which
            # is what every run before this argument existed did; the table then
            # has `depth` rows and `pe_slot` is the identity, so that path is
            # bit-identical to the old code.
            #
            # With a subset, the table shrinks to one row per insertion point
            # rather than keeping `depth` rows and skipping some.  Keeping the
            # full table would leave rows that no forward ever reads -- dead
            # parameters that the optimizer still carries -- and, worse, would
            # break what `indep` means: the cancellation term is "whatever the
            # stream is already carrying", which is the PREVIOUS INJECTED row,
            # not the previous block.  Indexing the difference by block would
            # make the stream carry pos_embed + delta[l] - delta[l-1] + ... at
            # the skipped positions, which is neither mode.
            #
            # So with S = {s_0 < s_1 < ...}: block s_j is where increment j
            # lands, and every block from s_j up to s_{j+1} - 1 carries
            # pos_embed + delta[j].  One correction, held for a stretch of
            # blocks, instead of one per block.
            self.pe_layers = self._resolve_pe_layers(pe_layers, depth)
            rows = len(self.pe_layers)
            # block index -> row of the table, or -1 for "inject nothing here".
            # A buffer, not a plain list, so it moves with the model and shows
            # up in the state_dict for anyone reading a checkpoint later.
            slot = torch.full((depth,), -1, dtype=torch.long)
            for j, layer in enumerate(self.pe_layers):
                slot[layer] = j
            self.register_buffer('pe_slot', slot, persistent=False)
            if pe_per_modality:
                # Leading axis is the spectrum; the rest matches the shared
                # shape so everything downstream indexes the same way once the
                # row has been picked.
                self.pos_delta = nn.Parameter(torch.zeros(
                    pe_per_modality, rows, 1, num_patches + 1, embed_dim))
            else:
                self.pos_delta = nn.Parameter(
                    torch.zeros(rows, 1, num_patches + 1, embed_dim))
            if rows != depth:
                print('PE layers: increments at blocks {} of {} -- {} rows, '
                      '{:,} parameters against {:,} at every block.'.format(
                          self.pe_layers, depth, rows,
                          self.pos_delta.numel(),
                          self.pos_delta.numel() // rows * depth))
        else:
            self.pos_delta = None
            self.pe_layers = []
        # Twin-anchored modality residuals: one bank of layer-wise increments
        # per NON-reference modality.  Same actuator as pos_delta -- zero init,
        # added to the stream before every block -- but selected per image by
        # its spectrum instead of applied to all of them.
        #
        # The reference modality deliberately has no bank.  Its features are the
        # coordinate frame the others are corrected into, so "the reference
        # stays put" is a property of the parameterisation, not something a loss
        # has to defend.  That is the one structural fix for what sank the text
        # target: there the anchors were free to move and the whole system
        # drifted onto one point.
        #
        # Rows follow DATASETS.MODALITIES order with the reference stripped, so
        # row k is modality k+1.  Zero-initialised and drawing nothing from the
        # RNG, so step 0 is the untouched backbone, bit for bit.
        self.mod_delta_modalities = int(mod_delta_modalities)
        if self.mod_delta_modalities > 0:
            self.mod_delta = nn.Parameter(
                torch.zeros(self.mod_delta_modalities, depth, num_patches + 1, embed_dim))
        else:
            self.mod_delta = None

        if pe_freeze_base and self.pos_embed is not None:
            # load_param writes through state_dict(), which hands back detached
            # views, so freezing here survives the pretrained load.
            self.pos_embed.requires_grad_(False)

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        # NOTE: everything below must stay AFTER self.apply(self._init_weights),
        # which re-initialises every nn.Linear in the tree and would otherwise
        # destroy TPCG's identity initialisation.
        if pe_type == 'chartpe':
            self.chart_generator = TopologyPreservingChart(
                embed_dim, hidden=chart_hidden, grid_h=self.grid_h, grid_w=self.grid_w,
                theta_max=math.radians(chart_theta_max_deg), s_max=chart_scale_max,
                input_norm=chart_input_norm, zero_based=pe_zero_based)
        else:
            self.chart_generator = None
        if pe_type != 'learnable':
            # [1, 1 + N, 2]; row 0 is the CLS placeholder (zero phase -> untouched).
            grid = build_centered_grid(self.grid_h, self.grid_w, zero_based=pe_zero_based)
            grid = torch.cat([torch.zeros(1, 2), grid], dim=0).unsqueeze(0)
            self.register_buffer('chart_grid', grid, persistent=False)
        self.last_chart_stats = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _build_chart(self, patch_tokens):
        """patch_tokens: [B, N, D] (no CLS) -> chart [B, 1 + N, 2], fp32.

        Row 0 is the CLS placeholder and stays zero. The same chart is reused
        by every block, so TPCG runs once per image, not once per layer.
        """
        B, N, _ = patch_tokens.shape
        if self.pe_type == 'rope_fixed':
            return self.chart_grid.expand(B, -1, -1)

        # Phases must be computed in fp32; the tokens arrive as fp16 under AMP.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            patch_map = patch_tokens.float().reshape(B, self.grid_h, self.grid_w, -1)
            q, stats = self.chart_generator(patch_map)
            cls_slot = q.new_zeros(B, 1, 2)
            chart = torch.cat([cls_slot, q.reshape(B, N, 2)], dim=1)

        # Health signals (doc section 8.3): spacings must stay off the floor,
        # theta/s must not saturate, and |q - grid| should creep up then settle.
        with torch.no_grad():
            dx, dy, s = stats['dx'], stats['dy'], stats['scale']
            self.last_chart_stats = {
                'min_dx': dx.min().item(),
                'min_dy': dy.min().item(),
                # Spacing non-uniformity per image; 1.0 means a perfectly regular axis.
                'dx_skew': (dx.max(dim=1).values / dx.min(dim=1).values).mean().item(),
                'dy_skew': (dy.max(dim=1).values / dy.min(dim=1).values).mean().item(),
                'theta_max': stats['theta'].abs().max().item(),
                # Signed mean: a value near zero means the batch splits between
                # stretching and squeezing, which is what a content-driven chart
                # should do; a strong bias means a dataset-level constant offset.
                's_mean': s.mean().item(),
                's_max': s.abs().max().item(),
                'q_dev': (chart[:, 1:] - self.chart_grid[:, 1:]).abs().mean().item(),
            }
        return chart

    @staticmethod
    def _resolve_pe_layers(pe_layers, depth):
        """Validate MODEL.PE_LAYERS and return it sorted, or all of them.

        Everything here is a configuration mistake that would otherwise produce
        a plausible run: an out-of-range index silently dropped, a duplicate
        quietly allocating a row nothing reaches, an empty-after-filtering list
        turning the whole mechanism off while the log still says 'indep'.
        """
        if pe_layers is None:
            return list(range(depth))
        wanted = [int(i) for i in pe_layers]
        if not wanted:
            return list(range(depth))
        bad = [i for i in wanted if i < 0 or i >= depth]
        if bad:
            raise ValueError(
                'MODEL.PE_LAYERS holds block indices outside [0, {}): {}.  They '
                'are zero-based positions in the transformer, not layer counts.'
                .format(depth, bad))
        if len(set(wanted)) != len(wanted):
            raise ValueError(
                'MODEL.PE_LAYERS has duplicates: {}.  Each block takes at most '
                'one increment, so a repeat would allocate a row no forward '
                'reads.'.format(wanted))
        return sorted(wanted)

    def _layerwise_deltas(self):
        """What to add before each block, [depth, 1, 1 + N, D], or None.

        These two modes are defined by what the residual stream should *carry*
        at block l, not by what gets added there -- and the two differ, because
        the stream is never normalised.  Our blocks are pre-norm::

            a    = attn(norm1(x))      # norm feeds the branch, not the stream
            m_in = x + a               # x itself comes through untouched
            out  = m_in + mlp(norm2(m_in))

        so anything added before block l is still there at block l+1, exactly.
        (Verified: zeroing both branch output projections makes blk(x) == x to
        the last bit.)  Adding delta[l] at every block therefore accumulates on
        its own, and the two strategies have to be built accordingly:

        ``chain``   the correction flows forward -- block l carries
                    ``pos_embed + delta[0] + ... + delta[l]``.
                    Adding delta[l] gives exactly that, for free.

        ``indep``   each block corrects the original independently -- block l
                    carries ``pos_embed + delta[l]`` and nothing else.
                    The stream already holds delta[l-1], so it has to be
                    cancelled: add ``delta[l] - delta[l-1]``.

        Both are zero at initialisation either way, so step 0 still reproduces
        the single-injection baseline bit for bit.

        (Strictly, what block 0 carries is ``ln_pre(patches + pos_embed)``
        rather than the raw table -- ln_pre runs before the loop.  That is the
        anchor as the model sees it; the increments are what these modes are
        about.)

        With MODEL.PE_LAYERS naming a subset, "block l" above reads "insertion
        point j": the table has one row per insertion point, and `indep`
        cancels against the previously INJECTED row rather than the previous
        block -- which is the same statement, since the stream between two
        insertion points is carrying exactly that previous row and nothing
        else.  The arithmetic below is unchanged; only the length of axis `ax`
        is.
        """
        if self.pos_delta is None:
            return None
        if self.pe_layerwise == 'chain':
            return self.pos_delta
        d = self.pos_delta
        # Depth is axis 0 normally and axis 1 once a modality axis is in front.
        # The subtraction is per row either way: each spectrum's stack of
        # increments cancels only against its own previous layer, never across
        # spectra, which is what keeps the three deltas independent.
        ax = 1 if self.pe_per_modality else 0
        zero = d.new_zeros(*d.shape[:ax], 1, *d.shape[ax + 1:])
        prev = torch.cat([zero, d.narrow(ax, 0, d.shape[ax] - 1)], dim=ax)
        return d - prev

    def forward_features(self, x, camera_id, modal_id, view_id, delta_gate=None):
        """delta_gate: None, or [B] / [B,1,1] scaling the layer-wise increments
        per image.  None means "apply them everywhere", which is what every run
        before the text-alignment work did and must stay bit-identical."""
        B = x.shape[0]
        x = self.patch_embed(x)

        chart = self._build_chart(x) if self.pe_type != 'learnable' else None

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat([cls_tokens, x], dim=1)

        if self.pos_embed is None:
            pass  # rotary-only: position is injected inside every attention layer
        elif self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        if self.ln_pre is not None:
            x = self.ln_pre(x)

        deltas = self._layerwise_deltas()
        if deltas is not None and delta_gate is not None:
            # [B] -> [B,1,1] so it broadcasts over tokens and channels.  Kept
            # out of the loop: the reshape is the same for all twelve blocks.
            delta_gate = delta_gate.to(x.dtype).reshape(-1, 1, 1)

        # Per-image modality bank.  `modal_id` is 0-based over
        # DATASETS.MODALITIES, 0 being the reference; the clamp keeps the
        # gather in range and `mod_keep` zeroes the reference rows, so the
        # reference modality provably receives nothing regardless of what the
        # bank holds.  Both are computed once rather than per block.
        mod_idx = mod_keep = None
        if self.mod_delta is not None and modal_id is not None:
            modal_id = modal_id.reshape(-1).long()
            n_mod = self.mod_delta_modalities + 1
            if int(modal_id.max()) >= n_mod or int(modal_id.min()) < 0:
                raise ValueError(
                    'modal_label out of range: got [{}, {}] for {} modalities '
                    '(0 is the reference)'.format(int(modal_id.min()),
                                                  int(modal_id.max()), n_mod))
            mod_idx = (modal_id - 1).clamp_min(0)
            mod_keep = (modal_id > 0).to(x.dtype).reshape(-1, 1, 1)

        if self.pe_per_modality and deltas is not None:
            if modal_id is None:
                # Picking a row by default would silently make every spectrum
                # read the same delta -- the very thing this mode exists to
                # stop -- and the log would look exactly like a correct run.
                raise ValueError(
                    'MODEL.PE_PER_MODALITY needs modal_label on every forward, '
                    'including evaluation; none was passed')
            pm_idx = modal_id.reshape(-1).long()
            if int(pm_idx.max()) >= self.pe_per_modality or int(pm_idx.min()) < 0:
                raise ValueError(
                    'modal_label out of range: got [{}, {}] for {} per-modality '
                    'deltas'.format(int(pm_idx.min()), int(pm_idx.max()),
                                    self.pe_per_modality))

        for i, blk in enumerate(self.blocks):
            # `row` is which increment belongs at this block, or -1 for none.
            # With MODEL.PE_LAYERS unset every block has its own row and this is
            # just `i`, so the arithmetic below is the same as it always was.
            row = int(self.pe_slot[i]) if deltas is not None else -1
            if row >= 0:
                # [n_mod, rows, 1, N+1, D] -> [n_mod, N+1, D] -> [B, N+1, D],
                # which is the shape the shared path's deltas[row] broadcasts to.
                step = (deltas[:, row, 0].index_select(0, pm_idx)
                        if self.pe_per_modality else deltas[row])
                x = x + (step if delta_gate is None else step * delta_gate)
            if mod_idx is not None:
                x = x + self.mod_delta[:, i].index_select(0, mod_idx) * mod_keep
            x = blk(x, chart=chart)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x, cam_label=None, modal_label=None, view_label=None,
                delta_gate=None):
        x = self.forward_features(x, cam_label, modal_label, view_label,
                                  delta_gate=delta_gate)
        return x

    def _load_rope_freqs(self, freqs):
        """Scatter RoPE-ViT's single `freqs` tensor into our per-block rope modules.

        Theirs is built as ``stack(per_layer, dim=1).view(2, depth, -1)`` where
        each per-layer slab is ``[2, heads, head_dim // 2]``, so the trailing axis
        is head-major.  Ours keeps ``[heads, head_dim // 2, 2]`` per block, which
        is exactly that slab permuted -- no interpolation, no reordering.
        """
        if freqs.dim() != 3 or freqs.shape[0] != 2:
            raise RuntimeError('unexpected freqs shape {}; expected [2, depth, heads*pairs]'
                               .format(tuple(freqs.shape)))
        _, depth, flat = freqs.shape
        if depth != len(self.blocks):
            raise RuntimeError('checkpoint has {} rope layers, model has {}'
                               .format(depth, len(self.blocks)))
        loaded = 0
        for n, blk in enumerate(self.blocks):
            rope = getattr(blk.attn, 'rope', None)
            if rope is None:
                continue                       # learnable PE: nothing to receive them
            heads, pairs, _ = rope.freqs.shape
            if heads * pairs != flat:
                raise RuntimeError('freqs width {} does not match {} heads x {} pairs'
                                   .format(flat, heads, pairs))
            rope.freqs.data.copy_(freqs[:, n, :].view(2, heads, pairs).permute(1, 2, 0))
            loaded += 1
        print('Loaded RoPE frequencies into %d / %d blocks.' % (loaded, len(self.blocks)))
        return loaded

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
        if isinstance(param_dict, torch.nn.Module):
            # OpenAI ships CLIP as a TorchScript archive.  torch.load does not
            # fail on one -- it quietly dispatches to torch.jit.load and hands
            # back a module -- so the check has to be on the type, not on an
            # exception that never fires.
            param_dict = param_dict.state_dict()
        if 'visual.conv1.weight' in param_dict:
            print('OpenAI CLIP checkpoint detected (visual.* image tower)')
            self._load_clip_visual(param_dict)
            return
        count=0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'teacher' in param_dict: ### for dino
            obj = param_dict["teacher"]
            print('Convert dino model......')
            newmodel = {}
            for k, v in obj.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if not k.startswith("backbone."):
                    continue
                old_k = k
                k = k.replace("backbone.", "")
                newmodel[k] = v
                param_dict = newmodel
        # RoPE-ViT / DeiT-III checkpoints (the *_LS releases) need three fixups
        # that no other source does.  Detect them structurally rather than by
        # filename: LayerScale is the thing that makes them what they are.
        deit3 = 'blocks.0.gamma_1' in param_dict
        if deit3:
            print('DeiT-III / RoPE-ViT checkpoint detected (LayerScale present)')
            if self.blocks[0].gamma_1 is None:
                raise RuntimeError(
                    'checkpoint carries LayerScale but the model was built without it; '
                    'set MODEL.LAYER_SCALE True (the rest of these weights were '
                    'trained with gamma_1/gamma_2 in place and are not valid without them)')
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if k in ('freqs_t_x', 'freqs_t_y'):
                # RoPE-ViT's own coordinate grid, sized for its 14x14 layout; we
                # rebuild ours from the actual patch grid in _build_chart.
                continue
            if k == 'freqs':
                count += self._load_rope_freqs(v)
                continue
            if deit3 and k == 'pos_embed' and self.pos_embed is not None:
                # DeiT-III's table covers patches only -- CLS gets no positional
                # embedding at all.  Prepending a zero row both reproduces that
                # and lets resize_pos_embed below work unmodified, since it
                # expects row 0 to be the CLS slot.
                v = torch.cat([v.new_zeros(1, 1, v.shape[-1]), v], dim=1)
            if k == 'pos_embed' and self.pos_embed is None:
                # Rotary modes have no position table; skip before the resize
                # branch below (it would dereference self.pos_embed.shape).
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x, self.hw_ratio)
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))


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


def vit_base_in(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, ics_embedding=False, \
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature, hw_ratio=1, **kwargs)

    return model

def vit_base_clip(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    """CLIP-ViT-B/16 image tower, shaped to accept OpenAI's released weights.

    Same 768/12/12 geometry as vit_base_in, but with the three things CLIP does
    differently: QuickGELU instead of GELU, an extra LayerNorm before block 0,
    and eps 1e-5 rather than 1e-6 in every LayerNorm.  Pair it with
    INPUT.PIXEL_MEAN/STD set to CLIP's own statistics -- the weights were fitted
    to that normalisation.
    """
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, ics_embedding=False, \
        norm_layer=partial(nn.LayerNorm, eps=1e-5), sie_xishu=sie_xishu, local_feature=local_feature, hw_ratio=1, \
        clip_style=True, act_layer=QuickGELU, **kwargs)

    return model

def vit_ics_lup(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, ics_embedding=True, \
        norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu, local_feature=local_feature, hw_ratio=2, **kwargs)

    return model


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
