import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_clip

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super().__init__()
        if cfg.MODEL.TRANSFORMER_TYPE != 'vit_base_clip':
            raise ValueError('This branch supports A / CLIP ViT-B/16 only')
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.base = factory['vit_base_clip'](
            img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
            trajectory_enabled=cfg.MODEL.TRAJECTORY.ENABLED,
            acceleration_mix=cfg.MODEL.TRAJECTORY.ACCELERATION_MIX,
            gain_bound=cfg.MODEL.TRAJECTORY.GAIN_BOUND)
        if cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            self.base.load_param(cfg.MODEL.PRETRAIN_PATH)
        elif cfg.MODEL.PRETRAIN_CHOICE not in ('self', 'no'):
            raise ValueError('PRETRAIN_CHOICE must be imagenet, self or no')
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.full_enabled = cfg.MODEL.FULL.ENABLED
        if self.full_enabled:
            if self.base.trajectory is None or not cfg.DATASETS.AERIAL_CAMS:
                raise ValueError('Full needs Trajectory and zero-based aerial camera IDs')
            from .backbones.clip_text import CLIPTextEncoder, ViewPrompts
            # Text initialization must not perturb the existing visual RNG stream.
            # Forward DropPath is NOT replayed: retain the reference VTC semantics.
            with torch.random.fork_rng(devices=[]):
                clip_sd = torch.load(cfg.MODEL.FULL.TEXT_CLIP_PATH, map_location='cpu')
                if isinstance(clip_sd, nn.Module):
                    clip_sd = clip_sd.state_dict()
                text = CLIPTextEncoder()
                text.load_clip(clip_sd)
                self.prompts = ViewPrompts(text, n_ctx=4)
            self.register_buffer('logit_scale', clip_sd['logit_scale'].exp().float())
            self.register_buffer('full_aerial_cams', torch.tensor(cfg.DATASETS.AERIAL_CAMS))
            self.rgb_index = list(cfg.DATASETS.MODALITIES).index('RGB')
            del clip_sd
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self, x=None, label=None, camids=None, mode=0, trajectory_gate=1.0):
        # Preserve modality-major concatenation and the public train/eval API.
        if mode == 0:
            x = torch.cat(list(x), dim=0)
        global_feat = self.base(x, trajectory_gate=trajectory_gate)
        feat = self.bottleneck(global_feat)
        if mode == 0:
            cls_score = self.classifier(feat)
            if not self.full_enabled:
                return cls_score, global_feat, feat
            per_modality = x.shape[0] // len(camids)
            start = self.rgb_index * per_modality
            rows = slice(start, start + per_modality)
            # RGB only, no detach, no RNG replay; the before pass bypasses BN.
            feat_before = self.base(x[rows], trajectory_gate=0.0)
            return cls_score, global_feat, feat, {
                'feat_after': global_feat[rows], 'feat_before': feat_before,
                'is_aerial': torch.isin(camids[self.rgb_index].to(x.device), self.full_aerial_cams),
                'text': self.prompts(), 'logit_scale': self.logit_scale,
                'proj': self.base.clip_proj,
            }
        return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        """Load backbone and bottleneck, preserving the historical classifier skip.

        Retrieval uses no classifier. Report missing and incompatible tensors.
        """
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        own = self.state_dict()
        full_keys = {k for k in own if k.startswith('prompts.') or k in ('logit_scale', 'full_aerial_cams')}
        saved_full = {k.replace('module.', '') for k in param_dict
                      if k.replace('module.', '').startswith('prompts.')
                      or k.replace('module.', '') in ('logit_scale', 'full_aerial_cams')}
        if full_keys != saved_full:
            raise RuntimeError('Full checkpoint/config mismatch: use the matching full or trajectory-only config')
        if full_keys and not torch.equal(param_dict.get('full_aerial_cams', param_dict.get('module.full_aerial_cams')).cpu(), self.full_aerial_cams.cpu()):
            raise RuntimeError('Full aerial camera mapping differs from checkpoint')
        trajectory_keys = {k for k in own if k.startswith('base.trajectory.')}
        checkpoint_trajectory = {k.replace('module.', '') for k in param_dict
                                 if k.replace('module.', '').startswith('base.trajectory.')}
        if trajectory_keys != checkpoint_trajectory:
            raise RuntimeError('Trajectory checkpoint/config mismatch: use the training config for evaluation. '
                               'Fresh Trajectory training starts from CLIP PRETRAIN_CHOICE=imagenet.')
        for k, v in param_dict.items():
            if k.replace('module.', '') == 'base.trajectory.spec':
                if not torch.equal(v.cpu(), own['base.trajectory.spec'].cpu()):
                    raise RuntimeError('Trajectory beta/gain bound differs from checkpoint; use its training config.')

        loaded, skipped, unexpected, mismatched = 0, [], [], []
        for k, v in param_dict.items():
            key = k.replace('module.', '')
            if 'classifier' in key:
                skipped.append(key)
                continue
            if key not in own:
                unexpected.append(key)
                continue
            if own[key].shape != v.shape:
                mismatched.append('{} {} vs {}'.format(key, tuple(v.shape), tuple(own[key].shape)))
                continue
            own[key].copy_(v)
            loaded += 1

        missing = [k for k in own if k not in {x.replace('module.', '') for x in param_dict}]
        print('Loading pretrained model from {}'.format(trained_path))
        print('  loaded {} / {} tensors | {} classifier rows skipped (identity counts differ)'
              .format(loaded, len(own), len(skipped)))
        for label, items in (('not in this model', unexpected),
                             ('shape mismatch', mismatched),
                             ('left at init', missing)):
            if items:
                print('  {}: {}{}'.format(label, items[:6], ' ...' if len(items) > 6 else ''))
        if mismatched:
            # Do not evaluate partially initialized weights after a shape mismatch.
            raise RuntimeError(
                'shape mismatch loading {}: {}.  The checkpoint was trained under a '
                'different architecture than this config builds. Loading anyway leaves weights at '
                'their initialisation and report a number for the wrong model.'
                .format(trained_path, mismatched[:6]))
        if loaded == 0:
            raise RuntimeError(
                'nothing loaded from {} -- every key was unknown. This usually means the '
                'file is a backbone-only checkpoint (keys like `blocks.0...`), which needs '
                'MODEL.PRETRAIN_CHOICE "imagenet", not "self".'.format(trained_path))

def make_model(cfg, num_class, camera_num, view_num):
    return build_transformer(num_class, camera_num, view_num, cfg,
                             {'vit_base_clip': vit_base_clip})
