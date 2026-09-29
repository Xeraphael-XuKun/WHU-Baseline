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
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
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
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            self.load_param(cfg.MODEL.PRETRAIN_PATH)

    def forward(self, x=None, label=None, camids=None, mode=0):
        # Preserve modality-major concatenation and the public train/eval API.
        if mode == 0:
            x = torch.cat(list(x), dim=0)
        global_feat = self.base(x)
        feat = self.bottleneck(global_feat)
        if mode == 0:
            return self.classifier(feat), global_feat, feat
        return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        """Load backbone and bottleneck, preserving the historical classifier skip.

        Retrieval uses no classifier. Report missing and incompatible tensors.
        """
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        own = self.state_dict()

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
