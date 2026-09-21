import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_base_in, vit_base_clip, vit_ics_lup

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

def _read_clip_checkpoint(path):
    """OpenAI ships CLIP as a TorchScript archive; torch.load does not fail on
    one, it quietly dispatches to torch.jit.load and returns a module.  Same
    handling as TransReID.load_param -- see the note there."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    if 'token_embedding.weight' not in obj:
        raise RuntimeError(
            '{} has no text tower (no token_embedding.weight); MODEL.TEXT_CLIP_PATH '
            'must point at the original CLIP release, not at one of our checkpoints'
            .format(path))
    return obj


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))
        print('using positional encoding: {}{}{}{}{}{}{}{}'.format(
            cfg.MODEL.PE_TYPE,
            ' (+ape)' if cfg.MODEL.PE_KEEP_APE else '',
            ' (+innorm)' if cfg.MODEL.CHART_INPUT_NORM else '',
            ' (+zerobase)' if cfg.MODEL.PE_ZERO_BASED else '',
            ' (+layerscale)' if cfg.MODEL.LAYER_SCALE else '',
            ' (+layerwise:{})'.format(cfg.MODEL.PE_LAYERWISE) if cfg.MODEL.PE_LAYERWISE != 'none' else '',
            ' (+gate:alpha=0)' if cfg.MODEL.ROPE_GATE else '',
            ' (+frozenbase)' if cfg.MODEL.PE_FREEZE_BASE else ''))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # ---- twin-anchored modality residuals ------------------------------
        # Row order of the bank is DATASETS.MODALITIES with the reference
        # stripped, and the batch is laid out modality-major in that same order
        # (train_collate_fn + the torch.cat in forward), so "modality index" is
        # one thing throughout.  v1 requires the reference to be first: any
        # other position would silently shift every bank row by one.
        self.modalities = list(cfg.DATASETS.MODALITIES)
        self.use_mod_delta = cfg.MODEL.MOD_DELTA
        self.test_mod_delta = cfg.TEST.MOD_DELTA
        n_mod_delta = 0
        if self.use_mod_delta:
            if cfg.MODEL.MOD_DELTA_REF != self.modalities[0]:
                raise ValueError(
                    'MODEL.MOD_DELTA_REF {!r} must be the FIRST entry of '
                    'DATASETS.MODALITIES {} -- bank row k is modality k+1, so a '
                    'reference anywhere else would attach every bank to the '
                    'wrong spectrum'.format(cfg.MODEL.MOD_DELTA_REF, self.modalities))
            if len(self.modalities) < 2:
                raise ValueError('MODEL.MOD_DELTA needs at least two modalities')
            n_mod_delta = len(self.modalities) - 1

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        pe_type=cfg.MODEL.PE_TYPE,
                                                        pe_keep_ape=cfg.MODEL.PE_KEEP_APE,
                                                        chart_hidden=cfg.MODEL.CHART_HIDDEN,
                                                        chart_theta_max_deg=cfg.MODEL.CHART_THETA_MAX_DEG,
                                                        chart_scale_max=cfg.MODEL.CHART_SCALE_MAX,
                                                        chart_input_norm=cfg.MODEL.CHART_INPUT_NORM,
                                                        rope_theta=cfg.MODEL.ROPE_THETA,
                                                        rope_freq_trainable=cfg.MODEL.ROPE_FREQ_TRAINABLE,
                                                        rope_gate=cfg.MODEL.ROPE_GATE,
                                                        pe_zero_based=cfg.MODEL.PE_ZERO_BASED,
                                                        layer_scale=cfg.MODEL.LAYER_SCALE,
                                                        layer_scale_init=cfg.MODEL.LAYER_SCALE_INIT,
                                                        pe_layerwise=cfg.MODEL.PE_LAYERWISE,
                                                        pe_freeze_base=cfg.MODEL.PE_FREEZE_BASE,
                                                        mod_delta_modalities=n_mod_delta,
                                                        pe_layers=list(cfg.MODEL.PE_LAYERS),
                                                        pe_per_modality=(len(self.modalities)
                                                                         if cfg.MODEL.PE_PER_MODALITY
                                                                         else 0))

        if cfg.MODEL.PE_LAYERS and cfg.MODEL.PE_LAYERWISE == 'none':
            # The list would be read, validated and then never used, and the run
            # would report the plain baseline under the ablation's name.
            raise ValueError(
                'MODEL.PE_LAYERS says where to inject the positional increments, '
                "but MODEL.PE_LAYERWISE is 'none' so there are none to inject")

        if self.use_mod_delta:
            print('Mod-delta: banks for {} (reference {} gets none), {:,} params'
                  .format(self.modalities[1:], self.modalities[0],
                          self.base.mod_delta.numel()))

        if pretrain_choice not in ('imagenet', 'self', 'no'):
            # Previously any other value fell through in silence and trained
            # from scratch, which looks exactly like a normal run in the log.
            raise ValueError(
                "MODEL.PRETRAIN_CHOICE must be 'imagenet' (backbone-only weights), "
                "'self' (a checkpoint from this codebase) or 'no', got {!r}".format(
                    pretrain_choice))

        if pretrain_choice == 'imagenet':
            # A backbone-only checkpoint (ImageNet ViT, CLIP, DeiT-III, ...):
            # keys look like `blocks.0.attn.qkv.weight`, so only `base` is touched
            # and it is safe to do here, before the heads exist.
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))
        elif pretrain_choice == 'no':
            print('PRETRAIN_CHOICE is "no": training from random initialisation')

        self.num_classes = num_classes

        from loss.make_loss import ce_modality_groups, ce_slot_count
        self.ce_slots = ce_slot_count(cfg)
        self.classifier = nn.Linear(self.in_planes,
                                    self.num_classes * self.ce_slots, bias=False)
        self.classifier.apply(weights_init_classifier)
        if self.ce_slots > 1:
            # Spell the grouping out by name.  `CE_SPLIT_MODALITY: True` reads
            # the same in the log whether the spectra are split three ways or
            # two, and the two runs differ by 500 classes.
            spectra = '-'
            if cfg.MODEL.CE_SPLIT_MODALITY:
                mods, g = list(cfg.DATASETS.MODALITIES), ce_modality_groups(cfg)
                spectra = ' | '.join(
                    '+'.join(m for m, gi in zip(mods, g) if gi == k)
                    for k in range(max(g) + 1))
            print('CE split: {} identities x {} slots = {} classes '
                  '(view={}, spectra: {})'.format(
                      self.num_classes, self.ce_slots,
                      self.num_classes * self.ce_slots,
                      cfg.MODEL.CE_SPLIT_VIEW, spectra))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # ---- one positional delta per spectrum -------------------------------
        self.pe_per_modality = bool(cfg.MODEL.PE_PER_MODALITY)
        if self.pe_per_modality:
            if cfg.MODEL.PE_LAYERWISE == 'none':
                raise ValueError('MODEL.PE_PER_MODALITY needs MODEL.PE_LAYERWISE '
                                 'indep or chain -- there is no delta to split')
            print('PE per modality: one independent delta per spectrum {}, '
                  'indep differencing runs inside each -- {:,} parameters against '
                  '{:,} shared.  Every forward, evaluation included, needs the '
                  'modality index.'
                  .format(self.modalities, len(self.modalities) * 1188864, 1188864))

        # ---- viewpoint gate on the positional delta -------------------------
        self.pe_view_gate = cfg.MODEL.PE_VIEW_GATE
        if self.pe_view_gate not in ('none', 'aerial', 'ground'):
            raise ValueError("MODEL.PE_VIEW_GATE must be 'none', 'aerial' or "
                             "'ground', got {!r}".format(self.pe_view_gate))
        if self.pe_view_gate != 'none':
            if cfg.MODEL.PE_LAYERWISE == 'none':
                raise ValueError('MODEL.PE_VIEW_GATE gates the layer-wise delta, '
                                 'but MODEL.PE_LAYERWISE is none -- there is '
                                 'nothing to gate')
            if not list(cfg.DATASETS.AERIAL_CAMS):
                raise ValueError('MODEL.PE_VIEW_GATE needs DATASETS.AERIAL_CAMS; '
                                 'without it every image counts as ground and the '
                                 'gate would silently switch the delta off entirely')
            if cfg.MODEL.TEXT_ALIGN and self.pe_view_gate == 'ground':
                # before and after would be the same tensor on the aerial rows,
                # and the table asks it to read as both anchors at once.
                raise ValueError(
                    "MODEL.PE_VIEW_GATE 'ground' with MODEL.TEXT_ALIGN True is "
                    'self-contradictory: the aerial rows take the delta-free path '
                    'in both passes, so A_before and A_after are one tensor with '
                    'two different targets, and the correction has no actuator')

        # ---- view alignment against CLIP text anchors -----------------------
        self.text_align = cfg.MODEL.TEXT_ALIGN
        self.text_target = cfg.MODEL.TEXT_TARGET
        self.text_supervision = cfg.MODEL.TEXT_SUPERVISION
        if self.text_supervision not in ('contrast', 'direct'):
            raise ValueError("MODEL.TEXT_SUPERVISION must be 'contrast' or "
                             "'direct', got {!r}".format(self.text_supervision))
        if self.text_supervision == 'direct' and self.text_target != 'view':
            # modality_align reads feat_before unconditionally, and 'direct'
            # does not produce one.  The spectrum axis has its own before/after
            # meaning ("still reads as its own spectrum"), so a direct variant
            # there is a separate design, not this key.
            raise ValueError(
                "MODEL.TEXT_SUPERVISION 'direct' is defined for TEXT_TARGET "
                "'view' only, got {!r}".format(self.text_target))
        if (self.text_align and self.text_supervision == 'contrast'
                and cfg.MODEL.PE_LAYERWISE == 'none'
                and not cfg.MODEL.TEXT_ALLOW_NO_ACTUATOR):
            # The same pathology the PE_VIEW_GATE 'ground' guard above refuses,
            # in its strongest form: with no delta at all, `feat_before` and
            # `feat_after` are one tensor for EVERY row, and A_before/A_after
            # hand it two opposite targets.  The two cross entropies cancel at
            # p = 0.5, so what the arm actually trains is "aerial features sit
            # equidistant from the two anchors" -- a real constraint, but not
            # the one the 2x2 was designed to express, and nothing in the log
            # would say so.
            raise ValueError(
                "MODEL.TEXT_ALIGN with PE_LAYERWISE 'none' has no actuator: "
                'before and after are the same tensor, so the aerial rows carry '
                "two opposite targets.  Use MODEL.TEXT_SUPERVISION 'direct' "
                '(one target per row, no before pass), or set '
                'MODEL.TEXT_ALLOW_NO_ACTUATOR True if the degenerate arm is '
                'deliberately what you want to measure.')
        self.prompts = None
        if self.text_align and self.text_target not in ('view', 'modality'):
            raise ValueError("MODEL.TEXT_TARGET must be 'view' or 'modality', "
                             "got {!r}".format(self.text_target))
        if (list(cfg.MODEL.TEXT_MODALITY_TARGETS)
                and not (self.text_align and self.text_target == 'modality')):
            # The mtext lesson: TEXT_MODALITY was set in that config, ignored,
            # and nothing in the log said so -- the run looked like the one
            # that was intended from start to finish.
            raise ValueError(
                'MODEL.TEXT_MODALITY_TARGETS only means anything under '
                "MODEL.TEXT_ALIGN True and MODEL.TEXT_TARGET 'modality'; "
                'setting it otherwise would be silently ignored')
        if self.text_align:
            from .backbones.clip_text import CLIPTextEncoder, ViewPrompts
            if getattr(self.base, 'clip_proj', None) is None:
                raise ValueError(
                    'MODEL.TEXT_ALIGN needs the CLIP joint-space projection; use '
                    'TRANSFORMER_TYPE vit_base_clip (visual.proj -> clip_proj)')
            # Deliberately NOT PRETRAIN_PATH: on the WHU-MARS legs that points
            # at our own CARGO checkpoint, which carries no text tower.
            if not cfg.MODEL.TEXT_CLIP_PATH:
                raise ValueError(
                    'MODEL.TEXT_ALIGN needs MODEL.TEXT_CLIP_PATH pointing at ViT-B-16.pt; '
                    'PRETRAIN_PATH is the image-side checkpoint and has no text tower')
            clip_sd = _read_clip_checkpoint(cfg.MODEL.TEXT_CLIP_PATH)
            text = CLIPTextEncoder()
            text.load_clip(clip_sd)
            modalities = list(cfg.DATASETS.MODALITIES)
            mod_words = list(cfg.MODEL.TEXT_MODALITY_WORDS) or None
            if self.text_target == 'modality':
                # One axis, not two: the first template slot holds the spectrum
                # and there is no viewpoint word at all.  Three anchors, one
                # per spectrum, and every image in the batch is supervised.
                if not mod_words:
                    raise ValueError(
                        "MODEL.TEXT_TARGET 'modality' needs MODEL.TEXT_MODALITY_WORDS "
                        "-- one single-token spectrum word per DATASETS.MODALITIES entry")
                self.prompts = ViewPrompts(text, template=cfg.MODEL.TEXT_TEMPLATE,
                                           n_ctx=cfg.MODEL.TEXT_N_CTX,
                                           view_words=mod_words, modality_words=None,
                                           slot_name='modality')
            else:
                # An empty MODEL.TEXT_MODALITY_WORDS means no modality slot: two
                # anchors, one supervised modality.  Naming the spectra turns the
                # task six-way and supervises all three.
                self.prompts = ViewPrompts(text, template=cfg.MODEL.TEXT_TEMPLATE,
                                           n_ctx=cfg.MODEL.TEXT_N_CTX,
                                           modality_words=mod_words)
            # CLIP's learned temperature, exp(logit_scale).  Frozen: it was
            # calibrated on 400M pairs and is not ours to retune.
            scale = clip_sd.get('logit_scale')
            self.register_buffer('logit_scale',
                                 scale.exp().float() if scale is not None
                                 else torch.tensor(100.0), persistent=False)
            self.n_text_modality = self.prompts.n_modality
            if self.text_target == 'modality':
                if len(mod_words) != len(modalities):
                    raise ValueError(
                        'MODEL.TEXT_MODALITY_WORDS has {} entries but DATASETS.MODALITIES '
                        'has {}; anchor row m is the anchor for modality m, so a mismatch '
                        'would attach the wrong spectrum word to a whole modality'
                        .format(len(mod_words), len(modalities)))
                if cfg.MODEL.TEXT_MODALITY not in modalities:
                    raise ValueError('MODEL.TEXT_MODALITY {!r} is not in DATASETS.MODALITIES {}'
                                     .format(cfg.MODEL.TEXT_MODALITY, modalities))
                # Under this target TEXT_MODALITY names the reference spectrum.
                # It is what `drift` is measured against -- the guard reading
                # that the reference itself has not wandered -- and it is the
                # default correction target for every other spectrum.
                self.text_target_row = modalities.index(cfg.MODEL.TEXT_MODALITY)
                # ... but the target can also be given per spectrum, so that a
                # pair which is already aligned is not asked to move.  Empty
                # reproduces "everything towards the reference" exactly, which
                # is what keeps whu_mcam's log comparable.
                rows = [int(r) for r in cfg.MODEL.TEXT_MODALITY_TARGETS]
                if rows:
                    if len(rows) != len(modalities):
                        raise ValueError(
                            'MODEL.TEXT_MODALITY_TARGETS has {} entries for the {} '
                            'modalities {} -- it is parallel to DATASETS.MODALITIES'
                            .format(len(rows), len(modalities), modalities))
                    if min(rows) < 0 or max(rows) >= len(modalities):
                        raise ValueError(
                            'MODEL.TEXT_MODALITY_TARGETS entries index the anchor '
                            'rows, so they must lie in [0, {}); got {}'
                            .format(len(modalities), rows))
                else:
                    rows = [self.text_target_row] * len(modalities)
                self.text_target_rows = rows
                print('Text correction (spectrum axis): {}'.format(', '.join(
                    '{} -> {}'.format(m, modalities[r])
                    for m, r in zip(modalities, rows))))
                self.text_modality_index = None      # every row is supervised
                # `modality` in _text_subset keys off this, and here it has to
                # be the real count so each row gets its own before-anchor.
                self.n_text_modality = len(mod_words)
            elif mod_words:
                if len(mod_words) != len(modalities):
                    raise ValueError(
                        'MODEL.TEXT_MODALITY_WORDS has {} entries but DATASETS.MODALITIES '
                        'has {}; they index the same axis, so a mismatch would attach '
                        'the wrong anchor to a whole spectrum'
                        .format(len(mod_words), len(modalities)))
                self.text_modality_index = None          # all of them
            else:
                if cfg.MODEL.TEXT_MODALITY not in modalities:
                    raise ValueError('MODEL.TEXT_MODALITY {!r} is not in DATASETS.MODALITIES {}'
                                     .format(cfg.MODEL.TEXT_MODALITY, modalities))
                self.text_modality_index = modalities.index(cfg.MODEL.TEXT_MODALITY)
            aerial = sorted(set(int(c) for c in cfg.DATASETS.AERIAL_CAMS))
            if not aerial and self.text_target == 'view':
                raise ValueError(
                    'MODEL.TEXT_ALIGN needs DATASETS.AERIAL_CAMS; without it every '
                    'image counts as ground and the aerial->ground term never fires')
            self.register_buffer('aerial_cams', torch.tensor(aerial).long(), persistent=False)
            if self.text_target == 'modality':
                print('Text align: target=modality, {} anchors {}, corrected towards '
                      '{!r} (row {}), logit_scale {:.2f}'
                      .format(len(mod_words), mod_words, cfg.MODEL.TEXT_MODALITY,
                              self.text_target_row, float(self.logit_scale)))
            else:
                print('Text align: target=view, {}, aerial cams {} (zero-based), '
                      'logit_scale {:.2f}'
                      .format('all {} modalities {}'.format(len(mod_words), mod_words)
                              if mod_words else
                              'modality {} (index {}) only'.format(
                                  cfg.MODEL.TEXT_MODALITY, self.text_modality_index),
                              aerial, float(self.logit_scale)))

        if self.pe_view_gate != 'none':
            if not hasattr(self, 'aerial_cams'):
                # TEXT_ALIGN registers this buffer; without it nothing else does,
                # and the gate would have no camera list to test against.
                self.register_buffer(
                    'aerial_cams',
                    torch.tensor(sorted(set(int(c) for c in cfg.DATASETS.AERIAL_CAMS))).long(),
                    persistent=False)
            print('PE view gate: the delta applies to {} images only (aerial cams '
                  '{}, zero-based).  The other viewpoint takes the delta-free path '
                  'in training AND at inference, so this model needs a camera id '
                  'at test time -- the ungated one does not.'
                  .format(self.pe_view_gate, cfg.DATASETS.AERIAL_CAMS))

        if pretrain_choice == 'self':
            # Deliberately last: this checkpoint carries `bottleneck.*` as well
            # as `base.*`, and those two modules do not exist until the lines
            # above have run.  Loading any earlier silently restores the
            # backbone only and leaves the bottleneck at its fresh init.
            self.load_param(model_path)
        
    def _text_subset(self, camids, per_modality, device):
        """Which rows of the concatenated batch feed the text loss, which of
        them are aerial, and which modality each belongs to.

        Modalities are concatenated along the batch in DATASETS.MODALITIES
        order (see build_transformer.forward's torch.cat and
        train_collate_fn), so row i belongs to modality i // per_modality.

        With `text_modality_index` set, only that one modality is supervised --
        CLIP has never seen infrared or thermal, so asking its frozen text
        tower whether such an image reads as a ground viewpoint has no
        grounding.  With modality anchors the template carries a learnable
        slot for the spectrum instead, and all three are supervised against
        their own pair of anchors.
        """
        cams = torch.cat([c.to(device) for c in camids], dim=0)
        if self.text_modality_index is None:
            rows = torch.arange(cams.shape[0], device=device)
        else:
            start = self.text_modality_index * per_modality
            rows = torch.arange(start, start + per_modality, device=device)
        if self.aerial_cams.numel():
            is_aerial = torch.isin(cams[rows], self.aerial_cams.to(device))
        else:
            # Only the view target needs this; under 'modality' the camera is
            # never consulted and an empty AERIAL_CAMS is legitimate.
            is_aerial = torch.zeros_like(rows, dtype=torch.bool)
        # Modality of each selected row; a single supervised modality always
        # maps to anchor block 0, which is what the two-anchor form expects.
        modality = (rows // per_modality if self.n_text_modality > 1
                    else torch.zeros_like(rows))
        return rows, is_aerial, modality

    def _view_gate(self, camids, n_rows, device):
        """[N] of 1.0 where the delta applies and 0.0 where it does not.

        `camids` is a list of per-modality tensors during training and a single
        tensor at evaluation; both are concatenated the same way the batch is.
        Returns None when the gate is off, which keeps the ungated call bit for
        bit what it was.
        """
        if self.pe_view_gate == 'none':
            return None
        if camids is None:
            # Silently applying the delta everywhere would produce a run that
            # looks like this arm and is the ungated one.
            raise ValueError(
                'MODEL.PE_VIEW_GATE needs camera ids, but none were passed. '
                'At evaluation, processor.do_inference has to forward `camidt` '
                'to the model.')
        cams = (torch.cat([c.to(device) for c in camids], dim=0)
                if isinstance(camids, (list, tuple))
                else camids.to(device).reshape(-1))
        if cams.shape[0] != n_rows:
            raise ValueError('camera ids cover {} rows but the batch has {}'
                             .format(cams.shape[0], n_rows))
        is_aerial = torch.isin(cams, self.aerial_cams.to(device))
        keep = is_aerial if self.pe_view_gate == 'aerial' else ~is_aerial
        return keep.float()

    def forward(self, x=None, label=None, camids=None, mode=0):
        if mode==0:
            imgs = list(x)
            per_modality = imgs[0].shape[0]
            x = torch.cat(imgs, dim=0)
            # Modality-major: row i belongs to modality i // per_modality, the
            # same convention _text_subset documents.  None when the banks are
            # off, which keeps the old call bit-identical.
            # Row i belongs to modality i // per_modality.  Either the banks
            # or the per-spectrum deltas need it; None keeps the old call
            # bit-identical when neither is on.
            modal = (torch.arange(len(imgs), device=x.device)
                     .repeat_interleave(per_modality)
                     if (self.use_mod_delta or self.pe_per_modality) else None)
            gate = self._view_gate(camids, x.shape[0], x.device)
            global_feat = self.base(x, modal_label=modal, delta_gate=gate)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)

            if not self.text_align or camids is None:
                return cls_score, global_feat, feat

            rows, is_aerial, modality = self._text_subset(camids, per_modality, x.device)
            if self.text_supervision == 'direct':
                # No before pass: each supervised row gets ONE target and the
                # correction is not expressed as a contrast.  The delta is then
                # optional rather than load-bearing, which is the point -- this
                # is the mode that can run with PE_LAYERWISE 'none' and measure
                # what the text supervision is worth on its own.
                return cls_score, global_feat, feat, {
                    'feat_after': global_feat[rows],
                    'feat_before': None,
                    'is_aerial': is_aerial,
                    'modality': modality,
                    'n_view': self.prompts.n_view,
                    'n_modality': self.prompts.n_modality,
                    'text': self.prompts(),
                    'logit_scale': self.logit_scale,
                    'proj': self.base.clip_proj,
                    'supervision': 'direct',
                    'target': self.text_target,
                }
            # Second pass over the supervised rows with the increments switched
            # off.  f_before and f_after come from the same images and the same
            # weights, so the only thing that can separate them is the
            # positional delta -- which is what makes this a constraint on the
            # delta rather than on the backbone.
            # `rows // per_modality` is the TRUE spectrum of each selected row.
            # Not `modality` from _text_subset -- that one is the ANCHOR block,
            # which is all zeros under two-anchor supervision and would send
            # every row to the RGB delta.  The gate is zero here so nothing is
            # added either way, but the row still has to be selectable.
            feat_before = self.base(
                x[rows],
                modal_label=(rows // per_modality) if self.pe_per_modality else None,
                delta_gate=torch.zeros(rows.shape[0], device=x.device))
            return cls_score, global_feat, feat, {
                'feat_after': global_feat[rows],
                'feat_before': feat_before,
                'is_aerial': is_aerial,
                'modality': modality,
                'n_view': self.prompts.n_view,
                'n_modality': self.prompts.n_modality,
                'text': self.prompts(),
                'logit_scale': self.logit_scale,
                'proj': self.base.clip_proj,
                # Which supervision the processor should apply.  Kept in the
                # dict rather than read off the model so the loss functions
                # stay testable with a plain dict.
                'target': self.text_target,
                # The reference spectrum, used for the drift guard.
                'target_row': getattr(self, 'text_target_row', 0),
                # Per-spectrum correction target.  None means "everything
                # towards target_row", the only behaviour that existed before
                # MODEL.TEXT_MODALITY_TARGETS and the one whu_mcam ran.
                'target_rows': getattr(self, 'text_target_rows', None),
                'per_modality': per_modality,
            }
        else:
            # `mode` is the 1-based modality index the evaluator iterates with,
            # so mode-1 is the bank row convention.  TEST.MOD_DELTA False sends
            # nothing, which under SOLVER.MOD_DELTA_ONLY is the frozen baseline
            # exactly -- that is the check the whole design rests on.
            modal = None
            if self.use_mod_delta and self.test_mod_delta:
                modal = torch.full((x.shape[0],), max(int(mode) - 1, 0),
                                   dtype=torch.long, device=x.device)
            elif self.pe_per_modality:
                # `mode` is the 1-based modality index the evaluator iterates
                # with, the same convention the banks use.  Without this a model
                # trained with three deltas would be evaluated reading one.
                modal = torch.full((x.shape[0],), max(int(mode) - 1, 0),
                                   dtype=torch.long, device=x.device)
            # The gate has to be applied here too: a model trained to correct
            # only one viewpoint has to be evaluated the same way, which is why
            # this path now needs camera ids at all.
            gate = self._view_gate(camids, x.shape[0], x.device)
            global_feat = self.base(x, modal_label=modal, delta_gate=gate)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        """Load a checkpoint produced by this class (transfer between datasets).

        The classifier is skipped on purpose: it is one row per training
        identity, so CARGO's and WHU-MARS's never match.  Everything else --
        backbone and bottleneck -- transfers as long as the architecture and
        the input size agree.

        Reports what happened rather than assuming.  The old version indexed
        self.state_dict() directly and copied in silence, so a checkpoint with
        the wrong prefix (a backbone-only file, say) raised KeyError on the
        first tensor, and a *partially* matching one loaded whatever fit and
        left the rest at random init without a word.
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
            # A checkpoint this class produced has no benign shape mismatch: the
            # architecture either matches the config or it does not.  Printing
            # and carrying on leaves the offending tensor at its initialisation,
            # and for `pos_delta` -- which is the whole method -- that means the
            # evaluation reports the baseline under the arm's name.  Shapes that
            # legitimately differ (the classifier, one row per identity) are
            # skipped by name well above this.
            raise RuntimeError(
                'shape mismatch loading {}: {}.  The checkpoint was trained under a '
                'different architecture than this config builds -- MODEL.PE_LAYERWISE, '
                'PE_PER_MODALITY, MOD_DELTA, ROPE_GATE and the text anchor count all '
                'change tensor shapes.  Loading anyway would leave those weights at '
                'their initialisation and report a number for the wrong model.'
                .format(trained_path, mismatched[:6]))
        if loaded == 0:
            raise RuntimeError(
                'nothing loaded from {} -- every key was unknown. This usually means the '
                'file is a backbone-only checkpoint (keys like `blocks.0...`), which needs '
                'MODEL.PRETRAIN_CHOICE "imagenet", not "self".'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_base_clip': vit_base_clip,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
