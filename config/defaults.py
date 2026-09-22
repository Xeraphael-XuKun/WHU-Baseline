from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'transformer'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
# Split the identity classes the CROSS ENTROPY sees, so that it no longer
# has to map a person's aerial thermal image and their ground RGB image
# onto the same logit.  `label = pid * slots + slot`, where `slot` encodes
# the view and/or the spectrum.  The TRIPLET keeps the original pid, on
# purpose: the alignment burden moves onto it entirely, which is both the
# point of the experiment and its main risk.
#
# Why: every image of an identity -- three spectra, two viewpoints -- is
# currently forced into one class, and satisfying that approximately is one
# reason the whole feature cloud collapses into a narrow cone (mean pairwise
# cosine 0.9819, against 0.6179 for the original CLIP).  The margin that
# decides retrieval is only 28% -- d_same 0.1489 against d_diffpid 0.1906.
#
# Costs nothing at inference: TEST.NECK_FEAT 'before' means retrieval reads
# the pre-BNNeck feature and the classifier is never called.  load_param
# already skips `classifier` keys unconditionally, so the changed width does
# not break transfer from the CARGO checkpoint either.
_C.MODEL.CE_SPLIT_VIEW = False
_C.MODEL.CE_SPLIT_MODALITY = False
# Which CE slot each modality falls into, parallel to DATASETS.MODALITIES.
# Empty means one slot per modality, which is what whu_ce_mod ran; [0, 0, 1]
# merges RGB and IR into one class and leaves Thermal its own.
#
# Why the grouping exists.  whu_ce_mod (10.49 / 34.77 against the baseline's
# 10.23 / 26.68) bought its within-spectrum gains by breaking cross-spectrum
# retrieval, and the damage is not spread evenly.  Measured on the twin column
# of diag/analyse_modality.py section [2b] -- where the pair is the same
# capture at the same instant, so the spectrum is the only difference -- the
# pure spectrum cost as a fraction of the identity budget went
#
#     RGB <-> IR        -16%  ->  +58%     (13.57 of the 17.50 lost mAP, 78%)
#     RGB <-> Thermal   105%  ->  113%     ( 3.93, 22%)
#
# RGB and IR were FREE to match before the split: the same capture in the two
# spectra sat CLOSER than two RGB images of one person from one camera.  The
# split is what made them expensive.  Thermal was already past the budget and
# barely moved.  So group them to keep the split that paid (Thermal alone,
# Th->Th 11.61 -> 17.30) and drop the one that cost.
_C.MODEL.CE_MODALITY_GROUPS = []
# Let MODEL.CE_MODALITY_GROUPS skip an index, e.g. [0, 0, 2]: three slots are
# allocated, only two are ever a target, and the middle 500 classifier columns
# receive nothing but negative gradient for the whole run.
#
# Off by default because a gap is almost always a typo, and one that costs a
# whole run: the head silently comes out wider than the label space.  Turning
# this on says the sparsity is deliberate.
#
# What it is for.  [0, 0, 1] and [0, 1, 2] differ in TWO ways -- what the labels
# mean, and how wide the head is (1000 against 1500 columns).  [0, 0, 2] holds
# the meaning of the first and the width of the second, so the two factors come
# apart.
#
# MODEL.IF_LABELSMOOTH has to stay 'off' for the dead columns to be dead.  With
# smoothing on they each receive eps/N of the target mass and are trained
# toward, which is the opposite of the intended setup.
_C.MODEL.CE_ALLOW_SPARSE_SLOTS = False
# Which text anchor each spectrum has to read as AFTER the correction, parallel
# to DATASETS.MODALITIES.  Empty means every spectrum is corrected towards
# MODEL.TEXT_MODALITY -- what whu_mcam ran -- and only applies under
# MODEL.TEXT_TARGET 'modality'.
#
#   []          RGB -> "color"   IR -> "color"      Thermal -> "color"
#   [0, 1, 0]   RGB -> "color"   IR -> "infrared"   Thermal -> "color"
#
# Why the second form exists.  It mirrors CE_MODALITY_GROUPS on the text axis,
# for the same measured reason: section [2b] of the diagnostics put the pure
# spectrum cost of RGB<->IR at MINUS 16% of the identity budget on the
# baseline -- the same capture in the two spectra sits CLOSER than two RGB
# images of one person from one camera.  Asking a text loss to align a pair
# that is already free is redundant work, and whu_ce_mod showed what happens
# when that pair is disturbed: 78% of the lost cross-modal mAP came from it.
# So [0, 1, 0] leaves RGB and IR alone -- their before and after cells become
# the same demand, "stay put" -- and asks only Thermal to move.
#
# The before cells are unchanged and still target each spectrum's own anchor,
# so the anti-collapse guard holds either way.
_C.MODEL.TEXT_MODALITY_TARGETS = []
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.PCA_LOSS_WEIGHT = 0.0
_C.MODEL.GPD_MOMENTUM = 0.2

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# ChartPE Parameter
# Positional encoding path: 'learnable' | 'rope_fixed' | 'chartpe'
#   learnable  : original additive pos_embed table (unchanged baseline)
#   rope_fixed : mixed 2D RoPE on Q/K with a fixed centered integer patch grid
#   chartpe    : same RoPE, coordinates generated per-image by TPCG
_C.MODEL.PE_TYPE = 'learnable'
# Keep the additive pos_embed table alongside rotary PE ("ape" variants in
# RoPE-ViT). Replacing the table outright discards the pretrained positional
# weights, which is costly when fine-tuning rather than pretraining from
# scratch; keeping it makes the rotary path purely additive capability.
_C.MODEL.PE_KEEP_APE = False
# TPCG bottleneck width (small on purpose: limits identity leakage into coords)
_C.MODEL.CHART_HIDDEN = 64
# Restricted global rotation / area-preserving scale limits
_C.MODEL.CHART_THETA_MAX_DEG = 30.0
_C.MODEL.CHART_SCALE_MAX = 0.35
# Instance-normalise TPCG's input so the chart head cannot see which sensor took
# the picture.  Only touches the copy fed to the chart generator; the backbone
# still receives the untouched patch tokens.
_C.MODEL.CHART_INPUT_NORM = False
# RoPE frequency base (10.0 = RoPE-ViT 'mixed' setting) and whether to train it
_C.MODEL.ROPE_THETA = 10.0
_C.MODEL.ROPE_FREQ_TRAINABLE = False
# Zero-initialised per-layer gain on the rotary phase -- the rotary counterpart
# of PE_LAYERWISE 'indep'.  With it on, alpha = 0 at step 0 means cos = 1 and
# sin = 0, so Q and K come out untouched and the model IS the baseline; the
# rotation then grows from zero and every gain is attributable to it alone.
# Off (the default) applies the rotation at full strength from the first step,
# which throws CLIP's pretrained attention off-distribution immediately -- that
# arm is worth running as a control, not as the method.
#
# Requires PE_TYPE != 'learnable' (there is no rotation to gate otherwise) and
# is only bit-identical to the baseline when PE_KEEP_APE is True, since
# dropping the pretrained position table is itself a change.
_C.MODEL.ROPE_GATE = False
# Patch coordinates start at 0 instead of being centred on the grid.  Only
# matters for CLS<->patch attention (patch<->patch sees differences only, and a
# constant shift cancels there), but RoPE-ViT trains with t_x = (t % end_x), so
# transferring its weights faithfully requires matching that origin.
_C.MODEL.PE_ZERO_BASED = False
# LayerScale (CaiT / DeiT-III `gamma_1`, `gamma_2`).  Required to load any of the
# *_LS checkpoints -- the rest of those weights were learned with it in place.
_C.MODEL.LAYER_SCALE = False
_C.MODEL.LAYER_SCALE_INIT = 1e-4

# Layer-wise positional residuals.  The baseline injects pos_embed once, before
# block 0; these modes re-inject a learned increment before every block while
# keeping the pretrained table as the anchor.  Only the increments are new
# parameters, and they start at zero, so step 0 reproduces the baseline exactly.
#
# Defined by what the residual stream CARRIES at block l -- which is not the
# same as what gets added there, because pre-norm blocks never normalise the
# stream, so every addition persists.  See TransReID._layerwise_deltas.
#   'none'  : baseline, single injection
#   'indep' : block l carries pos_embed + delta[l]              (独立叠加)
#   'chain' : block l carries pos_embed + delta[0..l] summed     (沿初始流转)
_C.MODEL.PE_LAYERWISE = 'none'
# WHICH blocks get an increment.  Zero-based positions in the transformer;
# empty means every block, which is what every run before this key did and is
# bit-identical to it.
#
# The table holds one row per entry here, not `depth` rows with some of them
# skipped: an unread row would be a parameter the optimizer carries and no
# forward touches, and `indep`'s cancellation term is "what the stream is
# already carrying", which is the previous INJECTED row.  So with [0, 3, 6, 9]
# blocks 0-2 carry pos_embed + delta[0], blocks 3-5 carry pos_embed + delta[1],
# and so on -- one correction held for a stretch of blocks.
#
# THE ABLATION THIS EXISTS FOR.  The delta is 1,188,864 parameters spread over
# twelve blocks and nothing has ever tested whether it needs to be everywhere:
#
#   [0]                    99,072  a second input positional table, in effect
#   [11]                   99,072  same size, other end -- the matched pair
#   [0, 3, 6, 9]          396,288
#   [0, 2, 4, 6, 8, 10]   594,432
#   []                  1,188,864  every block; whu_text_lam50_clip, 14.11
#
# [0] against [11] holds the parameter count fixed and moves only the position,
# so it separates "where" from "how many"; the four counts together say whether
# the answer is simply monotone in size.
_C.MODEL.PE_LAYERS = []
# How the text anchors supervise a row.
#
#   'contrast'  the 2x2.  Each supervised row is read TWICE -- once with the
#               increments gated off (`feat_before`) and once normally
#               (`feat_after`) -- and the four cells are
#                   A_before -> aerial   A_after -> ground
#                   G_before -> ground   G_after -> ground
#               Only A_after asks for a change; the other three ask for things
#               to stay put.  Because the two passes share images AND weights,
#               the only thing that can separate them is the positional delta,
#               which is what makes the loss a statement about the delta.
#
#   'direct'    one reading, one target: aerial rows read as ground, ground
#               rows read as ground.  No second forward.  The delta becomes
#               optional rather than load-bearing, so this is the mode that can
#               run with PE_LAYERWISE 'none' and answer "what is the text
#               supervision worth on its own".
#
# WHAT 'direct' GIVES UP.  A_before -> aerial is the only cell in the 2x2 whose
# target differs from the others, and it is therefore the entire brake against
# collapse: it says "still admit this is an aerial photo" while A_after says
# "read as ground".  Take it away and every supervised row wants the same
# anchor, which is trivially satisfied by mapping every feature onto one point.
# The contrastive form of text_align_loss does not help here -- pushing away
# from the other anchor is also satisfied by the collapse when no row wants
# that other anchor.  What is left holding the features apart is CE + triplet.
#
# This failure has happened before, on whu_mod_lam50: the anchors went rank-1
# and the arm lost 1.22 mAP.  The monitor is already in the log every
# LOG_PERIOD -- `d_diffpid` on the 768d geometry line.  If it falls toward
# d_same the run is collapsing, and that is itself the finding.
_C.MODEL.TEXT_SUPERVISION = 'contrast'
# Opt-in for the degenerate arm: TEXT_ALIGN with PE_LAYERWISE 'none' under
# 'contrast' supervision, where before and after are the same tensor and the
# aerial rows carry two opposite targets.  Refused by default because nothing
# in the log would say the 2x2 had quietly become "sit between the anchors".
_C.MODEL.TEXT_ALLOW_NO_ACTUATOR = False
# Freeze the inherited pos_embed table so gradients reach only the increments.
# Off by default: the baseline trains its table, and leaving it trainable keeps
# the layer-wise comparison a single-variable one.
_C.MODEL.PE_FREEZE_BASE = False
# Restrict the positional delta to one viewpoint: 'none' | 'aerial' | 'ground'.
#
# The delta is one set of parameters shared by every image, so under VTC the
# G_after cell is the only thing keeping it from dragging ground images around
# -- a soft guard, not a structural one.  Gating on 'aerial' makes the guard
# structural instead: ground images take the delta-free path in both the main
# forward and the before pass, so the delta contributes nothing to them either
# way -- by construction rather than by training.
#
# The logged `drift` still will not read exactly zero.  MODEL.DROP_PATH is 0.1
# and the two passes cover different row counts, so they draw different
# stochastic-depth masks; the residue is regularisation noise, not the delta
# moving ground features.  In eval mode, where drop-path is off, the two paths
# are bit-identical -- which is the form tests/test_view_gate.py pins.
#
# 'ground' is implementable but not interpretable under TEXT_ALIGN: the aerial
# rows would then take the same path in before and after, and the supervision
# table asks that one tensor to read as "aerial" and as "ground" at once.  That
# is the contradiction do_train already refuses when PE_LAYERWISE is 'none'.
#
# COST: a gated model needs to know each image's viewpoint AT INFERENCE, which
# the ungated one does not.  Camera ids are available in every ReID benchmark
# and the junk rule already uses them, but using them inside the model is a
# different claim from using them in the protocol; say so in the paper.
_C.MODEL.PE_VIEW_GATE = 'none'
# One independent layer-wise delta per spectrum instead of one shared by all.
#
# `pos_delta` grows a leading modality axis, so its shape goes from
# [depth, 1, N+1, D] to [n_modalities, depth, 1, N+1, D] -- 1.19 M parameters
# become 3.57 M -- and each image reads the row its spectrum indexes.  The
# indep differencing runs inside each row on its own, so "the net offset block
# l carries" is per spectrum, which is the thing being made independent.
#
# NOT the same as MODEL.MOD_DELTA.  That one keeps the shared delta and adds a
# correction for the non-reference spectra, leaving RGB with the shared one
# alone.  The two span the same function class, but this parametrisation gives
# each spectrum a delta that stands by itself, which is what makes "RGB reads
# its own" true of the parameters and not only of the arithmetic.
#
# WHAT IT COSTS.  Three times the new parameters, and a reviewer will ask
# whether the gain is routing or capacity.  There is no clean capacity control
# -- the shared delta's size is fixed by the architecture -- so the answer has
# to come from the trained norms: three rows that end up near-identical mean
# the routing bought nothing, and three that differ mean the spectra really did
# want different corrections.
#
# Requires the modality index at every forward, INCLUDING evaluation, the same
# way MODEL.PE_VIEW_GATE requires the camera id.
_C.MODEL.PE_PER_MODALITY = False

# ---- twin-anchored modality residuals ---------------------------------------
# One bank of layer-wise increments per non-reference modality (the actuator
# from direction one, aimed at the spectrum), trained so that an IR/Thermal
# image plus its bank lands on the feature of the SAME-CAPTURE RGB frame.  The
# target is the twin feature, not a text anchor: mod_lam50 established that a
# shared (rank-1) target either saturates or collapses -- the twin is a rank-N
# target that carries identity by construction.  The reference modality has no
# bank at all; its features are the coordinate frame, so "RGB stays RGB" is
# structural rather than a loss term.
_C.MODEL.MOD_DELTA = False
# Which modality is the frame everything else is corrected into.  v1 only
# supports the first entry of DATASETS.MODALITIES (bank row order is derived
# from that list); the key exists so the constraint is stated, not implied.
_C.MODEL.MOD_DELTA_REF = 'RGB'

# ---- view alignment against CLIP text anchors -------------------------------
# Two learnable sentences give a pair of fixed points in CLIP's joint space, one
# reading as an aerial viewpoint and one as a ground viewpoint.  Each supervised
# image is scored before and after the layer-wise increments are applied:
#
#                 before (no delta)     after (delta applied)
#   aerial image    -> aerial                -> ground
#   ground image    -> ground                -> ground
#
# Only the top-right cell asks for a change; the rest ask things to stay put.
# Off by default -- with TEXT_ALIGN False nothing here is constructed and the
# state_dict is identical to a run without it.
_C.MODEL.TEXT_ALIGN = False
# What the delta is asked to correct.
#
#   'view'      the original: aerial images read as ground afterwards.  The
#               view matrix later showed there was very little to correct --
#               A->G 10.81 against G->G 10.70 on the same gallery -- which is
#               why the correction fired (margin crossed zero, push rose
#               monotonically with lambda) yet mAP moved 0.09.
#   'modality'  the same machinery aimed at the gap that is actually there:
#               cross-thermal cells sit at 40% of the thermal diagonal, while
#               RGB<->IR loses only 6%.  Infrared and thermal images read as
#               their own spectrum with the delta off and as RGB with it on;
#               RGB reads as RGB either way.
#
# Under 'modality' the first template slot holds the spectrum word rather than
# the viewpoint word, MODEL.TEXT_MODALITY_WORDS supplies one anchor per
# spectrum, and every image in the batch is supervised instead of one third.
_C.MODEL.TEXT_TARGET = 'view'
# Path to the ORIGINAL CLIP release.  Deliberately separate from PRETRAIN_PATH:
# on the WHU-MARS legs that points at our own CARGO checkpoint, which carries
# only the image tower.
_C.MODEL.TEXT_CLIP_PATH = ''
# Number of shared learnable context vectors in the "[X...]" slot.
_C.MODEL.TEXT_N_CTX = 4
# Which modality gets the text supervision.  CLIP was trained on RGB web
# images; whether an infrared or thermal crop "looks like a ground viewpoint"
# is a question its frozen text tower has no grounding to answer.
_C.MODEL.TEXT_MODALITY = 'RGB'
# The placeholder 'X' marks every learnable slot -- one for the view word, then
# TEXT_N_CTX shared ones.  Changing this moves the token positions, which
# ViewPrompts recomputes; the tokenizer must still map 'X' to a single token.
_C.MODEL.TEXT_TEMPLATE = 'a photo of a X view person with X X X X .'
# Naming the spectra adds a second learnable slot and one anchor per
# (modality, view) pair -- six instead of two -- and supervises all three
# modalities instead of TEXT_MODALITY alone.  Motivation: at two classes the
# task saturates by epoch 60 (all four accuracies 1.00, loss 0.000) and a
# saturated loss teaches nothing; six classes also let the correction be read
# per spectrum.  Costs a second forward pass over the whole batch rather than
# a third of it.  Empty = keep the two-anchor form.
# Must line up with DATASETS.MODALITIES entry for entry, and every word has to
# be a single CLIP token so it can seed its slot.
_C.MODEL.TEXT_MODALITY_WORDS = []

# _C.MODEL.PRETRAIN_HW_RATIO = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('MARS')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# Ordered modality folder names under each split directory, e.g. train/<modality>/*.jpg
_C.DATASETS.MODALITIES = ('RGB', 'IR', 'Thermal')
# Evaluation protocol, for datasets that define several.
#   CARGO      ALL / AA / GG / AG
#   WHU-MARS   ALL / GD.  GD is the paper's WHU-MARS-1000-GD column: the SAME
#              trained model, evaluated on ground + daytime queries and
#              galleries only.  It filters query and gallery, never train, so
#              it needs no retraining -- reeval.sh on an existing checkpoint.
_C.DATASETS.PROTOCOL = 'ALL'
# Which split directory to read under DATASETS.ROOT_DIR.  Empty means the
# dataset class's own default ('WHU-MARS', the 1,000-identity split).  Set it to
# 'WHU-MARS-2337' for the larger one.  A config key rather than an edit to
# whu_mars.py: the split is part of the experiment and has to show up in the
# log, and two runs that differ only by a commented-out line are impossible to
# tell apart afterwards.
_C.DATASETS.SUBDIR = ''
# Zero-based camera ids shot from the air; everything else counts as ground.
# WHU-MARS: c6/c7 -> [5, 6].  Confirmed by geometry rather than inferred --
# median crops are 36x57 and 36x31 against 58-68 x 133-173 on the ground, and
# c7's aspect ratio of 1.16 (wider than tall) only happens looking near-straight
# down; a standing person is taller than wide from every oblique angle.
_C.DATASETS.AERIAL_CAMS = []


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'PKM'
# Draw the three modalities of a training triplet from the SAME capture instead
# of three unrelated frames of the same person. WHU-MARS names files
# `{pid}_c{cam}_m{modality}_f{frame}.jpg`, so the pairing is recoverable; the
# released baseline never uses it on AS-ReID.
_C.DATALOADER.SYNC_FRAMES = False
# Replay one augmentation draw across a capture's three modalities instead of
# drawing per image.  Only meaningful with SYNC_FRAMES: the point of that flag
# is that the triplet differs only in spectrum, and the shipped pipeline
# (RandomHorizontalFlip p=0.5, RandomCrop over Pad 10, RandomErasing p=0.5)
# quietly reintroduces a difference -- at p=0.5, half of the triplets come out
# mirrored relative to each other.  A ranking loss tolerates that; a regression
# onto the reference-modality feature does not, and a per-modality constant
# cannot undo a mirror.  Off by default so every run recorded so far keeps its
# exact input distribution.
_C.DATALOADER.TIE_AUGMENTATION = False
# Whether the random ERASING is tied along with the rest.  True keeps the
# behaviour every run so far had; False shares the crop and the flip but lets
# each spectrum draw its own erased rectangle.
#
# Why the distinction is worth a key.  Tying the crop and the flip is what
# makes a twin pair differ only in spectrum, which is the point.  Tying the
# erasing goes further than intended: the erased region is filled with a
# constant, so both images end up carrying a pixel-for-pixel IDENTICAL patch,
# and the gap the twin loss exists to close is shrunk by hand.  Measured, the
# twin cosine reads 0.9932 in training against 0.9880 on clean test features.
#
# The three whu_tnce runs dodged that by setting INPUT.RE_PROB to 0, which
# removed a regulariser the whole recipe was tuned with and left their 2.1-mAP
# deficit against the baseline unattributable.  Untying is the fix that does
# not touch the recipe: the augmentation distribution stays exactly what every
# control used, and only the correlation between the three spectra changes.
_C.DATALOADER.TIE_ERASING = True
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# WHU-MARS view-stratified PKM sampler. These options are ignored by the
# original PKM path, so A_baseline_candidate keeps the teacher snapshot's
# sampler behavior unchanged.
_C.DATALOADER.PKM_VIEW = CN()
_C.DATALOADER.PKM_VIEW.NUM_CROSS_VIEW_PIDS = 8
_C.DATALOADER.PKM_VIEW.NUM_GROUND_ONLY_PIDS = 8
# Camera ids are zero-based after dataset parsing: c6/c7 -> 5/6.
_C.DATALOADER.PKM_VIEW.AERIAL_CAM_IDS = [5, 6]

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Learning rate multiplier for the TPCG chart generator (1.0 = same as backbone)
_C.SOLVER.CHART_LR_MULT = 1.0
# Learning rate multiplier for the layer-wise positional residuals (pos_delta).
# They start at zero while the backbone starts pretrained, so a larger value is
# a reasonable thing to try if the increments never move off the floor.
_C.SOLVER.PE_DELTA_LR_MULT = 1.0
# Weight on the view-alignment loss.  It is a two-class cross entropy (~0.69 at
# chance) against an identity loss over hundreds of classes (~6.2 at init,
# ~4.7 by the second epoch), so 1.0 puts it at roughly 14% of the total, not at
# parity.  0.0 disables it even when MODEL.TEXT_ALIGN is on.
_C.SOLVER.TEXT_LOSS_WEIGHT = 0.0
# Weight on the twin-anchored modality loss (1 - cos to the same-capture
# reference-modality feature).  Needs DATALOADER.SYNC_FRAMES -- without it rows
# i and i+64 are the same person at unrelated moments and the loss would pull
# those together, a different experiment entirely; the processor hard-fails on
# that combination.  Magnitude note: with L2-normalised features at d_same ~
# 0.148, the cross-thermal twin term starts around (0.148*1.125)^2/2 ~ 0.014,
# against CE+triplet at ~0.2-0.6 near convergence, hence the default sweep
# value of 25 in the config rather than 1.
_C.SOLVER.TWIN_LOSS_WEIGHT = 0.0
# Weight on the twin CONTRASTIVE loss (loss/twin_infonce.py).  Separate key
# from TWIN_LOSS_WEIGHT above because the two are different shapes and their
# magnitudes are not comparable: that one scores 1 - cos, which starts near
# 0.014 and wants a weight of ~25; this one is a cross entropy over 61
# candidates, starts near 1.8 at TNCE_TAU 0.003, and reaches parity with
# CE+triplet at 1.0.  Also unlike that one it does NOT require MODEL.MOD_DELTA
# -- the whole point is that the correction comes from the backbone, which a
# per-modality constant cannot express.  Needs DATALOADER.SYNC_FRAMES.
_C.SOLVER.TNCE_WEIGHT = 0.0
# Temperature.  Measured, not conventional: diag/twin_infonce_probe.py put the
# cosine gap between a twin and the hardest of 60 strangers at 0.0026, so 0.003
# turns that into a logit gap of order one.  The 0.05-0.07 of the contrastive
# literature leaves the loss 4.5% below its ceiling -- a flat softmax with
# effectively no gradient, which is how whu_sync's triplet failed.
_C.SOLVER.TNCE_TAU = 0.003
# Epochs over which TNCE_WEIGHT ramps linearly from 0.  0 disables the ramp.
#
# Why it exists.  TNCE_WEIGHT was sized against the loss a CONVERGED model
# produces -- diag/twin_infonce_probe.py measured 1.81 on whu_recipe_hihr's
# final weights -- but training starts from the CARGO checkpoint, where the
# twin ranks BELOW the average stranger and the loss opens at 10.53, six times
# higher.  At weight 1.0 that put the term at ~40% of the total in epoch 1 and
# left whu_tnce 18% behind the baseline at epoch 10, with every cell down
# 16-24% and only the cross-thermal four up 2.8%.
#
# A ramp addresses that directly: CE and the triplet establish an identity
# space first, and the alignment pressure arrives once the twin loss has fallen
# to the magnitude the weight was chosen for.  It also makes the early epochs
# comparable to the baseline cell for cell, so a deficit there means a bug
# rather than a trade-off.
_C.SOLVER.TNCE_WARMUP_EPOCHS = 0
# Freeze every parameter except the modality banks.  This is v1's central
# claim made mechanical: with the backbone untouched, an evaluation with
# TEST.MOD_DELTA False is the baseline bit for bit, and any cross-modality
# change is attributable to the banks alone.
_C.SOLVER.MOD_DELTA_ONLY = False
# Freeze the pretrained tower and leave `pos_delta` as its only trainable
# parameter; the fresh head (BNNeck + classifier) keeps training.
#
# This is the BLUNT strict-attribution arm, and it answers a narrower question
# than it looks like it does: nothing trains the tower, identity losses
# included, so the run sits at ~2.4 mAP and is not on the same scale as
# anything else in the tables.  SOLVER.TEXT_GRAD_SCOPE 'delta' is the sharp
# version -- CE and triplet train the tower as usual and only the TEXT loss is
# kept out of it -- and it is the one to reach for.  Superseded, kept because
# the runs exist.
#
# The text loss reaches the backbone through `feat_after`, on the same forward
# pass `pos_delta` is on, so no amount of *detaching* separates them.  That is
# where this arm's reasoning stopped, and it stopped one step too early:
# detaching cuts the forward graph, but the routing can be done in the backward
# instead (torch.autograd.grad against a chosen parameter list), which separates
# them exactly.  See SOLVER.TEXT_GRAD_SCOPE.
#
# With the tower frozen, VTC's only trainable target is `pos_delta`: `clip_proj` sits
# under `base.` and is frozen with it, the prompts are frozen here too, the
# text encoder was already frozen, and `feat_before` carries no gradient at all
# because its gate is zero.
#
# The head stays trainable on purpose.  These runs start from raw CLIP, so
# freezing the classifier would lock it at its random initialisation and the
# identity loss would be measuring nothing.  (SOLVER.MOD_DELTA_ONLY can freeze
# everything because that arm starts from a trained checkpoint.)
#
# Absolute mAP is not comparable to a trainable-tower run; the readout is the
# gap against an arm with the same freeze and TEXT_ALIGN off.
_C.SOLVER.PLD_ONLY = False
# Which parameters the TEXT loss is allowed to move.  CE and triplet are never
# affected by this key: they always reach everything that is trainable.
#
#   'all'     the text term is added into the total loss and backpropagated
#             with everything else.  What every run before this key did.
#   'delta'   the text term's gradient goes to `pos_delta` and the prompt
#             vectors ONLY.  The ViT blocks, patch_embed, pos_embed and
#             clip_proj receive nothing from it, while still being trained
#             normally by the identity losses.
#
# WHY IT IS NOT A DETACH.  `pos_delta` lives inside the tower -- it is injected
# between blocks -- so any graph cut that keeps the text loss off block 7 also
# keeps it off the delta.  The separation has to happen in the backward pass:
# torch.autograd.grad(text_loss, [delta, prompts]) computes that one loss's
# partials for that one list and writes nothing anywhere else, and the identity
# losses then back-propagate normally over the same retained graph.
#
# WHY THE PROMPTS COUNT AS "NOT THE BACKBONE".  view_ctx / modality_ctx /
# shared_ctx are the text loss's own apparatus -- they exist only because
# TEXT_ALIGN is on and they feed the frozen text tower, never the image path.
# Freezing them would pin the anchors at initialisation and change what the
# loss even is.  `prompts.text.*` (the CLIP text encoder) is excluded: it is
# frozen at construction and is not ours to move.
#
# READ IT AGAINST two arms that already exist: the same recipe with TEXT_ALIGN
# off (whu_pe_indep_clip, 13.29) and with the text gradient unrestricted
# (whu_text_lam50_clip, 14.11).  How much of that 0.82 survives is the answer
# to "can the correction be carried by the delta alone".
_C.SOLVER.TEXT_GRAD_SCOPE = 'all'
# Separate learning rate for the pretrained backbone (`base.*`).  HiHR (and
# CLIP-ReID before it) run the pretrained tower two orders of magnitude slower
# than the freshly initialised head -- 5e-6 against 3.5e-4 -- because a
# well-fitted checkpoint is easy to wreck in the first few hundred steps.
# 0.0 disables the split: everything gets BASE_LR, which is the old behaviour.
# Zero-initialised parameters that happen to live inside the backbone
# (pos_delta) are treated as new modules, not as pretrained weights.
_C.SOLVER.PRETRAINED_LR = 0.0
# Warm up over iterations instead of epochs.  HiHR uses "a warm-up strategy
# with 100 iterations before the cosine learning rate schedule"; at ~1.9k
# iterations per epoch that is 0.05 of an epoch, which epoch-granularity
# stepping simply cannot express.  > 0 switches the scheduler to counting
# updates; 0 keeps the old WARMUP_EPOCHS behaviour.
_C.SOLVER.WARMUP_ITERS = 0
# Cosine floor and warmup start, as fractions of each parameter group's own
# peak lr.  The defaults reproduce the hard-coded 0.002 / 0.01 the factory used
# before.  With a two-tier lr these have to be *fractions*: a single absolute
# floor would sit at 0.2% of the head's peak but 14% of the backbone's, giving
# the two groups differently shaped schedules for no reason.
_C.SOLVER.LR_MIN_FACTOR = 0.002
_C.SOLVER.WARMUP_LR_FACTOR = 0.01
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# cuDNN convolution autotuner. False is the reproducible A baseline setting;
# True reproduces the teacher train.py setting for the matched control.
_C.SOLVER.CUDNN_BENCHMARK = False
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

_C.SOLVER.LOSS_TYPE = "base"

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# 0 for none top_k evaluation, otherwise will only evaluate top_k results for mAP
_C.TEST.TOP_K_EVAL= 0
# Apply the per-modality banks at evaluation (the corrected, single-frame
# reading).  False evaluates the same weights with no correction -- under
# SOLVER.MOD_DELTA_ONLY that is the frozen baseline exactly, which is the
# theorem check: it must reproduce whu_recipe_hihr's numbers bit for bit.
_C.TEST.MOD_DELTA = True
# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# Which gallery entries count as junk for a query.  See utils.metrics.junk_mask.
#   'sysu'   : drop the query's whole camera (what WHU-MARS has always used)
#   'market' : drop only the query's own identity from that camera (Market-1501,
#              and what fast-reid scores CARGO and the AG-ReID benchmarks with)
# Getting this wrong does not raise -- it just reports a different number.
_C.TEST.METRIC = 'sysu'
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
