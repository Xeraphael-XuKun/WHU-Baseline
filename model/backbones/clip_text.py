"""CLIP's text tower, plus the two learnable view prompts built on top of it.

The tower is frozen and only ever sees two sentences, so it costs one forward
over 2 x 77 tokens per training step regardless of batch size.  What it
produces is a pair of fixed points in CLIP's 512-d joint space:

    "a photo of a [V_aerial] view person with [X1..X4]."   ->  T_aerial
    "a photo of a [V_ground] view person with [X1..X4]."   ->  T_ground

The two sentences differ in exactly one slot, so ``T_ground - T_aerial`` is a
pure viewpoint direction with nothing else mixed in.  That is the whole point:
the loss then asks an aerial image to sit on the aerial side before the
positional delta is applied and on the ground side after it, and the only thing
that can produce that displacement is the delta itself.

Everything here is verified against the real ViT-B-16.pt rather than assumed:
the vocabulary is 49,408 entries (exactly the row count of
``token_embedding.weight``), the template tokenises to 15 ids with the
learnable slots at positions [5, 9, 10, 11, 12] and the end-of-text marker at
14, and both `aerial` and `ground` are single tokens (12440 / 2461) so [V] can
be initialised from their real embeddings.
"""

import os

import torch
import torch.nn as nn

# Token positions in the tokenised template, measured with the vendored
# tokenizer rather than counted by hand.  A different template moves them, so
# ViewPrompts recomputes them at construction and only falls back to these as
# the expected value in tests.
TEMPLATE = 'a photo of a X view person with X X X X .'
PLACEHOLDER = 'X'
SOT, EOT = 49406, 49407
CONTEXT_LENGTH = 77


class QuickGELU(nn.Module):
    """CLIP's activation; see the note in vit_pytorch.py -- not nn.GELU."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """One text-tower block, named to match the checkpoint's own keys."""

    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), QuickGELU(), nn.Linear(width * 4, width))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x, attn_mask):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
        return x + self.mlp(self.ln_2(x))


class CLIPTextEncoder(nn.Module):
    """OpenAI CLIP's text tower, frozen.

    Transcribed from clip/model.py.  Two details matter and neither raises if
    wrong: the attention is **causal** (an upper-triangular -inf mask, because
    CLIP trained it autoregressively), and the sentence vector is read at the
    position of the end-of-text token, not at position 0 and not by pooling.
    """

    def __init__(self, width=512, layers=12, heads=8, vocab=49408,
                 context_length=CONTEXT_LENGTH, embed_dim=512):
        super().__init__()
        self.width, self.context_length = width, context_length
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads)
                                        for _ in range(layers)])
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        mask = torch.full((context_length, context_length), float('-inf'))
        self.register_buffer('attn_mask', torch.triu(mask, diagonal=1), persistent=False)

        # torch.empty hands back whatever was in the allocation, which in
        # practice is zeros -- and a zero text_projection makes every sentence
        # come out as the zero vector, identically, with nothing raising.  Use
        # CLIP's own initialisation so an unloaded tower is at least
        # well-formed, and track whether the real weights ever arrived.
        self._init_like_clip(width, layers)
        self.loaded = False
        # Frozen from construction, not from load_clip: the tower is 63M
        # parameters and make_optimizer sweeps up everything with
        # requires_grad, so a tower that is only frozen on the load path would
        # quietly join the optimizer whenever that path is skipped.
        self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def _init_like_clip(self, width, layers):
        """clip/model.py :: CLIP.initialize_parameters, text half."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (width ** -0.5) * ((2 * layers) ** -0.5)
        attn_std, fc_std = width ** -0.5, (2 * width) ** -0.5
        for blk in self.resblocks:
            nn.init.normal_(blk.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(blk.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(blk.mlp[0].weight, std=fc_std)
            nn.init.normal_(blk.mlp[2].weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=width ** -0.5)

    def forward(self, embeddings, eot_index):
        """embeddings: [N, ctx, width] (already looked up and spliced)."""
        x = embeddings + self.positional_embedding.to(embeddings.dtype)
        x = x.permute(1, 0, 2)                       # NLD -> LND, as CLIP does
        for blk in self.resblocks:
            x = blk(x, self.attn_mask.to(x.dtype))
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eot_index]
        return x @ self.text_projection

    def load_clip(self, param_dict):
        """Load the non-`visual.` half of a CLIP checkpoint.  Mirrors
        TransReID._load_clip_visual; every name was read off the real file."""
        pairs = [('token_embedding.weight', 'token_embedding.weight'),
                 ('positional_embedding', 'positional_embedding'),
                 ('ln_final.weight', 'ln_final.weight'),
                 ('ln_final.bias', 'ln_final.bias'),
                 ('text_projection', 'text_projection')]
        for n in range(len(self.resblocks)):
            src, dst = 'transformer.resblocks.{}.'.format(n), 'resblocks.{}.'.format(n)
            for suffix in ('attn.in_proj_weight', 'attn.in_proj_bias',
                           'attn.out_proj.weight', 'attn.out_proj.bias',
                           'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias'):
                pairs.append((src + suffix, dst + suffix))
            pairs.append((src + 'mlp.c_fc.weight', dst + 'mlp.0.weight'))
            pairs.append((src + 'mlp.c_fc.bias', dst + 'mlp.0.bias'))
            pairs.append((src + 'mlp.c_proj.weight', dst + 'mlp.2.weight'))
            pairs.append((src + 'mlp.c_proj.bias', dst + 'mlp.2.bias'))

        own, count = self.state_dict(), 0
        for src, dst in pairs:
            if src not in param_dict:
                raise RuntimeError('CLIP checkpoint is missing {}'.format(src))
            v = param_dict[src]
            if own[dst].shape != v.shape:
                raise RuntimeError('shape mismatch {} {} -> {} {}'.format(
                    src, tuple(v.shape), dst, tuple(own[dst].shape)))
            own[dst].copy_(v)
            count += 1
        self.freeze()                      # already frozen; kept explicit
        self.loaded = True
        print('CLIP text tower: loaded %d / %d tensors, frozen' % (count, len(pairs)))
        return count


class ViewPrompts(nn.Module):
    """The view sentences, with learnable slots spliced into a fixed template.

    Two shapes, chosen by whether `modality_words` is given:

    ``None``   two anchors, one per view.  Template
               "a photo of a [V] view person with [X...]".
    given      one anchor per (modality, view) pair -- six for three
               modalities.  Template
               "a photo of a [V] view [M] person with [X...]".

    The six-anchor form turns a two-class problem into a six-class one.  That
    matters because at two classes the task saturates almost immediately (all
    four accuracies hit 1.00 and the loss reaches 0.000 by epoch 60), after
    which the supervision has nothing left to teach.  It also lets the view
    correction be read per modality rather than as one average.

    Row order is modality-major: index = modality * 2 + view, so with the
    default words rows are (aerial RGB, ground RGB, aerial IR, ground IR, ...).

    The token *ids* never change, only the embeddings at the placeholder
    positions, so the end-of-text position stays put and the sentence vector
    keeps being read from the right place.

    Every slot that has a real word starts from that word's embedding -- CLIP
    already places `aerial`, `ground`, `rgb`, `infrared` and `thermal`
    sensibly, and all five happen to be single tokens.  The shared slots have
    no word to start from and get N(0, 0.02), as CoOp does.
    """

    def __init__(self, text_encoder, template=TEMPLATE, n_ctx=4,
                 view_words=('aerial', 'ground'), modality_words=None,
                 tokenizer=None, slot_name='view'):
        super().__init__()
        # `view_words` is really "the axis being corrected": the first
        # placeholder in the template.  Under MODEL.TEXT_TARGET 'modality' it
        # holds the spectrum words and there is no second axis, so `slot_name`
        # exists to keep the printed line honest -- "3 anchors (3 views x 1
        # modalities)" would be read wrong six months from now.
        self.slot_name = slot_name
        self.text = text_encoder
        self.n_ctx = n_ctx
        self.view_words = tuple(view_words)
        self.modality_words = tuple(modality_words) if modality_words else None
        self.n_view = len(self.view_words)
        self.n_modality = len(self.modality_words) if self.modality_words else 1
        self.n_anchor = self.n_view * self.n_modality

        tok = tokenizer if tokenizer is not None else build_tokenizer()
        ph = tok.encode(PLACEHOLDER)
        if len(ph) != 1:
            raise RuntimeError(
                'the placeholder {!r} takes {} tokens; it must take exactly one, '
                'or the learnable slots land at the wrong positions'
                .format(PLACEHOLDER, len(ph)))
        ph = ph[0]

        ids = [SOT] + tok.encode(template) + [EOT]
        slots = [i for i, t in enumerate(ids) if t == ph]
        want = 1 + (1 if self.modality_words else 0) + n_ctx
        if len(slots) != want:
            raise RuntimeError(
                'template has {} placeholders but this configuration needs {} '
                '(1 view{} + {} shared)'.format(
                    len(slots), want,
                    ' + 1 modality' if self.modality_words else '', n_ctx))
        self.view_slot = slots[0]
        self.modality_slot = slots[1] if self.modality_words else None
        self.ctx_slots = slots[2:] if self.modality_words else slots[1:]
        self.eot_index = len(ids) - 1

        padded = ids + [0] * (text_encoder.context_length - len(ids))
        self.register_buffer('token_ids', torch.tensor(padded).long(), persistent=False)

        width = text_encoder.width

        def from_words(words):
            with torch.no_grad():
                table = text_encoder.token_embedding.weight
                out = []
                for w in words:
                    t = tok.encode(w)
                    if len(t) != 1:
                        raise RuntimeError(
                            '{!r} takes {} tokens; a slot holds exactly one vector, so '
                            'it cannot be initialised from a multi-token word'
                            .format(w, len(t)))
                    out.append(table[t[0]].clone())
                return torch.stack(out)

        self.view_ctx = nn.Parameter(from_words(self.view_words))       # [n_view, width]
        self.modality_ctx = (nn.Parameter(from_words(self.modality_words))
                             if self.modality_words else None)          # [n_mod, width]
        self.shared_ctx = nn.Parameter(torch.empty(n_ctx, width).normal_(std=0.02))

        learn = [self.view_slot] + ([self.modality_slot] if self.modality_words else []) \
            + self.ctx_slots
        print('Text prompts: %d anchors (%d %ss x %d modalities), %d learnable '
              'slots at %s, EOT at %d' % (self.n_anchor, self.n_view,
                                          self.slot_name, self.n_modality,
                                          len(learn), learn, self.eot_index))

    def anchor_index(self, modality, view):
        """(modality, view) -> row of forward()'s output.  Modality-major."""
        return modality * self.n_view + view

    def forward(self):
        """-> [n_anchor, embed_dim], modality-major (see anchor_index)."""
        n = self.n_anchor
        ids = self.token_ids.unsqueeze(0).expand(n, -1)
        emb = self.text.token_embedding(ids).clone()             # [n, ctx, width]
        # Row r stands for modality r // n_view and view r % n_view.
        view_of = torch.arange(n, device=emb.device) % self.n_view
        emb[:, self.view_slot] = self.view_ctx[view_of]
        if self.modality_ctx is not None:
            mod_of = torch.arange(n, device=emb.device) // self.n_view
            emb[:, self.modality_slot] = self.modality_ctx[mod_of]
        emb[:, self.ctx_slots] = self.shared_ctx.unsqueeze(0).expand(n, -1, -1)
        eot = torch.full((n,), self.eot_index, dtype=torch.long, device=emb.device)
        return self.text(emb, eot)


def build_tokenizer():
    """The vendored CLIP tokenizer, with a readable error when ftfy is absent."""
    try:
        from .clip_tokenizer import SimpleTokenizer
    except ImportError:                                          # pragma: no cover
        from clip_tokenizer import SimpleTokenizer               # direct-run fallback
    vocab = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'bpe_simple_vocab_16e6.txt.gz')
    if not os.path.exists(vocab):
        raise RuntimeError('missing {} -- it ships alongside clip_tokenizer.py'.format(vocab))
    return SimpleTokenizer(vocab)
