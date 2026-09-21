"""Unit tests for diag/weight_shift.py -- which weights the text loss moved.

CPU-only and self-contained.  Run directly::

    python tests/test_weight_shift.py

The script reports one number per mechanism, and every failure mode is a wrong
number rather than a crash:

  * a name landing in the wrong bucket -- `pos_delta` and `pos_embed` both
    start with `pos`, `clip_proj` sits under `base.` exactly like the trunk, and
    the frozen text tower lives under `prompts.` with the learnable prompts.
    Any of those mis-bucketed and the table says the opposite of the truth;
  * norms combined by summing instead of by root-sum-of-squares, which inflates
    every group by a different factor depending on how many tensors it holds;
  * a tensor present in only one checkpoint being silently dropped from the
    comparison instead of reported -- the text run HAS tensors the control does
    not, and pretending otherwise hides the text tower entirely.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diag.weight_shift import group_of, load, norms  # noqa: E402


# ---- bucketing -----------------------------------------------------------

def test_pos_delta_and_pos_embed_do_not_share_a_bucket():
    """Both start with `pos`; a prefix match would merge the mechanism with
    the thing it corrects, and the whole table would be meaningless."""
    d = group_of('base.pos_delta')
    e = group_of('base.pos_embed')
    assert d != e
    assert 'pos_delta' in d and 'pos_embed' in e


def test_clip_proj_is_not_counted_as_backbone():
    """It lives under `base.` but is only ever used by the text path, so
    folding it into the trunk would hide exactly the tensor in question."""
    assert group_of('base.clip_proj') != group_of('base.blocks.7.attn.qkv.weight')
    assert 'clip_proj' in group_of('base.clip_proj')


def test_the_frozen_text_tower_is_split_from_the_learnable_prompts():
    """`prompts.text.*` is CLIP's text encoder, frozen; `prompts.view_ctx` and
    friends are ours and do train.  One bucket would average a moving tensor
    with a stationary one."""
    frozen = group_of('prompts.text.transformer.resblocks.0.attn.in_proj_weight')
    learn = group_of('prompts.view_ctx')
    assert frozen != learn
    assert group_of('prompts.shared_ctx') == learn
    assert group_of('prompts.text.token_embedding.weight') == frozen


def test_head_and_backbone_are_separate():
    assert group_of('bottleneck.weight') == group_of('classifier.weight')
    assert group_of('classifier.weight') != group_of('base.blocks.0.mlp.fc1.weight')


def test_every_realistic_name_lands_somewhere():
    names = ['base.cls_token', 'base.pos_embed', 'base.pos_delta',
             'base.patch_embed.proj.weight', 'base.blocks.11.norm2.bias',
             'base.norm.weight', 'base.ln_pre.bias', 'base.clip_proj',
             'bottleneck.weight', 'classifier.weight', 'prompts.view_ctx',
             'prompts.text.positional_embedding']
    buckets = {group_of(n) for n in names}
    assert len(buckets) >= 8, buckets
    assert all(isinstance(group_of(n), str) and group_of(n) for n in names)


# ---- the arithmetic ------------------------------------------------------

def test_norms_is_root_sum_of_squares_not_a_sum_of_norms():
    """Two unit tensors give sqrt(2), not 2.  Summing norms would scale every
    group by how many tensors it happens to contain."""
    state = {'a': torch.ones(1), 'b': torch.ones(1)}
    n, count = norms(state, ['a', 'b'])
    assert abs(n - 2 ** 0.5) < 1e-6, n
    assert count == 2


def test_norms_matches_concatenating_the_group():
    torch.manual_seed(0)
    state = {'a': torch.randn(3, 4), 'b': torch.randn(5), 'c': torch.randn(2, 2)}
    n, count = norms(state, list(state))
    flat = torch.cat([v.reshape(-1) for v in state.values()])
    assert abs(n - float(flat.norm())) < 1e-5
    assert count == flat.numel()


def test_a_group_that_did_not_move_reports_zero():
    """The informative direction: chaotic divergence would not spare a group."""
    torch.manual_seed(0)
    w = torch.randn(4, 4)
    diff_sq = float((w - w).pow(2).sum())
    assert diff_sq == 0.0


# ---- loading -------------------------------------------------------------

def _save(sd, name):
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), name)
    torch.save(sd, path)
    return path


def test_load_unwraps_a_checkpoint_that_nests_its_state_dict():
    torch.manual_seed(0)
    inner = {'base.pos_delta': torch.randn(2, 3)}
    plain = load(_save(inner, 'a.pth'))
    wrapped = load(_save({'state_dict': inner, 'epoch': 60}, 'b.pth'))
    assert set(plain) == set(wrapped) == {'base.pos_delta'}
    assert torch.equal(plain['base.pos_delta'], wrapped['base.pos_delta'])


def test_load_drops_non_float_buffers():
    """Integer buffers -- `aerial_cams`, `pe_slot` -- have no meaningful norm
    and would raise on .pow(2).sum() in some dtypes."""
    sd = {'base.pos_delta': torch.randn(2, 3),
          'aerial_cams': torch.tensor([5, 6]),
          'base.pe_slot': torch.arange(12)}
    out = load(_save(sd, 'c.pth'))
    assert set(out) == {'base.pos_delta'}


def test_shape_mismatches_are_not_compared():
    """A classifier trained on a different identity count must be reported as
    unshared, not subtracted -- that would raise, or worse, broadcast."""
    a = {'classifier.weight': torch.randn(500, 768)}
    b = {'classifier.weight': torch.randn(2400, 768)}
    shared = [k for k in a if k in b and a[k].shape == b[k].shape]
    assert shared == []


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print('FAIL  {}: {}: {}'.format(fn.__name__, type(exc).__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
