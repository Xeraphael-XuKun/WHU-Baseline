"""Unit tests for the Stage 0 diagnostics.

These run on the dev machine, which has no `ftfy` and therefore no real CLIP
tokenizer, so everything here uses a stub tokenizer and a tiny randomly
initialised text tower.  That is enough for what the tests are for: the
arithmetic, the guards, and above all the SENTENCE TABLE itself.

The sentence table is the part most likely to be wrong in a way nothing
notices.  A typo in one word, a group whose sentences differ in two places
instead of one, a STRIP entry naming a word that is not actually in its
sentence -- each of those produces a perfectly plausible number that means
something other than what the report claims it means.  The tests below check
the table as data.

Run:  python tests/test_prompt_probe.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diag.prompt_probe import (CALIBRATION, GROUPS, STRIP,  # noqa: E402
                               encode, off_diagonal, pairwise, strip_word)
from model.backbones.clip_text import CLIPTextEncoder  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


# 49408 because `encode` prepends SOT (49406) and appends EOT (49407); a
# smaller table would raise IndexError on every sentence.  Only the embedding
# is full size -- width 32 and 2 layers keep the tower at a couple of MB.
VOCAB = 49408
WORD_IDS = 900          # the stub's own words stay well below SOT/EOT


class StubTokenizer:
    """Deterministic word -> id, stable across calls and processes.

    Not a hash(): Python randomises str hashing per process, so a test that
    passed once could fail the next run for no reason at all.
    """

    def encode(self, s):
        out = []
        for w in s.split():
            n = 7
            for ch in w:
                n = (n * 131 + ord(ch)) % WORD_IDS
            out.append(n)
        return out


def tiny_tower():
    torch.manual_seed(0)
    return CLIPTextEncoder(width=32, layers=2, heads=4, vocab=VOCAB,
                           embed_dim=16).eval()


# --------------------------------------------------------------------------
# 1. the sentence table, checked as data
# --------------------------------------------------------------------------
def test_every_group_has_at_least_two_sentences():
    for label, sentences in GROUPS + [CALIBRATION]:
        assert len(sentences) >= 2, (label, sentences)


def test_the_single_word_groups_really_differ_in_one_word():
    """The cosine inside a group is only attributable to the axis word if
    everything else is held fixed.  Two sentences differing in two places would
    still produce a number, and the report would credit it to the wrong thing.

    T1 and T2 are deliberately exempt: they rewrite the whole sentence, which
    is what makes them a different experiment rather than a broken one.
    """
    exempt = {'T1  (模态命名式)', 'T2  (成像特性式)'}
    for label, sentences in GROUPS + [CALIBRATION]:
        if label in exempt:
            continue
        rows = [s.split() for s in sentences.values()]
        lengths = {len(r) for r in rows}
        assert lengths == {len(rows[0])}, (label, lengths)
        differ = [i for i in range(len(rows[0]))
                  if len({r[i] for r in rows}) > 1]
        assert len(differ) == 1, (label, differ,
                                  [[r[i] for r in rows] for i in differ])


def test_every_strip_word_is_a_standalone_word_of_its_sentence():
    for label, words in STRIP.items():
        sentences = dict(GROUPS)[label]
        for key, word in words.items():
            assert key in sentences, (label, key)
            assert word in sentences[key].split(), (label, key, word, sentences[key])


def test_strip_targets_the_position_the_group_varies():
    """The word removed must be the one the group is varying.  Removing some
    other fixed word would measure a different sentence's contribution and the
    table would still print three tidy numbers."""
    for label, words in STRIP.items():
        sentences = dict(GROUPS)[label]
        rows = {k: s.split() for k, s in sentences.items()}
        keys = list(rows)
        varying = [i for i in range(len(rows[keys[0]]))
                   if len({rows[k][i] for k in keys}) > 1]
        assert len(varying) == 1, (label, varying)
        i = varying[0]
        for k in keys:
            if k in words:
                assert rows[k][i] == words[k], (label, k, rows[k][i], words[k])


def test_the_calibration_pair_order_matches_what_main_reads():
    """main() reads the ceiling off [0,1] and the floor off [0,2] of the
    calibration matrix, which is only right while the dict is ordered
    dog, cat, car."""
    assert list(CALIBRATION[1]) == ['dog', 'cat', 'car'], list(CALIBRATION[1])


def test_no_sentence_is_long_enough_to_risk_truncation():
    """A crude word-count bound rather than a token count -- the real tokenizer
    is not available here.  CLIP's limit is 77 including SOT and EOT and BPE
    rarely more than doubles a word count, so 30 words leaves a wide margin and
    still catches a sentence that grew out of hand."""
    for label, sentences in GROUPS + [CALIBRATION]:
        for k, s in sentences.items():
            assert len(s.split()) <= 30, (label, k, len(s.split()))


# --------------------------------------------------------------------------
# 2. strip_word
# --------------------------------------------------------------------------
def test_strip_word_removes_exactly_one_token():
    assert strip_word('a photo of a thermal person', 'thermal') == \
        'a photo of a person'


def test_strip_word_refuses_a_substring():
    """'color' is inside 'colorful'; a substring replace would silently maul
    the sentence and the measured drop would be attributed to the wrong word."""
    try:
        strip_word('a colorful photo of a person', 'color')
    except ValueError:
        return
    raise AssertionError('a substring match was accepted')


def test_strip_word_removes_only_the_first_of_a_repeat():
    """T2's RGB sentence contains 'color' twice.  list.remove drops the first;
    the test states that rather than leaving it to be discovered later."""
    out = strip_word('a color photo showing color and texture', 'color')
    assert out == 'a photo showing color and texture', out


# --------------------------------------------------------------------------
# 3. encode
# --------------------------------------------------------------------------
def test_encode_returns_unit_vectors_and_token_counts():
    text, tok = tiny_tower(), StubTokenizer()
    sents = ['a photo of a dog', 'a photo of a cat']
    v, counts = encode(text, tok, sents)
    assert v.shape == (2, 16), v.shape
    assert torch.allclose(v.norm(dim=-1), torch.ones(2), atol=1e-5)
    # SOT + words + EOT
    assert counts == [len(s.split()) + 2 for s in sents], counts


def test_encode_is_deterministic_and_identical_sentences_give_cosine_one():
    text, tok = tiny_tower(), StubTokenizer()
    v, _ = encode(text, tok, ['a photo of a dog', 'a photo of a dog'])
    assert abs(float(v[0] @ v[1]) - 1.0) < 1e-5, float(v[0] @ v[1])


def test_encode_reads_the_vector_at_the_end_of_text_position():
    """Two sentences of different lengths must be read at their OWN eot, not
    at a shared one.  If the index were wrong, appending a word to a sentence
    would change its vector by far more than the word warrants -- and the
    whole report is about small differences."""
    text, tok = tiny_tower(), StubTokenizer()
    short, long_ = 'a photo of a dog', 'a photo of a dog in a park'
    v_pair, _ = encode(text, tok, [short, long_])
    v_alone, _ = encode(text, tok, [short])
    assert torch.allclose(v_pair[0], v_alone[0], atol=1e-5), \
        (v_pair[0] - v_alone[0]).abs().max()


def test_encode_refuses_a_sentence_that_would_be_truncated():
    text, tok = tiny_tower(), StubTokenizer()
    try:
        encode(text, tok, [' '.join(['word'] * 200)])
    except ValueError as e:
        assert 'truncated' in str(e), e
        return
    raise AssertionError('an over-length sentence was accepted')


# --------------------------------------------------------------------------
# 4. the matrix helpers
# --------------------------------------------------------------------------
def test_pairwise_is_symmetric_with_a_unit_diagonal():
    text, tok = tiny_tower(), StubTokenizer()
    v, _ = encode(text, tok, ['a photo of a dog', 'a photo of a cat',
                              'a photo of a car'])
    m = pairwise(v)
    assert torch.allclose(m, m.t(), atol=1e-6)
    assert torch.allclose(m.diagonal(), torch.ones(3), atol=1e-5)
    assert float(m.max()) <= 1.0 and float(m.min()) >= -1.0


def test_off_diagonal_returns_every_distinct_pair_once():
    m = torch.tensor([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    got = off_diagonal(m)
    assert len(got) == 3, got
    # float32 -> Python float is not exact; 0.2 comes back as 0.20000000298
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, [0.2, 0.3, 0.4])), got


# --------------------------------------------------------------------------
# 5. Stage 0.3 -- the confusion matrix
# --------------------------------------------------------------------------
def test_confusion_counts_a_perfect_classifier_onto_the_diagonal():
    import numpy as np
    from diag.modality_zeroshot import confusion
    proj = torch.eye(3)
    anchors = torch.eye(3)
    feat = torch.eye(3)[[0, 1, 2, 0]]          # each row sits on its own anchor
    mods = torch.tensor([0, 1, 2, 0])
    mat = confusion(feat, mods, proj, anchors, 3)
    assert np.array_equal(mat, np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])), mat


def test_confusion_rows_are_true_and_columns_are_predicted():
    """A transpose leaves the trace unchanged, so the accuracy would look
    identical while every off-diagonal claim in the report was backwards.
    Planted asymmetrically so only the correct orientation passes."""
    import numpy as np
    from diag.modality_zeroshot import confusion
    proj = torch.eye(3)
    anchors = torch.eye(3)
    feat = torch.eye(3)[[1, 1, 1]]             # everything predicted as spectrum 1
    mods = torch.tensor([0, 0, 2])             # ... while truly 0, 0, 2
    mat = confusion(feat, mods, proj, anchors, 3)
    assert mat[0, 1] == 2 and mat[2, 1] == 1, mat
    assert mat[1, 0] == 0 and mat[1, 2] == 0, mat
    assert int(np.trace(mat)) == 0, mat


def test_confusion_uses_the_projection_it_is_given():
    """proj is CLIP's visual.proj, 768 -> 512; if it were ignored the code
    would still run whenever the dimensions happened to match."""
    import numpy as np
    from diag.modality_zeroshot import confusion
    swap = torch.eye(3)[[1, 0, 2]]             # a projection that swaps 0 and 1
    anchors = torch.eye(3)
    feat = torch.eye(3)[[0, 0]]
    mods = torch.tensor([0, 0])
    mat = confusion(feat, mods, swap, anchors, 3)
    assert mat[0, 1] == 2, mat                 # projected onto anchor 1, not 0


def test_spectrum_groups_are_exactly_the_three_spectrum_ones():
    from diag.modality_zeroshot import SPECTRUM_KEYS, spectrum_groups
    labels = [g[0] for g in spectrum_groups()]
    assert 'ours-view  (text_lam50 的两个锚点)' not in labels, labels
    for label, sentences in spectrum_groups():
        # main() indexes with `sentences[k] for k in SPECTRUM_KEYS`, so both the
        # membership and the ORDER have to hold, or a whole row of the
        # confusion matrix is attributed to the wrong spectrum.
        assert tuple(sentences) == SPECTRUM_KEYS, (label, tuple(sentences))
    assert len(labels) >= 3, labels


# --------------------------------------------------------------------------
# 6. Stage 0.3 -- concentration, the measurement that names the culprit
# --------------------------------------------------------------------------
def test_mean_pairwise_cos_is_one_for_identical_directions():
    from diag.modality_zeroshot import mean_pairwise_cos
    v = torch.ones(50, 8) * 3.0                # same direction, different norms
    assert abs(mean_pairwise_cos(v) - 1.0) < 1e-9, mean_pairwise_cos(v)


def test_mean_pairwise_cos_is_zero_for_an_orthonormal_set():
    from diag.modality_zeroshot import mean_pairwise_cos
    assert abs(mean_pairwise_cos(torch.eye(8))) < 1e-9


def test_mean_pairwise_cos_matches_the_brute_force_matrix():
    """The identity ||sum u||^2 - N is exact, but it is also the kind of
    shortcut that is quietly wrong by a factor of two."""
    from diag.modality_zeroshot import mean_pairwise_cos
    torch.manual_seed(1)
    v = torch.randn(23, 5)
    u = torch.nn.functional.normalize(v.double(), dim=-1)
    m = u @ u.t()
    n = u.shape[0]
    want = float((m.sum() - n) / (n * (n - 1)))
    assert abs(mean_pairwise_cos(v) - want) < 1e-12, (mean_pairwise_cos(v), want)


def test_concentration_detects_a_projection_that_annihilates_the_spread():
    """The case the whole measurement exists for.

    768-d features spread across two axes; the projection keeps only the first.
    The cloud is genuinely varied before the projection and is one direction
    after it -- which is exactly the "fell into the null space" verdict, and it
    must not be reachable by a merely concentrated input.
    """
    from diag.modality_zeroshot import concentration
    n = 60
    feat = torch.zeros(n, 4)
    feat[:, 0] = 1.0
    feat[:, 1] = torch.linspace(-3.0, 3.0, n)   # the varying direction
    proj = torch.zeros(4, 2)
    proj[0, 0] = 1.0                            # keeps axis 0, drops axis 1
    mods = torch.arange(n) % 3
    c = concentration(feat, mods, proj, 3)
    assert c['cos768'] < 0.7, c['cos768']
    assert c['cos512'] > 0.999, c['cos512']


def test_concentration_leaves_an_innocent_projection_alone():
    from diag.modality_zeroshot import concentration
    torch.manual_seed(2)
    feat = torch.randn(60, 4)
    mods = torch.arange(60) % 3
    c = concentration(feat, mods, torch.eye(4), 3)
    assert abs(c['cos768'] - c['cos512']) < 1e-9, c
    assert abs(c['gain'] - 1.0) < 1e-9, c['gain']


def test_concentration_gain_is_the_norm_ratio():
    from diag.modality_zeroshot import concentration
    feat = torch.randn(20, 4)
    mods = torch.arange(20) % 2
    c = concentration(feat, mods, 0.25 * torch.eye(4), 2)
    assert abs(c['gain'] - 0.25) < 1e-9, c['gain']


def test_centroid_cos_is_symmetric_with_a_unit_diagonal():
    import numpy as np
    from diag.modality_zeroshot import centroid_cos
    torch.manual_seed(3)
    v = torch.randn(30, 6)
    mods = torch.arange(30) % 3
    m = centroid_cos(v, mods, 3)
    assert np.allclose(m, m.T, atol=1e-12)
    assert np.allclose(np.diag(m), 1.0, atol=1e-9)


def test_centroid_cos_separates_planted_spectra():
    from diag.modality_zeroshot import centroid_cos
    v = torch.zeros(30, 3)
    mods = torch.arange(30) % 3
    for m in range(3):
        v[mods == m, m] = 1.0                   # each spectrum on its own axis
    mat = centroid_cos(v, mods, 3)
    assert abs(mat[0, 1]) < 1e-9 and abs(mat[0, 2]) < 1e-9, mat


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d prompt-probe tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
