"""Unit tests for the arms that fill the paper's remaining empty cells.

CPU-only and self-contained.  Run directly::

    python tests/test_main_table_arms.py

Four arms, three bases:

    whu_pemod_mtext_clip        the headline, already run: 14.42 / 37.13
    whu2337_pemod_mtext_clip    the main table's 2337 column
    whu_pemod_mtext_s2_clip     the same recipe at seed 2025
    whu_pemod_notext_clip       MPRC with no text -- the MODULE table's row 2
                                and the interaction 2x2's fourth corner

The last one is derived from the plain residual baseline rather than from the
headline, because that is the control it is read against (13.29) and because
switching TGVA off inside the full method would leave a dozen text keys in the
file with nothing reading them.

Everything asserted here is a way for one of these runs to come back wrong
WITHOUT failing: a config that differs from its base in a second key measures
two things at once and attributes neither; a seed arm that forgot to change the
seed reproduces its control bit for bit and looks like a triumph; a 2337 arm
still pointing at the 1,000-identity directory trains the wrong split and
reports a number for the wrong column; and a no-text arm that kept TEXT_ALIGN
would answer "what is MPRC worth alone" with a number that includes TGVA.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFGS = os.path.join(ROOT, 'configs')

BASE = 'hihr_whu_pemod_mtext_clip.yml'
SPLIT = 'hihr_whu2337_pemod_mtext_clip.yml'
SEED = 'hihr_whu_pemod_mtext_s2_clip.yml'
# The module-ablation arm and the plain residual baseline it is read against.
PLAIN = 'hihr_whu_pe_indep_clip.yml'
NOTEXT = 'hihr_whu_pemod_notext_clip.yml'


def keys(name):
    """Every `KEY: value` line, comments stripped.  Ordering is irrelevant."""
    out = {}
    for line in open(os.path.join(CFGS, name), encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        m = re.match(r'^(\s*)([A-Z_]+):\s*(.+)$', line)
        if m and m.group(3).strip():
            out[m.group(2)] = m.group(3).strip()
    return out


def differs(a, b):
    ka, kb = keys(a), keys(b)
    return {k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k)}


# ---- single-factor, both of them ----------------------------------------

def test_the_2337_arm_changes_only_the_split():
    d = differs(BASE, SPLIT)
    assert d == {'SUBDIR', 'OUTPUT_DIR'}, sorted(d)
    assert keys(SPLIT)['SUBDIR'] == "'WHU-MARS-2337'"
    assert 'SUBDIR' not in keys(BASE), \
        'the base must inherit the default split, or this is not a one-key diff'


def test_the_seed_arm_changes_only_the_seed():
    d = differs(BASE, SEED)
    assert d == {'SEED', 'OUTPUT_DIR'}, sorted(d)


def test_the_seed_actually_changes():
    """An arm that kept 1234 would reproduce its control BIT FOR BIT -- training
    here is deterministic -- and the perfect agreement would read as a
    replication rather than as a copy."""
    src = open(os.path.join(ROOT, 'config', 'defaults.py'), encoding='utf-8').read()
    assert '_C.SOLVER.SEED = 1234' in src, 'the default moved; this test is stale'
    assert keys(SEED)['SEED'] == '2025', keys(SEED).get('SEED')


def test_the_seed_reaches_torch_numpy_and_random():
    """SOLVER.SEED is only worth changing if it seeds everything that draws."""
    src = open(os.path.join(ROOT, 'train.py'), encoding='utf-8').read()
    body = src[src.index('def set_seed('):]
    body = body[:body.index('\ndef ', 1)] if '\ndef ' in body[1:] else body[:600]
    for call in ('torch.manual_seed', 'torch.cuda.manual_seed_all',
                 'np.random.seed', 'random.seed'):
        assert call in body, '{} is not seeded'.format(call)
    assert 'set_seed(cfg.SOLVER.SEED)' in src


# ---- the method itself must survive the copy ----------------------------

def test_all_three_arms_are_the_same_method():
    """The keys that DEFINE the method have to be identical across the three."""
    method = ('PE_LAYERWISE', 'PE_PER_MODALITY', 'TEXT_ALIGN', 'TEXT_TEMPLATE',
              'TEXT_MODALITY_WORDS', 'TEXT_LOSS_WEIGHT', 'TEXT_N_CTX',
              'MAX_EPOCHS', 'IMS_PER_BATCH', 'BASE_LR', 'PRETRAINED_LR',
              'TRANSFORMER_TYPE', 'PRETRAIN_CHOICE')
    base = keys(BASE)
    for name in (SPLIT, SEED):
        k = keys(name)
        for m in method:
            assert m in base, '{} is missing from the base config'.format(m)
            assert k.get(m) == base[m], \
                '{}: {} is {!r}, base has {!r}'.format(name, m, k.get(m), base[m])


def test_the_method_is_configuration_b():
    """Per-modality residuals AND six anchors -- either one alone is a null
    result (14.09 and 14.21 against the 14.11 control), so an arm that lost one
    of them would be measuring a different, weaker method under this name."""
    base = keys(BASE)
    assert base['PE_PER_MODALITY'] == 'True'
    assert base['PE_LAYERWISE'] == "'indep'"
    assert base['TEXT_ALIGN'] == 'True'
    assert 'TEXT_MODALITY_WORDS' in base, 'six anchors need the spectrum words'
    assert 'TEXT_MODALITY' not in base, \
        "TEXT_MODALITY pins supervision to one spectrum; the six-anchor form " \
        "must not carry it (make_model reads mod_words instead)"


def test_output_dirs_are_distinct():
    outs = {n: keys(n)['OUTPUT_DIR'] for n in (BASE, SPLIT, SEED, NOTEXT, PLAIN)}
    assert len(set(outs.values())) == 5, outs


def test_the_runner_knows_all_four():
    src = open(os.path.join(ROOT, 'run_hihr.sh'), encoding='utf-8').read()
    for name in (BASE, SPLIT, SEED, NOTEXT):
        mode = name[len('hihr_'):-len('.yml')]
        assert '{})'.format(mode) in src, \
            '{} has no branch in run_hihr.sh'.format(mode)


# ---- the module-ablation arm --------------------------------------------

def test_the_notext_arm_differs_from_the_plain_baseline_in_one_key():
    plain, arm = keys(PLAIN), keys(NOTEXT)
    differs = {k for k in set(plain) | set(arm) if plain.get(k) != arm.get(k)}
    assert differs == {'PE_PER_MODALITY', 'OUTPUT_DIR'}, sorted(differs)
    assert arm['PE_PER_MODALITY'] == 'True'


def test_the_notext_arm_carries_no_text_supervision():
    """The point of the arm.  With TEXT_ALIGN it would answer "what is MPRC
    worth alone" with a number that includes TGVA."""
    arm = keys(NOTEXT)
    assert arm.get('TEXT_ALIGN') != 'True'
    for k in ('TEXT_MODALITY_WORDS', 'TEXT_TEMPLATE', 'TEXT_LOSS_WEIGHT',
              'TEXT_N_CTX', 'TEXT_CLIP_PATH'):
        assert k not in arm, \
            '{} is dead weight without TEXT_ALIGN and reads as if it did something'.format(k)


def test_the_notext_arm_is_the_same_residual_as_the_headline():
    """Same actuator, text removed -- otherwise row 2 and row 4 of the module
    table are not on the same axis."""
    head, arm = keys(BASE), keys(NOTEXT)
    for k in ('PE_LAYERWISE', 'PE_PER_MODALITY', 'PE_FREEZE_BASE',
              'TRANSFORMER_TYPE', 'PRETRAIN_CHOICE', 'STRIDE_SIZE',
              'MAX_EPOCHS', 'IMS_PER_BATCH', 'BASE_LR', 'PRETRAINED_LR'):
        assert arm.get(k) == head.get(k), \
            '{}: notext has {!r}, headline has {!r}'.format(k, arm.get(k), head.get(k))


def test_the_plain_baseline_is_the_shared_residual():
    """13.29 is the SHARED residual; if it ever gained PE_PER_MODALITY the
    interaction table would be comparing a cell against itself."""
    plain = keys(PLAIN)
    assert plain.get('PE_PER_MODALITY') != 'True'
    assert plain['PE_LAYERWISE'] == "'indep'"
    assert plain.get('TEXT_ALIGN') != 'True'


def test_the_four_corners_of_the_interaction_table_are_four_configs():
    """shared/per-spectrum x no-text/six-anchor.  Three are already measured;
    this asserts they are genuinely four distinct configurations."""
    corners = {
        'shared, no text': PLAIN,
        'per-spectrum, no text': NOTEXT,
        'shared, six anchors': 'hihr_whu_mtext_lam50_clip.yml',
        'per-spectrum, six anchors': BASE,
    }
    seen = {}
    for label, name in corners.items():
        k = keys(name)
        sig = (k.get('PE_PER_MODALITY') == 'True', k.get('TEXT_ALIGN') == 'True')
        assert sig not in seen, '{} and {} are the same corner'.format(label, seen.get(sig))
        seen[sig] = label
    assert len(seen) == 4, seen


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
