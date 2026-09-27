"""Static experiment-contract checks for B0-W and R-UAD.

These checks compare the fully resolved YACS configurations.  They protect the
two intended comparisons from silently acquiring an extra variable:

  A -> B0-W: only Triplet -> WRT
  B0-W -> R-UAD: only ProCA + GPD
"""

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_value(text):
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip()


def load(name):
    """Parse the two-level YAML subset used by the experiment configs.

    Keeping this checker on the Python standard library lets it run before the
    server environment is activated; training itself still uses YACS.
    """
    values = {}
    section = None
    path = os.path.join(ROOT, 'configs', name)
    with open(path, encoding='utf-8') as handle:
        for raw in handle:
            line = raw.split('#', 1)[0].rstrip()
            if not line:
                continue
            top = re.match(r'^([A-Z_]+):(?:\s*(.+))?$', line)
            if top:
                if top.group(2):
                    values[top.group(1)] = parse_value(top.group(2))
                    section = None
                else:
                    section = top.group(1)
                continue
            nested = re.match(r'^  ([A-Z_]+):\s*(.+)$', line)
            if nested and section:
                values['{}.{}'.format(section, nested.group(1))] = \
                    parse_value(nested.group(2))
    return values


def changed(a, b):
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def test_b0_w_changes_only_metric_loss():
    a = load('A_baseline_candidate.yml')
    b0w = load('B0_WRT.yml')
    assert changed(a, b0w) == {
        'MODEL.METRIC_LOSS_TYPE',
        'OUTPUT_DIR',
    }
    assert b0w['MODEL.METRIC_LOSS_TYPE'] == 'wrt'
    assert b0w['SOLVER.LOSS_TYPE'] == 'base'


def test_r_uad_changes_only_the_uad_objective():
    b0w = load('B0_WRT.yml')
    uad = load('R_UAD.yml')
    # GPD_MOMENTUM is written explicitly in R-UAD but equals the project
    # default inherited by B0-W, so it is not an effective config difference.
    explicit = changed(b0w, uad)
    assert explicit == {
        'MODEL.GPD_MOMENTUM',
        'MODEL.PCA_LOSS_WEIGHT',
        'SOLVER.LOSS_TYPE',
        'OUTPUT_DIR',
    }
    defaults = open(os.path.join(ROOT, 'config', 'defaults.py'),
                    encoding='utf-8').read()
    assert '_C.MODEL.GPD_MOMENTUM = 0.2' in defaults
    assert uad['MODEL.METRIC_LOSS_TYPE'] == 'wrt'
    assert uad['MODEL.PCA_LOSS_WEIGHT'] == 0.01
    assert uad['MODEL.GPD_MOMENTUM'] == 0.2
    assert uad['SOLVER.LOSS_TYPE'] == 'base+pca+gpd'


def test_shared_formal_protocol_is_locked():
    for name in ('B0_WRT.yml', 'R_UAD.yml'):
        cfg = load(name)
        assert cfg['MODEL.TRANSFORMER_TYPE'] == 'vit_base_clip'
        assert cfg['INPUT.PIXEL_MEAN'] == [0.48145466, 0.4578275, 0.40821073]
        assert cfg['INPUT.PIXEL_STD'] == [0.26862954, 0.26130258, 0.27577711]
        assert cfg['DATALOADER.SAMPLER'] == 'PKM'
        assert cfg['DATALOADER.NUM_INSTANCE'] == 4
        assert cfg['SOLVER.IMS_PER_BATCH'] == 64
        assert cfg['SOLVER.MAX_EPOCHS'] == 60
        assert cfg['SOLVER.SEED'] == 1234
        assert cfg['TEST.NECK_FEAT'] == 'before'
        assert cfg['TEST.FEAT_NORM'] == 'yes'
        assert cfg['TEST.METRIC'] == 'sysu'


if __name__ == '__main__':
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith('test_')]
    for test in tests:
        test()
        print('PASS  {}'.format(test.__name__))
    print('{} / {} passed'.format(len(tests), len(tests)))
