"""Unit tests for the CARGO dataset and the two evaluation conventions.

CPU-only and self-contained: builds a synthetic CARGO tree on disk, so the
parser and the protocols are verified without the real 108k images.
Run directly::

    python tests/test_cargo.py

Two classes of silent failure are covered:

  * an off-by-one in the underscore index relabels the entire dataset and
    still trains happily, just to a meaningless number;
  * the SYSU junk rule (this repo's default) scores several points higher than
    the Market-1501 rule that fast-reid uses for CARGO, and nothing warns you.
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# datasets/__init__.py pulls in make_dataloader, which needs timm and torchvision.
# Neither has anything to do with parsing filenames, so stand in a bare package
# object: `from .bases import ...` inside cargo.py still resolves through
# __path__, but __init__.py never runs.  Keeps these tests runnable anywhere.
_pkg = types.ModuleType('datasets')
_pkg.__path__ = [os.path.join(ROOT, 'datasets')]
sys.modules.setdefault('datasets', _pkg)

from datasets.cargo import CARGO  # noqa: E402
from utils.metrics import eval_func, junk_mask  # noqa: E402


def build_tree(root, per_cam=2, pids=(3, 7, 11)):
    """CARGO/{train,query,gallery}/Cam1..Cam13/Cam{c}_{t}_{pid}_{n}.jpg"""
    base = os.path.join(root, 'CARGO')
    for split in ('train', 'query', 'gallery'):
        for cam in range(1, 14):
            d = os.path.join(base, split, 'Cam{}'.format(cam))
            os.makedirs(d)
            for pid in pids:
                for n in range(per_cam):
                    name = 'Cam{}_0001_{}_{}.jpg'.format(cam, pid, n)
                    open(os.path.join(d, name), 'wb').close()
    return base


class Tree(object):
    def __enter__(self):
        self.root = tempfile.mkdtemp()
        build_tree(self.root)
        return self.root

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def load(root, protocol='ALL'):
    return CARGO(root=root, verbose=False, modalities=['RGB'], protocol=protocol)


def rows(ds, split='train'):
    return getattr(ds, split)['RGB']


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_filename_parsing_matches_the_official_loader():
    """pid = 3rd underscore field, camid = digits after 'Cam'."""
    pid, camid, aerial = CARGO.parse('/x/y/Cam12_0001_4321_7.jpg')
    assert (pid, camid, aerial) == (4321, 12, False)
    pid, camid, aerial = CARGO.parse('Cam1_0001_9_0.jpg')
    assert (pid, camid, aerial) == (9, 1, True)
    # The boundary the whole aerial/ground split hangs on.
    assert CARGO.parse('Cam5_0_1_0.jpg')[2] is True
    assert CARGO.parse('Cam6_0_1_0.jpg')[2] is False


def test_bad_filenames_raise_instead_of_guessing():
    for name in ('Cam3_0001.jpg', 'C3_0001_5_0.jpg', 'Cam3_0001_xx_0.jpg'):
        try:
            CARGO.parse(name)
        except ValueError:
            pass
        else:
            raise AssertionError('{!r} should have raised'.format(name))


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------

def test_all_protocol_keeps_every_camera():
    with Tree() as root:
        ds = load(root, 'ALL')
        assert len(rows(ds)) == 13 * 3 * 2
        assert sorted({c for _, _, c, _ in rows(ds)}) == list(range(13))   # zero-based
        assert ds.num_train_pids == 3


def test_aa_and_gg_filter_train_and_test_alike():
    with Tree() as root:
        aa = load(root, 'AA')
        gg = load(root, 'GG')
        for split in ('train', 'query', 'gallery'):
            assert len(rows(aa, split)) == 5 * 3 * 2, split
            assert len(rows(gg, split)) == 8 * 3 * 2, split
        assert sorted({c for _, _, c, _ in rows(aa)}) == [0, 1, 2, 3, 4]
        assert sorted({c for _, _, c, _ in rows(gg)}) == [5, 6, 7, 8, 9, 10, 11, 12]


def test_ag_collapses_cameras_onto_the_two_views():
    """The camera id *is* the view here -- that is what makes it cross-view."""
    with Tree() as root:
        ag = load(root, 'AG')
        assert len(rows(ag)) == 13 * 3 * 2          # AG keeps the full set
        assert sorted({c for _, _, c, _ in rows(ag)}) == [0, 1]
        aerial = [r for r in rows(ag) if r[2] == 0]
        assert len(aerial) == 5 * 3 * 2


def test_view_is_recorded_in_the_track_slot():
    with Tree() as root:
        ds = load(root, 'ALL')
        for path, _, _, view in rows(ds):
            cam = CARGO.parse(path)[1]
            assert view == (1 if cam <= 5 else 2)
        assert ds.num_train_vids == 2


def test_train_is_relabelled_but_test_is_not():
    with Tree() as root:
        ds = load(root, 'ALL')
        assert sorted({p for _, p, _, _ in rows(ds, 'train')}) == [0, 1, 2]
        assert sorted({p for _, p, _, _ in rows(ds, 'query')}) == [3, 7, 11]


def test_single_modality_contract():
    """The pipeline is keyed on a modality list; CARGO must present exactly one."""
    with Tree() as root:
        ds = load(root)
        assert list(ds.train.keys()) == ['RGB']
        try:
            CARGO(root=root, verbose=False, modalities=['RGB', 'IR'])
        except ValueError as e:
            assert 'single-modality' in str(e), str(e)
        else:
            raise AssertionError('a multi-modality list should have raised')


def test_rejects_unknown_protocol_and_missing_tree():
    with Tree() as root:
        try:
            load(root, 'A2G')
        except ValueError as e:
            assert 'PROTOCOL' in str(e), str(e)
        else:
            raise AssertionError('unknown protocol should have raised')
    try:
        load(tempfile.mkdtemp())
    except RuntimeError as e:
        assert 'not available' in str(e), str(e)
    else:
        raise AssertionError('missing dataset should have raised')


# --------------------------------------------------------------------------
# evaluation conventions
# --------------------------------------------------------------------------

def test_junk_mask_conventions_differ_exactly_where_expected():
    g_pids = np.array([1, 1, 2, 2])
    g_cams = np.array([0, 1, 0, 1])
    market = junk_mask(g_pids, g_cams, q_pid=1, q_camid=0, metric='market')
    sysu = junk_mask(g_pids, g_cams, q_pid=1, q_camid=0, metric='sysu')
    # Market drops only the query's own identity on the query's camera.
    assert market.tolist() == [True, False, False, False]
    # SYSU drops that whole camera, distractor (pid 2, cam 0) included.
    assert sysu.tolist() == [True, False, True, False]
    try:
        junk_mask(g_pids, g_cams, 1, 0, metric='cargo')
    except ValueError as e:
        assert 'TEST.METRIC' in str(e), str(e)
    else:
        raise AssertionError('unknown metric should have raised')


def test_sysu_scores_higher_than_market_on_the_same_ranking():
    """Why this is a setting and not a constant."""
    # One query (pid 1, cam 0).  The ranking puts a same-camera distractor
    # ahead of the true cross-camera match: SYSU deletes that distractor and
    # sees a perfect ranking, Market keeps it and does not.
    q_pids, q_cams = np.array([1]), np.array([0])
    g_pids = np.array([2, 1])
    g_cams = np.array([0, 1])
    indices = np.array([[0, 1]])          # distractor first, then the match

    _, ap_sysu, _ = eval_func(indices, q_pids, g_pids, q_cams, g_cams, metric='sysu')
    _, ap_market, _ = eval_func(indices, q_pids, g_pids, q_cams, g_cams, metric='market')
    assert ap_sysu == 1.0
    assert abs(ap_market - 0.5) < 1e-9
    assert ap_sysu > ap_market


def test_relevant_set_is_identical_under_both_rules():
    """Only the junk mask changes; a same-pid cross-camera hit counts either way."""
    q_pids, q_cams = np.array([1]), np.array([0])
    g_pids = np.array([1, 1, 2])
    g_cams = np.array([0, 1, 1])
    indices = np.array([[1, 0, 2]])       # the cross-camera match ranked first
    cmc_s, ap_s, _ = eval_func(indices, q_pids, g_pids, q_cams, g_cams, metric='sysu')
    cmc_m, ap_m, _ = eval_func(indices, q_pids, g_pids, q_cams, g_cams, metric='market')
    assert ap_s == ap_m == 1.0
    assert cmc_s[0] == cmc_m[0] == 1.0


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
