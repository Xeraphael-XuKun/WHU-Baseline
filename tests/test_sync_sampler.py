"""Unit tests for DATALOADER.SYNC_FRAMES.

CPU-only, no dataset needed. Run directly::

    python tests/test_sync_sampler.py

The failure this guards against is silent: if the triplets are not actually
frame-aligned, training proceeds normally and only the science is wrong. So the
synthetic source deliberately shuffles each modality's list independently -- the
i-th RGB entry is NOT the i-th IR entry -- which means a sampler that pairs by
position instead of by filename will fail here.
"""

import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from datasets.sampler import PKMSampler  # noqa: E402
except ImportError:
    # `datasets/__init__` pulls in make_dataloader, which needs timm. The sampler
    # itself needs nothing beyond torch/numpy, so load the file on its own and the
    # test stays runnable on a machine without the training dependencies.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'pkm_sampler', os.path.join(ROOT, 'datasets', 'sampler.py'))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    PKMSampler = _mod.PKMSampler

MODALITIES = ('RGB', 'IR', 'Thermal')
PATTERN = re.compile(r'^(\d+)_c(\d+)_m(\d+)_f(\d+)\.jpg$')


def make_source(n_pid=8, n_cam=2, n_frame=8, seed=0, drop=()):
    """WHU-MARS-shaped index: {modality: [(path, pid, camid, mindex), ...]}."""
    src = {m: [] for m in MODALITIES}
    for pid in range(n_pid):
        for cam in range(1, n_cam + 1):
            for f in range(n_frame):
                for mi, m in enumerate(MODALITIES, 1):
                    if (pid, cam, f, m) in drop:
                        continue
                    name = '%04d_c%d_m%d_f%06d.jpg' % (pid, cam, mi, f)
                    src[m].append(('/data/train/%s/%s' % (m, name), pid, cam - 1, mi))
    rng = random.Random(seed)
    for m in MODALITIES:           # independent order per modality, as in the real loader
        rng.shuffle(src[m])
    return src


def capture_of(src, m, index):
    """(pid, cam, frame) of one entry, parsed from its filename."""
    g = PATTERN.match(os.path.basename(src[m][index][0]))
    return (g.group(1), g.group(2), g.group(4))


def test_sync_triplets_share_a_capture():
    src = make_source()
    random.seed(0)
    s = PKMSampler(src, batch_size=8, num_instances=4, modalities=MODALITIES,
                   sync_frames=True)
    out = list(iter(s))
    assert out, 'sampler produced nothing'
    for triplet in out:
        caps = [capture_of(src, m, i) for m, i in zip(MODALITIES, triplet)]
        assert len(set(caps)) == 1, 'triplet spans captures {}'.format(caps)
    print('   %d triplets, every one frame-aligned' % len(out))


def test_default_is_not_synchronised():
    """The old behaviour must survive untouched, and the test must be able to see it."""
    src = make_source()
    random.seed(0)
    s = PKMSampler(src, batch_size=8, num_instances=4, modalities=MODALITIES)
    assert not hasattr(s, 'sync_index'), 'sync index built when the flag is off'
    out = list(iter(s))
    aligned = sum(1 for t in out
                  if len({capture_of(src, m, i) for m, i in zip(MODALITIES, t)}) == 1)
    # Same person, unrelated frames: a few coincidences are fine, most must differ.
    assert aligned < 0.25 * len(out), \
        'default path looks synchronised ({} / {}), test cannot tell the modes apart' \
        .format(aligned, len(out))
    print('   %d triplets, only %d incidentally aligned (default = unsynchronised)'
          % (len(out), aligned))


def test_identity_is_consistent_within_a_triplet():
    src = make_source()
    for sync in (False, True):
        random.seed(0)
        s = PKMSampler(src, batch_size=8, num_instances=4, modalities=MODALITIES,
                       sync_frames=sync)
        for triplet in iter(s):
            pids = {src[m][i][1] for m, i in zip(MODALITIES, triplet)}
            assert len(pids) == 1, 'triplet spans identities {}'.format(pids)


def test_epoch_length_matches_default():
    """Switching modes must not resize the epoch.

    If it did, `sync` vs `default` would differ by the number of gradient steps as
    well as by the pairing, and the comparison would stop being single-variable.

    Run at a realistic shape (120 identities, batch 64, K=4 -> 16 identities per
    batch). At toy scale the tail of the greedy batching loop -- which drops the
    leftover groups once fewer than `num_pids_per_batch` identities still have
    any, and which predates this change -- is a large enough fraction to move
    with the RNG. Here it is a fixed 3.3% in both modes.
    """
    src = make_source(n_pid=120, n_cam=2, n_frame=8)
    counts, declared = {}, {}
    for sync in (False, True):
        random.seed(0)
        s = PKMSampler(src, batch_size=64, num_instances=4, modalities=MODALITIES,
                       sync_frames=sync)
        random.seed(0)
        declared[sync], counts[sync] = len(s), len(list(iter(s)))
    assert declared[False] == declared[True], declared
    assert counts[False] == counts[True], counts
    print('   epoch length identical in both modes: %d declared, %d yielded'
          % (declared[True], counts[True]))


def test_incomplete_captures_are_dropped():
    """A capture missing one modality must be skipped, not paired with a stand-in."""
    drop = {(0, 1, f, 'Thermal') for f in range(8)}      # pid 0, cam 1 loses Thermal
    src = make_source(drop=drop)
    random.seed(0)
    s = PKMSampler(src, batch_size=8, num_instances=4, modalities=MODALITIES,
                   sync_frames=True)
    total = sum(len(v) for v in s.sync_index.values())
    assert total == 8 * 2 * 8 - 8, total          # n_pid * n_cam * n_frame - dropped
    for triplet in iter(s):
        caps = [capture_of(src, m, i) for m, i in zip(MODALITIES, triplet)]
        assert len(set(caps)) == 1, caps
    print('   %d complete captures kept, 8 incomplete dropped' % total)


def test_deterministic_under_a_seed():
    src = make_source()
    runs = []
    for _ in range(2):
        random.seed(7)
        s = PKMSampler(src, batch_size=8, num_instances=4, modalities=MODALITIES,
                       sync_frames=True)
        random.seed(7)
        runs.append(list(iter(s)))
    assert runs[0] == runs[1], 'sampler is not reproducible under a fixed seed'


# ---------------------------------------------------------------------------
# the shipped run.  Until hihr_whu_sync.yml existed this sampler was dead code:
# SYNC_FRAMES was implemented, tested and never switched on by any config.  The
# run that uses it changes nothing else, so these two guard the claim that it is
# a single-variable experiment.
# ---------------------------------------------------------------------------
def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def test_the_sync_config_differs_from_its_control_by_one_training_key():
    import yaml
    a = yaml.safe_load(_read('configs', 'hihr_whu_recipe_hihr.yml'))
    b = yaml.safe_load(_read('configs', 'hihr_whu_sync.yml'))

    def flat(d, prefix=''):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(flat(v, prefix + k + '.'))
            else:
                out[prefix + k] = v
        return out

    fa, fb = flat(a), flat(b)
    assert fb['DATALOADER.SYNC_FRAMES'] is True
    assert 'DATALOADER.SYNC_FRAMES' not in fa, 'the control must not set it'
    # AERIAL_CAMS is read only by the evaluator's view matrix and OUTPUT_DIR is
    # a path, so neither touches training.  Anything else appearing here would
    # make the comparison two-variable.
    added = set(fb) - set(fa) - {'DATALOADER.SYNC_FRAMES', 'DATASETS.AERIAL_CAMS'}
    changed = {k for k in fa if k in fb and fa[k] != fb[k]} - {'OUTPUT_DIR'}
    assert not added, added
    assert not changed, changed
    assert not set(fa) - set(fb), set(fa) - set(fb)


def test_launcher_knows_the_sync_run():
    assert 'whu_sync)' in _read('run_hihr.sh')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print('FAIL  {}: {}'.format(fn.__name__, exc))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
