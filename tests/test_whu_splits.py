# encoding: utf-8
"""Two WHU-MARS splits from one class, and the GD evaluation protocol.

Two things can go wrong here in ways no error message would reveal:

  * the split is chosen by editing a line in whu_mars.py, so two runs produce
    logs that look identical and are not comparable;
  * the GD filter is applied to train as well as to query/gallery, which turns
    a re-evaluation into a different experiment -- the paper's Table 3 is
    explicit that GD evaluates *the WHU-MARS-1000-trained model*.

Both are pinned below.  The filter tests run against the real dataset when it
is present and skip cleanly when it is not, so this file is useful on the dev
machine and on a laptop alike.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported by file, not as `datasets.whu_mars`: the package __init__ pulls in
# make_dataloader, which needs timm.  Nothing here touches a transform, so the
# dependency would only stop this file from running on a machine without it.
import importlib.util  # noqa: E402
import types  # noqa: E402

_DS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
# A stand-in package so whu_mars.py's `from .bases import ...` resolves without
# executing datasets/__init__.py.
_pkg = types.ModuleType('_ds')
_pkg.__path__ = [_DS]
sys.modules['_ds'] = _pkg
_spec = importlib.util.spec_from_file_location('_ds.whu_mars', os.path.join(_DS, 'whu_mars.py'))
_mod = importlib.util.module_from_spec(_spec)
sys.modules['_ds.whu_mars'] = _mod
_spec.loader.exec_module(_mod)
WHU_MARS = _mod.WHU_MARS

PASS = FAIL = SKIP = 0
ROOTS = [r'E:\Code\CVPR\Datasets', '/mnt/cache/wanghanzhi/Datasets']
MODS = ('RGB', 'IR', 'Thermal')


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('PASS  %s' % name)
    else:
        FAIL += 1
        print('FAIL  %s  %s' % (name, detail))


def skip(name, why):
    global SKIP
    SKIP += 1
    print('SKIP  %s  (%s)' % (name, why))


def find_root(subdir='WHU-MARS'):
    for r in ROOTS:
        if os.path.isdir(os.path.join(r, subdir, 'train', 'RGB')):
            return r
    return None


def pids_cams(split):
    """-> {(pid, camid_zero_based)} over every modality of one split dict."""
    out = set()
    for m in MODS:
        for _, pid, camid, _ in split.get(m, []):
            out.add((pid, camid))
    return out


# --------------------------------------------------------------- the protocol
def test_gd_filters_evaluation_only():
    """The paper: GD "evaluates the WHU-MARS-1000-trained model using ground and
    daytime queries and galleries only".  Train must come out identical."""
    root = find_root()
    if root is None:
        return skip('gd filters evaluation only', 'WHU-MARS not on this machine')
    a = WHU_MARS(root=root, modalities=MODS, verbose=False)
    b = WHU_MARS(root=root, modalities=MODS, protocol='GD', verbose=False)
    check('GD leaves train untouched',
          {m: len(a.train[m]) for m in MODS} == {m: len(b.train[m]) for m in MODS},
          ({m: len(a.train[m]) for m in MODS}, {m: len(b.train[m]) for m in MODS}))
    check('GD shrinks the query set',
          sum(len(b.query[m]) for m in MODS) < sum(len(a.query[m]) for m in MODS))
    check('GD shrinks the gallery set',
          sum(len(b.gallery[m]) for m in MODS) < sum(len(a.gallery[m]) for m in MODS))


def test_gd_keeps_only_ground_cameras_and_the_daytime_window():
    """Both halves of the filter, read off the surviving rows rather than off
    the source.  camid is zero-based by the time it reaches the dataset, so the
    one-based c<=5 rule shows up here as <= 4."""
    root = find_root()
    if root is None:
        return skip('gd keeps ground + daytime only', 'WHU-MARS not on this machine')
    b = WHU_MARS(root=root, modalities=MODS, protocol='GD', verbose=False)
    lo, hi = WHU_MARS.GD_PID_RANGE
    for name, split in (('query', b.query), ('gallery', b.gallery)):
        rows = pids_cams(split)
        check('GD %s: every camera is ground' % name,
              all(c <= WHU_MARS.GD_MAX_GROUND_CAM - 1 for _, c in rows),
              sorted({c for _, c in rows}))
        check('GD %s: every identity is in the daytime window' % name,
              all(lo <= p < hi for p, _ in rows),
              sorted({p for p, _ in rows})[:5])
        check('GD %s: something survived' % name, len(rows) > 0)


def test_all_protocol_is_unchanged():
    """Everything measured so far ran under ALL.  Adding GD must not have moved
    it -- these are the numbers every WHU-MARS result in the diary rests on."""
    root = find_root()
    if root is None:
        return skip('ALL is unchanged', 'WHU-MARS not on this machine')
    a = WHU_MARS(root=root, modalities=MODS, verbose=False)
    check('ALL train: 30,711 captures per spectrum',
          all(len(a.train[m]) == 30711 for m in MODS),
          {m: len(a.train[m]) for m in MODS})
    check('ALL train: 500 identities',
          len({p for p, _ in pids_cams(a.train)}) == 500)
    check('ALL query: 2,135 per spectrum',
          all(len(a.query[m]) == 2135 for m in MODS),
          {m: len(a.query[m]) for m in MODS})
    check('ALL gallery: 31,203 per spectrum',
          all(len(a.gallery[m]) == 31203 for m in MODS),
          {m: len(a.gallery[m]) for m in MODS})
    check('ALL keeps the aerial cameras',
          max(c for _, c in pids_cams(a.query)) >= WHU_MARS.GD_MAX_GROUND_CAM)


def test_an_unknown_protocol_is_refused():
    """CARGO's AA/GG/AG mean nothing here.  Silently treating them as ALL would
    produce a full-split number under a name that promises a subset."""
    root = find_root()
    if root is None:
        return skip('unknown protocol refused', 'WHU-MARS not on this machine')
    for bad in ('AA', 'GG', 'AG', 'gd_daytime'):
        try:
            WHU_MARS(root=root, modalities=MODS, protocol=bad, verbose=False)
            check('protocol %r is refused' % bad, False, 'accepted it')
        except ValueError:
            check('protocol %r is refused' % bad, True)


def test_protocol_is_case_insensitive():
    root = find_root()
    if root is None:
        return skip('protocol case', 'WHU-MARS not on this machine')
    a = WHU_MARS(root=root, modalities=MODS, protocol='gd', verbose=False)
    b = WHU_MARS(root=root, modalities=MODS, protocol='GD', verbose=False)
    check('lower-case gd works', all(len(a.query[m]) == len(b.query[m]) for m in MODS))


# ------------------------------------------------------------------ the split
def test_subdir_selects_the_split():
    """The 2337 split lives in a sibling directory; SUBDIR picks it."""
    root = find_root('WHU-MARS-2337')
    if root is None:
        return skip('subdir selects the split', 'WHU-MARS-2337 not on this machine')
    d = WHU_MARS(root=root, modalities=MODS, subdir='WHU-MARS-2337', verbose=False)
    check('2337 train: 1,000 identities',
          len({p for p, _ in pids_cams(d.train)}) == 1000,
          len({p for p, _ in pids_cams(d.train)}))
    check('2337 test: 1,337 identities',
          len({p for p, _ in pids_cams(d.gallery)}) == 1337,
          len({p for p, _ in pids_cams(d.gallery)}))
    tt = sum(len(d.train[m]) + len(d.gallery[m]) for m in MODS)
    check('2337 train+test: 434,620 images (the paper count)', tt == 434620, tt)
    check('2337 spectra hold unequal counts (this split is not aligned)',
          len({len(d.train[m]) for m in MODS}) > 1,
          {m: len(d.train[m]) for m in MODS})
    # Every train identity must appear in every spectrum, or PKMSampler divides
    # by zero when it tries to fill that identity's quota.
    per = {m: {p for _, p, _, _ in d.train[m]} for m in MODS}
    allp = set().union(*per.values())
    check('2337 train: every identity appears in all three spectra',
          all(per[m] == allp for m in MODS),
          {m: len(allp - per[m]) for m in MODS})


def test_an_empty_gd_result_is_refused():
    """A window that matches nothing would report a meaningless mAP instead of
    failing.  Checked by moving the window somewhere no identity lives."""
    root = find_root()
    if root is None:
        return skip('empty GD refused', 'WHU-MARS not on this machine')
    keep = WHU_MARS.GD_PID_RANGE
    try:
        WHU_MARS.GD_PID_RANGE = (10 ** 8, 10 ** 9)
        try:
            WHU_MARS(root=root, modalities=MODS, protocol='GD', verbose=False)
            check('an empty GD query set raises', False, 'it loaded happily')
        except RuntimeError:
            check('an empty GD query set raises', True)
    finally:
        WHU_MARS.GD_PID_RANGE = keep


# ------------------------------------------------------------------- configs
def test_the_2337_configs_differ_only_where_intended():
    import yaml

    def flat(n, p=()):
        out = {}
        if isinstance(n, dict):
            for k, v in n.items():
                out.update(flat(v, p + (str(k),)))
        else:
            out['.'.join(p)] = n
        return out

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
    load = lambda n: flat(yaml.safe_load(open(os.path.join(here, 'hihr_%s.yml' % n),
                                              encoding='utf-8')))
    for new, base in (('whu2337_text_lam50', 'whu_text_lam50_clip'),
                      ('whu2337_mtext_lam50', 'whu_mtext_lam50_clip'),
                      ('whu2337_ce_mod_g2_mtext', 'whu_ce_mod_g2_mtext_clip')):
        a, b = load(base), load(new)
        check('%s: adds only DATASETS.SUBDIR' % new,
              set(b) - set(a) == {'DATASETS.SUBDIR'}, sorted(set(b) - set(a)))
        diff = {k for k in a if k in b and a[k] != b[k]}
        check('%s: changes only OUTPUT_DIR' % new, diff == {'OUTPUT_DIR'}, sorted(diff))
        check('%s: points at the 2337 split' % new,
              b['DATASETS.SUBDIR'] == 'WHU-MARS-2337', b['DATASETS.SUBDIR'])
        check('%s: SYNC_FRAMES stays off (the spectra are not aligned here)' % new,
              b.get('DATALOADER.SYNC_FRAMES', False) is False,
              b.get('DATALOADER.SYNC_FRAMES'))
        check('%s: still raw CLIP init' % new,
              b['MODEL.PRETRAIN_CHOICE'] == 'imagenet' and 'ViT-B-16' in b['MODEL.PRETRAIN_PATH'])


def main():
    for fn in (test_gd_filters_evaluation_only,
               test_gd_keeps_only_ground_cameras_and_the_daytime_window,
               test_all_protocol_is_unchanged,
               test_an_unknown_protocol_is_refused,
               test_protocol_is_case_insensitive,
               test_subdir_selects_the_split,
               test_an_empty_gd_result_is_refused,
               test_the_2337_configs_differ_only_where_intended):
        fn()
    total = PASS + FAIL
    print('\n%d/%d passed%s' % (PASS, total, '  (%d skipped)' % SKIP if SKIP else ''))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
