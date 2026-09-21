"""Unit tests for verify_upload.sh itself.

It is the script that stands between an incomplete upload and a wasted night,
and until 2026-08-13 nothing checked IT.  Three of its seventy-two assertions
were wrong, and all three failed against a tree that was perfectly correct:

    'after_tgt = rows[modality]'      [modality] is a character class to grep,
    'CE_MODALITY_GROUPS: [0, 0, 1]'   and so is [0, 1]
    'cp -r'                           matched the COMMENT in run_hihr.sh that
                                      explains why cp -r was removed

That failure mode is worse than no check.  It reports "stale or missing" for a
file that is neither, so the reply is to re-upload something already right --
and after two or three of those, the habit becomes ignoring the red lines.

The fix was to make every check a literal (`grep -qF`).  The test is to run all
of them against this tree, which IS the correct state by construction: every
check_has must hit and every check_no must not.  Substring matching in Python
is used rather than shelling out, because that is exactly what -F means, and a
divergence between the two would be the bug this file exists to catch -- so the
-F itself is asserted separately.

Run:  python tests/test_verify_upload.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print('  ok    %s' % name)
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print('  FAIL  %s\n          %s: %s' % (name, type(exc).__name__, exc))


def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding='utf-8').read()


CHECK_RE = re.compile(
    r'''^(check_has|check_no)\s+(\S+)\s+('[^']*'|"[^"]*")\s''', re.M)


def checks():
    out = []
    for kind, path, quoted in CHECK_RE.findall(_read('verify_upload.sh')):
        out.append((kind, path, quoted[1:-1]))
    return out


# --------------------------------------------------------------------------
# 1. every assertion holds against this tree
# --------------------------------------------------------------------------
def test_there_are_checks_to_run_at_all():
    """A parser that silently matched nothing would make every test below pass
    for the wrong reason."""
    assert len(checks()) > 50, len(checks())


def test_every_check_has_finds_its_string():
    wrong = []
    for kind, path, pat in checks():
        if kind != 'check_has':
            continue
        p = os.path.join(ROOT, *path.split('/'))
        if not os.path.exists(p):
            wrong.append((path, 'FILE MISSING'))
        elif pat not in _read(*path.split('/')):
            wrong.append((path, pat))
    assert not wrong, wrong


def test_every_check_no_finds_nothing():
    """The mirror.  These pin deletions, and a pattern that matches a comment
    about the deleted thing reports it as still present."""
    wrong = []
    for kind, path, pat in checks():
        if kind != 'check_no':
            continue
        if pat in _read(*path.split('/')):
            wrong.append((path, pat))
    assert not wrong, wrong


# --------------------------------------------------------------------------
# 2. the shell agrees with the matching done above
# --------------------------------------------------------------------------
def test_the_helpers_match_fixed_strings():
    """Python's `in` is -F.  If the script ever went back to a regex, the tests
    above would keep passing while the script itself broke on any pattern
    containing [ ] . * or $."""
    src = _read('verify_upload.sh')
    assert src.count('grep -qF "$2"') == 2, 'a helper is not using -F'
    assert 'grep -q "$2"' not in src.replace('grep -qF "$2"', ''), \
        'a helper still matches as a regex'


def test_the_patterns_that_broke_are_covered():
    """Named individually, because they are the reason this file exists and a
    silent reversion would otherwise only surface on the cluster."""
    src = _read('verify_upload.sh')
    for pat in ('after_tgt = rows[modality]', 'CE_MODALITY_GROUPS: [0, 0, 1]'):
        assert pat in src, pat
    # a comment explaining why cp -r went away is not cp -r coming back
    assert 'cp -r "$OUT"' in src, 'the deletion check lost its specificity'
    assert "check_no  run_hihr.sh            'cp -r'" not in src


def test_the_files_every_check_names_exist():
    missing = sorted({p for _k, p, _pat in checks()
                      if not os.path.exists(os.path.join(ROOT, *p.split('/')))})
    assert not missing, missing


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d verify_upload tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
