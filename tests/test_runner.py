"""Unit tests for run_hihr.sh's checkpoint rescue.

Why a shell function gets its own test file.  On 2026-08-12 the shared drive
filled up DURING a run: three checkpoints were copied back at the same moment
and all three landed truncated -- 54M, 73M and 54M against roughly 338M.  The
launcher said "done", the training log was flawless, and the damage only
surfaced hours later as `PytorchStreamReader failed reading zip archive`, by
which point the worker and its /tmp were gone and two hours of training with
them.

So the copy is now verified and retried, and that logic is worth executing
rather than eyeballing: a size comparison written the wrong way round, or a
retry loop that returns success on the last failed attempt, would restore
exactly the silence this is meant to remove.  The function is lifted out of
run_hihr.sh and run in a real bash subshell against real files.

Run:  python tests/test_runner.py
"""
import os
import re
import subprocess
import sys
import tempfile

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


def copy_back_source():
    """The COPY_TRIES default plus the whole copy_back function, verbatim.

    Sliced out of the launcher rather than duplicated here, so the tests can
    only ever pass against the code that actually ships.
    """
    src = _read('run_hihr.sh')
    i = src.index('COPY_TRIES=${COPY_TRIES:-')
    j = src.index('\n}\n', src.index('copy_back() {')) + 3
    return src[i:j]


def run_bash(script, **env):
    tmp = tempfile.mkdtemp()
    full = copy_back_source() + '\n' + script
    r = subprocess.run(['bash', '-c', full], capture_output=True, text=True,
                       env=dict(os.environ, T=tmp, **env))
    return r


# --------------------------------------------------------------------------
# 1. it executes
# --------------------------------------------------------------------------
def test_a_good_copy_succeeds_and_reports_the_size():
    r = run_bash('''
printf 'x%.0s' $(seq 1 5000) > "$T/src.pth"
copy_back "$T/src.pth" "$T/out/src.pth" && echo OK || echo BAD
''')
    assert 'OK' in r.stdout, (r.stdout, r.stderr)
    assert 'copied 5000 bytes' in r.stdout, r.stdout


def test_a_truncated_destination_is_overwritten_not_accepted():
    """The exact shape of the failure: a file is there, it is just short.  A
    check that only tested for existence would call this a success."""
    r = run_bash('''
printf 'x%.0s' $(seq 1 5000) > "$T/src.pth"
mkdir -p "$T/out"; printf 'short' > "$T/out/src.pth"
copy_back "$T/src.pth" "$T/out/src.pth" && echo OK || echo BAD
test "$(stat -c%s "$T/out/src.pth")" = 5000 && echo SIZE_OK || echo SIZE_BAD
''')
    assert 'OK' in r.stdout and 'SIZE_OK' in r.stdout, (r.stdout, r.stderr)


def test_a_copy_that_cannot_succeed_returns_non_zero():
    """And says so on stderr, loudly enough to act on -- with the source path,
    because the only window to rescue it is while this worker is alive."""
    r = run_bash('''
printf 'x%.0s' $(seq 1 5000) > "$T/src.pth"
printf 'blocker' > "$T/blocked"
copy_back "$T/src.pth" "$T/blocked/nested.pth" && echo BAD || echo OK
''', COPY_TRIES='1')
    assert 'OK' in r.stdout, (r.stdout, r.stderr)
    assert 'INCOMPLETE COPY' in r.stderr, r.stderr
    assert 'GAVE UP' in r.stderr, r.stderr
    assert 'src.pth' in r.stderr, 'the source path is not named in the message'


def test_the_retry_count_is_honoured():
    """COPY_TRIES has to reach the loop; a hardcoded count would make the
    two-hour window a fiction and the tests above still pass."""
    r = run_bash('''
printf 'x%.0s' $(seq 1 50) > "$T/src.pth"
printf 'blocker' > "$T/blocked"
copy_back "$T/src.pth" "$T/blocked/nested.pth" 2>&1 | grep -c INCOMPLETE
''', COPY_TRIES='2')
    assert r.stdout.strip().endswith('2'), r.stdout



def test_an_existing_destination_directory_does_not_nest():
    """The second way this went wrong, on the same day as the first.

    `cp -r src dst` copies src INTO dst when dst already exists, so a re-run
    produced out_hihr_X/out_hihr_X/transformer_60.pth while the truncated file
    from the first attempt stayed at the expected path -- and every load kept
    finding the broken one.  Copying to an explicit file path cannot do that,
    and this pins it.
    """
    r = run_bash('''
printf 'x%.0s' $(seq 1 5000) > "$T/src.pth"
mkdir -p "$T/out"                       # destination already exists
copy_back "$T/src.pth" "$T/out/src.pth" && echo OK || echo BAD
test -f "$T/out/src.pth" && echo FLAT_OK || echo FLAT_BAD
test -e "$T/out/out" && echo NESTED_BAD || echo NO_NEST_OK
''')
    assert 'OK' in r.stdout and 'FLAT_OK' in r.stdout, (r.stdout, r.stderr)
    assert 'NO_NEST_OK' in r.stdout, r.stdout


def test_a_stale_file_at_the_destination_is_detected_not_trusted():
    """Both failures presented as the same RuntimeError hours later.  The size
    check catches a leftover regardless of which one produced it."""
    r = run_bash('''
printf 'x%.0s' $(seq 1 5000) > "$T/src.pth"
mkdir -p "$T/out"; printf 'y%.0s' $(seq 1 900) > "$T/out/src.pth"
copy_back "$T/src.pth" "$T/out/src.pth" >/dev/null &&   test "$(stat -c%s "$T/out/src.pth")" = 5000 && echo OK || echo BAD
''')
    assert 'OK' in r.stdout, (r.stdout, r.stderr)


# --------------------------------------------------------------------------
# 2. the launcher uses it, and the old silent path is gone
# --------------------------------------------------------------------------
def test_the_unverified_recursive_copy_is_gone():
    src = _read('run_hihr.sh')
    assert 'cp -r "$OUT"' not in src, 'the old unchecked cp -r is still there'
    assert 'copy_back "$f"' in src, 'the launcher does not call copy_back'


def test_the_default_window_is_two_hours():
    """120 attempts x 60s.  Chosen with the user: idle GPU on failure is far
    cheaper than a retrained run."""
    src = _read('run_hihr.sh')
    assert 'COPY_TRIES=${COPY_TRIES:-120}' in src, 'the default window changed'
    assert 'sleep 60' in src


def test_a_failed_copy_makes_the_launcher_exit_non_zero():
    """Otherwise the task ends looking successful and the next thing anyone
    learns is a corrupt archive hours later."""
    src = _read('run_hihr.sh')
    i = src.index('copy_fail=1')
    j = src.index('exit 1', i)
    assert i < j, 'a failed copy does not reach a non-zero exit'
    assert 'DID NOT SURVIVE THE COPY' in src


def test_the_message_says_the_results_are_not_lost():
    """They are not: the mAP, the CMC and the 3x3 are all in the training log
    on the shared drive.  Only the weights, needed for feature diagnostics, are
    at risk -- and someone reading this at 3am should not have to work that
    out."""
    src = _read('run_hihr.sh')
    assert 'only the weights are at risk' in src


# --------------------------------------------------------------------------
# 3. run_seq.sh -- several runs in one task
# --------------------------------------------------------------------------
def _seq_sandbox(script, modes, stub='exit 0'):
    """A throwaway repo with stub configs and a stub run_hihr.sh.

    run_seq.sh has REPO and DATA hardcoded to the cluster, so they are
    rewritten here rather than parameterised in the shipped file -- a script
    whose paths can be overridden by the environment is a script that can be
    pointed at the wrong drive by accident.
    """
    t = tempfile.mkdtemp()
    os.makedirs(os.path.join(t, 'configs'))
    os.makedirs(os.path.join(t, 'D'))
    open(os.path.join(t, 'D', 'ViT-B-16.pt'), 'w').close()
    for m in modes:
        with open(os.path.join(t, 'configs', 'hihr_%s.yml' % m), 'w') as f:
            f.write("MODEL:\n  PRETRAIN_CHOICE: 'imagenet'\nSOLVER:\n  MAX_EPOCHS: 60\n")
    with open(os.path.join(t, 'run_hihr.sh'), 'w') as f:
        f.write('#!/usr/bin/env bash\n%s\n' % stub)
    src = _read('run_seq.sh')
    # Two Windows-isms, both silent if missed.  The sandbox path is a Windows
    # temp dir, so (a) bash eats its backslashes -- C:\Users\... reaches the
    # script as C:Usders... and every path check then fails for the wrong
    # reason -- and (b) re.sub reads backslashes in a replacement STRING as
    # escapes, where `\U` raises "bad escape".  Forward slashes fix the first,
    # lambda replacements the second.
    posix = t.replace(os.sep, '/')
    src = re.sub(r'(?m)^REPO=.*$', lambda _m: 'REPO=%s' % posix, src)
    src = re.sub(r'(?m)^DATA=.*$', lambda _m: 'DATA=%s/D' % posix, src)
    with open(os.path.join(t, 'run_seq.sh'), 'w') as f:
        f.write(src)
    r = subprocess.run(['bash', 'run_seq.sh'] + list(script), cwd=t,
                       capture_output=True, text=True)
    return t, r


def test_seq_preflight_refuses_before_starting_anything():
    """The point of the check.  A typo in the last of five modes otherwise
    surfaces eight hours in, with the machine already spent."""
    t, r = _seq_sandbox(['a', 'typo'], ['a'])
    assert r.returncode == 1, r.stdout
    assert 'MISSING CONFIG' in r.stdout, r.stdout
    assert 'nothing was started' in r.stdout, r.stdout
    assert 'START a' not in r.stdout, 'it began the queue anyway'


def test_seq_skips_a_run_whose_checkpoint_is_already_there():
    """Resume.  If the task is reclaimed at run four, relaunching the same
    command must not redo the first three."""
    t, _ = _seq_sandbox(['a'], ['a', 'b'])
    os.makedirs(os.path.join(t, 'out_hihr_b'))
    open(os.path.join(t, 'out_hihr_b', 'transformer_60.pth'), 'w').close()
    r = subprocess.run(['bash', 'run_seq.sh', 'a', 'b'], cwd=t,
                       capture_output=True, text=True)
    assert 'SKIP b' in r.stdout, r.stdout
    assert 'START b' not in r.stdout, r.stdout
    assert 'START a' in r.stdout, r.stdout


def test_seq_keeps_going_after_one_run_fails_and_still_exits_non_zero():
    """run_hihr.sh exits non-zero on a failed copy-back.  Under `set -e` that
    would abandon the remaining runs even though the machine is fine -- and the
    whole reason this script exists is that machines are scarce."""
    t, r = _seq_sandbox(['a', 'b', 'c'], ['a', 'b', 'c'],
                        stub='[ "$1" = a ] && exit 1\nexit 0')
    assert 'FAILED a' in r.stdout, r.stdout
    assert 'START b' in r.stdout and 'START c' in r.stdout, r.stdout
    assert 'DONE  c' in r.stdout, r.stdout
    assert r.returncode == 1, 'a failed run did not reach the exit status'
    assert 'relaunch the same command' in r.stdout


def test_seq_output_survives_a_successful_task():
    """Same reason reeval.sh tees: a task that succeeds keeps no downloadable
    log on this platform, so progress has to be on the shared drive."""
    src = _read('run_seq.sh')
    assert 'SEQ_LOG:-$REPO/run_seq_log.txt' in src
    assert 'tee -a "$LOG"' in src
    assert '^ *PRETRAIN_CHOICE:' in src, 'the config probe is not anchored'


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_')]
    print('%d runner tests\n' % len(tests))
    for name, fn in tests:
        check(name, fn)
    print('\n%d/%d passed' % (len(PASS), len(PASS) + len(FAIL)))
    sys.exit(1 if FAIL else 0)
