"""Process-tree cleanup for host-side evaluation commands (Linux)."""
import os
from pathlib import Path
import signal
import subprocess
import time


def _snapshot():
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(path.name)] = (int(fields[1]), int(fields[2]), fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return rows


def terminate_tree(root_pid, grace=2.0):
    """Drain the command's session and descendant job groups before returning.

    Callers launch the root with start_new_session=True. Children launched in
    separate job sessions are found before the root is killed. Zombies cannot
    execute or write and are left for their parent/init to reap.
    """
    rows = _snapshot()
    members = {root_pid} | {pid for pid, (_, group, _) in rows.items() if group == root_pid}
    while True:
        added = {pid for pid, (parent, _, _) in rows.items() if parent in members}
        if added <= members:
            break
        members |= added
    groups = {root_pid} | {rows[pid][1] for pid in members if pid in rows}
    groups.discard(os.getpgrp())

    def send(sig):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        for pid in members:
            if pid in rows and rows[pid][1] == os.getpgrp() and pid != os.getpid():
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        current = _snapshot()
        if not any(state != 'Z' and (pid in members or group in groups)
                   for pid, (_, group, state) in current.items()):
            return
        time.sleep(0.05)
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = _snapshot()
        if not any(state != 'Z' and (pid in members or group in groups)
                   for pid, (_, group, state) in current.items()):
            return
        time.sleep(0.05)
    raise RuntimeError(f'Could not drain evaluation process tree {root_pid}')


def run_command(cmd, timeout=None, cwd=None, cancelled=lambda: False):
    """Capture output; deadline/cancellation always drains all child job groups."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    start = time.monotonic()
    code = None
    try:
        while True:
            if cancelled() or (timeout is not None and time.monotonic() - start >= timeout):
                code = 130 if cancelled() else 124
                terminate_tree(proc.pid)
                out, err = proc.communicate()
                err += '\nCancelled' if code == 130 else f'\nTimed out after {timeout}s'
                break
            try:
                out, err = proc.communicate(timeout=0.2)
                code = proc.returncode
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        terminate_tree(proc.pid)
        proc.wait()
    return subprocess.CompletedProcess(cmd, code, out, err)
