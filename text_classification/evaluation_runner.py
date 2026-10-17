"""Bounded evaluation attempts with isolated outputs and atomic publication."""
import asyncio
import json
import math
import os
from pathlib import Path
import uuid

from process_control import terminate_tree

OUTPUT_FLAGS = ('--log', '--save-memory', '--val-output', '--test-output')


def _option(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def _validate(flag, path, cmd):
    if not path.is_file():
        raise ValueError(f'Missing {flag}: {path}')
    if flag not in ('--val-output', '--test-output'):
        return
    data = json.loads(path.read_text())
    expected = _option(cmd, '--num-val' if flag == '--val-output' else '--num-test')
    acc, total = data.get('accuracy'), data.get('total')
    if not isinstance(acc, (int, float)) or not math.isfinite(acc) or not 0 <= acc <= 1:
        raise ValueError('Invalid result accuracy')
    if not isinstance(total, int) or total <= 0 or (expected is not None and total != int(expected)):
        raise ValueError(f'Incomplete result: total={total}, expected={expected}')
    for key, argument in [('dataset', '--dataset'), ('model', '--model'), ('memory', '--memory')]:
        wanted = _option(cmd, argument)
        if wanted is not None and data.get(key) != wanted:
            raise ValueError(f'Result identity mismatch: {key}')


async def run_job(cmd, log_path, max_retries=2, timeout=14400):
    """Only a successful, validated attempt can publish canonical outputs."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries + 1):
        attempt_dir = log_path.parent / '.attempts' / log_path.stem / uuid.uuid4().hex
        attempt_dir.mkdir(parents=True)
        attempt_cmd = list(cmd)
        outputs = []
        for flag in OUTPUT_FLAGS:
            if flag not in cmd:
                continue
            index = cmd.index(flag) + 1
            target = Path(cmd[index]).resolve()
            staged = attempt_dir / (flag.removeprefix('--') + '.artifact')
            attempt_cmd[index] = str(staged)
            outputs.append((flag, staged, target))
        if not any(flag in ('--val-output', '--test-output') for flag, _, _ in outputs):
            raise ValueError('Evaluation command has no score output')
        metadata = dict(attempt=attempt + 1, timeout_seconds=timeout, command=attempt_cmd,
                        outputs={flag: str(target) for flag, _, target in outputs})
        (attempt_dir / 'attempt.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nattempt={attempt + 1} timeout={timeout}s directory={attempt_dir}\n')
            log.flush()
            proc = await asyncio.create_subprocess_exec(
                *attempt_cmd, stdout=log, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)
            try:
                try:
                    code = await asyncio.wait_for(proc.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    code = 124
                except asyncio.CancelledError:
                    metadata['status'] = 'cancelled'
                    raise
            finally:
                # Reap before any retry or publication, even if uv exited first.
                await asyncio.to_thread(terminate_tree, proc.pid)
                await proc.wait()
            if code == 0:
                try:
                    for flag, staged, _ in outputs:
                        _validate(flag, staged, cmd)
                    # Publish state/trace first; each score file is the commit marker.
                    for flag, staged, target in outputs:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex)
                        # Preserve the attempt's own artifact for audit.
                        temporary.write_bytes(staged.read_bytes())
                        os.replace(temporary, target)
                    metadata['status'] = 'complete'
                except (OSError, ValueError, TypeError) as exc:
                    code = 65
                    log.write(f'Validation/publication failed: {exc}\n')
            metadata.update(exit_code=code, status=metadata.get('status', 'failed'))
            (attempt_dir / 'attempt.json').write_text(json.dumps(metadata, indent=2) + '\n')
            log.write(f'exit={code}\n')
        if code == 0:
            return True
    return False
