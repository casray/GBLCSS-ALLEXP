"""Archive an interrupted arm before restoring its own durable prefix log."""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARM = ROOT / 'fits/C1/coauthorcs/seed0/R0/pair0.4/S8_L1.0'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert ROOT.name == 'd1_c1_confirmation_20261024'
    assert ARM.resolve().is_relative_to(ROOT)
    assert not (HERE / 'receipt.json').exists(), 'Recovery already applied'
    lock = json.loads((ROOT / 'run.lock').read_text())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, lock['pid'])
    exit_code = None
    if handle:
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        kernel.CloseHandle(ctypes.c_void_p(handle))
        assert ok and code.value != 259, 'Lock process is active or unverifiable'
        exit_code = code.value
    else:
        assert ctypes.get_last_error() == 87, 'PID absence not established'
    sources = json.loads((ROOT / 'sources.json').read_text(encoding='utf-8'))
    assert all(digest(Path(p)) == h for p, h in sources.items()), 'Frozen source changed'
    result = json.loads((ARM / 'result.json').read_text())
    ck = torch.load(ARM / 'resume.pt', map_location='cpu', weights_only=False)
    lines = (ARM / 'epochs.jsonl').read_bytes().splitlines(keepends=True)
    logs = [json.loads(line) for line in lines]
    assert ck['epoch'] == result['epoch'] == 75
    assert len(logs) == 163
    assert [r['epoch'] for r in logs] == list(range(1, 164))
    for name, key in [('checkpoint.pt', 'checkpoint_sha256'), ('probabilities.pt', 'probabilities_sha256')]:
        assert digest(ARM / name) == result[key]
    assert ck['meta']['arm'] == 'S8_L1.0' and ck['meta']['seed'] == 0
    assert ck['meta']['source_sha256'] == digest(ROOT / 'sources.json')
    assert ck['best'] == max(row['validation_correct'] for row in logs[:75])
    assert ck['best_epoch'] == next(row['epoch'] for row in logs[:75] if row['validation_correct'] == ck['best'])
    archive = HERE / 'original'
    archive.mkdir()
    shutil.copytree(ARM, archive / 'arm')
    for name in ['run.lock', 'console.log', 'status.json', 'HANDOFF.md', 'environment.json',
                 'sources.json', 'fit_registry.json', 'failure_latest.json', 'failures.jsonl',
                 'D1_complete.json']:
        path = ROOT / name
        if path.exists():
            shutil.copy2(path, archive / name)
    manifest = {str(p.relative_to(archive)): digest(p) for p in archive.rglob('*') if p.is_file()}
    assert digest(archive / 'arm/epochs.jsonl') == digest(ARM / 'epochs.jsonl')
    receipt = dict(time=datetime.now(timezone.utc).isoformat(), reason='process_absent_external_interruption',
                   old_pid=lock['pid'], old_exit_code=exit_code, arm=str(ARM.relative_to(ROOT)), restored_epoch=75,
                   last_logged_epoch=163, observed_discarded_updates=88,
                   unlogged_updates='unknown', discarded_segment_seconds='not persisted; unknown',
                   state_source='same arm resume.pt: model, optimizer, best state, all RNG',
                   source_manifest_sha256=digest(ROOT / 'sources.json'), archive_hashes=manifest)
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (ARM / 'epochs.jsonl').write_bytes(b''.join(lines[:75]))
    assert len((ARM / 'epochs.jsonl').read_text().splitlines()) == 75
    assert digest(ARM / 'resume.pt') == manifest['arm\\resume.pt']
    (ROOT / 'run.lock').unlink()
    with (ROOT / 'interruptions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'archive_hashes'}, indent=2))


if __name__ == '__main__':
    main()
