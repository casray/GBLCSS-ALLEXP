"""Preserve an interrupted attempt and align only logs beyond a saved prefix."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = Path(json.loads((CODE/'config.json').read_text(encoding='utf-8'))['runtime'])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prefix_bytes(raw, epoch):
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Nonsequential log')
    if len(rows) < epoch:
        raise ValueError('Log shorter than checkpoint')
    return b''.join(lines[:epoch]), len(rows)-epoch


def processes():
    cmd = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?(\\.exe)?$' } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    p = subprocess.run(['powershell','-NoProfile','-Command',cmd], capture_output=True, text=True, check=True)
    rows = json.loads(p.stdout or '[]')
    rows = [rows] if isinstance(rows, dict) else rows
    byid = {int(r['ProcessId']): r for r in rows}
    own = {os.getpid()}
    current = os.getpid()
    while current in byid:
        parent = int(byid[current]['ParentProcessId'])
        if parent not in byid or parent in own:
            break
        own.add(parent)
        current = parent
    return [r for r in rows if int(r['ProcessId']) not in own]


def recover(apply=False):
    owners = processes()
    if owners:
        raise RuntimeError('Other Python owners present: '+json.dumps(owners))
    if (ROOT/'fits_complete.json').exists():
        raise RuntimeError('Fitting already complete')
    for manifest in ('sources.json','postprocessing_sources.json'):
        for name, digest in read(ROOT/manifest).items():
            if sha(Path(name)) != digest:
                raise ValueError('Frozen source mismatch: '+name)
    pending = []
    folders = list((ROOT/'fits').glob('*/*/seed*/*/*'))
    for f in folders:
        result = read(f/'result.json') if (f/'result.json').exists() else None
        raw = (f/'epochs.jsonl').read_bytes() if (f/'epochs.jsonl').exists() else b''
        if not result:
            if raw:
                raise ValueError('Uncheckpointed new trajectory: '+str(f))
            continue
        epoch = result['epoch']
        head, excess = prefix_bytes(raw, epoch)
        for name, field in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            if sha(f/name) != result[field]:
                raise ValueError('Artifact mismatch: '+str(f/name))
        if epoch < 200:
            import torch
            ck = torch.load(f/'resume.pt', map_location='cpu', weights_only=False)
            if ck['epoch'] != epoch or any(ck['meta'][k] != result[k] for k in ('seed','input_hash','target_hash','source_sha256','initial_hash')):
                raise ValueError('Resume passport mismatch: '+str(f))
            rel = f.relative_to(ROOT/'fits/teacher')
            snapshot = ROOT/'prefix_snapshots'/rel/str(epoch)
            m = read(snapshot/'manifest.json')
            if sha(f/'resume.pt') != m['resume_sha256'] or sha(f/'result.json') != m['result_sha256']:
                raise ValueError('Prefix snapshot mismatch: '+str(f))
        if excess:
            if epoch not in (25,75):
                raise ValueError('Unexpected interrupted epoch: '+str(f))
            pending.append((f, head, excess))
    summary = dict(time=datetime.now(timezone.utc).isoformat(), verified_trajectories=len(folders),
                   repairs=[dict(folder=str(f), saved_epoch=read(f/'result.json')['epoch'], replay_updates=n) for f,_,n in pending],
                   completed_cells=len(list((ROOT/'selections/teacher').rglob('*.json'))),
                   active_other_python=owners, science_changed=False, applied=apply)
    if not apply:
        print(json.dumps(summary, indent=2))
        return summary
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT/'interruption_recovery'/stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('run.lock','launcher.lock','status.json','launcher_status.json','refs.json','physical_fits.json',
                 'background_stdout.log','background_stderr.log','launcher_stdout.log','launcher_stderr.log'):
        if (ROOT/name).exists():
            shutil.copy2(ROOT/name, archive/name)
    for f,head,_ in pending:
        dest = archive/'attempt'/f.relative_to(ROOT)
        shutil.copytree(f, dest)
        hashes = {p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()}
        for name,digest in hashes.items():
            if sha(f/name) != digest:
                raise ValueError('Archive copy mismatch')
        (dest/'archive_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        temp = f/'epochs.jsonl.recovery.tmp'
        temp.write_bytes(head)
        temp.replace(f/'epochs.jsonl')
    for name in ('run.lock','launcher.lock'):
        if (ROOT/name).exists():
            (ROOT/name).replace(archive/(name+'.stale'))
    summary['archive'] = str(archive)
    summary['extra_interrupted_updates'] = sum(n for _,_,n in pending)
    summary['reason'] = 'Workers absent; process exit cause unknown. Restore log to exact optimizer/RNG prefix; preserve interrupted work as extra cost.'
    (archive/'recovery.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (ROOT/'recovery_latest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (CODE/'recovery_receipt.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    recover(parser.parse_args().apply)
