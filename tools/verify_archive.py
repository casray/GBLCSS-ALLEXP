"""Check source-byte manifests against the actual staged Git objects before upload."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.time()
    with (ROOT/'manifests/files.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len({r['source_relative'] for r in rows})==len(rows)
    source=[r for r in rows if r['action']=='upload']
    index=subprocess.run(['git','ls-files','-s','-z'],cwd=ROOT,capture_output=True,check=True).stdout
    entries={}
    for entry in index.split(b'\0'):
        if entry:
            head,path=entry.split(b'\t',1)
            entries[path.decode('utf-8')]=head.decode('ascii').split()[1]
    expected={r['archive_path'] for r in source}
    assert {p for p in entries if p.startswith('sources/')}==expected,'Staged source path set differs'
    total=0
    for i,row in enumerate(source,1):
        path=ROOT/row['archive_path']
        sha256=hashlib.sha256()
        gitblob=hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                sha256.update(block);gitblob.update(block)
        assert path.stat().st_size==int(row['bytes']),row['archive_path']
        assert sha256.hexdigest()==row['sha256'],'Source SHA256 differs: '+row['archive_path']
        assert gitblob.hexdigest()==entries[row['archive_path']],'Git filtering changed file bytes: '+row['archive_path']
        total+=path.stat().st_size
        if i%5000==0:
            print(f'VERIFIED {i}/{len(source)}',flush=True)
    summary=json.loads((ROOT/'manifests/archive_summary.json').read_text(encoding='utf-8'))
    assert summary['source_files']==len(rows)
    assert summary['copied_source_files']==len(source)
    assert summary['copied_source_bytes']==total
    assert not any(r['archive_path'] for r in rows if r['action']!='upload')
    supplemental=json.loads((ROOT/'manifests/supplemental_files.json').read_text(encoding='utf-8'))
    for row in supplemental:
        if row['action']!='upload':
            assert not row['archive_path']
            continue
        path=ROOT/row['archive_path']
        content=path.read_bytes()
        assert hashlib.sha256(content).hexdigest()==row['sha256']
        expected_blob=hashlib.sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()
        assert entries[row['archive_path']]==expected_blob
    report=dict(status='passed',source_manifest_rows=len(rows),uploaded_source_files=len(source),
                source_bytes=total,staged_files=len(entries),source_paths_exactly_match=True,
                source_sha256_exact=True,git_blob_bytes_exact=True,local_only_items_not_staged=True,
                supplemental_uploaded_files=sum(r['action']=='upload' for r in supplemental),
                seconds=time.time()-started,publication_status='verified locally; remote commit checked after push')
    (ROOT/'manifests/local_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
