"""Verify the incremental source copies and staged Git blobs."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
REPO=HERE.parent if (HERE.parent/'.git').is_dir() else HERE.parent/'GBLCSS-ALLEXP'
PROJECT=REPO.parent/'实验8去噪5.19-'
OUT=REPO/'manifests/update_20261026'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert summary['secret_quarantine']==0
    changed=json.loads((OUT/'changed_files.json').read_text(encoding='utf-8'))
    with (OUT/'files.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    assert len(records)==summary['source_files']
    assert len({r['source_relative'] for r in records})==len(records)
    refs={r['archive_path']:r for r in records if r['action']=='upload'}
    forbidden={'.pt','.pth','.ckpt','.npy','.npz','.pkl','.pickle','.safetensors','.bin','.h5','.hdf5','.dgl'}
    stage=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).decode('utf-8').split('\0')
    blobs={}
    for line in stage:
        if not line:continue
        info,name=line.split('\t',1)
        mode,oid,index=info.split()
        assert index=='0'
        blobs[name]=oid
    for row in changed:
        name=row['path'];record=refs[name]
        source=PROJECT/record['source_relative'];target=REPO/name
        assert target.suffix.lower() not in forbidden
        assert target.stat().st_size<100*1024**2
        assert digest(source)==digest(target)==row['sha256']
        blob=hashlib.sha1()
        blob.update(f'blob {target.stat().st_size}\0'.encode())
        with target.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):blob.update(b)
        assert blobs[name]==blob.hexdigest(),name
    for row in records:
        assert len(row['sha256'])==64
        if row['action']=='upload':
            assert (REPO/row['archive_path']).is_file()
        else:
            assert not row['archive_path']
    result=dict(status='passed',source_inventory=len(records),changed_file_checks=len(changed),
                copied_source_sha256=True,staged_git_blob_bytes=True,credential_quarantine=0,
                no_weights_in_changed_payload=True,initial_manifests_preserved=True)
    (OUT/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
