"""Final archive-scope and additional credential review; never edits original sources."""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re
import shutil
import time

from build_archive import DEST,HERE,CODE_EXT

EXTRA=[
    ('google_api_key',re.compile(rb'\bAIza[0-9A-Za-z_-]{35}\b')),
    ('huggingface_token',re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b')),
    ('slack_token',re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{25,}\b')),
    ('assigned_credential',re.compile(rb'''(?i)["']?(?:api_key|access_token|client_secret)["']?\s*[:=]\s*["']([A-Za-z0-9_./+-]{40,})["']''')),
]


def signatures(path,compressed=False):
    result=set()
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        tail=b''
        for block in iter(lambda:f.read(1024*1024),b''):
            chunk=tail+block
            for name,pattern in EXTRA:
                if pattern.search(chunk):
                    result.add(name)
            tail=chunk[-2048:]
    return sorted(result)


def main():
    if (DEST/'manifests/post_review.json').exists():
        raise RuntimeError('Final review already completed')
    with (DEST/'manifests/files.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    removed=[]
    checked=0
    for row in rows:
        if row['action']!='upload':
            continue
        relative=Path(row['source_relative'])
        parts={p.lower() for p in relative.parts}
        name=relative.name.lower()
        reason=None
        findings=[]
        if 'code_data' in parts and 'raw' in parts and relative.suffix.lower() not in CODE_EXT:
            reason='raw Planetoid dataset component retained locally'
        elif row['category']=='text' or (row['category']=='compressed_package' and name.endswith('.gz')):
            findings=signatures(DEST/row['archive_path'],row['category']=='compressed_package')
            checked+=1
            if findings:
                reason='additional credential signatures: '+','.join(findings)
        if reason:
            target=(DEST/row['archive_path']).resolve()
            if not target.is_relative_to((DEST/'sources').resolve()):
                raise RuntimeError('Refusing to remove outside created archive sources')
            target.unlink()
            row['action']='local_security_review' if findings else 'local_dataset'
            row['reason']=reason
            row['archive_path']=''
            removed.append(dict(source_relative=row['source_relative'],reason=reason,patterns=findings))
    with (DEST/'manifests/files.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (DEST/'manifests/retained_locally.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            if row['action']!='upload':
                item=dict(row);item['bytes']=int(item['bytes'])
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
    summary=json.loads((DEST/'manifests/archive_summary.json').read_text(encoding='utf-8'))
    old_count,old_bytes=summary['copied_source_files'],summary['copied_source_bytes']
    actions=defaultdict(lambda:dict(files=0,bytes=0))
    for row in rows:
        actions[row['action']]['files']+=1;actions[row['action']]['bytes']+=int(row['bytes'])
    summary['actions']=dict(actions)
    summary['copied_source_files']=actions['upload']['files']
    summary['copied_source_bytes']=actions['upload']['bytes']
    summary['credential_quarantined_file_count']=sum(r['action']=='local_security_review' for r in rows)
    summary['final_review_completed_unix']=time.time()
    (DEST/'manifests/archive_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=(DEST/'README.md').read_text(encoding='utf-8')
    readme=readme.replace(f'复制了 {old_count:,} 个源文件（{old_bytes/1e9:.3f} GB，十进制）。',
                          f'复制了 {summary["copied_source_files"]:,} 个源文件（{summary["copied_source_bytes"]/1e9:.3f} GB，十进制）。')
    readme+='\n另有9份下载目录中的GB-LCSS稿件原件，见 [补充材料](supplemental/README.md) 和独立校验清单。\n'
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    report=dict(status='passed',additional_text_files_scanned=checked,removed_created_copies=removed,
                original_source_files_modified=False,final_uploaded_source_files=summary['copied_source_files'])
    (DEST/'manifests/post_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(__file__,DEST/'tools/post_review.py')
    shutil.copy2(HERE/'test_archive.py',DEST/'tools/test_archive.py')
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
