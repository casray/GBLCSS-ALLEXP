"""Incremental archival snapshot using the owner's existing upload policy."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent/'.git').is_dir() else HERE.parent/'GBLCSS-ALLEXP'
PROJECT = REPO.parent/'实验8去噪5.19-'
WORK = Path('gb_c1_ratio_success_20260609/results_ratio_c1_20260609')
STAMP = '20261026'
OUT = REPO/'manifests'/('update_'+STAMP)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


policy = module('archive_policy_readonly',HERE/'build_archive.py')
inventory = module('archive_inventory_readonly',HERE/'inventory.py')


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def enumerate_sources():
    base = PROJECT/WORK
    for dirname in ('experiments','research','docs','scripts','tests','GB-LCSS_code'):
        for parent,dirs,files in os.walk(base/dirname):
            dirs[:] = sorted(d for d in dirs if d not in inventory.SKIP and d not in policy.JUNK_PARTS)
            for name in sorted(files):
                path=Path(parent)/name
                if not path.is_symlink():
                    yield path
    for path in sorted(base.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.md','.json','.toml','.yaml','.yml'}:
            yield path


def main():
    assert (REPO/'.git').is_dir()
    if OUT.exists():
        raise RuntimeError('Snapshot exists; inspect it before any explicit resume')
    oldstat = {r['relative']:r for r in map(json.loads,(HERE/'inventory.jsonl').read_text(encoding='utf-8').splitlines())}
    with (REPO/'manifests/files.csv').open(encoding='utf-8-sig',newline='') as f:
        oldhash = {r['source_relative']:r for r in csv.DictReader(f)}
    records,changed,quarantine=[],[],[]
    stats=Counter()
    for i,source in enumerate(enumerate_sources(),1):
        before=source.stat()
        relative=source.relative_to(PROJECT).as_posix()
        category=inventory.classify(source,before.st_size)
        action,reason=policy.policy(dict(relative=relative,category=category,bytes=before.st_size))
        old=oldstat.get(relative)
        prior=oldhash.get(relative)
        cachevalid=old and prior and old['mtime_ns']==before.st_mtime_ns and old['bytes']==before.st_size and int(prior['bytes'])==before.st_size
        digest=prior['sha256'] if cachevalid else policy.sha(source)
        target=REPO/'sources'/relative
        archive_path=''
        changed_kind=''
        if action=='upload':
            identical=target.exists() and policy.sha(target)==digest
            if not identical:
                if category=='text' or category=='compressed_package':
                    matches=policy.scan(source,category=='compressed_package')
                else:
                    matches=[]
                    if category=='other':
                        with source.open('rb') as f:
                            binary=b'\0' in f.read(8192)
                        if binary:
                            action,reason='local_binary_other','unclassified binary retained locally'
                        else:
                            matches=policy.scan(source)
                if matches:
                    action,reason='local_security_review','credential signature quarantine'
                    quarantine.append(dict(source_relative=relative,pattern_classes=matches))
            if action=='upload':
                archive_path=target.relative_to(REPO).as_posix()
                if not identical:
                    changed_kind='modified' if target.exists() else 'added'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
                    assert policy.sha(target)==digest
                    changed.append(dict(path=archive_path,sha256=digest,bytes=before.st_size,change=changed_kind))
        after=source.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),relative
        records.append(dict(source_relative=relative,archive_path=archive_path,bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,sha256=digest,category=category,action=action,reason=reason,
            sha256_provenance='initial_inventory_metadata_cache' if cachevalid else 'recomputed',change=changed_kind))
        stats[action]+=1
        if i%5000==0:
            print(json.dumps(dict(reviewed=i,changed=len(changed),actions=dict(stats))),flush=True)
    OUT.mkdir(parents=True)
    with (OUT/'files.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with (OUT/'retained_locally.jsonl').open('w',encoding='utf-8') as f:
        for r in records:
            if r['action']!='upload':
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
    dump(OUT/'changed_files.json',changed)
    dump(OUT/'security_exclusions.json',quarantine)
    summary=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),scope=str(WORK),
        source_files=len(records),actions=dict(stats),added=sum(r['change']=='added' for r in changed),
        modified=sum(r['change']=='modified' for r in changed),changed_bytes=sum(r['bytes'] for r in changed),
        retained_local_files=sum(r['action']!='upload' for r in records),
        retained_local_bytes=sum(r['bytes'] for r in records if r['action']!='upload'),
        secret_quarantine=len(quarantine),historical_archive_preserved=True,source_files_modified=False,
        initial_manifest='manifests/files.csv remains the initial snapshot; this folder records current scoped files',
        hash_policy='initial size+mtime exact matches reuse original SHA; new/changed files recomputed',
        allowed='code/config/results/logs/research records; weights/arrays/raw data stay local')
    dump(OUT/'summary.json',summary)
    shutil.copy2(Path(__file__),REPO/'tools'/Path(__file__).name)
    roots=REPO/'sources'/WORK/'experiments'
    lines=['# Experiment Index','',
        'Updated 2026-10-26. Includes completed, negative and exploratory branches. Start with each report and audit.',
        '', '| Experiment | Reports | Archived files |','|---|---|---:|']
    for folder in sorted(p for p in roots.iterdir() if p.is_dir()):
        docs=sorted((p for p in folder.glob('*.md')),key=lambda p:(0 if p.name in {'REPORT_zh.md','综合结论与接续.md','实验结论.md','REPORT.md','PARENT_REVIEW.md'} else 1,p.name))[:3]
        link=lambda p:p.relative_to(REPO).as_posix()
        links='; '.join(f'[{p.name}](<{link(p)}>)' for p in docs) or 'See directory'
        lines.append(f'| [{folder.name}](<{link(folder)}>) | {links} | {sum(1 for p in folder.rglob("*") if p.is_file())} |')
    (REPO/'EXPERIMENT_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
