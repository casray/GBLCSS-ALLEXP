"""Build a reviewed historical archive without editing original experiment files."""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=Path('C:/Users/Admin/Desktop/实验8去噪5.19-')
DEST=Path('C:/Users/Admin/Desktop/GBLCSS-ALLEXP')
WORKSPACE='gb_c1_ratio_success_20260609/results_ratio_c1_20260609'
CODE_EXT={'.py','.r','.jl','.m','.cpp','.c','.h','.hpp','.js','.ts','.sh','.ps1','.bat','.cmd','.md','.rst'}
JUNK_PARTS={'tectonic_cache','tectonic-cache','texmf','__pycache__','.ipynb_checkpoints',
            '.pytest_cache','.mypy_cache','.ruff_cache','node_modules','.venv','venv'}
BUILD_EXT={'.aux','.toc','.fls','.fdb_latexmk','.xdv','.dvi','.bcf','.blg','.idx','.ilg','.ind','.lof','.lot','.pyc'}
PATTERNS=[
    ('private_key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('github_token',re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b')),
    ('openai_style_key',re.compile(rb'\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{40,}\b')),
    ('aws_access_key',re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('credential_url',re.compile(rb'https?://[^\s/"\x27:@]{2,}:[^\s/"\x27@]{5,}@')),
    ('bearer_credential',re.compile(rb'(?i)\bBearer\s+(?:eyJ[A-Za-z0-9_.-]{35,}|[A-Za-z0-9_-]{48,})')),
]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def policy(row):
    rel=Path(row['relative'])
    parts=set(rel.parts)
    lower=rel.name.lower()
    ext=rel.suffix.lower()
    if row['category']=='sensitive_name':
        return 'local_sensitive','sensitive filename'
    if parts & JUNK_PARTS or ext in BUILD_EXT or lower.endswith('.synctex.gz'):
        return 'local_generated','dependency/compiler/cache artifact'
    if row['category']=='model_or_array':
        return 'local_artifact','weights/arrays/cache retained locally by user choice'
    if ('data' in parts or 'datasets' in parts) and ext not in CODE_EXT and lower not in {'license','license.txt','readme','readme.txt'}:
        return 'local_dataset','raw/downloaded dataset retained locally by user choice'
    if row['category']=='compressed_package':
        if lower.endswith(('.jsonl.gz','.json.gz','.csv.gz','.log.gz','.txt.gz')):
            return 'upload','compressed experimental text record'
        return 'local_package','source-package duplicate or unrelated reference bundle'
    if row['bytes']>=100*1024*1024:
        return 'local_oversize','exceeds GitHub per-file size limit'
    return 'upload','code/config/result/log/document/history'


def scan(path,compressed=False):
    findings=set()
    source=gzip.open(path,'rb') if compressed else Path(path).open('rb')
    with source:
        tail=b''
        for block in iter(lambda:source.read(1024*1024),b''):
            content=tail+block
            for name,pattern in PATTERNS:
                if pattern.search(content):
                    findings.add(name)
            tail=content[-1024:]
    return sorted(findings)


def main():
    if not (DEST/'.git').is_dir() or (DEST/'manifests/archive_summary.json').exists():
        raise RuntimeError('Require the independently cloned, not-yet-built archive repository')
    started=time.time()
    rows=[json.loads(line) for line in (HERE/'inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    for folder in ('sources','manifests','tools'):
        (DEST/folder).mkdir(exist_ok=True)
    manifest=[]
    quarantined=[]
    stats=defaultdict(lambda:dict(files=0,bytes=0))
    uploads=[]
    for i,row in enumerate(rows,1):
        source=PROJECT/row['relative']
        before=source.stat()
        action,reason=policy(row)
        if action=='upload':
            is_text=row['category']=='text'
            compressed=row['category']=='compressed_package'
            if is_text or compressed:
                findings=scan(source,compressed)
                if findings:
                    action,reason='local_security_review','potential credential signature: '+','.join(findings)
                    quarantined.append(dict(relative=row['relative'],patterns=findings,bytes=row['bytes']))
            elif row['category']=='other':
                sample=source.read_bytes()[:8192] if row['bytes']<8192 else source.open('rb').read(8192)
                if b'\0' in sample:
                    action,reason='local_binary_other','unclassified binary retained locally'
                else:
                    findings=scan(source)
                    if findings:
                        action,reason='local_security_review','potential credential signature: '+','.join(findings)
                        quarantined.append(dict(relative=row['relative'],patterns=findings,bytes=row['bytes']))
        digest=sha(source)
        destination='sources/'+row['relative'] if action=='upload' else ''
        if action=='upload':
            target=DEST/destination
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise RuntimeError('Refuse to overwrite existing destination: '+destination)
            shutil.copy2(source,target)
            if sha(target)!=digest:
                raise RuntimeError('Copied content differs: '+row['relative'])
            uploads.append(destination)
        after=source.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise RuntimeError('Source changed while archiving: '+row['relative'])
        record=dict(source_relative=row['relative'],archive_path=destination,bytes=after.st_size,sha256=digest,
                    category=row['category'],action=action,reason=reason)
        manifest.append(record)
        stats[action]['files']+=1;stats[action]['bytes']+=after.st_size
        if i%2000==0:
            print(f'REVIEWED {i}/{len(rows)} copied={len(uploads)} elapsed={time.time()-started:.1f}s',flush=True)
    with (DEST/'manifests/files.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]))
        writer.writeheader();writer.writerows(manifest)
    with (DEST/'manifests/retained_locally.jsonl').open('w',encoding='utf-8') as f:
        for r in manifest:
            if r['action']!='upload':
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=dict(destination='https://github.com/casray/GBLCSS-ALLEXP',source_root=str(PROJECT),
                 snapshot_started_unix=started,source_files=len(rows),source_bytes=sum(r['bytes'] for r in manifest),
                 actions=dict(stats),source_files_modified=False,excluded_hidden_directories=json.loads((HERE/'inventory_summary.json').read_text(encoding='utf-8'))['skipped_directories'],
                 credential_quarantined_file_count=len(quarantined),user_choice='code/results/records plus full local artifact manifest; no weights or raw datasets',
                 copied_source_files=len(uploads),copied_source_bytes=stats['upload']['bytes'],build_seconds=time.time()-started)
    (DEST/'manifests/archive_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'security_review.json').write_text(json.dumps(quarantined,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'upload_paths.json').write_text(json.dumps(uploads,ensure_ascii=False),encoding='utf-8')
    # Publish only pattern classes and paths, never matching credential values.
    (DEST/'manifests/security_exclusions.json').write_text(json.dumps(quarantined,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('inventory.py','build_archive.py','PLAN.md'):
        shutil.copy2(HERE/name,DEST/'tools'/name)
    experiments=PROJECT/WORKSPACE/'experiments'
    index=['# Historical Experiment Index','',
           'Directory names are historical IDs, not assertions that every branch succeeded. Read its reports and negative findings.',
           '', '| Experiment directory | Reports / entry points | Uploaded source files | Local-only artifacts |','|---|---|---:|---:|']
    dirs=sorted(p for p in experiments.iterdir() if p.is_dir())
    for folder in dirs:
        prefix=WORKSPACE+'/experiments/'+folder.name+'/'
        selected=[r for r in manifest if r['source_relative'].startswith(prefix)]
        candidates=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.md']
        preferred=sorted(candidates,key=lambda p:(0 if p.name in {'综合结论与接续.md','实验结论.md','REPORT.md','99_session_handoff.md','HANDOFF.md','PLAN.md'} else 1,p.name))[:4]
        links='; '.join(f'[{p.name}](sources/{prefix}{p.name})' for p in preferred) or 'See preserved directory and manifest'
        index.append(f'| [{folder.name}](sources/{WORKSPACE}/experiments/{folder.name}) | {links} | {sum(r["action"]=="upload" for r in selected)} | {sum(r["action"]!="upload" for r in selected)} |')
    (DEST/'EXPERIMENT_INDEX.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    root_source='sources/'+WORKSPACE
    readme=f'''# GBLCSS-ALLEXP

GB-LCSS and related GB–GCL research archive, assembled on 2026-10-21 from the owner's historical workspace.
本仓库保留历史代码、配置、日志、结果、实验复盘和稿件版本，包括失败实验；不是只展示最终成功结果的精简代码仓库。

## Start here / 阅读入口

- [实验目录索引](EXPERIMENT_INDEX.md)：当前主实验目录下的 {len(dirs)} 个历史分支。
- [项目核心上下文](sources/项目核心总结_新窗口上下文.md)、[项目交接](sources/99_session_handoff.md)。
- [历史实验记录](sources/experiment_history_and_lessons)、[文献研究地图](sources/01_literature_and_research_map.md)。
- [GB-LCSS implementation]({root_source}/GB-LCSS_code)、[全部主实验目录]({root_source}/experiments)。
- [后续研究与复盘]({root_source}/research)、[更早的GRACE实验](sources/gb_graph_aware_minival_6_1)。
- [近期R2机制对照]({root_source}/experiments/r2_mechanism_controls_20261021/综合结论与接续.md)。
- [边界诊断与粗化复盘]({root_source}/experiments/r2_boundary_diagnostic_20261021/综合结论与接续.md)。

## What is actually included / 归档范围

复制了 {len(uploads):,} 个源文件（{stats['upload']['bytes']/1e9:.3f} GB，十进制）。
`sources/` 下保留相对原项目根目录的路径，区分原方法、后续尝试、基线实现与历史版本。
模型权重、数组缓存和原始数据按所有者选择保留本地，**没有声称它们已上传**。
完整盘点 {len(rows):,} 个文件，逐文件路径、大小、SHA-256及上传/留存原因见 [files.csv](manifests/files.csv)。
[retained_locally.jsonl](manifests/retained_locally.jsonl) 专门列出本地留存项；
[archive_summary.json](manifests/archive_summary.json) 给出数量与排除范围。

环境、依赖缓存、Git内部对象、编译临时文件、重复分发压缩包不作为实验源文件上传。
潜在凭据签名匹配的文件需人工审核，未公开其内容；清单只给路径与检测类型。
参考论文库、NIDS和开放意图独立课题未并入本仓库。相关图对比前身与第三方对照源码保留供追溯。

## Reproduction and interpretation / 复现边界

这是忠实的历史快照，不是承诺一条命令即可重跑全部实验的统一软件包。旧脚本中的绝对Windows路径没有擅自改写。
原根目录为 `C:/Users/Admin/Desktop/实验8去噪5.19-`；迁移到其他机器需按各分支说明调整路径与环境，
并另行取得清单中的本地数据/权重。不要仅因目录里存在结果就认定该方向通过了成功门槛。
单种子、开发集反复使用、不同骨干/部署条件和负结果均须以原分支报告为准。

## Third-party material / 第三方代码

`sources/external/`、`upstream/`、`vendor/` 等保留对应源代码中已有的LICENSE及来源说明。
本仓库不为所有第三方文件统一重新许可，也不把基线源码宣称为本项目原创。

## Integrity / 完整性

源文件复制前后SHA-256一致，原实验文件不改动。上传完成后核对Git树中的文件字节与清单。
归档脚本及范围规则在 [tools/](tools/)，上传状态与远端核验信息随后写入仓库记录。
'''
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    (DEST/'.gitattributes').write_text('* -text\n',encoding='ascii')
    (DEST/'.gitignore').write_text('__pycache__/\n*.pyc\n.env\n.env.*\n*.pem\n*.pfx\n',encoding='ascii')
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
