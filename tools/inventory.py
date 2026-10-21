"""Read-only source inventory; only writes archive planning outputs beside this script."""
from __future__ import annotations

from collections import Counter,defaultdict
import json
import os
from pathlib import Path

PROJECT=Path('C:/Users/Admin/Desktop/实验8去噪5.19-')
OUTPUT=Path(__file__).resolve().parent
ROOTS=[
    'gb_c1_ratio_success_20260609','gb_graph_aware_minival_6_1',
    'gb_denoise_stage_a_negfree_vs_infonce','gb_denoise_stage_b_compactness',
    'gb_denoise_stage_c_confidence_soft_relation','gb_noisy_label_validation',
    'gb_graphcl_noisy_graphcls','external','experiment_history_and_lessons','docs',
    'deepresearch6.1','GPT深度思考包_SoftDist_20260629','next_paper',
]
SKIP={'.git','.svn','.hg','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache',
      '.venv','venv','env','node_modules','.claude','.codex','.aws','.ssh',
      '.drawio-tmp','.codex_tmp'}
TEXT={'.py','.md','.txt','.json','.jsonl','.csv','.tsv','.yaml','.yml','.toml','.ini','.cfg',
      '.sh','.ps1','.bat','.cmd','.tex','.bib','.sty','.cls','.bst','.bbl','.html','.htm','.xml',
      '.drawio','.svg','.log','.out','.err','.rst','.ipynb','.r','.jl','.m','.cpp','.c','.h','.hpp',
      '.js','.ts','.css','.sln','.vcxproj','.in','.sql','.gitignore','.gitattributes','.gitmodules',
      '.lock','.sum','.cmake','.patch','.diff','.srt'}
BINARY={'.pt','.pth','.ckpt','.npz','.npy','.pkl','.pickle','.bin','.h5','.hdf5','.safetensors','.dgl'}
PACKAGES={'.zip','.rar','.7z','.tar','.gz','.xz','.bz2','.tgz'}
MEDIA={'.png','.jpg','.jpeg','.pdf','.tif','.tiff','.webp','.gif','.eps','.docx','.xlsx','.pptx'}
SECRET_NAMES={'.env','.env.local','.env.production','.pypirc','.netrc','id_rsa','id_ed25519',
              'credentials','credentials.json','token.json','tokens.json','secrets.json'}


def classify(path,size):
    lower=path.name.lower()
    if lower in SECRET_NAMES or lower.startswith('.env.') or path.suffix.lower() in {'.pem','.key','.pfx','.p12'}:
        return 'sensitive_name'
    ext=path.suffix.lower()
    if ext in BINARY:
        return 'model_or_array'
    if ext in PACKAGES:
        return 'compressed_package'
    if ext in MEDIA:
        return 'document_or_media'
    if ext in TEXT or not ext or lower in {'license','makefile','dockerfile'}:
        return 'text'
    return 'other'


def main():
    if (OUTPUT/'inventory.jsonl').exists():
        raise RuntimeError('Inventory already exists; choose explicit refresh instead of overwriting')
    rows=[]
    errors=[]
    skipped=[]
    def add(path):
        try:
            size=path.stat().st_size
            rows.append(dict(relative=path.relative_to(PROJECT).as_posix(),bytes=size,
                             category=classify(path,size),mtime_ns=path.stat().st_mtime_ns))
        except OSError as e:
            errors.append(dict(path=str(path),error=type(e).__name__))
    for folder in ROOTS:
        base=PROJECT/folder
        if not base.is_dir():
            errors.append(dict(path=str(base),error='expected directory missing'))
            continue
        stack=[base]
        while stack:
            parent=stack.pop()
            try:
                with os.scandir(parent) as entries:
                    for entry in entries:
                        path=Path(entry.path)
                        if entry.is_symlink():
                            skipped.append(dict(path=path.relative_to(PROJECT).as_posix(),reason='symlink'))
                        elif entry.is_dir(follow_symlinks=False):
                            if entry.name in SKIP:
                                skipped.append(dict(path=path.relative_to(PROJECT).as_posix(),reason='cache/environment/private metadata'))
                            else:
                                stack.append(path)
                        elif entry.is_file(follow_symlinks=False):
                            add(path)
            except OSError as e:
                errors.append(dict(path=str(parent),error=type(e).__name__))
    for path in PROJECT.iterdir():
        if path.is_file() and not path.is_symlink():
            add(path)
    rows.sort(key=lambda x:x['relative'])
    with (OUTPUT/'inventory.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    groups=defaultdict(lambda:dict(files=0,bytes=0))
    tops=defaultdict(lambda:dict(files=0,bytes=0,text_files=0,text_bytes=0))
    for row in rows:
        g=groups[row['category']]
        g['files']+=1;g['bytes']+=row['bytes']
        top=row['relative'].split('/')[0]
        g=tops[top]
        g['files']+=1;g['bytes']+=row['bytes']
        if row['category']=='text':
            g['text_files']+=1;g['text_bytes']+=row['bytes']
    result=dict(source=str(PROJECT),files=len(rows),bytes=sum(r['bytes'] for r in rows),categories=dict(groups),
                roots=dict(sorted(tops.items(),key=lambda p:-p[1]['bytes'])),
                largest_files=sorted(rows,key=lambda r:-r['bytes'])[:30],errors=errors,skipped_directories=skipped)
    (OUTPUT/'inventory_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'skipped_directories','largest_files'}},ensure_ascii=False,indent=2))
    print('Largest files:',json.dumps(result['largest_files'][:8],ensure_ascii=False))


if __name__=='__main__':
    main()
