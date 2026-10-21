"""Preserve the explicitly GB-LCSS-named manuscript inputs found in Downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from build_archive import PATTERNS,DEST,sha

DOWNLOADS=Path('C:/Users/Admin/Downloads')


def main():
    selected=sorted(p for p in DOWNLOADS.iterdir() if p.is_file() and
                    (p.name.lower().startswith('gb-lcss') or p.name=='final_metadata_clean.zip'))
    rows=[]
    folder=DEST/'supplemental/submission_materials'
    folder.mkdir(parents=True,exist_ok=True)
    for p in selected:
        findings=set()
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as archive:
                for item in archive.infolist():
                    if item.file_size>100*1024*1024:
                        findings.add('unexpected_large_embedded_file')
                    if Path(item.filename).suffix.lower() in {'.tex','.bib','.md','.txt','.json','.xml','.py','.yaml','.yml'}:
                        content=archive.read(item)
                        for name,pattern in PATTERNS:
                            if pattern.search(content):
                                findings.add(name)
        action='local_security_review' if findings else 'upload'
        archive_path='supplemental/submission_materials/'+p.name if action=='upload' else ''
        if action=='upload':
            target=DEST/archive_path
            if target.exists():
                raise RuntimeError('Supplement already copied')
            shutil.copy2(p,target)
            assert sha(target)==sha(p)
        rows.append(dict(source=str(p),archive_path=archive_path,bytes=p.stat().st_size,sha256=sha(p),
                         action=action,credential_signatures=sorted(findings)))
    (DEST/'manifests/supplemental_files.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Supplemental submission inputs','',
           'These GB-LCSS-named files and final_metadata_clean.zip were found in Downloads, outside the main project root.',
           'They are preserved as manuscript-history originals, not claimed to be the current executable experiment package.',
           'Archive contents containing text were checked for credential signatures before inclusion.',
           'They are additional to the project-root files.csv inventory; hashes/provenance are in manifests/supplemental_files.json.',
           'No model weights or downloaded raw datasets were added by this step.','']
    for r in rows:
        lines.append(f'- {Path(r["source"]).name}: {r["action"]}, {r["bytes"]} bytes')
    (DEST/'supplemental/README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,DEST/'tools/add_submission_materials.py')
    print(json.dumps(dict(supplemental_files=len(rows),uploaded=sum(r['action']=='upload' for r in rows),
                          bytes=sum(r['bytes'] for r in rows if r['action']=='upload')),indent=2))


if __name__=='__main__':
    main()
