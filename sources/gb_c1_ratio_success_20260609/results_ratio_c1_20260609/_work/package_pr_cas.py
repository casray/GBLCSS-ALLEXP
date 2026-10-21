from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'GB-LCSS_PR_CAS_20260929'
OLD = ROOT / 'GB-LCSS_EM_submission_20260926'

for stem in ('main', 'supplementary'):
    source = DEST / f'{stem}_source'
    shutil.copy2(source / f'{stem}.pdf', DEST / f'{stem}.pdf')
    files = sorted(source.glob('*.tex')) + [source / 'cas-sc.cls', source / 'cas-common.sty']
    if stem == 'main':
        files += sorted(source.glob('fig*.pdf'))
        files += [source / 'references.bib', source / 'main.bbl', source / 'elsarticle-num.bst']
    with ZipFile(DEST / f'{stem}_source.zip', 'w', ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    print(f'{stem}_source.zip: {len(files)} files, flat structure')

for name in ('highlights.docx', 'highlights.pdf', 'highlights.tex', 'supplementary_configurations.zip'):
    shutil.copy2(OLD / name, DEST / name)

with ZipFile(ROOT / 'GB-LCSS_PR_CAS_20260929.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(DEST.iterdir()):
        if path.is_file():
            archive.write(path, path.name)
print('Delivery archive: GB-LCSS_PR_CAS_20260929.zip')
