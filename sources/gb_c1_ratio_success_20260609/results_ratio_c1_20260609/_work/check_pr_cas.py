from pathlib import Path
from zipfile import ZipFile
import json
import re

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'GB-LCSS_EM_submission_20260926'
NEW = ROOT / 'GB-LCSS_PR_CAS_20260929'
OUT = ROOT / '_work' / 'pr_cas_checks'
OUT.mkdir(exist_ok=True)
report = {}
for stem, marker in [('main', r'\section{Introduction}'), ('supplementary', r'\appendix')]:
    directory = NEW / f'{stem}_source'
    with ZipFile(OLD / f'{stem}_source.zip') as archive:
        old = archive.read(f'{stem}.tex').decode('utf-8-sig')
        new = (directory / f'{stem}.tex').read_text(encoding='utf-8-sig')
        body_old = old[old.index(marker):].replace('\r\n', '\n')
        body_new = new[new.index(marker):]
        body_old = body_old.replace(r'\bibliographystyle{gblcss-numeric}', r'\bibliographystyle{elsarticle-num}')
        assert body_old == body_new, f'{stem}: body changed'
        unchanged = []
        for name in archive.namelist():
            if name.endswith(('.tex', '.pdf', '.bib')) and name != f'{stem}.tex':
                assert archive.read(name) == (directory / name).read_bytes(), name
                unchanged.append(name)
    log = (directory / f'{stem}.log').read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'(?:Reference|Citation).*undefined', log)
    doc = fitz.open(directory / f'{stem}.pdf')
    outside = []
    for n, page in enumerate(doc):
        for x0, y0, x1, y1, text, *_ in page.get_text('blocks'):
            if x0 < -1 or x1 > page.rect.width + 1 or y0 < -1 or y1 > page.rect.height + 1:
                outside.append({'page': n + 1, 'text': text[:80]})
    for start in range(0, len(doc), 8):
        sheet = Image.new('RGB', (1200, 1760), 'white')
        draw = ImageDraw.Draw(sheet)
        for n in range(start, min(start + 8, len(doc))):
            pix = doc[n].get_pixmap(matrix=fitz.Matrix(0.7, 0.7), alpha=False)
            thumb = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            thumb.thumbnail((580, 410))
            j = n - start
            x, y = (j % 2) * 600, (j // 2) * 440
            draw.text((x + 12, y + 5), f'{stem} page {n + 1}', fill='black')
            sheet.paste(thumb, (x + (600 - thumb.width) // 2, y + 24))
        sheet.save(OUT / f'{stem}_{start + 1}.png')
    report[stem] = {'pages': len(doc), 'page_size': list(doc[0].rect),
                    'body_unchanged': True, 'unchanged_dependencies': len(unchanged),
                    'text_outside_page': outside,
                    'overfull_warnings': re.findall(r'Overfull[^\n]*', log)}
    doc[0].get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(OUT / f'{stem}_first.png')
    if stem == 'main':
        text = doc[0].get_text()
        assert 'huangbing@nau.edu.cn' in text
        assert 'ORCID(s):' not in text
        report[stem]['references'] = (directory / 'main.bbl').read_text().count(r'\bibitem')
(OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
