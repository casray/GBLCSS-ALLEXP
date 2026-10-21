from pathlib import Path
import difflib
import json
import re
import zipfile

from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_integrated_20260921'
NEW = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_integrated_v2_20260921'
OUT = Path(__file__).resolve().parent
changes = []
for name in ('main.tex', 'supplementary.tex', 'highlights.tex'):
    a, b = [(base / name).read_text(encoding='utf-8-sig') for base in (OLD, NEW)]
    if name != 'highlights.tex':
        for pattern in (r'\\begin\{equation\}.*?\\end\{equation\}',
                        r'\\(?:cite|label|ref\*?|eqref)\{[^}]+\}', r'\d+(?:\.\d+)?'):
            assert re.findall(pattern, a, re.S) == re.findall(pattern, b, re.S), name
    changes.extend(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                      fromfile='previous/' + name, tofile='revised/' + name))
    log = (NEW / name.replace('.tex', '.log')).read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'undefined references|(?:Citation|Reference) .* undefined|multiply defined|^! ', log, re.M)
for sub in ('figures', 'tables', 'algorithms'):
    for path in (OLD / sub).rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (NEW / path.relative_to(OLD)).read_bytes()
for name in ('references.bib', 'gblcss-numeric.bst'):
    assert (OLD / name).read_bytes() == (NEW / name).read_bytes()
items = re.findall(r'\\item (.*)', (NEW / 'highlights.tex').read_text())
assert all(len(s) <= 85 for s in items)
texts = {}
pages = {}
for stem in ('main', 'supplementary', 'highlights'):
    doc = PdfReader(NEW / f'{stem}.pdf')
    pages[stem] = len(doc.pages)
    texts[stem] = ' '.join(' '.join(p.extract_text().split()) for p in doc.pages)
    assert 'Appendix Appendix' not in texts[stem]
assert 'Appendix H.1 proves the local-vote error bound' in texts['supplementary']
assert 'Appendix H.2 derives the squared-error identity' in texts['supplementary']
affected = [('main', p) for p in (2, 7, 8, 17, 18, 19, 20)]
affected.extend([('supplementary', 10), ('highlights', 1)])
sheet = Image.new('RGB', (1260, 560 * ((len(affected) + 2) // 3)), 'white')
for i, (stem, number) in enumerate(affected):
    doc = pdfium.PdfDocument(NEW / f'{stem}.pdf')
    im = doc[number - 1].render(scale=1.4).to_pil().convert('RGB')
    im.save(OUT / f'{stem}_{number}.png')
    im.thumbnail((408, 532))
    x, y = (i % 3) * 420, (i // 3) * 560
    sheet.paste(im, (x, y + 22))
    ImageDraw.Draw(sheet).text((x + 8, y + 5), f'{stem} / {number}', fill='black')
sheet.save(OUT / 'affected_pages.jpg')
(OUT / 'changes.patch').write_text(''.join(changes), encoding='utf-8')
files = [NEW / n for n in ('main.tex', 'supplementary.tex', 'highlights.tex', 'main.pdf',
                           'supplementary.pdf', 'highlights.pdf', 'references.bib',
                           'gblcss-numeric.bst', 'README.md')]
for sub in ('tables', 'algorithms', 'figures'):
    files.extend(p for p in (NEW / sub).rglob('*') if p.is_file()
                 and p.suffix.lower() in ('.tex', '.pdf', '.png', '.drawio'))
archive = NEW.with_suffix('.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(NEW).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(z.read(n) == (NEW / n).read_bytes() for n in z.namelist())
print(json.dumps({'pages': pages, 'highlight_lengths': [len(s) for s in items],
                  'equations_numbers_citations_figures_tables_algorithms_unchanged': True,
                  'appendix_references_verified_in_pdf': True,
                  'archive': str(archive), 'archive_files': len(files)}, ensure_ascii=False))
