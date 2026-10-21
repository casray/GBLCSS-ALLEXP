from pathlib import Path
import json
import re

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_checked_20260921'
NEW = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_integrated_20260921'
OUT = Path(__file__).resolve().parent


def text(path):
    return path.read_text(encoding='utf-8-sig')


checks = {}
for name in ('main.tex', 'supplementary.tex'):
    a, b = text(OLD / name), text(NEW / name)
    equations = r'\\begin\{equation\}.*?\\end\{equation\}'
    ae = re.findall(equations, a, re.S)
    be = re.findall(equations, b, re.S)
    assert ae == be, f'Equation change in {name}'
    checks[name + '_unchanged_equations'] = len(ae)
    for source in re.findall(r'\\input\{([^}]+)\}', b):
        assert (NEW / source).exists(), source
    assert re.findall(r'\\label\{([^}]+)\}', a) == re.findall(r'\\label\{([^}]+)\}', b)
    log = text(NEW / name.replace('.tex', '.log'))
    assert not re.search(r'(undefined references|Citation .* undefined|Reference .* undefined|multiply defined|^! )', log, re.M), name
    checks[name + '_warnings'] = [line for line in log.splitlines()
                                if any(s in line for s in ('Warning:', 'Overfull', 'Underfull'))]

table_count = 0
for path in (OLD / 'tables').glob('*.tex'):
    old, new = text(path), text(NEW / 'tables' / path.name)
    if path.name != 'table_F_1.tex':
        assert old == new, path.name
    else:
        assert re.findall(r'\d+(?:\.\d+)?', old) == re.findall(r'\d+(?:\.\d+)?', new)
    table_count += 1
checks['unchanged_numeric_tables'] = table_count
for path in (OLD / 'algorithms').glob('*'):
    if path.is_file():
        assert path.read_bytes() == (NEW / 'algorithms' / path.name).read_bytes(), path.name
assert (OLD / 'references.bib').read_bytes() == (NEW / 'references.bib').read_bytes()
assert (OLD / 'gblcss-numeric.bst').read_bytes() == (NEW / 'gblcss-numeric.bst').read_bytes()
checks['algorithms_bibliography_unchanged'] = True
main = text(NEW / 'main.tex')
old_cites = set(re.findall(r'ref\d+', text(OLD / 'main.tex')))
assert old_cites == set(re.findall(r'ref\d+', main))
checks['cited_keys'] = len(old_cites)
items = re.findall(r'\\item (.*)', text(NEW / 'highlights.tex'))
checks['highlight_lengths'] = [len(item) for item in items]
assert all(len(item) <= 85 for item in items)

for stem in ('main', 'supplementary', 'highlights'):
    rows = []
    with pdfplumber.open(NEW / f'{stem}.pdf') as doc:
        for i, page in enumerate(doc.pages):
            words = page.extract_words()
            assert words, f'Blank page: {stem} {i+1}'
            content_words = [w for w in words if w['bottom'] < page.height - 60]
            bottom = max(w['bottom'] for w in content_words)
            rows.append({'page': i + 1, 'words': len(words), 'bottom': round(bottom, 1),
                         'tail_gap': round(page.height - 79.4 - bottom, 1),
                         'first': ' '.join(w['text'] for w in words[:14])})
    checks[stem + '_pages'] = rows
    doc = pdfium.PdfDocument(NEW / f'{stem}.pdf')
    thumbs = []
    for i in range(len(doc)):
        page = doc[i]
        im = page.render(scale=0.65).to_pil().convert('RGB')
        tile = Image.new('RGB', (420, 560), '#dddddd')
        im.thumbnail((408, 532))
        tile.paste(im, ((420-im.width)//2, 22))
        ImageDraw.Draw(tile).text((8, 5), f'{stem} / {i+1}', fill='black')
        thumbs.append(tile)
        if stem == 'main' and i in (0, 1, 2, 9, 10, 17, 18, 19):
            page.render(scale=1.3).to_pil().save(OUT / f'{stem}_{i+1}.png')
    for start in range(0, len(thumbs), 8):
        chunk = thumbs[start:start+8]
        sheet = Image.new('RGB', (420*4, 560*((len(chunk)+3)//4)), 'white')
        for i, tile in enumerate(chunk):
            sheet.paste(tile, ((i%4)*420, (i//4)*560))
        sheet.save(OUT / f'{stem}_contact_{start//8+1}.jpg')

(OUT / 'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
