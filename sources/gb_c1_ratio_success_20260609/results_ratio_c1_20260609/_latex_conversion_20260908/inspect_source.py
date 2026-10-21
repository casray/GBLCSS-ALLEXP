import json
import zipfile
from pathlib import Path
from lxml import etree as ET

ROOT = Path(__file__).resolve().parent
doc = json.loads((ROOT / 'source.json').read_text(encoding='utf-8'))

def text(x):
    if isinstance(x, list):
        return ''.join(text(v) for v in x)
    if not isinstance(x, dict):
        return ''
    kind, c = x.get('t'), x.get('c')
    if kind in ('Str', 'Code'):
        return c if kind == 'Str' else c[1]
    if kind in ('Space', 'SoftBreak', 'LineBreak'):
        return ' '
    if kind == 'Math':
        return '$' + c[1] + '$'
    if kind in ('Link', 'Span', 'Header'):
        return text(c[1] if kind in ('Link','Span') else c[2])
    if kind in ('Strong', 'Emph', 'Para', 'Plain', 'SmallCaps', 'Superscript', 'Subscript'):
        return text(c)
    if kind == 'Image':
        return '[IMAGE ' + c[2][0] + ']'
    return ''

lines = []
for i, b in enumerate(doc['blocks']):
    if b['t'] == 'Table':
        c=b['c']
        rows=c[3][1] + [r for body in c[4] for r in body[2]+body[3]] + c[5][1]
        lines.append(f'{i} TABLE cols={len(c[2])}, rows={len(rows)} caption={text(c[1][1])}')
        for row in rows:
            lines.append(' | '.join(text(cell[4]) for cell in row[1]))
    elif b['t'] == 'Figure':
        lines.append(f'{i} FIGURE: {text(b["c"][1][1])}')
    else:
        lines.append(f'{i} {b["t"]}: {text(b)}')
(ROOT/'source_inventory.txt').write_text('\n'.join(lines),encoding='utf-8')
with zipfile.ZipFile(r'C:\Users\Admin\Downloads\GB-LCSS_英文稿0907 (1).docx') as z:
    xml=ET.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    print({'word_tables':len(xml.findall('.//w:tbl',ns)),
           'word_math':len(xml.findall('.//m:oMath',ns)),
           'tracked_deletions':len(xml.findall('.//w:del',ns)),
           'tracked_insertions':len(xml.findall('.//w:ins',ns))})
print('Inventory:', ROOT/'source_inventory.txt')
