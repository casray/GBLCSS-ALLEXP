"""Rasterize the native Word previews for visual verification."""
from pathlib import Path
import json

from pdf2image import convert_from_path
import pdfplumber
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / 'word_qa'
summary = {}
for stem in ['main', 'supplement']:
    pages = convert_from_path(str(root / (stem + '.pdf')), dpi=125)
    for number, page in enumerate(pages, 1):
        page.save(root / f'{stem}-{number:02}.png')
    for start in range(0, len(pages), 6):
        sheet = Image.new('RGB', (1680, 1630), '#e3e3e3')
        draw = ImageDraw.Draw(sheet)
        for i, page in enumerate(pages[start:start+6]):
            thumb = page.copy()
            thumb.thumbnail((540, 770))
            x = (i % 3) * 560 + 10
            y = (i // 3) * 815 + 30
            sheet.paste(thumb, (x, y))
            draw.text((x, y-23), f'{stem} / {start+i+1}', fill='black')
        sheet.save(root / f'{stem}-contact-{start//6+1}.png')
    with pdfplumber.open(root / (stem + '.pdf')) as doc:
        summary[stem] = {'pages': len(doc.pages), 'pages_with_text_outside_paper': []}
        for i, page in enumerate(doc.pages, 1):
            bad = [c for c in page.chars if c.get('text', '').strip()
                   and (c['x0'] < -1 or c['x1'] > page.width+1
                        or c['top'] < -1 or c['bottom'] > page.height+1)]
            if bad:
                summary[stem]['pages_with_text_outside_paper'].append(i)
print(json.dumps(summary, indent=2))
(root / 'page_checks.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
