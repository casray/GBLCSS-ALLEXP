from pathlib import Path
import json
import fitz
from PIL import Image, ImageDraw

root = Path('GB-LCSS_submission_notation_final_20260925')
out = Path('tmp/pdfs/layout_review_20260925')
out.mkdir(parents=True, exist_ok=True)
summary = {}
for name in ('main', 'supplementary', 'highlights'):
    doc = fitz.open(root / f'{name}.pdf')
    records = []
    for index, page in enumerate(doc):
        page.get_pixmap(dpi=120, alpha=False).save(out / f'{name}_{index+1:02}.png')
        spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
        body = [s for s in spans if s['bbox'][1] < page.rect.height - 55]
        records.append({'page': index+1, 'body_bottom': round(max((s['bbox'][3] for s in body), default=0),1),
                        'head': page.get_text()[:140], 'tail': page.get_text()[-220:],
                        'min_font': round(min((s['size'] for s in body), default=0),2),
                        'fonts': sorted(set(s['font'] for s in spans))})
    for start in range(0, len(doc), 6):
        sheet = Image.new('RGB', (1080, 2200), '#dedede')
        draw = ImageDraw.Draw(sheet)
        for offset in range(min(6, len(doc)-start)):
            img = Image.open(out / f'{name}_{start+offset+1:02}.png')
            img.thumbnail((524, 685))
            x, y = (offset%2)*540+8, (offset//2)*733+30
            sheet.paste(img,(x,y))
            draw.text((x,y-21),f'{name} page {start+offset+1}',fill='black')
        sheet.save(out / f'{name}_sheet_{start//6+1}.png')
    fonts = {f[0]: f for page in doc for f in page.get_fonts(full=True)}
    summary[name] = {'pages':len(doc),'page_records':records,
                     'fonts':[{'name':f[3], 'extension':doc.extract_font(x)[1], 'embedded_bytes':len(doc.extract_font(x)[3])} for x,f in fonts.items()]}
(out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for name, data in summary.items():
    print(name, data['pages'], 'pages; unembedded fonts:',[f['name'] for f in data['fonts'] if not f['embedded_bytes']])
    for p in data['page_records']:
        print(p['page'],p['body_bottom'],p['min_font'],repr(p['head'][:75]))
