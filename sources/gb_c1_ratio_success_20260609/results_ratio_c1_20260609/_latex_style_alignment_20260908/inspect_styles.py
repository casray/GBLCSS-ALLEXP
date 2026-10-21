import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
with zipfile.ZipFile(ROOT / 'INS-D-26-4174.zip') as z:
    senior_bytes = z.read('INS-D-26-4174/main.pdf')
    senior_tex = z.read('INS-D-26-4174/main.tex').decode('utf-8-sig')

(HERE / 'senior_preamble.txt').write_text(senior_tex.split(r'\begin{document}')[0], encoding='utf-8')
result = {}
for name, data in [('senior', senior_bytes), ('current', (ROOT / 'GB-LCSS_Elsevier_LaTeX_Template/main.pdf').read_bytes())]:
    with pdfplumber.open(io.BytesIO(data)) as doc:
        p = doc.pages[0]
        font_counts = Counter((c['fontname'].split('+')[-1], round(c['size'], 2)) for c in p.chars if c['text'].strip())
        colors = Counter(str(c.get('non_stroking_color')) for c in p.chars if c['text'].strip())
        result[name] = {'page_size': (p.width, p.height), 'pages': len(doc.pages),
                        'fonts': [(str(k), v) for k, v in font_counts.most_common(12)], 'colors': dict(colors)}
        for query in ('Density peaks', 'Clustering is', 'In many deployment', 'Graph neural'):
            lines = p.extract_text_lines()
            for line in lines:
                if query in line['text']:
                    result[name][query] = {k: line[k] for k in ('text', 'x0', 'x1', 'top', 'bottom')}
        result[name]['body_lines'] = [{k: ln[k] for k in ('text', 'x0', 'x1', 'top', 'bottom')}
                                      for ln in p.extract_text_lines() if ln['top'] > 200][:8]
    pdf = pdfium.PdfDocument(data)
    pdf[0].render(scale=1.6).to_pil().save(HERE / f'{name}_page1.png')

(HERE / 'style_measurements.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
