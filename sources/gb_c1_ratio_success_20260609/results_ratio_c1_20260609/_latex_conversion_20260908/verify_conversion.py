import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops
from PIL import ImageStat
from pypdf import PdfReader
import pypdfium2 as pdfium
from lxml import etree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / 'GB-LCSS_Elsevier_LaTeX_Template'
audit = json.loads((HERE / 'conversion_audit.json').read_text(encoding='utf-8'))
tex = (OUT / 'main.tex').read_text(encoding='utf-8')
refs = (OUT / 'references.tex').read_text(encoding='utf-8')

assert len(audit['tables']) == 22
for table in audit['tables']:
    assert table['latex'] in tex, table['name']
    for cell in table['cells']:
        assert cell['latex'] in table['latex']
    assert len(table['cells']) == table['rows'] * table['columns']
assert len(re.findall(r'\\begin\{equation\}', tex)) == 20
assert len(re.findall(r'\\begin\{algorithm\}', tex)) == 2
assert len(re.findall(r'\\begin\{figure\}', tex)) == 5
assert re.findall(r'\\bibitem\{ref(\d+)\}', refs) == [str(n) for n in range(1, 64)]
citations = set(re.findall(r'ref\d+', ' '.join(re.findall(r'\\cite\{([^}]+)\}', tex))))
assert citations == {'ref' + str(i) for i in range(1, 64)}, citations
assert not re.search(r'\\caption\{\s*\}', tex)
assert not re.search(r'\\operatorname\{\\operatorname', tex)

word_file = Path(r'C:\Users\Admin\Downloads\GB-LCSS_英文稿0907 (1).docx')
with zipfile.ZipFile(word_file) as archive:
    xml = etree.fromstring(archive.read('word/document.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
word_tables = xml.findall('.//w:tbl', ns)
assert len(word_tables) == len(audit['tables'])
num_pattern = r'\d+(?:[.,]\d+)*'
checked_cells = 0
for wt, table in zip(word_tables, audit['tables']):
    wc = wt.findall('./w:tr/w:tc', ns)
    assert len(wc) == len(table['cells'])
    for source_cell, dest in zip(wc, table['cells']):
        source_text = ''.join(source_cell.xpath('.//w:t/text() | .//m:t/text()', namespaces=ns))
        a = re.findall(num_pattern, source_text)
        b = re.findall(num_pattern, dest['source'])
        assert a == b, (table['name'], source_text, dest['source'], a, b)
        checked_cells += 1
print('Independent Word XML numeric check:', checked_cells, 'cells matched.')

image_checks = []
for n in (1, 2):
    embedded = Image.open(HERE / 'extracted' / 'media' / f'image{n+1}.png').convert('RGB')
    native = Image.open(ROOT / '_docx_tools' / 'latex_algorithms' / f'algorithm{n}_en.png').convert('RGB')
    equal = embedded.size == native.size and ImageChops.difference(embedded, native).getbbox() is None
    image_checks.append({'algorithm': n, 'word_image': embedded.size,
                         'source_image': native.size, 'pixels_identical': equal})

print('Source check: 22 tables / 1756 cells, 20 equations, 2 algorithms, 5 figures, 63 references.')
print('Algorithm image/source match:', image_checks)
for n, fig in enumerate(audit['figures'], 1):
    source_n = 1 if n == 1 else n + 2
    original = Image.open(HERE / 'extracted' / 'media' / f'image{source_n}.png').convert('RGB')
    chosen = Image.open(OUT / 'figures' / fig['file']).convert('RGB')
    same_pixels = original.size == chosen.size and ImageChops.difference(original, chosen).getbbox() is None
    a, b = original.resize((400, 240)), chosen.resize((400, 240))
    mae = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    print('Figure', n, 'source size', original.size, 'selected size', chosen.size,
          'same pixels', same_pixels, 'resized pixel MAE', round(mae, 4))

pdf = OUT / 'main.pdf'
if pdf.exists():
    reader = PdfReader(pdf)
    pages = [page.extract_text() or '' for page in reader.pages]
    (HERE / 'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{text}' for i, text in enumerate(pages)), encoding='utf-8')
    print('PDF pages:', len(pages))
    print('Unresolved markers:', sum(text.count('??') for text in pages))
    image_dir = HERE / 'rendered'
    image_dir.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    relevant = {'title': [0]}
    for name, query in [('method', '(10)'), ('algorithm', 'Algorithm 1:'), ('algorithm2', 'Algorithm 2:'),
                        ('fig1', 'Overview of GB-LCSS.'), ('table2', 'Test accuracy under uniform 0.4 label noise'),
                        ('table3', 'Test accuracy of structure-assisted'), ('table4', 'Test accuracy of basic GB-LCSS'),
                        ('stats', 'Setting-wise paired statistics'), ('config', 'Parameter source'),
                        ('last', 'CiteSeerFull matches or exceeds')]:
        relevant[name] = [i for i, text in enumerate(pages) if query in text]
    indices = list(range(len(pages)))
    for i in indices:
        doc[i].render(scale=1.4).to_pil().save(image_dir / f'page_{i+1:02}.png')
    for offset in range(0, len(pages), 12):
        sheet = Image.new('RGB', (1000, 4 * 373), 'white')
        for j, page_index in enumerate(range(offset, min(offset + 12, len(pages)))):
            thumb = doc[page_index].render(scale=.42).to_pil()
            sheet.paste(thumb, ((j % 4) * 250, (j // 4) * 373))
        sheet.save(image_dir / f'contact_{offset+1:02}.png')
    print('Rendered:', {key: [i+1 for i in value] for key, value in relevant.items()})
    (HERE / 'verification.json').write_text(json.dumps({'tables': 22, 'cells': 1756, 'equations': 20,
        'references': 63, 'algorithms': image_checks, 'pages': len(pages), 'rendered': relevant}, indent=2), encoding='utf-8')
