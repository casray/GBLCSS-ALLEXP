from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re

from docx import Document
import pdfplumber
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/Admin/Desktop/GB-LCSS_PR_submission_credit_strict_20260930.zip')
archive_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
with ZipFile(SOURCE) as archive:
    assert archive.testzip() is None
    for item in archive.infolist():
        target = (BASE / 'package' / item.filename).resolve()
        if not target.is_relative_to((BASE / 'package').resolve()):
            raise ValueError(item.filename)
    archive.extractall(BASE / 'package')
ROOT = BASE / 'package' / 'GB-LCSS_PR_submission_credit_strict_20260930'
report = {'archive_sha256': archive_hash, 'files': sorted(p.name for p in ROOT.iterdir())}
sources = {p.name: p.read_text(encoding='utf-8') for p in ROOT.glob('*.tex')}
report['font_and_layout_commands'] = {
    n: [f'{i}: {line}' for i, line in enumerate(s.splitlines(), 1)
        if re.search(r'footnotesize|scriptsize|tiny|fontsize|baselinestretch|geometry|resizebox|scalebox|captionsetup|SetAlFnt|small', line)]
    for n, s in sources.items()
}
missing_inputs = []
for name, source in sources.items():
    for dependency in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', source):
        if not (ROOT / dependency).is_file() and not (ROOT / (dependency + '.tex')).is_file():
            missing_inputs.append([name, dependency])
report['missing_inputs'] = missing_inputs
citations = set()
for source in sources.values():
    for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', source):
        citations.update(k.strip() for k in group.split(','))
bib = (ROOT / 'references.bib').read_text(encoding='utf-8')
bibkeys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib))
bblkeys = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', (ROOT / 'main.bbl').read_text(encoding='utf-8')))
report['bibliography'] = {'cited': len(citations), 'bib': len(bibkeys), 'bbl': len(bblkeys),
                          'missing': sorted(citations - bibkeys), 'unused': sorted(bibkeys - citations),
                          'bbl_missing': sorted(citations - bblkeys)}
highlights = Document(ROOT / 'highlights.docx')
report['highlights'] = [{'text': p.text, 'characters': len(p.text)} for p in highlights.paragraphs if p.text.strip()]
report['source_flags'] = {}
for name, source in sources.items():
    matches = [f'{i}: {line}' for i, line in enumerate(source.splitlines(), 1)
               if re.search(r'TODO|FIXME|PLACEHOLDER|generative|ChatGPT|Claude|Codex|competing|availability|Data statement|credit|CRediT', line, re.I)]
    if matches:
        report['source_flags'][name] = matches
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', sources['main.tex'], re.S).group(1)
abstract = '\n'.join(line for line in abstract.splitlines() if not line.startswith('\\'))
report['abstract_words'] = len(abstract.split())
report['title_words'] = len(re.search(r'\\title\{([^}]+)\}', sources['main.tex']).group(1).split())
report['pdf'] = {}
for name in ['main.pdf', 'supplementary.pdf']:
    reader = PdfReader(ROOT / name)
    data = {'pages': len(reader.pages), 'metadata': {str(k): str(v) for k, v in (reader.metadata or {}).items()},
            'page_details': [], 'all_fonts': Counter(), 'all_sizes': Counter()}
    texts = []
    with pdfplumber.open(ROOT / name) as pdf:
        for idx, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ''
            texts.append(f'PAGE {idx}\n{text}')
            chars = [c for c in page.chars if c.get('text', '').strip()]
            sizes = Counter(round(c['size'], 2) for c in chars)
            fonts = Counter(c['fontname'] for c in chars)
            data['all_sizes'].update(sizes)
            data['all_fonts'].update(fonts)
            details = {
                'page': idx, 'width': float(page.width), 'height': float(page.height),
                'sizes': sizes.most_common(4),
                'captions': [line for line in text.splitlines() if re.match(r'^(Table|Fig\.|Algorithm)\s', line)],
                'bounds': [round(min(c['x0'] for c in chars), 2), round(min(c['top'] for c in chars), 2),
                           round(max(c['x1'] for c in chars), 2), round(max(c['bottom'] for c in chars), 2)] if chars else None,
                'unresolved': '??' in text,
                'last_lines': text.splitlines()[-2:],
            }
            data['page_details'].append(details)
            page.close()
    data['all_sizes'] = data['all_sizes'].most_common()
    data['all_fonts'] = data['all_fonts'].most_common()
    report['pdf'][name] = data
    (BASE / (name + '.txt')).write_text('\n\n'.join(texts), encoding='utf-8')
report['archive_unchanged'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == archive_hash
(BASE / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({
    'package': str(ROOT), 'archive_unchanged': report['archive_unchanged'],
    'bibliography': report['bibliography'], 'missing_inputs': missing_inputs,
    'highlights': report['highlights'], 'abstract_words': report['abstract_words'],
    'title_words': report['title_words'], 'source_flags': report['source_flags'],
    'pdf': {n: {'pages': d['pages'], 'fonts': d['all_fonts'][:8], 'sizes': d['all_sizes'][:7],
                'table_pages': [p for p in d['page_details'] if p['captions']],
                'unresolved_pages': [p['page'] for p in d['page_details'] if p['unresolved']]}
            for n, d in report['pdf'].items()},
}, ensure_ascii=True, indent=2))
