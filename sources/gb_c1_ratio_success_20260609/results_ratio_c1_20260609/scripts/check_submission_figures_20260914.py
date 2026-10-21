from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'revised_20260914'
WORK = ROOT / '.drawio-tmp' / 'submission_figures_20260914'


def main():
    png = OUT / 'fig1_gblcss_overview_drawio_v21_readable.png'
    with Image.open(png) as source:
        source.load()
        pixels = hashlib.sha256(source.tobytes()).hexdigest()
        metadata = PngImagePlugin.PngInfo()
        old_text = dict(source.text)
        for key, value in old_text.items():
            metadata.add_text(key, value)
        source.save(png, pnginfo=metadata, dpi=(300, 300))
    with Image.open(png) as check:
        assert hashlib.sha256(check.tobytes()).hexdigest() == pixels
        assert all(check.text.get(k) == v for k, v in old_text.items())

    files = []
    for pdf in sorted(OUT.glob('*.pdf')):
        with fitz.open(pdf) as document:
            assert len(document) == 1, pdf.name
            page = document[0]
            assert len(page.get_text().strip()) > 100, pdf.name
            assert not [w for w in page.get_text('words')
                        if w[0] < -1 or w[1] < -1 or w[2] > page.rect.width + 1
                        or w[3] > page.rect.height + 1], pdf.name
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
                WORK / (pdf.stem + '_pdf_review.png'))
        with Image.open(pdf.with_suffix('.png')) as image:
            image.load()
            assert image.width > 4000
            assert image.info.get('dpi', (0, 0))[0] >= 299
            files.append({'figure': pdf.stem, 'pixels': image.size, 'dpi': image.info['dpi']})
    assert len(files) == 5
    for record in ['data_checks.json', 'fig1_checks.json']:
        data = json.loads((WORK / record).read_text(encoding='utf-8'))
        if record.startswith('data'):
            assert data['unchanged_source_data']
            for name, value in data['sources'].items():
                assert hashlib.sha256((ROOT / 'figures/source_data' / name).read_bytes()).hexdigest() == value
        else:
            assert data['source_unchanged']
            assert abs(sum(data['soft_target']) - 1) < 1e-12
    (WORK / 'export_checks.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    main()
