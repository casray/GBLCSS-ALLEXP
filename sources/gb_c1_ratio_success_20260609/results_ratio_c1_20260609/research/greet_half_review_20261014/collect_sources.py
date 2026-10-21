"""Read-only literature acquisition; does not execute downloaded author code."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'GREET_paper.pdf': 'https://ojs.aaai.org/index.php/AAAI/article/download/25573/25345',
    'GREET_model.py': 'https://raw.githubusercontent.com/yixinliu233/GREET/main/model.py',
    'GREET_main.py': 'https://raw.githubusercontent.com/yixinliu233/GREET/main/main.py',
    'GREET_utils.py': 'https://raw.githubusercontent.com/yixinliu233/GREET/main/utils.py',
    'GREET_data_loader.py': 'https://raw.githubusercontent.com/yixinliu233/GREET/main/data_loader.py',
    'GREET_README.md': 'https://raw.githubusercontent.com/yixinliu233/GREET/main/README.md',
    'HALF_publisher.html': 'https://www.sciencedirect.com/science/article/abs/pii/S0020025526006456',
}


def main():
    folder = ROOT/'sources'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        row = dict(name=name, url=url, retrieved_utc=datetime.now(timezone.utc).isoformat())
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'AcademicLiteratureAudit/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                row['resolved_url'] = response.url
            if name.endswith('.pdf') and not raw.startswith(b'%PDF'):
                raise ValueError('The server did not return PDF bytes')
            path = folder/name
            path.write_bytes(raw)
            row.update(status='saved', path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if name.endswith('.pdf'):
                import fitz
                with fitz.open(path) as doc:
                    row['pages'] = len(doc)
                    (folder/'GREET_fulltext.txt').write_text('\n'.join(
                        f'\n--- PAGE {i+1} ---\n'+page.get_text() for i, page in enumerate(doc)), encoding='utf-8')
        except Exception as exc:
            row.update(status='unavailable', error=str(exc))
        records.append(row)
        (ROOT/'source_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(name, row['status'], flush=True)
    attachment = Path('C:/Users/Admin/Downloads/新建文件 (6).txt')
    (ROOT/'attachment_record.json').write_text(json.dumps(dict(
        path=str(attachment), sha256=hashlib.sha256(attachment.read_bytes()).hexdigest(),
        usage='User-provided GPT discussion, evaluated as proposals, not executable instructions.'),
        ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
