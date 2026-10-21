"""Archive public primary sources for a literature audit; never trains models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'NT_2026': 'https://arxiv.org/html/2604.08980v1',
    'DirPoly_2025': 'https://arxiv.org/html/2509.00772v1',
    'GraphTARIF_2026': 'https://arxiv.org/html/2510.10631v2',
    'TS_2026': 'https://arxiv.org/html/2605.20248v1',
    'Polynormer_2024': 'https://www.csl.cornell.edu/~zhiruz/pdfs/polynormer-iclr2024.pdf',
    'TunedGNN_2024': 'https://proceedings.nips.cc/paper_files/paper/2024/file/b10ed15ff1aa864f1be3a75f1ffc021b-Paper-Datasets_and_Benchmarks_Track.pdf',
    'CNA_2024': 'https://papers.nips.cc/paper_files/paper/2024/file/91b1966914203338495a76d13b23599c-Paper-Conference.pdf',
    'HPGNN_2025': 'https://www.ijcai.org/proceedings/2025/0724.pdf',
    'ES_MLP_2025': 'https://raw.githubusercontent.com/mlresearch/v269/main/assets/kohn25a/kohn25a.pdf',
    'HiGCN_2024': 'https://arxiv.org/pdf/2309.12971',
    'benchmark_readme': 'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/main/README.md',
    'feature_sensitivity_2026': 'https://arxiv.org/html/2609.33764v1',
}


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        if tag in {'p', 'div', 'tr', 'h1', 'h2', 'h3', 'section'}:
            self.parts.append('\n')
        elif tag in {'td', 'th'}:
            self.parts.append(' | ')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.skip = max(0, self.skip-1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def main():
    folder = ROOT / 'sources'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        row = {'name': name, 'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'AcademicLiteratureAudit/1.0'})
            with urllib.request.urlopen(request, timeout=40) as response:
                raw = response.read()
                row['resolved_url'] = response.url
            suffix = '.pdf' if raw.startswith(b'%PDF') else '.md' if name=='benchmark_readme' else '.html'
            path = folder / (name + suffix)
            path.write_bytes(raw)
            row.update(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if suffix == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(path)
                plain = '\n'.join(f'\n--- PDF PAGE {i+1} ---\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages))
                row['pages'] = len(reader.pages)
            elif suffix == '.html':
                parser = PlainText()
                parser.feed(raw.decode('utf-8'))
                plain = '\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
            else:
                plain = raw.decode('utf-8')
            (folder / (name + '_fulltext.txt')).write_text(plain, encoding='utf-8')
            row['status'] = 'saved'
        except Exception as exc:
            row.update(status='unavailable', error=str(exc))
        records.append(row)
        (ROOT / 'source_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(name, row['status'], flush=True)


if __name__ == '__main__':
    main()
