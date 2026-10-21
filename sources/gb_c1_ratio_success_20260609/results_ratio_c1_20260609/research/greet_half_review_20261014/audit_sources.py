"""Check literature integrity and local interfaces without importing models."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for item in sources:
        if item['status'] == 'saved':
            assert sha(Path(item['path'])) == item['sha256']
    attachment = json.loads((ROOT/'attachment_record.json').read_text(encoding='utf-8'))
    assert sha(Path(attachment['path'])) == attachment['sha256']
    expected = {
        'representation.py': 'load_frozen_representation',
        'granular_ball.py': 'build_granular_balls',
        'targets.py': 'build_leave_one_out_targets',
        'training.py': 'train_x_only_mlp',
        'diagnostics.py': 'target_metrics',
    }
    interfaces = []
    for name, function in expected.items():
        path = WORKSPACE/'GB-LCSS_code/src/gblcss'/name
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
        interfaces.append(dict(path=str(path), function=function, line=definition.lineno, sha256=sha(path)))
    reports = [ROOT/'00_建议与执行顺序.md', ROOT/'01_两篇文献与代码核查.md',
               ROOT/'02_独立验证与融合实验设计.md']
    for path in reports:
        content = path.read_text(encoding='utf-8')
        assert '\ufffd' not in content
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if not target.startswith(('https://', 'http://', '#')):
                assert (path.parent/target).exists(), target
    record = dict(status='passed', verified_utc=datetime.now(timezone.utc).isoformat(),
                  training_started=False, author_code_executed=False,
                  source_files_verified=sum(r['status']=='saved' for r in sources),
                  unavailable_sources=[r for r in sources if r['status']!='saved'],
                  interfaces=interfaces,
                  environment_modules={name:importlib.util.find_spec(name) is not None
                                       for name in ['torch', 'torch_geometric', 'dgl', 'fitz']},
                  documents=[dict(path=str(p), sha256=sha(p)) for p in reports],
                  scope='Source and interface checks only; no model reproduction or empirical effectiveness claim.')
    (ROOT/'validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:record[key] for key in ['status','training_started','author_code_executed',
                                               'source_files_verified','environment_modules','scope']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
