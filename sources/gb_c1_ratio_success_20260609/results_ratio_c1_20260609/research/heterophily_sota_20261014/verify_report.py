"""Validate archived sources and reported local scores; performs no training."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifests = [ROOT/'source_manifest.json', ROOT/'source_manifest_followup.json',
                 ROOT/'frontiers/source_manifest.json', ROOT/'supplement/source_manifest.json',
                 ROOT/'sources/CNA_code_manifest.json']
    verified, unavailable = [], []
    for manifest in manifests:
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            if row.get('status') == 'unavailable':
                unavailable.append({'url': row['url'], 'error': row.get('error')})
                continue
            path = Path(row['path'])
            if not path.is_absolute():
                path = WORKSPACE/path
            assert digest(path) == row['sha256'], f'Archive changed: {path}'
            verified.append(str(path))
    with (ROOT/'local_clean_summary.csv').open(encoding='utf-8', newline='') as f:
        scores = list(csv.DictReader(f))
    aliases = {'MLP_CE': ['CE'], 'GB_LCSS_original': ['GB64', 'OLD64'],
               'GB_LCSS_support16': ['SQRT_SUPPORT']}
    for row in scores:
        path = Path(row['source'])
        assert digest(path) == row['source_sha256'], f'Local results changed: {path}'
        with path.open(encoding='utf-8-sig', newline='') as f:
            raw = [r for r in csv.DictReader(f) if r['noise'] == 'clean'
                   and r['dataset'] == row['dataset'] and r['arm'] in aliases[row['method']]]
        values = [100*float(r['test_accuracy']) for r in raw]
        assert len(values) == int(row['n'])
        assert math.isclose(statistics.mean(values), float(row['mean_percent']), abs_tol=1e-8)
        if len(values) > 1:
            assert math.isclose(statistics.stdev(values), float(row['sample_std_percent']), abs_tol=1e-8)
        else:
            assert not row['sample_std_percent'], 'Do not invent a one-run std'
    protocols = json.loads((ROOT/'protocol_audit.json').read_text(encoding='utf-8'))
    for row in protocols['protocols']:
        assert digest(Path(row['source'])) == row['sha256']
    means = {(r['dataset'], r['method']): float(r['mean_percent']) for r in scores}
    paired_mean_differences = {dataset: means[dataset, 'GB_LCSS_original']-means[dataset, 'MLP_CE']
                              for dataset in ['roman', 'texas', 'wisconsin', 'corafull', 'coauthorcs']}
    reference_gaps = {method: score-means['roman', 'GB_LCSS_original'] for method, score in
                     [('TunedGCN', 91.27), ('Polynormer-r', 92.55), ('GraphTARIF', 93.23), ('Dir-NT', 94.77)]}
    documents = [ROOT/'00_异配图SOTA与GBLCSS干净对比.md', ROOT/'01_强方法与机制核读.md']
    for path in documents:
        content = path.read_text(encoding='utf-8')
        assert '54.81±0.11' not in content
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('https://', 'http://', '#')):
                continue
            assert (path.parent/target).exists(), f'Missing report link: {target}'
    text = (ROOT/'sources/Polynormer_official_2024_fulltext.txt').read_text(encoding='utf-8')
    assert re.search(r'Polynormer-r\s+92\.55±\s*0\.37\s+54\.81±\s*0\.49', text)
    out = {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
           'training_started': False, 'source_files_sha_verified': len(verified),
           'preserved_unavailable_sources': unavailable, 'local_score_rows_recomputed': len(scores),
           'protocol_files_sha_verified': len(protocols['protocols']),
           'gb_minus_ce_mean_pp': paired_mean_differences,
           'roman_cross_paper_reference_gaps_pp_not_matched_experiments': reference_gaps,
           'documents': [{'path': str(p), 'sha256': digest(p)} for p in documents],
           'scope': 'Integrity and arithmetic checks, not a reproduction of external papers or a proof of global SOTA.'}
    (ROOT/'validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
