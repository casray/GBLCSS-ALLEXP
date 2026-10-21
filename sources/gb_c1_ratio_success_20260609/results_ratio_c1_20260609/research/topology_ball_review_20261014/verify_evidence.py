"""Read-only source checks and tiny CPU target examples; no model training."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE/'GB-LCSS_code/src'))
from gblcss.targets import build_leave_one_out_targets


def main():
    paths = [Path(r'C:\Users\Admin\Downloads\新建文件 (5).txt'),
             Path(r'C:\Users\Admin\Downloads\Wang 等 - 2026 - Multi-Granularity Graph Contrastive Learning Framework via Granular Ball on Heterogeneous Graphs.pdf'),
             Path(r'C:\Users\Admin\Downloads\final_metadata_clean.zip'),
             WORKSPACE/'GB-LCSS_code/src/gblcss/targets.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/experiment.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball_adaptive.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/training.py']
    sources = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    with zipfile.ZipFile(paths[2]) as z:
        manuscript = z.read('GB-LCSS_PR_submission_FINAL_20260930/main.tex').decode('utf-8')
        assert '74.48 minutes' in manuscript
        assert '\\Delta_n>0' in manuscript
    # One clean class-A root with four clean class-B voters in the same region.
    labels = torch.tensor([0, 1, 1, 1, 1])
    train = np.arange(5)
    result = build_leave_one_out_targets(labels, np.zeros(5, dtype=np.int64), train, 5)
    expected = torch.tensor([0., 1., 0., 0., 0.])
    torch.testing.assert_close(result.soft_target[0], expected, rtol=0, atol=0)
    assert result.alpha[0].item() == 1.0
    brier = float((result.soft_target[0]-torch.tensor([1., 0., 0., 0., 0.])).square().sum())
    assert brier == 2.0
    # Splitting down to two nodes per ball gives only one other vote and falls back.
    small_labels = torch.tensor([0, 1, 2, 3])
    small = build_leave_one_out_targets(small_labels, np.array([0, 0, 1, 1]), np.arange(4), 5)
    torch.testing.assert_close(small.alpha, torch.zeros(4), rtol=0, atol=0)
    # Valid nontraining class changes do not influence the training targets.
    all_labels = torch.tensor([0, 1, 1, 1, 1, 2, 3])
    assignment = np.zeros(7, dtype=np.int64)
    original = build_leave_one_out_targets(all_labels, assignment, train, 5)
    modified_labels = all_labels.clone()
    modified_labels[5:] = torch.tensor([4, 0])
    modified = build_leave_one_out_targets(modified_labels, assignment, train, 5)
    torch.testing.assert_close(original.soft_target[:5], modified.soft_target[:5], rtol=0, atol=0)
    out = {'status': 'passed', 'training_started': False, 'sources': sources,
           'manuscript_checks': {'corafull_grace_minutes': 74.48, 'positive_margin_assumption': True},
           'tiny_cpu_examples': {'wrong_unanimous_vote': {'root_alpha': 1.0, 'root_target': expected.tolist(),
                                 'root_clean_brier': brier},
                                 'one_other_vote_falls_back': True, 'nontrain_label_changes_ignore_training_targets': True},
           'boundary': 'Analytic behavior checks of existing targets, not a BFS implementation or dataset experiment.'}
    (ROOT/'evidence.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'sources'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
