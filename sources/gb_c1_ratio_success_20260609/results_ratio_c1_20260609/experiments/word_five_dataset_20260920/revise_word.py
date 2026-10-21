"""Update the paired diagnostics and their interpretation, preserving the main study."""
from collections import Counter
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import statistics
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

HERE = Path(__file__).resolve().parent
DL = Path('C:/Users/Admin/Downloads')
RESULTS = HERE.parent / 'submission_checks_20260920_extension/combined_five_datasets'
spec = importlib.util.spec_from_file_location('helpers', HERE.parent / 'submission_checks_20260919/revise_word.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
DATASETS = ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonphoto']
NAMES = ['Cora', 'CiteSeer', 'CoraFull', 'Coauthor-CS', 'Amazon-Photo']
RED = 'C00000'
CHANGES = []


def read_csv(name):
    with (RESULTS / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


summary = {(r['dataset'], r['arm']): r for r in read_csv('summary_verified.csv')}
theory = {r['dataset']: r for r in read_csv('theory_diagnostics.csv')}
raw = read_csv('per_seed_results.csv')
assert len(raw) == 450
assert len({(r['dataset'], r['arm'], r['seed']) for r in raw}) == 450
for (dataset, arm), row in summary.items():
    seeds = [r for r in raw if r['dataset'] == dataset and r['arm'] == arm]
    assert sorted(int(r['seed']) for r in seeds) == list(range(5))
    values = [float(r['test_accuracy']) for r in seeds]
    assert abs(statistics.mean(values) - float(row['accuracy_mean'])) < 1e-12
    assert abs(statistics.stdev(values) - float(row['accuracy_sd'])) < 1e-12
for dataset in DATASETS:
    for seed in range(5):
        pair = [r for r in raw if r['dataset'] == dataset and int(r['seed']) == seed]
        assert len({r['split_hash'] for r in pair}) == 1
        assert len({r['noise_hash'] for r in pair}) == 1


def text(p):
    return ''.join(p._p.xpath('.//w:t/text() | .//m:t/text()'))


def find(doc, prefix):
    matches = [p for p in doc.paragraphs if text(p).startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]


def red(element):
    for run in element.iter():
        if run.tag not in (qn('w:r'), qn('m:r')):
            continue
        pr = run.find(qn('w:rPr'))
        if pr is None:
            pr = OxmlElement('w:rPr')
            run.insert(1 if len(run) and run[0].tag == qn('m:rPr') else 0, pr)
        color = pr.find(qn('w:color'))
        if color is None:
            color = OxmlElement('w:color')
            pr.append(color)
        color.attrib.clear()
        color.set(qn('w:val'), RED)


def rewrite(p, value, slots=None):
    old = text(p)
    assert not p._p.xpath('.//w:drawing | .//w:fldChar | .//w:commentRangeStart')
    runs = p._p.xpath('./w:r[w:t]')
    props = deepcopy(runs[0].rPr) if runs and runs[0].rPr is not None else None
    for child in list(p._p):
        if child.tag not in (qn('w:pPr'), qn('w:bookmarkStart'), qn('w:bookmarkEnd')):
            p._p.remove(child)
    for i, part in enumerate(value.split('$')):
        if not part:
            continue
        if i % 2:
            p._p.append(deepcopy(slots[int(part)]) if slots is not None else h.inline_math(part))
        else:
            r = p.add_run(part)
            if props is not None:
                r._r.insert(0, deepcopy(props))
    red(p._p)
    CHANGES.append({'old': old, 'new': text(p)})


def remove(p):
    CHANGES.append({'old': text(p), 'new': ''})
    p._p.getparent().remove(p._p)


def accuracy(dataset, arm):
    r = summary[dataset, arm]
    return f"{100 * float(r['accuracy_mean']):.2f} ± {100 * float(r['accuracy_sd']):.2f}"


def balls(dataset, arm):
    r = summary[dataset, arm]
    lo, hi = int(float(r['balls_min'])), int(float(r['balls_max']))
    return str(lo) if lo == hi else f'{lo}–{hi}'


def replace_table(doc, index, headers, rows, widths, size=9.5):
    old = doc.tables[index]
    new = h.new_table(doc, headers, rows, widths, size)
    # All replacement tables retain the manuscript's white, three-rule layout.
    for row in new.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Pt(0)
                p.paragraph_format.left_indent = Pt(0)
                p.paragraph_format.right_indent = Pt(0)
    for r in new._tbl.iter(qn('m:r')):
        pr = r.find(qn('w:rPr'))
        if pr is None:
            pr = OxmlElement('w:rPr')
            r.insert(1, pr)
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), str(round(size * 2)))
        pr.append(sz)
    red(new._tbl)
    old._tbl.addprevious(new._tbl)
    old._tbl.getparent().remove(old._tbl)
    return new


MAIN_IN = DL / 'GB-LCSS_theory_experiments_integrated_20260920_red.docx'
SUPP_IN = DL / 'GB-LCSS_supplementary_theory_experiments_integrated_20260920_red.docx'
MAIN_OUT = DL / 'GB-LCSS_five_dataset_experiments_revised_20260920_red.docx'
SUPP_OUT = DL / 'GB-LCSS_supplementary_five_dataset_experiments_revised_20260920_red.docx'


def edit_main():
    d = Document(MAIN_IN)
    p = find(d, 'The event includes a tie')
    slots = p._p.xpath('./m:oMath')
    assert len(slots) == 2
    rewrite(p, 'The event includes a tie for the most votes. Equation (10) preserves the maximizing class set of $0$, so the same bound applies to $1$. With the class count, noise rate, and positive margin fixed, the exponential term decreases as more independent votes are available. The margin also matters: adding nodes from competing classes can weaken the evidence for the current node. Appendix H.1 gives the proof.', slots)
    p = find(d, 'Equation (16) expresses')
    rewrite(p, 'Equation (16) relates soft-target error to the errors of the observed label and the local consensus, together with their disagreement. It applies directly to the consensus weight and tie-handling rule in Eqs. (9)–(12). Combining these sources can correct corrupted labels and can also perturb initially correct labels. The overall change in target error depends on the balance of these effects. Appendix H.2 derives the identity and a sufficient condition for lower conditional expected target error under symmetric noise. Section 4.3.1 and Appendix I.2 examine the two effects empirically.')
    rewrite(find(d, 'A further analysis on Cora'), 'Across the five datasets under uniform 0.4 noise, we also separate the reduction in error on corrupted labels from the increase on initially correct labels (Appendix I.2). The reduction exceeds the increase in all 25 dataset–seed runs.')
    rewrite(find(d, 'The representation determines'), 'The representation determines which nodes are grouped together and which labels contribute to each target. Table 7 compares GRACE, raw node attributes $X$, and two-step propagated attributes $A^2X$ on all five datasets under uniform 0.4 noise. Each condition was rerun with five paired seeds and the same splitting parameters (Appendix I.1). For $A^2X$, the original adjacency matrix multiplies the attributes twice, with no added self-loops or adjacency normalization. The resulting node vectors are L2-normalized as in Eq. (3).')
    rewrite(find(d, 'Table 7. Classification accuracy'), 'Table 7. Classification accuracy (%) and ball counts for three representation spaces under uniform 0.4 noise. Accuracy is mean ± sample standard deviation over five paired seeds; ball counts give the range across seeds.')
    rewrite(find(d, 'With the same splitting parameters'), 'Raw $X$ produced one ball on every dataset and seed. In this case, each training node obtains its consensus from the labels of all other training nodes in the graph. On Cora, CiteSeer, and CoraFull, $A^2X$ also produced one ball, giving the same targets and accuracy as raw $X$. On Coauthor-CS and Amazon-Photo, propagation enabled 50–59 and 64 balls, respectively, and improved accuracy over raw $X$. GRACE achieved the highest mean accuracy among the three representations on all five datasets, exceeding $A^2X$ by 1.47 and 1.38 percentage points on the latter two. Thus, simple propagation can support local grouping on some datasets, while the learned representation gave the most effective supervision in this comparison.')
    p = find(d, 'We next varied the GRACE pretraining seed')
    old = text(p)
    new = old.replace('We next varied the GRACE pretraining seed, using', 'We separately examined GRACE pretraining variability on Cora, CiteSeer, and CoraFull, using')
    rewrite(p, new)

    p = find(d, 'We varied one parameter at a time')
    rewrite(p, 'We examine how the parameters affect the formation of local groups, the availability of label evidence, and its weight in the target. Table I.1 varies one parameter at a time on all five datasets under uniform 0.4 noise, using fixed pretrained representations and five paired seeds.')
    p2 = find(d, 'Requiring more votes can leave fewer nodes')
    rewrite(p2, 'The ball budget and splitting threshold determine how finely the nodes are grouped. On CoraFull, reducing $B_max$ from 64 to 32 lowered mean accuracy by 2.10 percentage points; increasing it to 128 raised accuracy by 1.57 points. The larger budget left Cora unchanged and slightly lowered accuracy on Coauthor-CS and Amazon-Photo. Raising $δ$ from 0.05 to 0.10 prevented splitting on the first four datasets, with losses of 3.04–13.15 points. Amazon-Photo still formed 64 balls and changed by −0.18 points. Lowering $δ$ to 0.01 changed accuracy by at most 0.04 points across the five datasets.')
    p3 = h.add_after(p2, '', template=p2)
    rewrite(p3, 'The voting threshold $s_min$ controls how many other labeled nodes are needed to use consensus. Raising it from 2 to 10 reduced the fraction of training nodes receiving a positive consensus weight to 56.53% on Cora and 58.32% on CiteSeer, accompanied by accuracy losses of 7.27 and 5.22 points. Coverage remained at least 89.96% on the other three datasets, where accuracy changed by at most 0.38 points. The value of a stricter voting threshold therefore depends on the available labels within each group.')
    p4 = h.add_after(p3, '', template=p2)
    rewrite(p4, 'Giving more weight to the most-voted class also had dataset-specific effects. Increasing $β$ from 0.5 to 1 raised mean accuracy on four datasets, most noticeably on CiteSeer (+1.74 points), and lowered it by 0.14 points on CoraFull. Changes to $n_min$ likewise had mixed effects. Several configurations improved on the common default, with different settings yielding the highest mean on different datasets. The clearest losses arose when local groups failed to form or too few nodes could use consensus.')

    rows = [[name, *[accuracy(ds, a) for a in ['default', 'raw_x', 'a2x']], *[balls(ds, a) for a in ['default', 'raw_x', 'a2x']]] for ds, name in zip(DATASETS, NAMES)]
    t = replace_table(d, 6, ['Dataset', 'GRACE\naccuracy', 'Raw X\naccuracy', 'A²X\naccuracy', 'GRACE\nballs', 'Raw X\nballs', 'A²X\nballs'], rows, [.99, 1.13, 1.13, 1.13, .74, .74, .74])
    for row in t.rows[1:]:
        for r in row.cells[1].paragraphs[0].runs:
            r.bold = True
    d.save(MAIN_OUT)


def edit_supp():
    d = Document(SUPP_IN)
    rewrite(find(d, 'These experiments use Cora'), 'The representation and parameter comparisons cover the five datasets in Table 2 under uniform 0.4 noise, with seeds 0–4 and the protocol in Section 4.1. Each dataset uses its archived GRACE representation. Variants share the same data split, noisy labels, and initialization seed with the default configuration, and retain the MLP architecture, optimizer, 200-epoch budget, and checkpoint-selection rule.')
    rewrite(find(d, 'Targets were computed on the CPU'), 'For these paired reruns, targets were computed on the CPU and the MLP was trained on the AMD Radeon RX 7700 XT. The software and thread settings were PyTorch 2.9.1+rocm7.2.1 with four host threads and scikit-learn 1.8.0 with six threads for 2-means. Tables 7 and I.1 use the rerun default as their reference; its results differ slightly from the main comparisons, which used GPU target construction.')
    rewrite(find(d, 'The default values are'), 'The default parameters are listed in Section 4.1.4. Each sensitivity row changes only the indicated parameter. The threshold $s_min$ counts votes after excluding the current node. With $β=0$, the empirical class distribution still contributes to the target. Random grouping preserves the default ball sizes; raw $X$ and $A^2X$ change the representation used for grouping (Section 4.3.3).')
    rewrite(find(d, 'Table I.1.'), 'Table I.1. Parameter sensitivity and controls under uniform 0.4 noise. Accuracy (%) is mean ± sample standard deviation over five paired seeds. Bold denotes the highest mean in each column.')
    remove(find(d, 'Configurations with identical target tensors'))
    rewrite(find(d, 'A larger ball budget helps'), 'The ball counts help explain the effect of the budget. Increasing $B_max$ to 128 left Cora at 52–53 balls and CiteSeer at 63–66, while the other three datasets reached 128. The finer partition improved CoraFull on all five seeds, with a paired gain of 1.57 ± 0.54 percentage points and a decrease in target Brier score from 0.6321 to 0.6026. Coauthor-CS and Amazon-Photo instead had small decreases in mean accuracy. Additional groups were therefore useful to different degrees across datasets.')
    rewrite(find(d, 'The effects of majority-class weighting'), 'Reducing $n_min$ from 32 to 1 changed mean accuracy by −1.65, +0.70, −0.05, −0.31, and −0.47 points on Cora, CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo, respectively. At $s_min=10$, the fractions of training nodes receiving a positive consensus weight were 56.53%, 58.32%, 99.13%, 98.98%, and 89.96% in the same order. These counts connect the sensitivity results to how many nodes can receive supervision from their local group.')
    remove(find(d, 'Raising the voting threshold leaves'))
    rewrite(find(d, 'Equation (H.5) separates'), 'Equation (H.5) separates the change in target error into the reduction on corrupted labels and the increase on initially correct labels. For each seed, both components are divided by the total number of training nodes; their difference is the net reduction in Brier score. Table I.2 reports their means over five seeds for the default configuration. Clean training labels provide the reference for this diagnostic.')
    rewrite(find(d, 'In all 15 runs'), 'The reduction on corrupted labels exceeded the increase on initially correct labels in all 25 dataset–seed runs. Mean net reductions ranged from 0.1690 on CoraFull to 0.4877 on Coauthor-CS. Random grouping also lowered Brier scores, yet yielded much lower classification accuracy (Table I.1). Target quality and downstream accuracy therefore provide complementary views of the supervision.')
    rewrite(find(d, 'We also examined the local-class condition'), 'For Proposition 1, we also counted training nodes whose true class was more frequent than every competing class among the other training nodes in their ball, using clean labels. The proportions averaged 75.64%, 65.02%, 36.13%, 88.94%, and 84.76% on Cora, CiteSeer, CoraFull, Coauthor-CS, and Amazon-Photo, respectively. The proposition applies to these positive-margin nodes. The Brier decomposition in Table I.2 includes all training nodes.')

    arms = ['default', 'beta_0', 'beta_0.25', 'beta_0.75', 'beta_1', 'max_balls_32', 'max_balls_128', 'min_train_per_ball_5', 'min_train_per_ball_10', 'min_ball_size_1', 'min_ball_size_16', 'min_ball_size_64', 'split_improvement_0.01', 'split_improvement_0.1', 'mlp_ce', 'shuffled', 'raw_x', 'a2x']
    labels = ['Default', '$β=0$', '$β=0.25$', '$β=0.75$', '$β=1$', '$B_max=32$', '$B_max=128$', '$s_min=5$', '$s_min=10$', '$n_min=1$', '$n_min=16$', '$n_min=64$', '$δ=0.01$', '$δ=0.10$', 'MLP+CE', 'Random grouping', 'Raw $X$', '$A^2X$']
    rows = [[label, *[accuracy(ds, arm) for ds in DATASETS]] for label, arm in zip(labels, arms)]
    t = replace_table(d, 14, ['Configuration', *NAMES], rows, [1.29, 1.06, 1.06, 1.06, 1.06, 1.06])
    for j, ds in enumerate(DATASETS, 1):
        best = max(float(summary[ds, arm]['accuracy_mean']) for arm in arms)
        for i, arm in enumerate(arms, 1):
            if float(summary[ds, arm]['accuracy_mean']) == best:
                for r in t.rows[i].cells[j].paragraphs[0].runs:
                    r.bold = True
    rows = [[name, *[f"{float(theory[ds][key]):.4f}" for key in ['noisy_brier', 'target_brier', 'corrupted_gain', 'clean_cost', 'net_gain']]] for ds, name in zip(DATASETS, NAMES)]
    replace_table(d, 15, ['Dataset', 'One-hot\nlabel', 'Soft\ntarget', 'Reduction on\ncorrupted labels', 'Error increase on\ncorrect labels', 'Net\nreduction'], rows, [1.1, .8, .8, 1.45, 1.55, .9])
    d.save(SUPP_OUT)


def canonical(element):
    return etree.tostring(element, method='c14n', exclusive=True)


def verify():
    report = {'paired_rows_verified': len(raw), 'summary_rows_verified': len(summary), 'outputs': {}}
    for label, source, target, changed_tables in [('main', MAIN_IN, MAIN_OUT, {6}), ('supplement', SUPP_IN, SUPP_OUT, {14, 15})]:
        old, new = Document(source), Document(target)
        revised_old_texts = {r['old'] for r in CHANGES}
        kept = Counter(canonical(p._p) for p in new.paragraphs)
        for p in old.paragraphs:
            if text(p) not in revised_old_texts:
                value = canonical(p._p)
                assert kept[value] > 0, (label, text(p)[:120])
                kept[value] -= 1
        assert len(old.tables) == len(new.tables)
        for i, (a, b) in enumerate(zip(old.tables, new.tables)):
            if i not in changed_tables:
                assert canonical(a._tbl) == canonical(b._tbl), (label, i)
        # Display equations and their numbering are unchanged; inline symbols in
        # the rewritten prose may change in number as the sentences are shortened.
        get_equations = lambda doc: [canonical(x) for x in doc.element.body.xpath('.//m:oMathPara')]
        assert get_equations(old) == get_equations(new), label
        refs = lambda doc: [text(p) for p in doc.paragraphs if text(p).startswith('[')]
        assert refs(old) == refs(new)
        with ZipFile(source) as a, ZipFile(target) as b:
            media = [n for n in a.namelist() if n.startswith('word/media/')]
            assert all(a.read(n) == b.read(n) for n in media)
        assert not any('$' in text(p) for p in new.paragraphs)
        report['outputs'][label] = {'file': str(target), 'unchanged_tables_verified': len(new.tables) - len(changed_tables), 'display_equations_preserved': len(get_equations(new)), 'media_preserved': len(media), 'references_unchanged': True}
    # Re-read the exported cells to check every new result, not just the source data.
    md, sd = Document(MAIN_OUT), Document(SUPP_OUT)
    for i, ds in enumerate(DATASETS, 1):
        for j, arm in enumerate(['default', 'raw_x', 'a2x'], 1):
            assert md.tables[6].cell(i, j).text == accuracy(ds, arm)
        assert sd.tables[15].cell(i, 5).text == f"{float(theory[ds]['net_gain']):.4f}"
    arms = ['default', 'beta_0', 'beta_0.25', 'beta_0.75', 'beta_1', 'max_balls_32', 'max_balls_128', 'min_train_per_ball_5', 'min_train_per_ball_10', 'min_ball_size_1', 'min_ball_size_16', 'min_ball_size_64', 'split_improvement_0.01', 'split_improvement_0.1', 'mlp_ce', 'shuffled', 'raw_x', 'a2x']
    for i, arm in enumerate(arms, 1):
        for j, ds in enumerate(DATASETS, 1):
            assert sd.tables[14].cell(i, j).text == accuracy(ds, arm)
    report['changes'] = CHANGES
    (HERE / 'revision_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    edit_main()
    edit_supp()
    verify()
