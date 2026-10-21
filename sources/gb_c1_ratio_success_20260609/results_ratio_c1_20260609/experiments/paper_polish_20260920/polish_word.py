"""Polish the added experiments and mark this round's edits in red."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path('C:/Users/Admin/Downloads')
HELPER = HERE.parent/'submission_checks_20260919/revise_word.py'
spec = importlib.util.spec_from_file_location('word_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
RED = 'C00000'
CHANGES = []


def all_text(p):
    return ''.join(p._p.xpath('.//w:t/text() | .//m:t/text()'))


def xml_color(rpr):
    e = rpr.find(qn('w:color'))
    if e is None:
        e = OxmlElement('w:color')
        rpr.append(e)
    e.attrib.clear()
    e.set(qn('w:val'), RED)


def color_element(element):
    # OMML runs use Word run properties for the display color.
    for r in element.iter():
        if r.tag not in {qn('w:r'), qn('m:r')}:
            continue
        pr = r.find(qn('w:rPr'))
        if pr is None:
            pr = OxmlElement('w:rPr')
            position = 1 if len(r) and r[0].tag == qn('m:rPr') else 0
            r.insert(position, pr)
        xml_color(pr)


def rewrite(p, text):
    old = all_text(p)
    assert not p._p.xpath('.//w:drawing | .//w:fldChar | .//w:commentRangeStart')
    old_runs = p._p.xpath('./w:r[w:t]')
    run_properties = deepcopy(old_runs[0].rPr) if old_runs and old_runs[0].rPr is not None else None
    for child in list(p._p):
        if child.tag not in {qn('w:pPr'), qn('w:bookmarkStart'), qn('w:bookmarkEnd')}:
            p._p.remove(child)
    for index, part in enumerate(text.split('$')):
        if not part:
            continue
        if index % 2:
            p._p.append(helper.inline_math(part))
        else:
            run = p.add_run(part)
            if run_properties is not None:
                run._r.insert(0,deepcopy(run_properties))
    color_element(p._p)
    CHANGES.append({'old': old, 'new': all_text(p)})


def first_sentence(p):
    old = all_text(p)
    first = p._p.xpath('./w:r')[0]
    first.find(qn('w:t')).text = 'GB-LCSS uses '
    for child in p._p:
        if child.tag == qn('w:pPr'):
            continue
        text = ''.join(e.text or '' for e in child.iter(qn('w:t')))
        if text.startswith(' as the default settings.'):
            tail = deepcopy(child)
            tail.find(qn('w:t')).text = text[len(' as the default settings.'):]
            child.find(qn('w:t')).text = ' as the default settings.'
            child.addnext(tail)
            color_element(child)
            break
        color_element(child)
    else:
        raise AssertionError('Default sentence end not found')
    CHANGES.append({'old':old, 'new':all_text(p)})


MAIN_REWRITES = {
    'The supplementary runs under uniform 0.4':
        'A further analysis on Cora, CiteSeer, and CoraFull separates the reduction in error on corrupted labels from the increase on initially correct labels (Appendix I.2). Under uniform 0.4 noise, the reduction outweighed the increase in all 15 runs.',
    'The representation used for grouping determines':
        'The representation determines which nodes are grouped together and hence which labels contribute to each target. Table 7 compares GRACE with raw node attributes $X$ and two-step propagated attributes $A^2X$ on Cora, CiteSeer, and CoraFull under uniform 0.4 noise. Propagation uses the original adjacency matrix without added self-loops or adjacency normalization, followed by L2 normalization of the node vectors as in Eq. (3). All three conditions use five paired seeds and the common setup in Appendix I.1. The ball counts show the range across seeds.',
    'Under the same construction settings':
        'With the same splitting parameters, raw $X$ and $A^2X$ both produced a single ball in all 15 runs. Each node therefore drew its consensus from all other training nodes rather than a local subset. The identical partitions produced identical soft targets and classification accuracy for each paired seed. GRACE produced multiple balls and higher accuracy on all three datasets. Under the chosen splitting rule, its learned representation therefore enabled more useful local grouping than either of the two simple representations.',
    'We also examine pretraining variability':
        'We next varied the GRACE pretraining seed, using three pretrained representations and three downstream seeds for each. The resulting partitions differed: their adjusted Rand index [74], a measure of agreement between partitions, ranged from 0.247 to 0.365. Classification accuracy, however, was relatively stable. After averaging over downstream seeds, its standard deviation across pretraining seeds was 0.13, 0.45, and 0.60 percentage points on Cora, CiteSeer, and CoraFull, respectively. GB-LCSS exceeded MLP+CE for every pretraining seed. Thus, the tested representations yielded different node assignments but similarly effective supervision under uniform 0.4 noise.',
    'We varied one parameter at a time':
        'We varied one parameter at a time on Cora, CiteSeer, and CoraFull under uniform 0.4 noise, keeping the pretrained representations fixed and using five paired seeds (Appendix I.1). Increasing the ball budget $B_max$ from 64 to 128 changed mean accuracy by 0.00, 0.21, and 1.57 percentage points, respectively. Cora still formed 52–53 balls, whereas CoraFull used all 128 balls and improved on every seed. Increasing the majority-class weight $β$ from 0.5 to 1 improved CiteSeer by 1.74 points, with little change on the other two datasets.',
    'The support and split thresholds have clearer failure cases':
        'Requiring more votes can leave fewer nodes able to use local consensus. Raising $s_min$ from 2 to 10 reduced this coverage to 56.53% on Cora and 58.32% on CiteSeer, while accuracy fell by 7.27 and 5.22 points. The split threshold also mattered: $δ=0.10$ prevented all splits on the three datasets, whereas $δ=0.01$ produced the same targets as the default 0.05. Changes to the minimum ball size $n_min$ had mixed effects across datasets. In these tests, the largest losses occurred when a setting prevented local groups from forming or substantially reduced the number of nodes receiving consensus supervision.',
}

SUPP_REWRITES = {
    'We evaluated Cora, CiteSeer, and CoraFull under uniform 0.4':
        'These experiments use Cora, CiteSeer, and CoraFull with uniform 0.4 noise on training labels, a 20%/10%/70% split, clean validation labels, and seeds 0–4. Each dataset retains the frozen GRACE representation used in the main comparisons. Each parameter variant is paired with the default configuration using the same split, noise realization, and initialization seed. The target rule, MLP architecture, optimizer, 200-epoch budget, and validation-based checkpoint selection remain unchanged.',
    'For these runs, targets were computed':
        'Targets were computed on the CPU, and the MLP was trained on an AMD Radeon RX 7700 XT using PyTorch 2.9.1+rocm7.2.1. The scikit-learn 1.8.0 implementation of 2-means used six CPU threads; PyTorch used four host threads. The default configuration was rerun with this setup and serves as the reference for all variants in Table 7 and Table I.1.',
    'The default configuration is':
        'The default values are $β=0.5$, $B_max=64$, $n_min=32$, $δ=0.05$, and $s_min=2$. Each sensitivity row changes only the indicated parameter. The voting threshold $s_min$ applies after the current node is excluded. Setting $β=0$ removes the extra weight on the most-voted class but retains the empirical local distribution. Random grouping preserves the default ball sizes; raw $X$ and $A^2X$ replace the representation used for grouping, as described in Section 4.3.3.',
    'When two configurations produced exactly':
        'Configurations with identical target tensors for the same dataset and seed shared one training result. The 270 evaluated combinations therefore comprise 214 training runs and 56 reuses after exact target equality was verified.',
    'The effect of the ball budget depends':
        'A larger ball budget helps only if it permits further splitting. Cora formed 52–53 balls with either budget, so increasing $B_max$ from 64 to 128 left its targets unchanged. CiteSeer formed 63–64 and 63–66 balls, respectively. CoraFull reached both budgets: using 128 balls improved accuracy on all five seeds, with a paired gain of 1.57 ± 0.54 percentage points, and lowered the mean target Brier score from 0.6321 to 0.6026. Finer groups were therefore beneficial on CoraFull in this comparison.',
    'Increasing β to 1':
        'The effects of majority-class weighting and minimum ball size differed across datasets. With $β=1$, CiteSeer improved on all five seeds, by 1.74 ± 0.53 points in paired differences, while the mean changes on Cora and CoraFull were +0.04 and −0.14 points. Reducing $n_min$ to 1 changed accuracy by −1.65, +0.70, and −0.05 points on Cora, CiteSeer, and CoraFull, respectively. Neither change improved all three datasets.',
    'A high support requirement reduces':
        'Raising the voting threshold leaves more nodes with their observed labels as targets. At $s_min=10$, the fractions of training nodes with a positive consensus weight were 56.53%, 58.32%, and 99.13% on Cora, CiteSeer, and CoraFull, respectively. Accuracy changed by −7.27, −5.22, and −0.10 points. The larger losses on Cora and CiteSeer accompanied the lower coverage of consensus supervision. A stricter split threshold, $δ=0.10$, prevented splitting on every seed and produced the same single-ball targets as raw $X$ and $A^2X$. In contrast, $δ=0.01$ gave the same targets as the default $δ=0.05$ in all 15 runs.',
    'Table I.2 evaluates the decomposition':
        'Equation (H.5) separates the change in target error into two parts: the reduction on corrupted labels and the increase on initially correct labels. Table I.2 reports both for the default configuration. For each seed, each part is summed over the corresponding nodes and divided by the total number of training nodes, so their difference gives the net reduction in Brier score. The table averages these quantities over five seeds. Clean training labels are used only to evaluate target error and the local class counts examined below.',
    'The reduction on corrupted labels exceeds':
        'In all 15 runs, the reduction on corrupted labels outweighed the increase on correct labels. The targets therefore had lower aggregate error even though some initially correct annotations were perturbed. Random grouping also lowered Brier scores, but its classification accuracy was much lower than that of GB-LCSS (Table I.1). This comparison shows why lower target error alone does not explain the benefit of grouping related nodes.',
    'The positive-margin condition in Proposition 1':
        'We also examined the local-class condition in Proposition 1. Using clean labels for this diagnostic, we counted training nodes whose true class had more votes than any other class among the other training nodes in their ball. Averaged over seeds, these fractions were 75.64%, 65.02%, and 36.13% on Cora, CiteSeer, and CoraFull, respectively. The condition is node-specific: a dominant class within a ball can provide useful evidence for members of that class while disagreeing with other members.',
}


def find(doc, prefix):
    candidates = [p for p in doc.paragraphs if all_text(p).startswith(prefix)]
    assert len(candidates) == 1, (prefix, len(candidates))
    return candidates[0]


def canonical(element):
    copy = deepcopy(element)
    for color in list(copy.iter(qn('w:color'))):
        color.getparent().remove(color)
    return etree.tostring(copy, method='c14n', exclusive=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(stem, rewrites, is_main):
    suffix = '_revised_20260919.docx' if is_main else '_20260919.docx'
    source = DOWNLOADS/f'{stem}{suffix}'
    output = DOWNLOADS/f'{stem}_polished_20260920_red.docx'
    old_hash = digest(source)
    doc = Document(source)
    untouched = {p._p:etree.tostring(p._p, method='c14n', exclusive=True) for p in doc.paragraphs}
    changed = set()
    for prefix, replacement in rewrites.items():
        if replacement is None:
            continue
        p = find(doc, prefix)
        changed.add(p._p)
        rewrite(p, replacement)
        if prefix == 'The effect of the ball budget depends':
            p.paragraph_format.keep_together = True
    if is_main:
        p = find(doc, 'Unless varied in Section 4.3.4')
        changed.add(p._p)
        first_sentence(p)
    else:
        cell = doc.tables[9].cell(12,2)
        assert cell.text == 'Chosen on Cora and CiteSeer, then fixed'
        rewrite(cell.paragraphs[0], 'Set during method development on Cora and CiteSeer; used unchanged in the main comparisons')
    for p in doc.paragraphs:
        if p._p not in changed:
            assert etree.tostring(p._p,method='c14n',exclusive=True) == untouched[p._p]
    doc.save(output)
    assert digest(source) == old_hash
    return source, output


def verify(source, output, is_main):
    old, new = Document(source), Document(output)
    assert len(old.tables) == len(new.tables)
    changed_tables = {9} if not is_main else set()
    for index,(a,b) in enumerate(zip(old.tables,new.tables)):
        if index not in changed_tables:
            assert etree.tostring(a._tbl,method='c14n',exclusive=True) == etree.tostring(b._tbl,method='c14n',exclusive=True)
        else:
            for ri,(ra,rb) in enumerate(zip(a.rows,b.rows)):
                for ci,(ca,cb) in enumerate(zip(ra.cells,rb.cells)):
                    if (ri,ci) != (12,2):
                        assert etree.tostring(ca._tc,method='c14n',exclusive=True) == etree.tostring(cb._tc,method='c14n',exclusive=True)
    with ZipFile(source) as a, ZipFile(output) as b:
        media = [n for n in a.namelist() if n.startswith('word/media/')]
        assert all(a.read(n)==b.read(n) for n in media)
    if is_main:
        def refs(d):
            i = next(i for i,p in enumerate(d.paragraphs) if p.text=='References')
            return [etree.tostring(p._p) for p in d.paragraphs[i:]]
        assert refs(old)==refs(new)
    def displayed_math(d):
        return [canonical(p) for p in d.element.xpath('.//m:oMathPara')]
    assert displayed_math(old)==displayed_math(new)
    return {'source':str(source),'output':str(output),'tables':len(new.tables),
            'table_numbers_unchanged':True,'all_media_unchanged':True,
            'displayed_equations_unchanged':True,'references_unchanged':True}


if __name__=='__main__':
    main = build('GB-LCSS_experiments',MAIN_REWRITES,True)
    supp = build('GB-LCSS_supplementary_experiments',SUPP_REWRITES,False)
    checks = [verify(*main,True),verify(*supp,False)]
    (HERE/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    (HERE/'changed_paragraphs.json').write_text(json.dumps(CHANGES,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))
