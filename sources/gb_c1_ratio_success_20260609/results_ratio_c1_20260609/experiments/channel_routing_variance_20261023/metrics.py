"""Hard-prediction complementarity and exact switching accounting."""
import numpy as np


def protocol_cell(dataset, student_seed, condition):
    return dataset, 0, condition


def accuracy_f1(y, pred, classes=None):
    y, pred = np.asarray(y), np.asarray(pred)
    classes = int(max(y.max(), pred.max())) + 1 if classes is None else classes
    cm = np.bincount(y * classes + pred, minlength=classes * classes).reshape(classes, classes)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(classes, dtype=float), where=den > 0)
    return dict(accuracy=float(np.mean(pred == y)), macro_f1=float(f1.mean()))


def pair_metrics(y, a, c):
    y, a, c = np.asarray(y), np.asarray(a), np.asarray(c)
    ac, cc = a == y, c == y
    union = float(np.mean(ac | cc))
    best = float(max(ac.mean(), cc.mean()))
    return dict(union=union, best_single=best, headroom=union-best,
                disagreement=float(np.mean(a != c)), both_wrong=float(np.mean(~ac & ~cc)),
                a_only=float(np.mean(ac & ~cc)), c_only=float(np.mean(~ac & cc)),
                both_correct=float(np.mean(ac & cc)))


def route_metrics(y, a, c, switch):
    y, a, c, switch = map(np.asarray, (y, a, c, switch))
    switch = switch.astype(bool) & (a != c)
    rescued = int(np.sum(switch & (c == y) & (a != y)))
    harmed = int(np.sum(switch & (a == y) & (c != y)))
    pred = np.where(switch, c, a)
    return dict(accuracy=float(np.mean(pred == y)), rescued=rescued, harmed=harmed,
                gain=(rescued-harmed)/len(y), switched=int(switch.sum()), coverage=float(switch.mean()))


def stage0_gate(rows, primary=('cora','citeseer','corafull'), threshold=2.0, required=2):
    names = list(dict.fromkeys(r['dataset'] for r in rows))
    gains = {n:float(np.mean([r['gap_pp'] for r in rows if r['dataset']==n and r['condition']!='clean'])) for n in names}
    passed = sum(gains[n] >= threshold for n in primary) >= required
    return dict(passed=bool(passed), gains_pp=gains, threshold_pp=threshold, required=required,
                primary_datasets=list(primary), gate_split='test', exploratory=True,
                decision='routing eligible' if passed else 'stop; current cross-channel advantage not established')
