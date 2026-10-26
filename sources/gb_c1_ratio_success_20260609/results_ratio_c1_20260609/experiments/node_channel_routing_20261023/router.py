"""Validation-only nested routing; serialized models require no pickle."""
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'channel_routing_variance_20261023'))
from routing_kernels import student_features, choose_threshold, choose_weight

CONFIG = json.loads((ROOT/'config.json').read_text())
FEATURES = ['A_max', 'A_entropy', 'A_margin', 'C_max', 'C_entropy',
            'C_margin', 'JS', 'agreement']


def fit_model(x, y):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return dict(kind='constant', value=int(y[0]) if len(y) else 0,
                    n=int(len(y)), positives=int(y.sum()))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=CONFIG['logistic_C'], solver='lbfgs',
                               max_iter=CONFIG['max_iter'], class_weight=None)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    return dict(kind='logistic', mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                n=int(len(y)), positives=int(y.sum()), iterations=int(model.n_iter_[0]))


def predict_model(model, x):
    if model['kind'] == 'constant':
        return np.full(len(x), model['value'], dtype=float)
    return expit(((x-np.asarray(model['mean']))/np.asarray(model['scale'])) @
                 np.asarray(model['coef']) + model['intercept'])


def apply_route(a, c, score, threshold):
    threshold = float('inf') if threshold is None else threshold
    return np.where((score >= threshold) & (a != c), c, a)


def fit_local(x, a, c, y):
    usable = (a == y) ^ (c == y)
    return fit_model(x[usable], (c[usable] == y[usable]).astype(int))


def oof(x, a, c, y, folds):
    scores = np.zeros(len(y))
    models = []
    for train, hold in KFold(folds, shuffle=True, random_state=CONFIG['fold_seed']).split(x):
        m = fit_local(x[train], a[train], c[train], y[train])
        scores[hold] = predict_model(m, x[hold])
        models.append(dict(model=m, train=train.tolist(), hold=hold.tolist()))
    return scores, models


def threshold(y, a, c, scores):
    t = choose_threshold(y, a, c, scores)
    return None if np.isinf(t) else t


def fit_router(a_prob, c_prob, valid_indices, valid_labels):
    """Only validation labels enter this API; test labels are unavailable here."""
    x = student_features(a_prob, c_prob)
    a, c = a_prob.argmax(1), c_prob.argmax(1)
    vi = np.asarray(valid_indices)
    y = np.asarray(valid_labels)
    xv, av, cv = x[vi], a[vi], c[vi]
    nested = np.full(len(vi), -1, dtype=int)
    global_pred = nested.copy(); nested05 = nested.copy()
    fold_ids = nested.copy(); nested_score = np.zeros(len(vi))
    records = []
    for k, (train, hold) in enumerate(KFold(CONFIG['outer_folds'], shuffle=True,
                                          random_state=CONFIG['fold_seed']).split(vi)):
        inner, models = oof(xv[train], av[train], cv[train], y[train], CONFIG['inner_folds'])
        t = threshold(y[train], av[train], cv[train], inner)
        m = fit_local(xv[train], av[train], cv[train], y[train])
        score = predict_model(m, xv[hold])
        nested[hold] = apply_route(av[hold], cv[hold], score, t)
        nested05[hold] = apply_route(av[hold], cv[hold], score, .5)
        nested_score[hold] = score; fold_ids[hold] = k
        w = choose_weight(y[train], a_prob[vi[train]], c_prob[vi[train]])
        global_pred[hold] = (w*a_prob[vi[hold]]+(1-w)*c_prob[vi[hold]]).argmax(1)
        records.append(dict(train=train.tolist(), hold=hold.tolist(), model=m,
                            inner_models=models, threshold=t, global_weight=w))
    full_oof, full_models = oof(xv, av, cv, y, CONFIG['outer_folds'])
    t = threshold(y, av, cv, full_oof)
    final = fit_local(xv, av, cv, y)
    return dict(score=predict_model(final, x), threshold=t, model=final,
                nested_pred=nested, nested_score=nested_score, nested05=nested05,
                global_nested=global_pred, fold_ids=fold_ids, fold_models=records,
                full_oof=full_oof, full_oof_models=full_models,
                global_weight=choose_weight(y, a_prob[vi], c_prob[vi]))
