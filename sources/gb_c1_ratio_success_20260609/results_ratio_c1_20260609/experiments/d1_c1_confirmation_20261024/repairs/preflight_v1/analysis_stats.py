"""Seed is the repeat unit; conditions are averaged inside each seed."""
import numpy as np


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values)>1 else None


def seed_macro(rows, key):
    return [float(np.mean([r[key] for r in rows if r['seed']==s])) for s in sorted({r['seed'] for r in rows})]
