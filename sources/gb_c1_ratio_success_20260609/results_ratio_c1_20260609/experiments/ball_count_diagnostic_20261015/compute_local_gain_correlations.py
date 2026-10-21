"""Size-normalized sensitivity check for post-hoc geometry/target gain correlations."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import data_tools as dt


def main():
    root=dt.ROOT/'split_diagnostic';source=root/'per_split_utility.csv';digest=dt.d.sha(source)
    rows=[]
    for key,g in pd.read_csv(source).groupby(['dataset','space','noise','seed']):
        x=g.relative_sse_gain.to_numpy();y=g.brier_gain_per_parent_train.to_numpy()
        rho=float(spearmanr(x,y).statistic) if len(g)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else None
        rows.append(dict(zip(['dataset','space','noise','seed'],key))|{'spearman_relative_sse_vs_local_brier_gain':rho})
    pd.DataFrame(rows).to_csv(root/'local_normalized_gain_correlations.csv',index=False)
    if dt.d.sha(source)!=digest:raise ValueError('Split input changed')
    dt.d.write(root/'local_correlation_manifest.json',dict(status='passed',script_sha256=dt.d.sha(__file__),
        source_sha256=digest,cells=len(rows),x='relative SSE reduction',y='Brier reduction per affected parent training node',
        aggregation='Spearman per tree/condition, then descriptive mean/std over protocol seeds; no significance claim'))


if __name__=='__main__':main()
