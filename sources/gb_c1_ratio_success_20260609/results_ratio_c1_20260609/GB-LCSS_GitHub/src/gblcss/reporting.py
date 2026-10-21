"""CSV summaries of fresh runner rows; no archived results are read."""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy import stats


GROUP = ['suite', 'dataset', 'method', 'variant', 'noise_type', 'noise_rate',
         'validation_protocol']
SETTING = [column for column in GROUP if column != 'method']
PAIR = SETTING + ['seed']
COMPARATORS = ['mlp_ce', 'jocor', 'gblcss_shuffled']
B_COLUMNS = SETTING + ['comparator', 'seeds', 'mean_difference_pp',
                      'ci95_low_pp', 'ci95_high_pp', 'positive_seeds',
                      'p_value_two_sided', 'holm_adjusted_p']
D_GROUP = ['suite', 'dataset', 'variant', 'validation_protocol']
D_COLUMNS = D_GROUP + ['comparator', 'pairs', 'seeds', 'noise_settings',
                       'mean_difference_pp']
METADATA = {'best_epoch', 'epoch', 'epochs', 'max_epochs', 'parameter',
            'parameters', 'params', 'num_params', 'n_params', 'num_parameters',
            'n_parameters', 'learning_rate', 'weight_decay', 'hidden_dim',
            'num_layers', 'batch_size'}


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise ValueError('Duplicate column names')
    missing = set(GROUP + ['seed', 'test_accuracy', 'best_epoch']) - {'variant'} - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame.copy()
    if 'variant' not in frame:
        frame['variant'] = 'default'
    frame['variant'] = frame.variant.fillna('default').replace('', 'default')
    if frame[GROUP + ['seed']].isna().any().any():
        raise ValueError('Grouping identifiers and seed must not be missing')
    if not frame.validation_protocol.isin(['clean', 'noisy']).all():
        raise ValueError('validation_protocol must be clean or noisy')
    for column in ['test_accuracy', 'noise_rate']:
        try:
            frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{column} must be finite numeric values in [0, 1]') from exc
        if not (np.isfinite(frame[column]) & frame[column].between(0, 1)).all():
            raise ValueError(f'{column} must be finite numeric values in [0, 1]')
    if frame.duplicated(GROUP + ['seed']).any():
        raise ValueError('Duplicate run rows for grouping identifiers and seed')
    if (frame.loc[frame.suite == 'clean', 'noise_rate'] != 0).any():
        raise ValueError('The clean suite requires zero training-label noise')
    return frame


def _paired(frame: pd.DataFrame, comparator: str) -> pd.DataFrame:
    left = frame.loc[frame.method == 'gblcss', PAIR + ['test_accuracy']]
    right = frame.loc[frame.method == comparator, PAIR + ['test_accuracy']]
    paired = left.merge(right, on=PAIR, how='outer', validate='one_to_one',
                        suffixes=('_gblcss', '_comparator'), indicator=True)
    if left.empty or right.empty or (paired['_merge'] != 'both').any():
        unmatched = paired.loc[paired['_merge'] != 'both', PAIR + ['_merge']]
        raise ValueError(f'Pairing mismatch for gblcss vs {comparator}: exact setting/seed '
                         f'identities required; {unmatched.head(5).to_dict("records")}')
    paired['difference_pp'] = 100 * (paired.test_accuracy_gblcss - paired.test_accuracy_comparator)
    return paired


def _appendix_b(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = frame.loc[frame.suite == 'matched']
    if matched.empty:
        return pd.DataFrame(columns=B_COLUMNS)
    for comparator in COMPARATORS:
        for key, group in _paired(matched, comparator).groupby(SETTING, sort=True):
            delta = group.difference_pp
            n, mean = len(delta), float(delta.mean())
            low = high = p_value = np.nan
            if n > 1:
                std = float(delta.std(ddof=1))
                if std == 0:
                    low = high = mean
                    p_value = 1.0 if mean == 0 else 0.0
                else:
                    sem = std / np.sqrt(n)
                    margin = float(stats.t.ppf(.975, n - 1)) * sem
                    low, high = mean - margin, mean + margin
                    p_value = float(2 * stats.t.sf(abs(mean / sem), n - 1))
            rows.append(dict(zip(SETTING, key)) | {
                'comparator': comparator, 'seeds': n, 'mean_difference_pp': mean,
                'ci95_low_pp': low, 'ci95_high_pp': high,
                'positive_seeds': int((delta > 0).sum()),
                'p_value_two_sided': p_value, 'holm_adjusted_p': np.nan,
            })
    result = pd.DataFrame(rows, columns=B_COLUMNS)
    # Undefined single-seed tests are not members of the tested Holm family.
    for _, family in result.groupby('comparator', sort=True):
        p = family.p_value_two_sided.dropna().sort_values()
        adjusted = np.minimum(1, np.maximum.accumulate(p.to_numpy() * np.arange(len(p), 0, -1)))
        result.loc[p.index, 'holm_adjusted_p'] = adjusted
    return result.sort_values(SETTING + ['comparator'])


def _appendix_d(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for suite, comparator in [('leave_one_out', 'gblcss_self_vote'), ('clean', 'mlp_ce')]:
        subset = frame.loc[frame.suite == suite]
        if subset.empty:
            continue
        for key, group in _paired(subset, comparator).groupby(D_GROUP, sort=True):
            rows.append(dict(zip(D_GROUP, key)) | {
                'comparator': comparator, 'pairs': len(group),
                'seeds': group.seed.nunique(),
                'noise_settings': len(group[['noise_type', 'noise_rate']].drop_duplicates()),
                'mean_difference_pp': group.difference_pp.mean(),
            })
    return pd.DataFrame(rows, columns=D_COLUMNS)


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = set(GROUP + ['seed', 'test_accuracy']) | METADATA
    metrics = [column for column in frame.select_dtypes(include='number').columns
               if column not in excluded
               and not column.startswith(('parameter_', 'param_'))
               and not column.endswith(('_id', '_parameters', '_params', '_epochs'))]
    columns = GROUP + ['metric', 'count', 'mean', 'std']
    if not metrics:
        return pd.DataFrame(columns=columns)
    long = frame.melt(id_vars=GROUP, value_vars=metrics, var_name='metric', value_name='value')
    return (long.groupby(GROUP + ['metric'], as_index=False, observed=True)
            .agg(count=('value', 'count'), mean=('value', 'mean'), std=('value', 'std')))[columns]


def summarize(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write aggregate.csv, <suite>_table.csv, diagnostics_aggregate.csv and B/D CSVs.

    Accuracy means/stds in aggregate.csv are percentages (sample std, ddof=1).
    Suite tables have method/variant rows and four header levels: dataset,
    noise_type, noise_rate, validation_protocol. Missing variants use 'default'.
    Diagnostics retain their original units and exclude run/config metadata.
    B reports two-sided paired t-tests, unadjusted 95% CIs in percentage points,
    and Holm p-values across all tested settings separately per comparator.
    D averages paired differences across noise/seed within dataset/variant/protocol.
    All pair keys, including variants, must match exactly; no inner-join dropping.
    """
    frame = _validate(frame)
    appendix_b, appendix_d = _appendix_b(frame), _appendix_d(frame)
    aggregate = (frame.groupby(GROUP, as_index=False, observed=True)
                 .agg(seeds=('seed', 'nunique'),
                      mean_test_accuracy_pct=('test_accuracy', 'mean'),
                      std_test_accuracy_pct=('test_accuracy', 'std')))
    aggregate[['mean_test_accuracy_pct', 'std_test_accuracy_pct']] *= 100
    diagnostics = _diagnostics(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [('aggregate', aggregate), ('diagnostics_aggregate', diagnostics),
                        ('table_b1_paired_statistics', appendix_b),
                        ('table_d1_differences', appendix_d)]:
        table.to_csv(output_dir / f'{name}.csv', index=False, na_rep='NaN')
    for suite, group in aggregate.groupby('suite', sort=True, observed=True):
        group = group.assign(accuracy=[f'{mean:.2f} +- {std:.2f}' if pd.notna(std)
                                       else f'{mean:.2f} +- NaN'
                                       for mean, std in zip(group.mean_test_accuracy_pct,
                                                            group.std_test_accuracy_pct)])
        table = group.pivot(index=['method', 'variant'],
                            columns=['dataset', 'noise_type', 'noise_rate', 'validation_protocol'],
                            values='accuracy')
        table.to_csv(output_dir / f'{quote(str(suite), safe="")}_table.csv', na_rep='NaN')
    _paper_views(frame, aggregate, output_dir)


def _paper_views(frame, aggregate, output_dir):
    """Convenient main-table slices; all values come from the supplied runs."""
    names = {'matched': 'table3', 'graph_assisted': 'table4',
             'target_controls': 'table5', 'grouping': 'table6',
             'representations': 'table7'}
    for suite, name in names.items():
        selected = aggregate.loc[(aggregate.suite == suite) &
            (aggregate.noise_type == 'uniform') & (aggregate.noise_rate == .4)]
        if selected.empty:
            continue
        selected = selected.assign(accuracy=[f'{mean:.2f} +- {std:.2f}'
            for mean, std in zip(selected.mean_test_accuracy_pct, selected.std_test_accuracy_pct)])
        selected.pivot(index='method', columns='dataset', values='accuracy').to_csv(output_dir / f'{name}.csv')

    extra = frame.loc[frame.suite == 'extra']
    rows = []
    for dataset, group in extra.groupby('dataset'):
        means = group.groupby('method').test_accuracy.mean()
        baselines = means.drop(['gblcss', 'gblcss_shuffled'], errors='ignore')
        if baselines.empty or not {'gblcss', 'gblcss_shuffled'}.issubset(means.index):
            continue
        best = baselines.idxmax()
        pairs = _paired(group, best)
        rows.append(dict(dataset=dataset, best_baseline=best,
                         gblcss_accuracy_pct=100 * means['gblcss'],
                         baseline_accuracy_pct=100 * means[best],
                         difference_vs_baseline_pp=100 * (means['gblcss'] - means[best]),
                         wins=int((pairs.difference_pp > 0).sum()), seeds=len(pairs),
                         random_grouping_accuracy_pct=100 * means['gblcss_shuffled'],
                         difference_vs_random_pp=100 * (means['gblcss'] - means['gblcss_shuffled'])))
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'table_g2.csv', index=False)
