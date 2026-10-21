from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gblcss.reporting import summarize


COMPARATORS = ['mlp_ce', 'jocor', 'gblcss_shuffled']


def rows(method, values, **setting):
    return [dict(suite='matched', dataset='toy', method=method,
                 variant='default', noise_type='uniform', noise_rate=0.4,
                 validation_protocol='clean', seed=seed, test_accuracy=value,
                 best_epoch=10) | setting
            for seed, value in enumerate(values)]


def matched(proposed, comparator, **setting):
    return pd.DataFrame(rows('gblcss', proposed, **setting) + [
        row for method in COMPARATORS for row in rows(method, comparator, **setting)
    ])


def test_aggregate_pivots_and_generic_diagnostics(tmp_path):
    frame = pd.DataFrame(rows('mlp_ce', [0.7, 0.9], suite='main'))
    frame['target_quality_future'] = [0.2, 0.6]
    frame['target_sparse'] = [np.nan, 0.5]
    frame['epochs'] = 100
    frame['parameter'] = 0.1
    frame['parameter_count'] = 200
    frame['run_id'] = [1, 2]
    frame['note'] = 'not a metric'
    frame = frame.drop(columns='variant')
    original = frame.copy(deep=True)
    assert summarize(frame, tmp_path) is None
    pd.testing.assert_frame_equal(frame, original)
    aggregate = pd.read_csv(tmp_path / 'aggregate.csv')
    row = aggregate.iloc[0]
    assert row.variant == 'default'
    assert row.seeds == 2
    assert row.mean_test_accuracy_pct == pytest.approx(80)
    assert row.std_test_accuracy_pct == pytest.approx(np.std([70, 90], ddof=1))
    table = pd.read_csv(tmp_path / 'main_table.csv', header=[0, 1, 2, 3], index_col=[0, 1])
    assert table.loc[('mlp_ce', 'default')].iloc[0] == '80.00 +- 14.14'
    diagnostics = pd.read_csv(tmp_path / 'diagnostics_aggregate.csv').set_index('metric')
    assert set(diagnostics.index) == {'target_quality_future', 'target_sparse'}
    assert diagnostics.loc['target_quality_future', 'mean'] == pytest.approx(0.4)
    assert diagnostics.loc['target_quality_future', 'std'] == pytest.approx(np.std([.2, .6], ddof=1))
    assert diagnostics.loc['target_sparse', 'count'] == 1
    assert np.isnan(diagnostics.loc['target_sparse', 'std'])


def test_suites_variants_and_validation_are_not_pooled(tmp_path):
    frame = pd.DataFrame([
        row for suite in ['main', 'other'] for variant in ['default', 'changed']
        for protocol in ['clean', 'noisy']
        for row in rows('mlp_ce', [.6, .8], suite=suite, variant=variant,
                        validation_protocol=protocol)
    ])
    summarize(frame, tmp_path)
    aggregate = pd.read_csv(tmp_path / 'aggregate.csv')
    assert len(aggregate) == 8
    assert (aggregate.seeds == 2).all()
    table = pd.read_csv(tmp_path / 'main_table.csv', header=[0, 1, 2, 3], index_col=[0, 1])
    assert table.shape == (2, 2)


def test_paired_statistics_and_holm_per_comparator(tmp_path):
    rng = np.random.default_rng(39)
    records = []
    expected = {}
    for dataset in ['a', 'b']:
        for rate in [.2, .4]:
            base = rng.uniform(.6, .7, 6)
            records.extend(rows('gblcss', base, dataset=dataset, noise_rate=rate))
            for i, method in enumerate(COMPARATORS):
                reference = base - rng.normal(.01 * (i + 1), .04, 6)
                records.extend(rows(method, reference, dataset=dataset, noise_rate=rate))
                expected[dataset, rate, method] = base - reference
    frame = pd.DataFrame(records).sample(frac=1, random_state=17)
    summarize(frame, tmp_path)
    result = pd.read_csv(tmp_path / 'table_b1_paired_statistics.csv')
    assert len(result) == 12
    for row in result.itertuples():
        delta = expected[row.dataset, row.noise_rate, row.comparator]
        margin = stats.t.ppf(.975, 5) * delta.std(ddof=1) / np.sqrt(6)
        assert row.mean_difference_pp == pytest.approx(delta.mean() * 100)
        assert row.ci95_low_pp == pytest.approx((delta.mean() - margin) * 100)
        assert row.ci95_high_pp == pytest.approx((delta.mean() + margin) * 100)
        assert row.p_value_two_sided == pytest.approx(stats.ttest_1samp(delta, 0).pvalue)
        assert row.positive_seeds == (delta > 0).sum()
        assert row.seeds == 6
    for _, group in result.groupby('comparator'):
        group = group.sort_values('p_value_two_sided')
        adjusted = np.minimum(1, np.maximum.accumulate(group.p_value_two_sided.to_numpy() * [4, 3, 2, 1]))
        np.testing.assert_allclose(group.holm_adjusted_p, adjusted)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, -.01, 1.01, 'bad'])
def test_rejects_invalid_accuracy_before_writing(tmp_path, value):
    frame = matched([.7], [.6])
    frame['test_accuracy'] = frame.test_accuracy.astype(object)
    frame.loc[0, 'test_accuracy'] = value
    with pytest.raises(ValueError, match='test_accuracy'):
        summarize(frame, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_duplicate_rejected_after_default_variant_normalization(tmp_path):
    frame = pd.DataFrame(rows('mlp_ce', [.7], suite='main') * 2)
    frame.loc[0, 'variant'] = None
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        summarize(frame, tmp_path)


@pytest.mark.parametrize('change', ['identity', 'missing', 'extra', 'setting', 'method'])
def test_exact_pairing_required(tmp_path, change):
    frame = matched([.7, .8], [.6, .7])
    comparator = frame.method == 'mlp_ce'
    index = frame.index[comparator][0]
    if change == 'identity':
        frame.loc[index, 'seed'] = 99
    elif change == 'missing':
        frame = frame.drop(index)
    elif change == 'extra':
        frame = pd.concat([frame, frame.loc[[index]].assign(seed=99)])
    elif change == 'setting':
        frame.loc[comparator, 'validation_protocol'] = 'noisy'
    else:
        frame = frame.loc[~comparator]
    with pytest.raises(ValueError, match='[Pp]air|[Ss]eed'):
        summarize(frame, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_single_seed_nan_does_not_contaminate_holm(tmp_path):
    frame = pd.concat([matched([.7], [.6], dataset='single'),
                       matched([.5, .75], [.5, .75], dataset='zero'),
                       matched([.5, .75], [.25, .5], dataset='constant')])
    summarize(frame, tmp_path)
    result = pd.read_csv(tmp_path / 'table_b1_paired_statistics.csv')
    single = result[result.dataset == 'single']
    assert single[['ci95_low_pp', 'ci95_high_pp', 'p_value_two_sided', 'holm_adjusted_p']].isna().all().all()
    zero = result[result.dataset == 'zero']
    assert (zero[['p_value_two_sided', 'holm_adjusted_p']] == 1).all().all()
    assert (zero[['ci95_low_pp', 'ci95_high_pp']] == 0).all().all()
    constant = result[result.dataset == 'constant']
    assert (constant[['p_value_two_sided', 'holm_adjusted_p']] == 0).all().all()
    assert (constant[['ci95_low_pp', 'ci95_high_pp']] == 25).all().all()
    aggregate = pd.read_csv(tmp_path / 'aggregate.csv')
    assert aggregate.loc[aggregate.dataset == 'single', 'std_test_accuracy_pct'].isna().all()


def appendix_d_frame():
    return pd.DataFrame(
        rows('gblcss', [.7, .9], suite='leave_one_out', noise_rate=.2)
        + rows('gblcss_self_vote', [.6, .7], suite='leave_one_out', noise_rate=.2)
        + rows('gblcss', [.5, .7], suite='leave_one_out', noise_rate=.4)
        + rows('gblcss_self_vote', [.4, .6], suite='leave_one_out', noise_rate=.4)
        + rows('gblcss', [.8, .9], suite='clean', noise_rate=0)
        + rows('mlp_ce', [.9, .9], suite='clean', noise_rate=0)
        + rows('gblcss', [.1, .1], suite='unrelated'))


def test_appendix_d_averages_paired_noise_seed_differences(tmp_path):
    summarize(appendix_d_frame(), tmp_path)
    result = pd.read_csv(tmp_path / 'table_d1_differences.csv').set_index('suite')
    assert set(result.index) == {'leave_one_out', 'clean'}
    assert result.loc['leave_one_out', 'mean_difference_pp'] == pytest.approx(12.5)
    assert result.loc['leave_one_out', 'pairs'] == 4
    assert result.loc['leave_one_out', 'noise_settings'] == 2
    assert result.loc['clean', 'mean_difference_pp'] == pytest.approx(-5)


def test_appendix_d_also_requires_exact_pairing(tmp_path):
    frame = appendix_d_frame()
    frame.loc[frame.method == 'gblcss_self_vote', 'seed'] += 10
    with pytest.raises(ValueError, match='[Pp]air|[Ss]eed'):
        summarize(frame, tmp_path)


def test_clean_suite_rejects_nonzero_training_noise(tmp_path):
    frame = appendix_d_frame()
    frame.loc[frame.suite == 'clean', 'noise_rate'] = .2
    with pytest.raises(ValueError, match='clean'):
        summarize(frame, tmp_path)


@pytest.mark.parametrize('column', ['suite', 'seed', 'best_epoch'])
def test_required_columns(tmp_path, column):
    with pytest.raises(ValueError, match=column):
        summarize(matched([.7], [.6]).drop(columns=column), tmp_path)


def test_empty_input_writes_readable_headers(tmp_path):
    summarize(matched([.7], [.6]).iloc[:0], tmp_path)
    for name in ['aggregate.csv', 'diagnostics_aggregate.csv',
                 'table_b1_paired_statistics.csv', 'table_d1_differences.csv']:
        assert pd.read_csv(tmp_path / name).empty


def test_summarizer_never_reads_reference_csvs(tmp_path, monkeypatch):
    def forbidden_read(*args, **kwargs):
        pytest.fail('summarize must not read CSV inputs or archived results')

    monkeypatch.setattr(pd, 'read_csv', forbidden_read)
    summarize(matched([.7, .8], [.6, .65]), tmp_path)
    assert (tmp_path / 'aggregate.csv').is_file()


def test_all_single_seed_tests_remain_undefined_even_for_zero_delta(tmp_path):
    summarize(matched([.7], [.7]), tmp_path)
    result = pd.read_csv(tmp_path / 'table_b1_paired_statistics.csv')
    assert len(result) == 3
    assert result[['ci95_low_pp', 'ci95_high_pp', 'p_value_two_sided',
                   'holm_adjusted_p']].isna().all().all()


def test_appendix_d_keeps_variants_and_protocols_separate(tmp_path):
    frame = pd.concat([appendix_d_frame().assign(variant=variant, validation_protocol=protocol)
                       for variant in ['default', 'alternative'] for protocol in ['clean', 'noisy']])
    summarize(frame, tmp_path)
    result = pd.read_csv(tmp_path / 'table_d1_differences.csv')
    assert len(result) == 8
    assert (result.loc[result.suite == 'leave_one_out', 'pairs'] == 4).all()
    assert (result.loc[result.suite == 'clean', 'pairs'] == 2).all()


@pytest.mark.parametrize('column,value', [('validation_protocol', 'unknown'), ('seed', None)])
def test_invalid_group_identifiers_are_rejected(tmp_path, column, value):
    frame = matched([.7], [.6])
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        summarize(frame, tmp_path)


def test_accuracy_boundaries_are_valid(tmp_path):
    summarize(pd.DataFrame(rows('mlp_ce', [0, 1], suite='main')), tmp_path)
    result = pd.read_csv(tmp_path / 'aggregate.csv').iloc[0]
    assert result.mean_test_accuracy_pct == 50
    assert result.std_test_accuracy_pct == pytest.approx(100 / np.sqrt(2))
