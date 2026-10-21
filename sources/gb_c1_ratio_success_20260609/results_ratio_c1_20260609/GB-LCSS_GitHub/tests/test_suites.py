from pathlib import Path

import pytest

from gblcss.suites import build_plan

ROOT = Path(__file__).resolve().parents[1]


def test_matched_suite_preserves_full_grid():
    plan = build_plan(ROOT, ['matched'])
    assert len(plan) == 5 * 5 * 5 * 9
    assert {job['noise_type'] for job in plan} == {'uniform', 'pair', 'instance'}
    assert {job['seed'] for job in plan} == set(range(5))


def test_overrides_and_sensitivity():
    plan = build_plan(ROOT, ['sensitivity'], datasets=['cora'], seeds=[0])
    assert len(plan) == 14
    assert len({job['variant'] for job in plan}) == 14
    assert all(job['noise_rate'] == .4 for job in plan)


def test_suite_selection_is_strict():
    with pytest.raises(ValueError, match='suite'):
        build_plan(ROOT, ['typo'])
    with pytest.raises(ValueError, match='dataset'):
        build_plan(ROOT, ['matched'], datasets=['typo'])
    with pytest.raises(ValueError, match='seed'):
        build_plan(ROOT, ['matched'], seeds=[0, 0])


def test_validation_pairs_clean_and_noisy():
    plan = build_plan(ROOT, ['validation'], datasets=['cora'], seeds=[0])
    assert len(plan) == 12
    assert {job['validation_protocol'] for job in plan} == {'clean', 'noisy'}
