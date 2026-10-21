import numpy as np
import pandas as pd
import pytest

from scripts.summarize_mlp62_vqgraph_sampled_rocm import (
    FORMAL_DATASETS,
    aggregate_results,
    pair_with_ball_teacher,
    validate_formal_rows,
)


def _formal_rows():
    rows = []
    for rate in (0.4, 0.6):
        for dataset in FORMAL_DATASETS:
            for seed in range(5):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": "vqgraph_sampled_faithful",
                        "seed": seed,
                        "noise_type": "uniform",
                        "noise_rate": rate,
                        "noise_scope": "train",
                        "status": "ok",
                        "test": 0.5 + 0.01 * seed,
                        "split_hash": f"split-{dataset}-{seed}",
                        "noise_hash": f"noise-{dataset}-{rate}-{seed}",
                        "vqgraph_active_codes": 2,
                        "vqgraph_teacher_parameter_device": "cuda:0",
                        "vqgraph_student_parameter_device": "cuda:0",
                        "g2m_train_uses_graph": 1,
                        "g2m_inference_uses_graph": 0,
                    }
                )
    return pd.DataFrame(rows)


def _ball_rows(formal):
    return pd.DataFrame(
        {
            "dataset": formal["dataset"],
            "seed": formal["seed"],
            "noise_rate": formal["noise_rate"],
            "method": "mlp_distill_formal_lcss_x",
            "test": formal["test"] + 0.2,
            "split_hash": formal["split_hash"],
            "noise_hash": formal["noise_hash"],
        }
    )


def test_formal_validation_excludes_corafull_by_contract():
    formal = _formal_rows()
    audit = validate_formal_rows(formal)
    assert len(audit) == 40
    assert audit["passed"].all()

    extra = pd.concat(
        [formal, formal.iloc[[0]].assign(dataset="corafull")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_formal_rows(extra)


def test_aggregation_and_hash_paired_ball_comparison():
    formal = _formal_rows()
    aggregate = aggregate_results(formal)
    assert len(aggregate) == 8
    assert (aggregate["count"] == 5).all()

    paired = pair_with_ball_teacher(formal, _ball_rows(formal))
    assert len(paired) == 40
    assert paired["hashes_paired"].all()
    assert np.allclose(paired["delta_pp"], -20.0)


def test_pairing_rejects_hash_mismatch():
    formal = _formal_rows()
    controls = _ball_rows(formal)
    controls.loc[0, "noise_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        pair_with_ball_teacher(formal, controls)
