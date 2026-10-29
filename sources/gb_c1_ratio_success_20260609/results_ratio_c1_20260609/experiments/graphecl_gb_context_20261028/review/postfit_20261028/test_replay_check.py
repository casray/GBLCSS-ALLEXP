import pytest
import torch
from replay_check import assess_replay


def test_cpu_near_tie_with_exact_gpu_reload_is_reported():
    saved=torch.tensor([[.50000006,.49999994],[.8,.2]])
    cpu=torch.tensor([[.49999997,.50000003],[.8,.2]])
    result=assess_replay(saved,cpu,saved.clone())
    assert result['cpu_disagreement_nodes']==[0]
    assert result['gpu_bitwise_equal']
    assert result['saved_margin_to_cpu_winner'][0]<=2*result['cpu_row_error'][0]


def test_gpu_prediction_artifact_change_is_rejected():
    saved=torch.tensor([[.7,.3]])
    with pytest.raises(AssertionError,match='GPU reload'):
        assess_replay(saved,saved,torch.tensor([[.700001,.299999]]))


def test_large_cpu_probability_error_is_rejected():
    saved=torch.tensor([[.7,.3]])
    with pytest.raises(AssertionError,match='CPU probability'):
        assess_replay(saved,torch.tensor([[.701,.299]]),saved)


def test_nonfinite_prediction_is_rejected():
    saved=torch.tensor([[.7,.3]])
    with pytest.raises(AssertionError,match='finite'):
        assess_replay(saved,torch.tensor([[float('nan'),.3]]),saved)
