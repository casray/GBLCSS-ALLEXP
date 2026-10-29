import copy
import pytest
import torch
from schedule import convert_checkpoint


def fixture_checkpoint():
    cfg = {"epochs": 900, "hidden": 2048, "lam": .1}
    old = dict(expected=dict(dataset="amazonpho", seed=1, method="graphecl", config=cfg),
               epoch=900, model={"w": torch.tensor([1.])},
               optimizer={"state": {0: {"step": torch.tensor(900.)}}}, rng={"cpu": torch.tensor([7])})
    expected = dict(dataset="amazonpho", seed=1, arm="ecl", config=cfg, receipt_sha256="new", stage="formal")
    return old, expected


def test_cached_checkpoint_preserves_training_state_without_aliasing():
    old, expected = fixture_checkpoint()
    converted = convert_checkpoint(old, expected)
    assert converted["expected"] == expected and converted["epoch"] == 900
    assert torch.equal(converted["model"]["w"], old["model"]["w"])
    assert torch.equal(converted["rng"]["cpu"], old["rng"]["cpu"])
    converted["optimizer"]["state"][0]["step"].add_(1)
    assert old["optimizer"]["state"][0]["step"].item() == 900
    assert old["expected"]["method"] == "graphecl"


@pytest.mark.parametrize("mutation", ["seed", "config", "epoch"])
def test_reject_incompatible_cached_baseline(mutation):
    old, expected = fixture_checkpoint()
    old = copy.deepcopy(old)
    if mutation == "seed": old["expected"]["seed"] = 0
    if mutation == "config": old["expected"]["config"]["hidden"] = 1024
    if mutation == "epoch": old["epoch"] = 450
    with pytest.raises(ValueError, match="Cached baseline"):
        convert_checkpoint(old, expected)
