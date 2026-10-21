from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_single_exp_limits_label_noise_to_training_nodes():
    source = (REPO_ROOT / "single_exp.py").read_text(encoding="utf-8")

    assert "eligible_mask=data.train_masks" in source


def test_total_exp_limits_label_noise_to_training_nodes():
    source = (REPO_ROOT / "total_exp.py").read_text(encoding="utf-8")

    assert "eligible_mask=dataset.train_masks" in source
