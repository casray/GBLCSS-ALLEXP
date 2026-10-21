from gb_stage_b.cli import main
from gb_stage_b.utils import read_json
from gb_stage_b.utils import write_json


def test_cli_smoke_run(tmp_path):
    config = tmp_path / "stage_b.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: B1
ssl_objective: negfree
seed: 0
epochs: 4
warmup_epochs: 2
hidden_dim: 16
proj_dim: 8
device: cpu
experiment_group: aug_weak
variant: negfree_plain
ball_algorithm: gain_split
edge_drop_1: 0.1
edge_drop_2: 0.2
feat_mask_1: 0.1
feat_mask_2: 0.2
lambda_gb: 0.005
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )

    main(["run", "--config", str(config)])

    run_dirs = list((tmp_path / "results").glob("Synthetic/B1/seed_0/*"))
    assert run_dirs
    assert (run_dirs[0] / "config.json").exists()
    assert (run_dirs[0] / "metrics.json").exists()
    assert (run_dirs[0] / "history.json").exists()
    assert (run_dirs[0] / "embedding.pt").exists()


def test_cli_summarize_existing_results(tmp_path):
    run_dir = tmp_path / "results" / "Synthetic" / "B0" / "seed_0" / "run"
    write_json(
        run_dir / "metrics.json",
        {
            "dataset": "Synthetic",
            "method": "B0",
            "seed": 0,
            "ssl_objective": "negfree",
            "variant": "negfree_plain",
            "experiment_group": "aug_weak",
            "linear_eval_acc": 0.5,
            "mean_radius": 0.3,
            "mean_average_angular_distance": 0.2,
            "effective_rank": 4.0,
            "feature_std_mean": 0.1,
        },
    )

    main(["summarize", "--results", str(tmp_path / "results")])

    assert (tmp_path / "results" / "stage_b_summary.csv").exists()
    assert (tmp_path / "results" / "stage_b_grouped_summary.csv").exists()


def test_cli_accepts_c1_gated_overrides(tmp_path):
    config = tmp_path / "stage_c.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: C1
ssl_objective: negfree
seed: 0
epochs: 0
warmup_epochs: 0
phase2_epochs: 1
hidden_dim: 16
proj_dim: 8
device: cpu
gain_split_min_ball_size: 4
phase2_ball_refresh: 1
train_ratio: 0.6
val_ratio: 0.2
test_ratio: 0.2
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )

    main(
        [
            "run",
            "--config",
            str(config),
            "--method",
            "C1_gated",
            "--gate-tau-cov",
            "4",
            "--gate-tau-conf",
            "0.7",
            "--gate-alpha",
            "0.25",
        ]
    )

    run_dir = list((tmp_path / "results").glob("Synthetic/C1_gated/seed_0/*"))[0]
    cfg = read_json(run_dir / "config.json")
    assert cfg["method"] == "C1_gated"
    assert cfg["gate_tau_cov"] == 4
    assert cfg["gate_tau_conf"] == 0.7
    assert cfg["gate_alpha"] == 0.25


def test_cli_accepts_c1_weighted_overrides(tmp_path):
    config = tmp_path / "stage_c.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: C1
ssl_objective: negfree
seed: 0
epochs: 0
warmup_epochs: 0
phase2_epochs: 1
hidden_dim: 16
proj_dim: 8
device: cpu
gain_split_min_ball_size: 4
phase2_ball_refresh: 1
train_ratio: 0.6
val_ratio: 0.2
test_ratio: 0.2
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )

    main(
        [
            "run",
            "--config",
            str(config),
            "--method",
            "C1_weighted",
            "--weighted-k-ref",
            "4",
            "--weighted-floor",
            "0.2",
        ]
    )

    run_dir = list((tmp_path / "results").glob("Synthetic/C1_weighted/seed_0/*"))[0]
    cfg = read_json(run_dir / "config.json")
    assert cfg["method"] == "C1_weighted"
    assert cfg["weighted_k_ref"] == 4
    assert cfg["weighted_floor"] == 0.2


def test_b1_smoke_run_activates_compactness_after_warmup(tmp_path):
    config = tmp_path / "stage_b.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: B1
ssl_objective: negfree
seed: 0
epochs: 4
warmup_epochs: 2
hidden_dim: 16
proj_dim: 8
lambda_gb: 0.005
reliable_ball_min_size: 3
reliable_ball_max_size: 64
device: cpu
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )

    main(["run", "--config", str(config)])

    run_dir = list((tmp_path / "results").glob("Synthetic/B1/seed_0/*"))[0]
    history = read_json(run_dir / "history.json")["history"]
    metrics = read_json(run_dir / "metrics.json")
    post_warmup = [row for row in history if row["epoch"] >= 2]
    assert max(row["covered_nodes"] for row in post_warmup) > 0
    assert metrics["compact_nonzero_epochs"] > 0
    assert metrics["covered_nodes_max"] > 0


def test_cli_rebuild_balls_from_saved_embedding(tmp_path):
    config = tmp_path / "stage_b.yaml"
    config.write_text(
        f"""
dataset: Synthetic
method: B0
ssl_objective: infonce
seed: 0
epochs: 2
warmup_epochs: 1
hidden_dim: 16
proj_dim: 8
device: cpu
ball_algorithm: gain_split
results_dir: {tmp_path / "results"}
""",
        encoding="utf-8",
    )

    main(["run", "--config", str(config)])
    run_dir = list((tmp_path / "results").glob("Synthetic/B0/seed_0/*"))[0]

    main(
        [
            "rebuild-balls",
            "--run-dir",
            str(run_dir),
            "--min-ball-size",
            "4",
            "--max-depth",
            "8",
            "--min-split-gain",
            "0.001",
        ]
    )

    outputs = list(run_dir.glob("ball_ablation/*.json"))
    assert outputs
    metrics = read_json(outputs[0])
    assert metrics["ball_algorithm"] == "gain_split"
    assert metrics["gain_split_min_ball_size"] == 4
