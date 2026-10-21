from pathlib import Path

from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _gb_value(conf, key):
    gb = conf.gb
    return gb[key] if isinstance(gb, dict) else getattr(gb, key)


def test_corrected_knn_configs_exclude_self_and_use_tie_safe_targets():
    for dataset in DATASETS:
        conf = load_conf(None, "knn_mlp_softdist_beta05", dataset)
        assert _gb_value(conf, "exclude_self") is True
        assert _gb_value(conf, "tie_mode") == "distribution_only"


def test_kmeans_configs_cover_all_main_datasets_and_use_tie_safe_targets():
    for dataset in DATASETS:
        config_path = (
            REPO_ROOT
            / "config"
            / "kmeans_mlp_softdist_beta05"
            / f"kmeans_mlp_softdist_beta05_{dataset}.yaml"
        )
        assert config_path.exists()
        conf = load_conf(None, "kmeans_mlp_softdist_beta05", dataset)
        assert _gb_value(conf, "tie_mode") == "distribution_only"


def test_knn_teacher_configs_exclude_query_node():
    for dataset in (*DATASETS, "pubmed"):
        conf = load_conf(None, "mlp_distill_knn_xh", dataset)
        assert _gb_value(conf, "exclude_self") is True
        assert float(_gb_value(conf, "target_beta")) == 0.0
