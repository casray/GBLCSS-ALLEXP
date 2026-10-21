from __future__ import annotations

import importlib
from argparse import Namespace
from pathlib import Path

import torch

from utils.tools import load_conf


class TinyData:
    def __init__(self):
        self.n_nodes = 6
        self.n_classes = 3
        self.dim_feats = 4
        self.feats = torch.tensor(
            [
                [1.0, 0.0, 0.2, 0.1],
                [0.8, 0.1, 0.0, 0.2],
                [0.0, 1.0, 0.1, 0.2],
                [0.1, 0.8, 0.2, 0.0],
                [0.2, 0.1, 1.0, 0.0],
                [0.0, 0.2, 0.8, 0.1],
            ]
        )
        self.labels = torch.tensor([0, 0, 1, 1, 2, 2])
        self.noisy_label = torch.tensor([0, 1, 1, 1, 2, 2])
        self.train_masks = torch.tensor([0, 1, 2])
        self.val_masks = torch.tensor([3])
        self.test_masks = torch.tensor([4, 5])
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
        )
        self.adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1]),
            (self.n_nodes, self.n_nodes),
        ).coalesce()


def tiny_conf(method: str) -> Namespace:
    return Namespace(
        model={
            "method": method,
            "n_feat": 4,
            "n_classes": 3,
            "n_hidden": 8,
            "n_layer": 2,
            "dropout": 0.0,
            "teacher_hidden": 8,
            "teacher_n_layer": 2,
            "teacher_dropout": 0.0,
            "teacher_epochs": 2,
            "teacher_lr": 0.01,
            "teacher_weight_decay": 0.0,
            "distill_lambda": 0.5,
            "distill_temperature": 0.8,
            "reliability_noise_level": 1.0,
            "reliability_init_power": 1.0,
            "reliability_momentum": 0.99,
            "reliability_bins": 5,
            "reliability_update_start": 1,
            "codebook_size": 4,
            "vq_decay": 0.8,
            "vq_commitment_weight": 0.25,
            "vq_lambda_node": 0.001,
            "vq_lambda_edge": 0.03,
            "token_temperature": 4.0,
            "lambda_soft_labels": 0.0,
            "lambda_soft_tokens": 1e-8,
            "batch_size": 6,
        },
        training={
            "lr": 0.01,
            "weight_decay": 0.0,
            "n_epochs": 2,
            "patience": 2,
            "criterion": "metric",
            "debug": False,
        },
        dataset={"sparse": True},
        analysis={"flag": False},
    )


def test_official_source_commits_are_pinned():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    assert module.KRD_SOURCE_COMMIT == "32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21"
    assert module.VQGRAPH_SOURCE_COMMIT == "60aa1c99782d542c316a37f0394eba934bcf97b0"


def test_faithful_methods_are_separate_and_registered():
    standard = importlib.import_module("scripts.run_standard_downstream")
    expected = {
        "krd_faithful": "official-core faithful protocol port",
        "vqgraph_faithful": "official-core faithful protocol port",
    }
    for method, status in expected.items():
        assert standard.PREDICTOR_MODULES[method] == "predictor.G2MFaithful_Predictor"
        capability = standard.METHOD_CAPABILITIES[method]
        assert capability["implementation_status"] == status
        assert capability["training_uses_graph"] is True
        assert capability["inference_uses_graph"] is False
        assert capability["uses_unlabeled_nodes"] is True


def test_vqgraph_cora_config_matches_the_paper_main_setting():
    root = Path(__file__).resolve().parents[1]
    conf = load_conf(str(root / "config" / "vqgraph_faithful" / "vqgraph_faithful_cora.yaml"))

    assert conf.model["teacher_architecture"] == "sage_gcn_aggregate_fullbatch_port"
    assert conf.model["teacher_hidden"] == 128
    assert conf.model["teacher_dropout"] == 0.0
    assert conf.model["teacher_lr"] == 0.01
    assert conf.model["teacher_weight_decay"] == 0.0005
    assert conf.model["codebook_size"] == 2048
    assert conf.model["n_hidden"] == 128
    assert conf.model["dropout"] == 0.4
    assert conf.training["lr"] == 0.005
    assert conf.training["weight_decay"] == 0.001


def test_mlp61_five_dataset_configs_have_preregistered_parameter_sources():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "krd_faithful": {
            "cora": ("KRD_official_best_parameters_tran_GCN", None),
            "citeseer": ("KRD_official_best_parameters_tran_GCN", None),
            "corafull": (
                "preregistered_nearest_node_count_mapping",
                "coauthor-cs",
            ),
            "coauthorcs": ("KRD_official_best_parameters_tran_GCN", None),
            "amazonpho": ("KRD_official_best_parameters_tran_GCN", None),
        },
        "vqgraph_faithful": {
            "cora": ("VQGraph_ICLR2024_Tables10_12", None),
            "citeseer": ("VQGraph_ICLR2024_Tables10_12", None),
            "corafull": (
                "preregistered_nearest_node_count_mapping",
                "pubmed",
            ),
            "coauthorcs": (
                "preregistered_nearest_node_count_mapping",
                "pubmed",
            ),
            "amazonpho": ("VQGraph_ICLR2024_Tables10_12", None),
        },
    }

    for method, datasets in expected.items():
        for dataset, (source, mapped_from) in datasets.items():
            conf = load_conf(
                str(root / "config" / method / f"{method}_{dataset}.yaml")
            )
            assert conf.model["parameter_source"] == source
            assert conf.model.get("parameter_mapping_dataset") == mapped_from


def test_mlp61_krd_official_dataset_parameters_match_source_json():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "cora": (3, 1024, 0.5, 0.3, 0.5, 0.8, 0.0005),
        "citeseer": (2, 256, 0.4, 0.6, 0.0, 0.9, 0.0005),
        "coauthorcs": (2, 256, 0.5, 0.6, 0.3, 1.1, 0.001),
        "amazonpho": (2, 1024, 0.4, 0.4, 0.4, 0.8, 0.0),
    }

    for dataset, values in expected.items():
        conf = load_conf(
            str(
                root
                / "config"
                / "krd_faithful"
                / f"krd_faithful_{dataset}.yaml"
            )
        )
        actual = (
            conf.model["n_layer"],
            conf.model["n_hidden"],
            conf.model["teacher_dropout"],
            conf.model["dropout"],
            conf.model["distill_lambda"],
            conf.model["distill_temperature"],
            conf.training["weight_decay"],
        )
        assert actual == values


def test_mlp61_vqgraph_paper_dataset_parameters_match_table12():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "cora": (2048, 0.005, 0.001, 0.4),
        "citeseer": (4096, 0.01, 0.005, 0.6),
        "amazonpho": (8192, 0.001, 0.001, 0.1),
    }

    for dataset, values in expected.items():
        conf = load_conf(
            str(
                root
                / "config"
                / "vqgraph_faithful"
                / f"vqgraph_faithful_{dataset}.yaml"
            )
        )
        actual = (
            conf.model["codebook_size"],
            conf.training["lr"],
            conf.training["weight_decay"],
            conf.model["dropout"],
        )
        assert actual == values


def test_krd_edge_loss_matches_official_equation():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    student = torch.tensor([[2.0, 0.1], [0.2, 1.5], [1.0, 0.8]])
    teacher = torch.tensor([[1.7, 0.3], [0.1, 1.8], [0.9, 0.7]])
    pairs = torch.tensor([[0, 1, 2], [1, 2, 2]])
    tau = 0.8

    actual = module.krd_edge_kl(student, teacher, pairs, tau)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student[pairs[0]] / tau, dim=1),
        torch.log_softmax(teacher[pairs[1]] / tau, dim=1),
        reduction="batchmean",
        log_target=True,
    )
    assert torch.allclose(actual, expected)


def test_krd_sampling_uses_neighbor_reliability_and_restores_self_pairs():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    # Pairs follow the official convention: row 0 is the student target and
    # row 1 is the teacher knowledge point sampled from its neighborhood.
    nonself = torch.tensor([[0, 1, 2], [1, 2, 0]])
    reliability = torch.tensor([1.0, 0.0, 1.0])
    pairs = module.sample_krd_pairs(nonself, reliability, num_nodes=3)

    assert torch.equal(pairs[:, -3:], torch.arange(3).repeat(2, 1))
    assert torch.equal(pairs[:, :-3], torch.tensor([[1, 2], [2, 0]]))


def test_vqgraph_teacher_uses_the_vendored_official_quantizer():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    teacher = module._VQGraphTeacher(
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        dropout=0.0,
        codebook_size=4,
        vq_decay=0.8,
        commitment_weight=0.25,
        lambda_node=0.001,
        lambda_edge=0.03,
    )

    assert teacher.quantizer.__class__.__name__ == "VectorQuantize"
    assert teacher.quantizer.__class__.__module__ == "vqgraph_official_vq"


def test_vqgraph_graph_inputs_are_bidirected_and_have_self_loops():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    predictor = module.vqgraph_faithful_Predictor(
        tiny_conf("vqgraph_faithful"), TinyData(), device="cpu"
    )

    edges, dense = predictor._graph_inputs()
    edge_set = set(map(tuple, edges.t().tolist()))
    assert all((node, node) in edge_set for node in range(6))
    assert all((destination, source) in edge_set for source, destination in edge_set)
    assert torch.equal(dense, dense.t())
    assert torch.equal(torch.diag(dense), torch.ones(6))


def test_vqgraph_soft_token_loss_matches_official_temperature_scaling():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    projected_student = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher_similarities = torch.tensor([[2.0, 0.5], [0.2, 1.7]])
    temperature = 4.0

    actual = module.vqgraph_token_kl(
        projected_student,
        codebook,
        teacher_similarities,
        temperature,
    )
    student_scores = -torch.cdist(projected_student, codebook, p=2) / temperature
    target = torch.softmax(teacher_similarities / temperature, dim=-1)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student_scores, dim=-1),
        target,
        reduction="batchmean",
    ) * temperature**2
    assert torch.allclose(actual, expected)


def test_faithful_students_ignore_adjacency_at_inference():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    data = TinyData()
    for cls in [module.krd_faithful_Predictor, module.vqgraph_faithful_Predictor]:
        predictor = cls(tiny_conf(cls.__name__.replace("_Predictor", "")), data, device="cpu")
        with_adj, _, _ = predictor.get_prediction(data.feats, data.adj)
        without_adj, _, _ = predictor.get_prediction(data.feats, None)
        assert torch.allclose(with_adj, without_adj)
