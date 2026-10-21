from __future__ import annotations

import importlib
from pathlib import Path

import torch

from utils.tools import load_conf


def _chain_edges() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )


def test_sampler_is_deterministic_and_keeps_outputs_first():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edge_index = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    sampler = module.CpuNeighborSampler(edge_index, num_nodes=4, fanouts=(1, 2), seed=17)

    first = sampler.sample(torch.tensor([2, 0]), epoch=3, batch_id=4)
    second = sampler.sample(torch.tensor([2, 0]), epoch=3, batch_id=4)

    assert first.output_count == 2
    assert first.global_nodes[:2].tolist() == [2, 0]
    assert torch.equal(first.global_nodes, second.global_nodes)
    assert torch.equal(first.edge_index, second.edge_index)
    assert int(first.edge_index.max()) < first.num_nodes


def test_sampled_graph_is_bidirected_and_every_local_node_has_positive_degree():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edge_index = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    sampler = module.CpuNeighborSampler(edge_index, num_nodes=4, fanouts=(2, 2), seed=9)
    sample = sampler.sample(torch.tensor([1, 2]), epoch=0, batch_id=0)
    edges = set(map(tuple, sample.edge_index.t().tolist()))

    assert all((destination, source) in edges for source, destination in edges)
    degree = torch.bincount(sample.edge_index.flatten(), minlength=sample.num_nodes)
    assert bool((degree > 0).all())


def test_local_union_preserves_official_multiedges_and_dense_multiplicity():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edge_index = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    sampler = module.CpuNeighborSampler(edge_index, num_nodes=4, fanouts=(-1,), seed=2)
    sample = sampler.sample(torch.tensor([0]), epoch=0, batch_id=0)
    dense = module.local_dense_adjacency(sample.edge_index, sample.num_nodes, device="cpu")

    assert dense[0, 0].item() == 2.0
    assert torch.equal(dense, dense.t())


def test_dense_reconstruction_is_local_not_full_graph_sized():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edge_index = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    sampler = module.CpuNeighborSampler(edge_index, num_nodes=4, fanouts=(1,), seed=2)
    sample = sampler.sample(torch.tensor([0]), epoch=0, batch_id=0)
    dense = module.local_dense_adjacency(sample.edge_index, sample.num_nodes, device="cpu")

    assert dense.shape == (sample.num_nodes, sample.num_nodes)
    assert sample.num_nodes < 4
    assert torch.equal(dense, dense.t())


def test_graph_conv_matches_dgl_both_degree_normalization():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edges = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
    )
    conv = module.DGLBothNormGraphConv(2, 2, activation=None)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[1.0, 0.5], [0.25, 1.0]]))
        conv.bias.copy_(torch.tensor([0.1, -0.2]))

    src, dst = edges
    out_degree = torch.bincount(src, minlength=4).float().clamp_min(1)
    in_degree = torch.bincount(dst, minlength=4).float().clamp_min(1)
    messages = (features * out_degree.rsqrt().unsqueeze(1)) @ conv.weight
    expected = torch.zeros(4, 2)
    expected.index_add_(0, dst, messages[src])
    expected = expected * in_degree.rsqrt().unsqueeze(1) + conv.bias

    assert torch.allclose(conv(features, edges), expected, atol=1e-6)


def test_full_neighbor_mode_collects_every_one_hop_neighbor():
    module = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    edge_index = module.bidirected_with_self_loops(_chain_edges(), num_nodes=4)
    sampler = module.CpuNeighborSampler(edge_index, num_nodes=4, fanouts=(-1,), seed=3)
    sample = sampler.sample(torch.tensor([1]), epoch=0, batch_id=0)

    assert sample.global_nodes.tolist() == [1, 0, 2]


def test_sampled_teacher_outputs_only_batch_targets_and_quantizes_local_graph():
    module = importlib.import_module("predictor.G2MFaithful_Predictor")
    sampling = importlib.import_module("predictor.module.vqgraph_neighbor_sampling")
    teacher = module._VQGraphSampledTeacher(
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
    features = torch.randn(3, 4)
    edges = sampling.bidirected_with_self_loops(
        torch.tensor([[0, 1], [1, 2]]),
        num_nodes=3,
    )
    dense = sampling.local_dense_adjacency(edges, 3, device="cpu")

    outputs = teacher(features, edges, dense, output_count=2)
    logits, auxiliary, similarities, codebook, indices = outputs[:5]

    assert logits.shape == (2, 3)
    assert auxiliary.numel() == 1
    assert similarities.shape == (3, 4)
    assert codebook.shape == (4, 4)
    assert indices.numel() == 3
    assert teacher.quantizer.__class__.__module__ == "vqgraph_official_vq"


def test_sampled_method_is_separately_registered_and_x_only_at_inference():
    standard = importlib.import_module("scripts.run_standard_downstream")
    method = "vqgraph_sampled_faithful"
    assert standard.PREDICTOR_MODULES[method] == "predictor.G2MFaithful_Predictor"
    capability = standard.METHOD_CAPABILITIES[method]
    assert capability["implementation_status"] == "faithful protocol reimplementation"
    assert capability["training_uses_graph"] is True
    assert capability["inference_uses_graph"] is False
    assert capability["uses_unlabeled_nodes"] is True


def test_sampled_configs_preserve_mlp61_parameters_and_add_sampling_contract():
    root = Path(__file__).resolve().parents[1]
    for dataset in ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho"):
        old = load_conf(
            str(root / "config" / "vqgraph_faithful" / f"vqgraph_faithful_{dataset}.yaml")
        )
        new = load_conf(
            str(
                root
                / "config"
                / "vqgraph_sampled_faithful"
                / f"vqgraph_sampled_faithful_{dataset}.yaml"
            )
        )
        for key in (
            "parameter_source",
            "parameter_mapping_dataset",
            "teacher_hidden",
            "teacher_dropout",
            "teacher_epochs",
            "teacher_patience",
            "teacher_lr",
            "teacher_weight_decay",
            "codebook_size",
            "vq_decay",
            "vq_commitment_weight",
            "vq_lambda_node",
            "vq_lambda_edge",
            "batch_size",
            "lambda_soft_labels",
            "lambda_soft_tokens",
            "token_temperature",
        ):
            assert new.model.get(key) == old.model.get(key)
        assert new.model["teacher_architecture"] == "sampled_local_graphconv_pytorch_rocm"
        assert new.model["teacher_fanouts"] == [5, 5]
        assert new.model["teacher_batch_size"] == 512
        assert new.model["sampling_device"] == "cpu"
        assert new.model["training_device"] == "cuda:0"
