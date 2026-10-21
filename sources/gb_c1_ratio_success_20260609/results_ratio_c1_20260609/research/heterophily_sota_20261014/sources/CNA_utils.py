import json
import os
import random
from typing import List, Union

import matplotlib.pyplot as plt
import networkx
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from numpy import genfromtxt
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import utils
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import (
    Amazon,
    CitationFull,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.datasets.graph_generator import GridGraph
from torch_geometric.transforms import BaseTransform


def set_manual_seed(seed=1):
    """
    The set_manual_seed function sets the seed for random number generation in Python,
    NumPy and PyTorch.

    :param seed: Set the seed for all of the random number generators used in pytorch
    :return: Nothing
    """
    assert type(isinstance(seed, int))
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.benchmark = True


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for various libraries to ensure reproducibility.

    This function sets the random seed for the NumPy, Python's built-in random module,
    PyTorch CPU and GPU, and other related configurations to ensure that random
    operations produce consistent results across different runs.

    Parameters:
        seed: The seed value to set for random number generation. Default is 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


mode = True


def set_mode(mode_value):
    global mode
    mode = mode_value


def load_dataset(
    dataset_name, dataset_type="Planetoid"
):
    """
    The load_dataset function is used to load different datasets and split them into training, test, and validation sets.
    :param dataset_type: to set teh dataset type
    :param dataset_name: Name of the dataset. Possible values: "Cora", "CiteSeer", "PubMed".
    :return: A dataset object
    """
    dataset_dir = os.path.join(f"../datasets/{dataset_type}", dataset_name)
    dataset_path = os.path.join(os.getcwd(), dataset_dir)
    total_nodes = 0
    if not os.path.exists(os.path.join(os.getcwd(), f"../datasets/{dataset_type}")):
        os.makedirs(os.path.join(os.getcwd(), f"../datasets/{dataset_type}"))
    if dataset_type == "CitationFull":
        assert dataset_name in ["Cora", "CiteSeer", "PubMed", "Cora_ML", "DBLP"]
        data, num_classes = load_full_dataset(name=dataset_name, root=dataset_path)
        dataset = [data, num_classes]
    elif dataset_type == "Amazon":
        assert dataset_name in ["Computers", "Photo"]
        data, num_classes = load_amazon_dataset(
            name=dataset_name,
            root=dataset_path
        )
        dataset = [data, num_classes]
    elif dataset_type == "WikipediaNetwork":
        assert dataset_name in ["Chameleon", "Squirrel"]
        data, num_classes = load_wikipedianetwork_dataset(
            name=dataset_name, root=dataset_path
        )
        dataset = [data, num_classes]
    elif dataset_type == "WebKB":
        assert dataset_name in ["Cornell", "Texas", "Wisconsin"]
        data, num_classes = load_webkb_dataset(
            name=dataset_name, root=dataset_path
        )
        dataset = [data, num_classes]
    elif dataset_type == "PygNodePropPredDataset":
        assert dataset_name in ["ogbn-arxiv", "ogbn-proteins"]
        data, num_classes = load_pyg_node_prop_pred_dataset(
            name=dataset_name, root=dataset_path
        )
        dataset = [data, num_classes]

    else:
        assert dataset_type == "Planetoid" and dataset_name in [
            "Cora",
            "CiteSeer",
            "PubMed",
        ]
        if not os.path.exists(dataset_path):
            dataset = Planetoid(root=dataset_path, name=dataset_name.lower())
            save_path = os.path.join(dataset_path, f"{dataset_name.lower()}.pt")
            torch.save(dataset, save_path)
        else:
            dataset = torch.load(
                os.path.join(dataset_path, f"{dataset_name.lower()}.pt")
            )

        total_nodes = dataset[0].num_nodes

        print("============================================")
        print("Node number (totally): ", total_nodes)
        print("============================================")

        if mode:
            # Calculate the number of nodes for each split based on the given percentages
            train_percentage, test_percentage, valid_percentage = (80, 10, 10)
            num_train_nodes = int(total_nodes * (train_percentage / 100))
            num_test_nodes = int(total_nodes * (test_percentage / 100))
            num_valid_nodes = int(total_nodes * (valid_percentage / 100))

            # Update masks accordingly
            dataset[0].train_mask.fill_(False)
            dataset[0].train_mask[:num_train_nodes] = 1
            dataset[0].val_mask.fill_(False)
            dataset[0].val_mask[num_train_nodes : num_train_nodes + num_valid_nodes] = 1
            dataset[0].test_mask.fill_(False)
            dataset[0].test_mask[
                num_train_nodes
                + num_valid_nodes : num_train_nodes
                + num_valid_nodes
                + num_test_nodes
            ] = 1

            dataset[0].transform = T.NormalizeFeatures()

        train_mask = dataset[0].train_mask
        num_train_nodes = torch.sum(train_mask).item()
        test_mask = dataset[0].test_mask
        num_test_nodes = torch.sum(test_mask).item()
        valid_mask = dataset[0].val_mask
        num_valid_nodes = torch.sum(valid_mask).item()

        print("Number of train nodes after:", num_train_nodes)
        print("Number of test nodes after:", num_test_nodes)
        print("Number of validation nodes after:", num_valid_nodes)

        print(
            f"Portions: train set = {(num_train_nodes / total_nodes) * 100 :.2f}%, "
            f"test set = {(num_test_nodes / total_nodes) * 100 :.2f}%, "
            f"validation set ={(num_valid_nodes / total_nodes) * 100 :.2f}%"
        )
        print("============================================")

    return dataset


def load_pyg_node_prop_pred_dataset(name="ogbn-arxiv", root="."):
    """
    Load the PygNodePropPredDataset dataset and generate train, validation, and test masks based on percentages.

    Args:
        name (str): Name of the dataset (default is "Chameleon").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = PygNodePropPredDataset(root=root, name=name, transform=T.TargetIndegree())
    data = dataset[0]
    data.y = data.y.squeeze(dim=1)
    if mode:
        train_percent, val_percent, test_percent = (80, 10, 10)
    else:
        train_percent, val_percent, test_percent = (60, 20, 20)
    assert train_percent + val_percent + test_percent == 100

    train_mask, val_mask, test_mask = generate_percent_split(
        data, dataset.num_classes, train_percent, val_percent
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    total_nodes = data.num_nodes
    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    return data, dataset.num_classes


def load_wikipedianetwork_dataset(name="Chameleon", root="."):
    """
    Load the WikipediaNetwork dataset and generate train, validation, and test masks based on percentages.

    Args:
        name (str): Name of the dataset (default is "Chameleon").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = WikipediaNetwork(root=root, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    if mode:
        train_percent, val_percent, test_percent = (80, 10, 10)
    else:
        train_percent, val_percent, test_percent = (40, 30, 30)
    assert train_percent + val_percent + test_percent == 100

    train_mask, val_mask, test_mask = generate_percent_split(
        data, dataset.num_classes, train_percent, val_percent
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    total_nodes = data.num_nodes
    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    return data, dataset.num_classes


def load_webkb_dataset(name="Texas", root="."):
    """
    Load the WebKB dataset and generate train, validation, and test masks
    based on percentages.

    Args:
        name (str): Name of the dataset (default is "Cornell").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = WebKB(root=root, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    print(data)
    if mode:
        if name == "Texas":
            train_percent, val_percent, test_percent = (70, 15, 15)
        else:
            train_percent, val_percent, test_percent = (80, 10, 10)
    else:
        train_percent, val_percent, test_percent = (40, 30, 30)
    assert train_percent + val_percent + test_percent == 100

    train_mask, val_mask, test_mask = generate_percent_split(
        data, dataset.num_classes, train_percent, val_percent
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    total_nodes = data.num_nodes
    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    return data, dataset.num_classes


def load_amazon_dataset(name="Computers", root="."):
    """
    Load the Amazon dataset and generate train, validation, and test masks
    based on percentages.

    Args:
        name (str): Name of the dataset (default is "Computers").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = Amazon(root, name, transform=T.TargetIndegree())
    data = dataset[0]
    num_classes = dataset.num_classes
    if mode:
        train_percent, val_percent, test_percent = (80, 10, 10)
    else:
        train_percent, val_percent, test_percent = (40, 30, 30)
    assert train_percent + val_percent + test_percent == 100

    train_mask, val_mask, test_mask = generate_percent_split(
        data, num_classes, train_percent, val_percent
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    total_nodes = data.num_nodes

    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    return data, num_classes


def load_full_dataset(name="Cora", root="."):
    """
    Load the CitationFull dataset and generate train, validation, and test
    masks based on percentages.

    Args:
        name (str): Name of the dataset (default is "Cora").
        root (str): Root directory for dataset storage.
    Returns:
        tuple: A tuple containing the Data object and the number of classes.
    """
    dataset = CitationFull(root, name, transform=T.NormalizeFeatures())
    data = dataset[0]

    num_classes = dataset.num_classes
    if mode:
        train_percent, val_percent, test_percent = (80, 10, 10)
    else:
        train_percent, val_percent, test_percent = (40, 30, 30)
    assert train_percent + val_percent + test_percent == 100

    train_mask, val_mask, test_mask = generate_percent_split(
        data, num_classes, train_percent, val_percent
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    total_nodes = data.num_nodes

    print("============================================")
    print("Node number (totally): ", total_nodes)
    print("============================================")

    return data, num_classes


def generate_percent_split(data, num_classes, train_percent=10, val_percent=10):
    """
    Generate train, validation, and test masks based on class-wise percentage split.

    Args:
        data (torch_geometric.data.Data): Input data object.
        num_classes (int): Number of classes in the dataset.
        train_percent (int): Percentage of data for training (default is 10).
        val_percent (int): Percentage of data for validation (default is 10).

    Returns:
        tuple: Tuple containing train, validation, and test masks.
    """
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)

    for c in range(num_classes):
        all_c_idx = torch.nonzero(data.y == c, as_tuple=True)[0].flatten()
        num_c = all_c_idx.size(0)

        train_num_per_c = num_c * train_percent // 100
        val_num_per_c = num_c * val_percent // 100

        perm = torch.randperm(all_c_idx.size(0))

        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True

        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True

    test_mask = ~test_mask
    return train_mask, val_mask, test_mask


def visualize_data(data, name, colors=None):
    """
    The visualize_data function takes in a dataset and the name of the dataset.
    It then creates a graph from that data, using NetworkX. It then uses
    spring_layout to create an xyz position for each node.
    The function plots these nodes as points on a 3D plot, with edges
    connecting them.

    :param data: Pass the data object to the function
    :param name: Determine the number of nodes in the graph
    :return: A 3d plot of the graph
    """
    plt.style.use("dark_background")
    edge_index = data.edge_index.numpy()
    num_nodes = 2708 if name.lower() == "cora" else data.num_nodes
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.transpose())
    pos = nx.spring_layout(G, dim=3, seed=779)
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        colors = np.array([np.linalg.norm(pos[v]) for v in sorted(G)])
        ax.scatter(*node_xyz.T, s=25, ec="w", c=colors, cmap="rainbow")
    else:
        ax.scatter(*node_xyz.T, s=25, ec="w", c=colors)

    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        ax.grid(False)
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    ax.set_axis_off()
    plt.title(name)
    _format_axes(ax)
    fig.tight_layout()

    plt.show()


def get_data(name="chameleon", split=0):
    """
    Retrieves and preprocesses data for the task Node Regression.

    :param name (str): The name of the graph dataset. Default is "chameleon".
    :param split (int): The split index for train-validation-test sets. Default is 0.
    :return data (torch_geometric.data.Data): Processed graph data with features, labels, and masks.
    """
    root = f"../datasets/Regression_datasets/data_npz/{name}/"
    if not os.path.exists(root):
        os.makedirs(root)
    dataset = WikipediaNetwork(root=root, name=name, transform=T.NormalizeFeatures())
    edges = genfromtxt(
        "../datasets/Regression_datasets/data/" + name + "_edges.csv", delimiter=","
    )[1:].astype(int)
    G = networkx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    data = utils.from_networkx(G)

    y = genfromtxt(
        "../datasets/Regression_datasets/data/" + name + "_target.csv", delimiter=","
    )[1:, -1].astype(int)
    y = y / np.max(y)
    data.y = torch.tensor(y).float()

    with open(
        "../datasets/Regression_datasets/data/" + name + "_features.json", "r"
    ) as myfile:
        file = myfile.read()
    obj = json.loads(file)

    if name == "chameleon":
        x = np.zeros((2277, 3132))
        for i in range(2277):
            feats = np.array(obj[str(i)])
            x[i, feats] = 1

    elif name == "squirrel":
        x = np.zeros((5201, 3148))
        for i in range(5201):
            feats = np.array(obj[str(i)])
            x[i, feats] = 1

    data.x = torch.tensor(x).float()

    path = "../datasets/Regression_datasets/data_npz/" + name
    splits_file = np.load(
        f"{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz"
    )

    train_mask = splits_file["train_mask"]
    val_mask = splits_file["val_mask"]
    test_mask = splits_file["test_mask"]

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data


class NormalizeFeaturesColumnwise(BaseTransform):
    """
    Normalizes the specified attributes column-wise in the given Data or
    HeteroData object.
    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        """
        Performs column-wise normalization on the specified attributes in
        the given Data or HeteroData.

        Args:
            data (Union[Data, HeteroData]): Input Data or HeteroData object
            to be normalized.

        Returns:
            Union[Data, HeteroData]: Normalized Data or HeteroData object.
        """
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    mean = value.mean(dim=0, keepdim=True)
                    std = value.std(dim=0, unbiased=False, keepdim=True).clamp_(
                        min=1e-5
                    )
                    value = (value - mean) / std
                    store[key] = value
        return data
