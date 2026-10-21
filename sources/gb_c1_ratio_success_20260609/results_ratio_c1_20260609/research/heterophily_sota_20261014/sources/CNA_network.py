import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch_sparse as ts
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import (
    ARMAConv,
    ChebConv,
    CuGraphSAGEConv,
    GATConv,
    GCNConv,
    GraphConv,
    LEConv,
    Linear,
    MFConv,
    SAGEConv,
    Sequential,
    SGConv,
    SSGConv,
    TransformerConv,
)

from utils.model_params import LayerType
from utils.model_params import ModelParams as mp


class Net(torch.nn.Module):
    """
    A neural network model for graph data with customizable layers and activations.

    Parameters:
        activation (callable): The activation function to use in intermediate layers.
        input_features (int): Number of input features (node features) in the graph.
        output_features (int): Number of output features (node classes) in the graph.
        hidden_features (int): Number of hidden features in the intermediate layers.
        num_layer (int): Number of layers in the model.
        layer_type (str, optional): The type of graph convolutional layer to use.
            Defaults to "GCNConv".

    Attributes:
        layer_type (str): The type of graph convolutional layer used in the model.
        activation (callable): The activation function used in intermediate layers.
        num_layer (int): Number of layers in the model.
        model (torch_geometric.nn.Sequential): The PyTorch Geometric Sequential model
            containing the layers of the network.

    """

    def __init__(
        self,
        activation,
        input_features,
        output_features,
        hidden_features,
        num_layer,
        layer_type=mp.model_type,
    ):
        super(Net, self).__init__()

        self.layer_type = layer_type
        self.activation = activation
        self.num_layer = num_layer
        self.model_ = self._build_sequential_container(input_features, hidden_features)
        self.out_layer = self._get_conv_layer(hidden_features, output_features)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        self.model_.reset_parameters()

    def forward(self, x, edge_index=None):
        """
        Perform a forward pass through the model.

        Parameters:
            x (torch.Tensor): Input node features in the graph.
            edge_index (torch.Tensor): Graph edge indices (COO format).

        Returns:
            torch.Tensor: Log softmax output of the model.
        """
        if self.layer_type == LayerType.LINEAR:
            h = self.model_(x)
            out = self.activation(h)
            out = self.out_layer(out)
        else:
            h = self.model_(x, edge_index)
            out = self.activation(h)
            out = self.out_layer(out, edge_index)

        return h, self.log_softmax(out)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(self.device)
                x = conv(x, batch.edge_index.to(self.device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[: batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0)
        return x_all

    def _build_sequential_container(self, input_features, hidden_features):
        """
        Build the PyTorch Geometric Sequential container with the specified layers.

        Parameters:
            input_features (int): Number of input features (node features) in the graph.
            hidden_features (int): Number of hidden features in the intermediate layers.

        Returns:
            torch_geometric.nn.Sequential: The PyTorch Geometric Sequential container
            containing the specified layers.
        """
        if self.layer_type == LayerType.LINEAR:
            if self.num_layer == 2:
                conv_list = [
                    (
                        self._get_conv_layer(input_features, hidden_features),
                        "x -> x0",
                    )
                ]
                return Sequential("x", conv_list)

            conv_list = [
                (
                    self._get_conv_layer(input_features, hidden_features),
                    "x -> x0",
                ),
            ]
            [
                conv_list.extend(
                    [
                        (self.activation, f"x{i - 1} -> x{i - 1}a"),
                        (
                            self._get_conv_layer(hidden_features, hidden_features),
                            f"x{i - 1}a -> x{i}",
                        ),
                    ]
                )
                for i in range(1, self.num_layer - 1)
            ]

            return Sequential("x", conv_list)

        else:
            if self.num_layer == 2:
                conv_list = [
                    (
                        self._get_conv_layer(input_features, hidden_features),
                        "x, edge_index -> x0",
                    )
                ]
                return Sequential("x, edge_index", conv_list)

            conv_list = [
                (
                    self._get_conv_layer(input_features, hidden_features),
                    "x, edge_index -> x0",
                ),
            ]
            [
                conv_list.extend(
                    [
                        (self.activation, f"x{i-1} -> x{i-1}a"),
                        (
                            self._get_conv_layer(hidden_features, hidden_features),
                            f"x{i - 1}a, edge_index -> x{i}",
                        ),
                    ]
                )
                for i in range(1, self.num_layer - 1)
            ]

            return Sequential("x, edge_index", conv_list)

    def _get_conv_layer(self, input_features, output_features):
        """
        Get the specified graph convolutional layer based on the layer_type.

        Parameters:
            input_features (int): Number of input features (node features) in the graph.
            output_features (int): Number of output features (node classes) in the graph.

        Returns:
            torch_geometric.nn.MessagePassing: The graph convolutional layer based on
            the layer_type.

        Raises:
            ValueError: If the specified layer_type is not supported.
        """
        if self.layer_type == LayerType.LINEAR:
            return Linear(input_features, output_features)
        elif self.layer_type == LayerType.GCNCONV:
            return GCNConv(input_features, output_features)
        elif self.layer_type == LayerType.SAGECONV:
            return SAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GATCONV:
            return GATConv(input_features, output_features)
        elif self.layer_type == LayerType.TRANSFORMERCONV:
            return TransformerConv(input_features, output_features)
        elif self.layer_type == LayerType.CHEBCONV:
            return ChebConv(input_features, output_features, K=3)
        elif self.layer_type == LayerType.CUGRAPHSAGECONV:
            return CuGraphSAGEConv(input_features, output_features)
        elif self.layer_type == LayerType.GRAPHCONV:
            return GraphConv(input_features, output_features)
        elif self.layer_type == LayerType.ARMACONV:
            return ARMAConv(input_features, output_features)
        elif self.layer_type == LayerType.SGCONV:
            return SGConv(input_features, output_features)
        elif self.layer_type == LayerType.MFCONV:
            return MFConv(input_features, output_features)
        elif self.layer_type == LayerType.SSGCONV:
            return SSGConv(input_features, output_features, alpha=0.5)
        elif self.layer_type == LayerType.LECONV:
            return LEConv(input_features, output_features)
        elif self.layer_type == LayerType.DIRSAGECONV:
            return DirSageConv(input_features, output_features)
        elif self.layer_type == LayerType.DIRGCNCONV:
            return DirGCNConv(input_features, output_features)
        else:
            raise ValueError(f"Unsupported layer_type: {self.layer_type}")


class DirSageConv(torch.nn.Module):
    """
    Implementation of a directional GraphSAGE convolutional layer.

    Args:
        input_features (int): The dimensionality of the input features.
        output_features (int): The dimensionality of the output features.
        alpha (float, optional): The balance parameter for combining
        source-to-target and target-to-source information.
            Default is 1.


    Methods:
        forward(x, edge_index): Forward pass of the directional GraphSAGE convolutional layer.
    """

    def __init__(self, input_features, output_features):
        super(DirSageConv, self).__init__()

        self.source_to_target = SAGEConv(
            input_features, output_features, flow="source_to_target", root_weight=False
        )
        self.target_to_source = SAGEConv(
            input_features, output_features, flow="target_to_source", root_weight=False
        )
        self.linear = Linear(input_features, output_features)
        self.alpha = 0.5

    def forward(self, x, edge_index):
        out = (
            self.linear(x)
            + (1 - self.alpha) * self.source_to_target(x, edge_index)
            + self.alpha * self.target_to_source(x, edge_index)
        )

        return out


class DirGCNConv(torch.nn.Module):
    """
    Implementation of a directed graph convolution layer.

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output features.
    """

    def __init__(self, input_dim, output_dim):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear transformations for source-to-destination and destination-to-source edges
        self.linear_src_to_dst = Linear(input_dim, output_dim)
        self.linear_dst_to_src = Linear(input_dim, output_dim)

        # Hyperparameter for combining source-to-destination and destination-to-source information
        self.alpha = 1.

        # Normalized adjacency matrices for source-to-destination and destination-to-source edges
        self.adjacency_norm, self.adjacency_transposed_norm = None, None

    def directed_norm(self, adjacency_matrix):
        """
        Normalize the adjacency matrix for directed edges.

        Args:
            adjacency_matrix (torch.sparse.Tensor): Sparse adjacency matrix.

        Returns:
            torch.sparse.Tensor: Normalized adjacency matrix.
        """
        in_deg = ts.sum(adjacency_matrix, dim=0)
        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        out_deg = ts.sum(adjacency_matrix, dim=1)
        out_deg_inv_sqrt = out_deg.pow_(-0.5)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        adjacency_matrix = ts.mul(adjacency_matrix, out_deg_inv_sqrt.view(-1, 1))
        adjacency_matrix = ts.mul(adjacency_matrix, in_deg_inv_sqrt.view(1, -1))

        return adjacency_matrix

    def forward(self, x, edge_index):
        """
        Forward pass of the directed graph convolution layer.

        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge index tensor.

        Returns:
            torch.Tensor: Output feature matrix.
        """
        if self.adjacency_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            # Create sparse adjacency matrices for source-to-destination and destination-to-source edges
            adjacency_matrix = ts.SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adjacency_norm = self.directed_norm(adjacency_matrix)

            adjacency_matrix_transposed = ts.SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adjacency_transposed_norm = self.directed_norm(adjacency_matrix_transposed)

        # Apply directed graph convolution
        src_to_dst_term = self.linear_src_to_dst(self.adjacency_norm @ x)
        dst_to_src_term = self.linear_dst_to_src(self.adjacency_transposed_norm @ x)
        return self.alpha * src_to_dst_term + (1 - self.alpha) * dst_to_src_term

