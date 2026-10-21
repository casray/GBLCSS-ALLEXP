"""Graph-MLP mechanism adaptation used in the matched-backbone experiments.

CE plus sampled edge-based contrastive loss, with an attribute-only MLP.
Source: NoisyGL/predictor/GraphFreeFamily_Predictor.py::graphmlp_Predictor.
See THIRD_PARTY_NOTICES.md for the retained MIT attribution.
"""
from copy import deepcopy

import torch
import torch.nn.functional as F

from ..models import MLP
from ..training import TrainResult, accuracy
from .structure import _split_index


def _forward(model, features):
    hidden = features
    for index, linear in enumerate(model.layers[:-1]):
        hidden = F.relu(linear(hidden))
        hidden = model.batch_norms[index](hidden)
        hidden = F.dropout(hidden, p=model.dropout, training=model.training)
    return model.layers[-1](hidden), hidden


def contrastive_loss(hidden, edges, batch_size, temperature):
    """Sample positive edges and independent negative nodes with replacement."""
    if edges.shape[1] == 0:
        raise ValueError('Graph-MLP requires training-time edges')
    selected = torch.randint(edges.shape[1], (batch_size,), device=hidden.device)
    source, target = edges[:, selected]
    negative = torch.randint(hidden.shape[0], (batch_size,), device=hidden.device)
    normalized = F.normalize(hidden, p=2, dim=1)
    positive_score = (normalized[source] * normalized[target]).sum(dim=1) / temperature
    negative_score = (normalized[source] * normalized[negative]).sum(dim=1) / temperature
    scores = torch.stack([positive_score, negative_score], dim=1)
    return F.cross_entropy(scores, torch.zeros(batch_size, dtype=torch.long, device=hidden.device))


def train_graph_mlp(data, settings, *, hidden=64, layers=2, dropout=.5,
                    learning_rate=.01, weight_decay=.0005, epochs=200, device='cpu'):
    batch_size = int(settings.get('graph_batch_size', 512))
    temperature = float(settings.get('temperature', .5))
    weight = float(settings.get('contrastive_weight', .5))
    if batch_size < 1 or temperature <= 0 or weight < 0 or epochs < 1:
        raise ValueError('invalid Graph-MLP training settings')
    x, clean = data.x.to(device), data.y.to(device)
    labels = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train, val, test = [_split_index(data, split, device) for split in ('train', 'val', 'test')]
    edges = data.edge_index.to(device)
    edges = edges[:, edges[0] != edges[1]]
    model = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation, best_epoch, best_state = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embeddings = _forward(model, x)
        loss = F.cross_entropy(logits[train], labels[train])
        loss = loss + weight * contrastive_loss(embeddings, edges, batch_size, temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at epoch {epoch + 1}')
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = accuracy(_forward(model, x)[0][val], labels[val])
        if validation > best_validation:
            best_validation, best_epoch = validation, epoch + 1
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():
        logits = _forward(model, x)[0]
    return TrainResult(accuracy(logits[train], labels[train]), best_validation,
                       accuracy(logits[test], clean[test]), best_epoch, logits.cpu())
