"""Fixed attribute/residual spaces built from features and graph edges only."""

from numbers import Integral

import numpy as np
from scipy import sparse
import torch


def _cpu_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalized_rows(value):
    # Float64 accumulation also handles very large or very small float32 rows.
    work = np.asarray(value, dtype=np.float64)
    norms = np.sqrt(np.einsum('ij,ij->i', work, work))
    np.divide(work, norms[:, None], out=work, where=norms[:, None] != 0)
    return work.astype(np.float32)


def structural_views(x, edge_index, *, shuffle_seed=20261015, block_rows=1024):
    """Return U, [U,V]/sqrt(2), and a fixed row-shuffled residual control.

    U=rowL2(x), V=rowL2(Px-x). P is the loop-free, deduplicated,
    undirected random-walk adjacency. Isolated nodes have P rows of zero.
    The residual uses the original feature magnitudes before normalization.
    Dense allocations scale with N*F; adjacency remains sparse throughout.
    """
    if isinstance(block_rows, bool) or not isinstance(block_rows, Integral):
        raise TypeError('block_rows must be a positive integer')
    if block_rows <= 0:
        raise ValueError('block_rows must be positive')
    if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, Integral):
        raise TypeError('shuffle_seed must be a nonnegative integer')
    if shuffle_seed < 0:
        raise ValueError('shuffle_seed must be nonnegative')

    features = _cpu_array(x)
    if features.ndim != 2 or min(features.shape) == 0:
        raise ValueError('x must have shape (positive N, positive F)')
    if features.dtype.kind not in 'fiu':
        raise TypeError('x must contain real numeric features')
    for start in range(0, features.shape[0], block_rows):
        if not np.isfinite(features[start:start + block_rows]).all():
            raise ValueError('x must contain only finite features')
    with np.errstate(over='ignore', invalid='ignore'):
        features = np.asarray(features, dtype=np.float32)
    for start in range(0, features.shape[0], block_rows):
        if not np.isfinite(features[start:start + block_rows]).all():
            raise ValueError('x must be representable as finite float32 features')

    edges = _cpu_array(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('edge_index must have shape (2, E)')
    if edges.dtype.kind not in 'iu':
        raise TypeError('edge_index must contain integer node indices')
    n, width = features.shape
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('edge_index endpoints must be in [0, N)')
    edges = np.asarray(edges, dtype=np.int64)
    source, target = edges[:, edges[0] != edges[1]]
    rows = np.concatenate((source, target))
    columns = np.concatenate((target, source))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(n, n), dtype=np.float64,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data.fill(1)
    degree = np.maximum(np.diff(adjacency.indptr), 1)

    original = np.empty((n, width), dtype=np.float32)
    residual = np.empty_like(original)
    # Sum before dividing, in float64: float32 reciprocal weights can create
    # rounding-only residuals for constant neighborhoods, then rowL2 amplifies
    # them to unit magnitude. One conversion avoids a full cast per CSR block.
    features64 = features.astype(np.float64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        original[start:stop] = _normalized_rows(features[start:stop])
        neighborhood = adjacency[start:stop] @ features64
        neighborhood /= degree[start:stop, None]
        difference = neighborhood - features64[start:stop]
        if not np.isfinite(difference).all():
            raise ValueError('neighbor averaging produced nonfinite residuals')
        residual[start:stop] = _normalized_rows(difference)
    del features64

    permutation = np.random.default_rng(int(shuffle_seed)).permutation(n).astype(np.int64)
    combined = np.empty((n, 2 * width), dtype=np.float32)
    shuffled = np.empty_like(combined)
    factor = np.float32(1 / np.sqrt(2))
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        combined[start:stop, :width] = original[start:stop] * factor
        combined[start:stop, width:] = residual[start:stop] * factor
        shuffled[start:stop, :width] = combined[start:stop, :width]
        shuffled[start:stop, width:] = residual[permutation[start:stop]] * factor
    return {'X': original, 'R': combined, 'SHUFFLE': shuffled,
            'residual': residual, 'permutation': permutation}
