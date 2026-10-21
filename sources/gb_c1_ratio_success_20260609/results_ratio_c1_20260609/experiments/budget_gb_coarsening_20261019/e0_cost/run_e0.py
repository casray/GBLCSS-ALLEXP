"""E0 smoke: compression headroom of label-free granular-ball coarsening.

Paired comparisons (see ../PLAN.md, E0):
  full GCN   vs coarse GCN (Pc = R P H)
  full SIGN  vs coarse SIGN (cached [X, PX, P^2X] aggregated by R)
Cut rule here is label-free (R0, D-only): plain best-first SSE bisection on
F = [U, PU, P^2U]. Labels and training positions are NOT used to build balls.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits
import heapq

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub/src"))
from gblcss.data import percent_split_indices, inject_train_label_noise, set_seed, GraphData  # noqa: E402

NOISYGL = PROJECT / "external/NoisyGL"
DATA_FILES = {
    "cora": NOISYGL / "data/Cora/processed/data.pt",
    "citeseer": NOISYGL / "data/CiteSeer/processed/data.pt",
    "pubmed": NOISYGL / "data/PubMed/processed/data.pt",
    "coauthorcs": NOISYGL / "data/CS/processed/data.pt",
    "physics": NOISYGL / "data/Physics/processed/data.pt",
    "amazonpho": NOISYGL / "data/Photo/processed/data.pt",
    "corafull": PROJECT / "gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt",
}
DEV = torch.device("cuda")


def sync():
    torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                sync()
                self.s = time.perf_counter()

            def __exit__(self, *a):
                sync()
                timer.t[name] = timer.t.get(name, 0.0) + time.perf_counter() - self.s
        return _C()


def load(dataset, noise, seed):
    from torch_geometric.data import Data
    stored = torch.load(DATA_FILES[dataset], map_location="cpu", weights_only=False)
    raw = Data.from_dict(stored[0]) if isinstance(stored[0], dict) else stored[0]
    n = raw.x.shape[0]
    idx = percent_split_indices(n, seed)
    masks = []
    for ind in idx:
        m = torch.zeros(n, dtype=torch.bool)
        m[ind] = True
        masks.append(m)
    data = GraphData(dataset, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                     train_index=torch.tensor(idx[0]), val_index=torch.tensor(idx[1]),
                     test_index=torch.tensor(idx[2]))
    kind, rate = noise
    inject_train_label_noise(data, kind, rate, seed)
    assert torch.equal(data.noisy_y[data.val_mask], data.y[data.val_mask])
    assert torch.equal(data.noisy_y[data.test_mask], data.y[data.test_mask])
    return data


def sym_norm_adj(edge_index, n):
    ei = edge_index.numpy()
    a = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float32)
    a.setdiag(0)
    a.eliminate_zeros()
    a = a + sp.eye(n, dtype=np.float32, format="csr")
    d = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr().astype(np.float32)


def to_torch_sparse(m):
    m = m.tocoo()
    i = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    v = torch.tensor(m.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, m.shape).coalesce().to(DEV)


def rownorm(z):
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(nrm, 1e-12)


def coords(x, p, dim):
    xs = sp.csr_matrix(rownorm(x.numpy()))
    k = min(dim, xs.shape[1] - 1)
    u = TruncatedSVD(n_components=k, random_state=0).fit_transform(xs).astype(np.float32)
    pu = p @ u
    ppu = p @ pu
    return np.hstack([rownorm(u), rownorm(pu), rownorm(ppu)]).astype(np.float32) / np.sqrt(3.0)


def bisect_tree(z, budget, min_leaf, seed=0):
    """Label-free best-first SSE bisection; returns assignment with <= budget balls."""
    n = z.shape[0]

    def sse(ind):
        sub = z[ind]
        return float(((sub - sub.mean(0)) ** 2).sum())

    leaves = {0: np.arange(n)}
    heap = [(-sse(leaves[0]), 0)]
    nxt = 1
    with threadpool_limits(limits=4):
        while len(leaves) < budget and heap:
            neg, lid = heapq.heappop(heap)
            ind = leaves[lid]
            if ind.size < 2 * min_leaf:
                continue
            lab = KMeans(n_clusters=2, n_init=3, max_iter=100, random_state=(seed + lid) % (2**31)).fit_predict(z[ind])
            left, right = ind[lab == 0], ind[lab == 1]
            if left.size < min_leaf or right.size < min_leaf:
                continue
            leaves.pop(lid)
            for child in (left, right):
                leaves[nxt] = child
                heapq.heappush(heap, (-sse(child), nxt))
                nxt += 1
    assign = np.empty(n, dtype=np.int64)
    for b, lid in enumerate(sorted(leaves)):
        assign[leaves[lid]] = b
    return assign


def coarse_ops(assign, p):
    n = assign.size
    m = int(assign.max()) + 1
    h = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), assign)), shape=(n, m))
    sizes = np.asarray(h.sum(0)).ravel()
    r = sp.diags(1.0 / sizes) @ h.T
    pc = (r @ p @ h).tocsr().astype(np.float32)
    return h.tocsr(), r.tocsr().astype(np.float32), pc


class GCN(nn.Module):
    def __init__(self, din, dh, dout, drop=0.5):
        super().__init__()
        self.l1, self.l2, self.drop = nn.Linear(din, dh), nn.Linear(dh, dout), drop

    def forward(self, x, p):
        h = F.relu(torch.sparse.mm(p, self.l1(x)))
        h = F.dropout(h, self.drop, self.training)
        return torch.sparse.mm(p, self.l2(h))


class MLP(nn.Module):
    def __init__(self, din, dh, dout, drop=0.5):
        super().__init__()
        self.l1, self.l2, self.drop = nn.Linear(din, dh), nn.Linear(dh, dout), drop

    def forward(self, x):
        h = F.dropout(F.relu(self.l1(x)), self.drop, self.training)
        return self.l2(h)


def train_loop(step_fn, eval_fn, epochs, patience):
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        step_fn()
        acc, state = eval_fn()
        if acc > best:
            best, best_state, bad = acc, state, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_state, ep + 1


def soft_targets(assign, m, y, train_idx, c):
    tgt = torch.zeros(m, c)
    a = torch.as_tensor(assign)[train_idx]
    tgt.index_add_(0, a, F.one_hot(y[train_idx], c).float())
    cnt = tgt.sum(1)
    lab = cnt > 0
    tgt[lab] = tgt[lab] / cnt[lab, None]
    return tgt, cnt, lab


def run_case(dataset, noise, seed, ratios, hidden, epochs, patience, dim, min_leaf, out):
    tm = Timer()
    data = load(dataset, noise, seed)
    n, c = data.num_nodes, data.num_classes
    y_noisy, y_clean = data.noisy_y, data.y
    tr, va, te = data.train_index, data.val_index, data.test_index
    with tm("pre_adj"):
        p = sym_norm_adj(data.edge_index, n)
        p_t = to_torch_sparse(p)
        x = data.x.to(DEV)
    with tm("pre_sign"):
        px = torch.sparse.mm(p_t, x)
        ppx = torch.sparse.mm(p_t, px)
        sign_feat = torch.cat([x, px, ppx], 1)
        del px, ppx
    rows = []
    yn, yc = y_noisy.to(DEV), y_clean.to(DEV)
    tr_d, va_d, te_d = tr.to(DEV), va.to(DEV), te.to(DEV)
    # warm-up so kernel compilation is not charged to the first timed arm
    warm = GCN(x.shape[1], hidden, c).to(DEV)
    for _ in range(3):
        F.cross_entropy(warm(x, p_t)[tr_d], yn[tr_d]).backward()
        F.cross_entropy(MLP(sign_feat.shape[1], hidden, c).to(DEV)(sign_feat[tr_d]), yn[tr_d]).backward()
    del warm
    sync()

    def acc(logits, idx):
        return float((logits[idx].argmax(1) == yc[idx]).float().mean())

    # ---- full GCN
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model = GCN(x.shape[1], hidden, c).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def step():
        model.train(); opt.zero_grad()
        F.cross_entropy(model(x, p_t)[tr_d], yn[tr_d]).backward(); opt.step()

    def ev():
        model.eval()
        with torch.no_grad():
            lg = model(x, p_t)
        return acc(lg, va_d), {k: v.clone() for k, v in model.state_dict().items()}
    t0 = time.perf_counter(); sync()
    best, state, ep = train_loop(step, ev, epochs, patience)
    sync(); t_train = time.perf_counter() - t0
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        test = acc(model(x, p_t), te_d)
    rows.append(dict(arm="full_gcn", M=n, ratio=1.0, train_s=t_train, epochs=ep, pre_s=tm.t["pre_adj"],
                     test=test, val=best, peak_gib=torch.cuda.max_memory_allocated() / 2**30))

    # ---- full SIGN head (trains on L labelled rows)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    head = MLP(sign_feat.shape[1], hidden, c).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=5e-4)
    ftr = sign_feat[tr_d]

    def step():
        head.train(); opt.zero_grad()
        F.cross_entropy(head(ftr), yn[tr_d]).backward(); opt.step()

    def ev():
        head.eval()
        with torch.no_grad():
            lg = head(sign_feat[va_d])
        return float((lg.argmax(1) == yc[va_d]).float().mean()), {k: v.clone() for k, v in head.state_dict().items()}
    t0 = time.perf_counter(); sync()
    best, state, ep = train_loop(step, ev, epochs, patience)
    sync(); t_train = time.perf_counter() - t0
    head.load_state_dict(state); head.eval()
    with torch.no_grad():
        test = acc(head(sign_feat), te_d)
    rows.append(dict(arm="full_sign", M=n, ratio=1.0, train_s=t_train, epochs=ep,
                     pre_s=tm.t["pre_adj"] + tm.t["pre_sign"], test=test, val=best,
                     peak_gib=torch.cuda.max_memory_allocated() / 2**30))

    # ---- coarse arms
    t0 = time.perf_counter()
    z = coords(data.x, p, dim)
    t_coords = time.perf_counter() - t0
    for ratio in ratios:
        budget = max(c, int(round(ratio * n)))
        t0 = time.perf_counter()
        assign = bisect_tree(z, budget, min_leaf, seed=seed)
        t_tree = time.perf_counter() - t0
        t0 = time.perf_counter()
        h, r, pc = coarse_ops(assign, p)
        m = pc.shape[0]
        pc_t, r_t = to_torch_sparse(pc), to_torch_sparse(r)
        xc = torch.sparse.mm(r_t, x)
        sign_c = torch.sparse.mm(r_t, sign_feat)
        tgt, cnt, lab = soft_targets(assign, m, y_noisy, tr, c)
        tgt, cnt, lab = tgt.to(DEV), cnt.to(DEV), lab.to(DEV)
        assign_d = torch.as_tensor(assign, device=DEV)
        sync(); t_ops = time.perf_counter() - t0
        lab_idx = torch.nonzero(lab).flatten()
        w = cnt[lab_idx] / cnt[lab_idx].sum()
        struct = dict(M=m, ratio=ratio, M_plus=int(lab.sum()), E=int(p.nnz), Ec=int(pc.nnz),
                      coords_s=t_coords, tree_s=t_tree, ops_s=t_ops)
        # clean-label soft-target Brier (diagnostic only)
        tgt_clean, _, _ = soft_targets(assign, m, y_clean, tr, c)
        node_tgt = tgt.cpu()[torch.as_tensor(assign)[tr]]
        struct["target_brier_vs_clean"] = float(((node_tgt - F.one_hot(y_clean[tr], c).float()) ** 2).sum(1).mean())

        # coarse GCN
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        model = GCN(x.shape[1], hidden, c).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        def step():
            model.train(); opt.zero_grad()
            lp = F.log_softmax(model(xc, pc_t)[lab_idx], 1)
            (-(tgt[lab_idx] * lp).sum(1) * w).sum().backward(); opt.step()

        def ev():
            model.eval()
            with torch.no_grad():
                lg = model(xc, pc_t)[assign_d]
            return acc(lg, va_d), {k: v.clone() for k, v in model.state_dict().items()}
        t0 = time.perf_counter(); sync()
        best, state, ep = train_loop(step, ev, epochs, patience)
        sync(); t_train = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        model.load_state_dict(state); model.eval()
        t0 = time.perf_counter(); sync()
        with torch.no_grad():
            lg_full = model(x, p_t)
        sync(); t_inf = time.perf_counter() - t0
        with torch.no_grad():
            lg_lift = model(xc, pc_t)[assign_d]
        rows.append(dict(arm="coarse_gcn", train_s=t_train, epochs=ep, pre_s=tm.t["pre_adj"],
                         test=acc(lg_full, te_d), test_lift=acc(lg_lift, te_d), val=best, peak_gib=peak,
                         infer_s=t_inf, **struct))

        # coarse SIGN head
        set_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        head = MLP(sign_feat.shape[1], hidden, c).to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=5e-4)
        fc = sign_c[lab_idx]

        def step():
            head.train(); opt.zero_grad()
            lp = F.log_softmax(head(fc), 1)
            (-(tgt[lab_idx] * lp).sum(1) * w).sum().backward(); opt.step()

        def ev():
            head.eval()
            with torch.no_grad():
                lg = head(sign_feat[va_d])
            return float((lg.argmax(1) == yc[va_d]).float().mean()), {k: v.clone() for k, v in head.state_dict().items()}
        t0 = time.perf_counter(); sync()
        best, state, ep = train_loop(step, ev, epochs, patience)
        sync(); t_train = time.perf_counter() - t0
        head.load_state_dict(state); head.eval()
        with torch.no_grad():
            test = acc(head(sign_feat), te_d)
        rows.append(dict(arm="coarse_sign", train_s=t_train, epochs=ep, pre_s=tm.t["pre_adj"] + tm.t["pre_sign"],
                         test=test, val=best, peak_gib=torch.cuda.max_memory_allocated() / 2**30, **struct))
    for row in rows:
        row.update(dataset=dataset, noise=f"{noise[0]}{noise[1]}", seed=seed, N=n, L=int(tr.numel()), C=c)
        out.write(json.dumps(row) + "\n")
        out.flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coauthorcs"])
    ap.add_argument("--noises", nargs="+", default=["uniform:0.0", "uniform:0.4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.1, 0.3])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--min-leaf", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "smoke_results.jsonl"))
    a = ap.parse_args()
    with open(a.out, "a", encoding="utf-8") as out:
        for ds in a.datasets:
            for nz in a.noises:
                kind, rate = nz.split(":")
                for s in a.seeds:
                    for row in run_case(ds, (kind, float(rate)), s, a.ratios, a.hidden, a.epochs, a.patience,
                                        a.dim, a.min_leaf, out):
                        keys = ["dataset", "noise", "seed", "arm", "ratio", "M", "test", "test_lift", "train_s",
                                "epochs", "peak_gib", "tree_s"]
                        print({k: (round(row[k], 4) if isinstance(row.get(k), float) else row.get(k)) for k in keys if k in row}, flush=True)


if __name__ == "__main__":
    main()
