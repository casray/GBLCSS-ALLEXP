"""Read audited v2 caches and prepare new label-free partition controls."""
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from experiment import atomic_json, file_hash
from legacy_api import array_hash


def rownorm(x):
    return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prepare(config_path, root, dataset):
    started = time.perf_counter()
    config = json.loads(Path(config_path).read_text(encoding='utf8'))
    old = Path(config['source_cache_root'])/dataset
    folder = Path(root)/'cache'/dataset
    folder.mkdir(parents=True, exist_ok=True)
    out = folder/'metadata.json'
    if out.exists():
        meta = json.loads(out.read_text(encoding='utf8'))
        if meta['config_sha256'] != file_hash(config_path):
            raise ValueError('prepared config drift')
        for name, digest in meta['new_files'].items():
            if file_hash(folder/name) != digest:
                raise ValueError('prepared partition drift '+name)
        previous=Path(meta['old_directory'])
        if file_hash(previous/'metadata.json')!=meta['original_metadata_sha256']:
            raise ValueError('original metadata drift')
        for name,record in meta['original_cache_files'].items():
            if file_hash(previous/name)!=record['sha256']:
                raise ValueError('original input drift '+name)
        return meta
    original = json.loads((old/'metadata.json').read_text(encoding='utf8'))
    for name in ['x.npy','y.npy','p.npz','tree.pkl']:
        if file_hash(old/name) != original['cache_files'][name]['sha256']:
            raise ValueError('original cache digest changed '+name)
    n = original['N']; budget = round(config['ratio']*n)
    gb = np.load(old/'cuts'/f'R0_M{budget}.npy')
    cut = json.loads((old/'cuts'/f'R0_M{budget}.json').read_text(encoding='utf8'))
    if array_hash(gb) != cut['assignment_sha256']:
        raise ValueError('original cut digest changed')
    if not np.array_equal(np.unique(gb), np.arange(budget)):
        raise ValueError('cut budget mismatch')
    np.save(folder/'granule.npy', gb)
    t_coord = time.perf_counter()
    x = np.load(old/'x.npy', mmap_mode='r'); p = sp.load_npz(old/'p.npz')
    with threadpool_limits(limits=config['cpu_threads']):
        u = TruncatedSVD(n_components=min(config['svd_dim'], x.shape[1]-1),
            random_state=config['tree_seed']).fit_transform(sp.csr_matrix(rownorm(x))).astype(np.float32)
        pu = p@u
        channels = [rownorm(a) for a in (u, pu, p@pu)]
        sses = [float(np.square(a-a.mean(0)).sum()) for a in channels]
        if not np.allclose(sses, original['root_channel_sses'], rtol=1e-5, atol=1e-6):
            raise ValueError('reconstructed coordinate normalization differs')
        coords = np.concatenate([a/np.sqrt(max(s,1e-12)) for a,s in zip(channels,sses)], axis=1)
        with (old/'tree.pkl').open('rb') as stream:
            tree = pickle.load(stream)
        probes = np.linspace(0,len(tree.members)-1,min(50,len(tree.members)),dtype=int)
        errors = []
        for node in probes:
            v = coords[tree.members[node]].astype(np.float64)
            errors.append(abs(float(np.square(v-v.mean(0)).sum())-float(tree.costs[node])))
        if max(errors)>1e-5:
            raise ValueError('reconstructed coordinates do not match original tree SSE')
        coord_s = time.perf_counter()-t_coord
        np.save(folder/'coordinates.npy',coords)
        t_km = time.perf_counter()
        km = KMeans(n_clusters=budget, random_state=config['tree_seed'], n_init=config['kmeans_n_init'],
            max_iter=config['kmeans_max_iter'], algorithm='lloyd').fit_predict(coords).astype(np.int64)
        km_s = time.perf_counter()-t_km
    if not np.array_equal(np.unique(km),np.arange(budget)):
        raise RuntimeError('k-means returned fewer groups than fixed budget; no silent fallback')
    np.save(folder/'kmeans.npy',km)
    meta = dict(dataset=dataset,N=n,D=original['D'],C=original['C'],M=budget,
        config_sha256=file_hash(config_path),old_directory=str(old),original_metadata=original,
        original_metadata_sha256=file_hash(old/'metadata.json'),
        original_cache_files={k:original['cache_files'][k] for k in ['x.npy','y.npy','p.npz','tree.pkl']},
        new_files={name:file_hash(folder/name) for name in ['granule.npy','kmeans.npy','coordinates.npy']},
        coordinates_s=coord_s,kmeans_s=km_s,coordinate_probe_max_sse_error=max(errors),
        original_cut_cold_s=cut.get('cut_cold_s',0),preparation_wall_s=time.perf_counter()-started,
        granule_sha256=array_hash(gb),kmeans_sha256=array_hash(km), labels_used=False)
    atomic_json(out,meta)
    print(json.dumps({'dataset':dataset,'M':budget,'coordinates_s':coord_s,'kmeans_s':km_s,
        'coordinate_probe_max_sse_error':max(errors)}),flush=True)
    return meta


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--run-root',required=True);parser.add_argument('--dataset',required=True)
    args=parser.parse_args();prepare(args.config,args.run_root,args.dataset)
