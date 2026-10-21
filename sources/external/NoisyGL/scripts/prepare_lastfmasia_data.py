from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


SNAP_URL = "https://snap.stanford.edu/data/lastfm_asia.zip"
PAPER_URL = "https://arxiv.org/abs/2005.07959"
PYG_ORIGINAL_URL = "https://graphmining.ai/datasets/ptg/lastfm_asia.npz"
OFFICIAL_COUNTS = {
    "num_nodes": 7624,
    "num_undirected_edges": 27806,
    "num_directed_edges": 55612,
    "num_unique_raw_features": 7842,
    "num_classes": 18,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _find_file(raw_dir: Path, filename: str) -> Path:
    matches = list(raw_dir.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} under {raw_dir}, found {len(matches)}"
        )
    return matches[0]


def build_lastfmasia_arrays(
    raw_dir: Path,
    n_components: int = 128,
    random_state: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    raw_dir = Path(raw_dir)
    edges_path = _find_file(raw_dir, "lastfm_asia_edges.csv")
    target_path = _find_file(raw_dir, "lastfm_asia_target.csv")
    features_path = _find_file(raw_dir, "lastfm_asia_features.json")

    with target_path.open("r", encoding="utf-8", newline="") as handle:
        target_rows = list(csv.DictReader(handle))
    target_by_id = {int(row["id"]): int(row["target"]) for row in target_rows}
    num_nodes = len(target_by_id)
    expected_node_ids = list(range(num_nodes))
    if sorted(target_by_id) != expected_node_ids:
        raise ValueError("target node IDs must be contiguous and start at zero")
    target = np.asarray(
        [target_by_id[node_id] for node_id in expected_node_ids],
        dtype=np.int64,
    )

    feature_map = json.loads(features_path.read_text(encoding="utf-8"))
    feature_by_id = {int(node_id): values for node_id, values in feature_map.items()}
    if sorted(feature_by_id) != expected_node_ids:
        raise ValueError("feature node IDs must match target node IDs")
    artist_ids = sorted(
        {int(artist_id) for values in feature_by_id.values() for artist_id in values}
    )
    artist_to_column = {
        artist_id: column for column, artist_id in enumerate(artist_ids)
    }
    rows: list[int] = []
    columns: list[int] = []
    for node_id in expected_node_ids:
        for artist_id in sorted({int(value) for value in feature_by_id[node_id]}):
            rows.append(node_id)
            columns.append(artist_to_column[artist_id])
    sparse_features = csr_matrix(
        (
            np.ones(len(rows), dtype=np.float32),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(num_nodes, len(artist_ids)),
        dtype=np.float32,
    )

    if not 0 < int(n_components) < min(sparse_features.shape):
        raise ValueError(
            "n_components must be positive and smaller than both sparse matrix dimensions"
        )
    svd = TruncatedSVD(
        n_components=int(n_components),
        algorithm="randomized",
        n_iter=5,
        random_state=int(random_state),
    )
    features = svd.fit_transform(sparse_features).astype(np.float32, copy=False)

    undirected_edges: list[tuple[int, int]] = []
    with edges_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = int(row["node_1"])
            target_node = int(row["node_2"])
            if source == target_node:
                continue
            if not 0 <= source < num_nodes or not 0 <= target_node < num_nodes:
                raise ValueError(f"edge contains out-of-range node: {(source, target_node)}")
            undirected_edges.append((source, target_node))
    edges = np.asarray(
        [
            directed_edge
            for source, target_node in undirected_edges
            for directed_edge in ((source, target_node), (target_node, source))
        ],
        dtype=np.int64,
    )

    arrays = {
        "features": features,
        "target": target,
        "edges": edges,
    }
    metadata = {
        "num_nodes": num_nodes,
        "num_undirected_edges": len(undirected_edges),
        "num_directed_edges": int(edges.shape[0]),
        "num_unique_raw_features": len(artist_ids),
        "num_classes": int(np.unique(target).size),
        "raw_feature_nonzeros": int(sparse_features.nnz),
        "svd_components": int(n_components),
        "svd_random_state": int(random_state),
        "svd_explained_variance_ratio_sum": float(
            svd.explained_variance_ratio_.sum()
        ),
    }
    return arrays, metadata


def prepare_lastfmasia_data(
    source_zip: Path,
    output_root: Path,
    n_components: int = 128,
    random_state: int = 0,
    validate_official_counts: bool = True,
) -> tuple[Path, Path]:
    source_zip = Path(source_zip).resolve()
    output_root = Path(output_root).resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    processed_cache = output_root / "processed" / "data.pt"
    if processed_cache.exists():
        raise RuntimeError(
            f"{processed_cache} exists; remove processed/data.pt before rebuilding "
            "the raw NPZ so PyG cannot silently reuse stale data"
        )

    with tempfile.TemporaryDirectory() as tmp:
        extracted_dir = Path(tmp)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(extracted_dir)
        arrays, dataset_metadata = build_lastfmasia_arrays(
            extracted_dir,
            n_components=n_components,
            random_state=random_state,
        )
        source_files = {
            filename: _sha256(_find_file(extracted_dir, filename))
            for filename in (
                "lastfm_asia_edges.csv",
                "lastfm_asia_target.csv",
                "lastfm_asia_features.json",
            )
        }

    if validate_official_counts:
        for key, expected in OFFICIAL_COUNTS.items():
            actual = dataset_metadata[key]
            if actual != expected:
                raise ValueError(
                    f"official count mismatch for {key}: {actual} != {expected}"
                )

    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    npz_path = raw_dir / "lastfm_asia.npz"
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "dataset_name": "LastFMAsia",
        "source": {
            "official_snap_url": SNAP_URL,
            "source_zip": str(source_zip),
            "source_zip_sha256": _sha256(source_zip),
            "source_file_sha256": source_files,
            "paper_url": PAPER_URL,
            "unavailable_pyg_original_url": PYG_ORIGINAL_URL,
        },
        "reconstruction": {
            "status": "deterministic_reconstruction_not_byte_identical_claim",
            "raw_feature_encoding": "binary user-artist incidence matrix",
            "dimensionality_reduction": "sklearn.decomposition.TruncatedSVD",
            "svd_algorithm": "randomized",
            "svd_n_iter": 5,
            "svd_n_components": int(n_components),
            "svd_random_state": int(random_state),
            "feature_normalization_before_svd": "none",
            "labels_used_for_feature_construction": False,
            "paper_protocol_evidence": (
                "The FEATHER paper states that supervised-model vertex features "
                "were reduced to 128 dimensions with Scikit-Learn Truncated SVD."
            ),
        },
        "dataset": dataset_metadata,
        "array_sha256": {
            key: _array_sha256(value) for key, value in arrays.items()
        },
        "npz_sha256": _sha256(npz_path),
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path = raw_dir / "lastfm_asia_reconstruction.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return npz_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the unavailable PyG LastFM-Asia NPZ from official SNAP data."
    )
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_zip is None:
        with tempfile.TemporaryDirectory() as tmp:
            source_zip = Path(tmp) / "lastfm_asia.zip"
            urllib.request.urlretrieve(SNAP_URL, source_zip)
            npz_path, metadata_path = prepare_lastfmasia_data(
                source_zip,
                args.output_root,
                n_components=args.n_components,
                random_state=args.random_state,
            )
    else:
        npz_path, metadata_path = prepare_lastfmasia_data(
            args.source_zip,
            args.output_root,
            n_components=args.n_components,
            random_state=args.random_state,
        )
    print(f"WROTE {npz_path}")
    print(f"WROTE {metadata_path}")


if __name__ == "__main__":
    main()
