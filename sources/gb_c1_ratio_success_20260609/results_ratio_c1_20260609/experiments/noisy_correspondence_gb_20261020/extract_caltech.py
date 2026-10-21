"""Extract the author's MATLAB v7.3 Caltech file without changing sample order."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))
import pyfive  # noqa: E402


def main() -> None:
    source = ROOT / "data" / "smalldata" / "Caltech101-20.mat"
    hdf = source.with_suffix(".h5")
    if not hdf.exists():
        with source.open("rb") as src, hdf.open("wb") as dst:
            src.seek(512)
            while block := src.read(1 << 20):
                dst.write(block)

    with pyfive.File(str(hdf)) as mat:
        ref_map = {address: key for key, address in mat["#refs#"]._links.items()}
        data_refs = mat["data"][:].reshape(-1)
        views = []
        for ref in data_refs:
            address = int.from_bytes(bytes(ref), "little")
            view = np.asarray(mat["#refs#"][ref_map[address]][:]).T
            views.append(np.asarray(view, dtype=np.float32))
        labels = np.asarray(mat["labels"][:], dtype=np.int32).reshape(-1)

    assert len(views) == 6
    assert all(view.shape[0] == labels.size for view in views)
    assert [view.shape[1] for view in views] == [48, 40, 254, 1984, 512, 928]
    assert labels.size == 2386 and np.unique(labels).size == 20
    assert all(np.isfinite(view).all() for view in views)
    out = ROOT / "data" / "Caltech101-20.npz"
    np.savez_compressed(out, **{f"view{i}": v for i, v in enumerate(views)}, labels=labels)
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(out),
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": int(labels.size),
        "classes": int(np.unique(labels).size),
        "view_dims": [int(v.shape[1]) for v in views],
        "selected_views": [3, 4],
        "reader": "pyfive v1.1.0, commit 5ac28a91e227dff5108a90febb5a7f7c3510476e",
    }
    (ROOT / "data" / "Caltech101-20_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
