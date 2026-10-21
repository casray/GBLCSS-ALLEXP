from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    environment = json.loads((ROOT / "environment.json").read_text(encoding="utf-8"))
    workspace = ROOT.parents[1]
    for relative, expected in environment["gblcss_source_hashes"].items():
        assert hashlib.sha256((workspace / relative).read_bytes()).hexdigest() == expected
    audits = []
    for path in sorted((ROOT / "runs").glob("*/*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "ok":
            continue
        if result["method"] == "tfr":
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            assert config["teacher"]["adapter_revision"] == 2
            assert config["teacher"]["graph_self_loops"] is False
        cached = torch.load(ROOT / "protocol_cache" / f"{result['dataset']}_u{result['noise_rate']}.pt",
                            map_location="cpu", weights_only=False)
        record = dict(dataset=result["dataset"], rate=result["noise_rate"], method=result["method"])
        for phase in ("teacher", "student"):
            probs = torch.load(path.parent / f"{phase}_probabilities.pt", map_location="cpu", weights_only=True)
            assert probs.shape == (result["num_nodes"], result["num_classes"])
            assert torch.isfinite(probs).all() and probs.min() >= 0
            error = float((probs.sum(1) - 1).abs().max())
            assert error < 1e-5
            pred = probs.argmax(1)
            for split, label_name in (("train", "noisy"), ("val", "clean"), ("test", "clean")):
                index = cached[split + "_index"]
                acc = float((pred[index] == cached[label_name][index]).float().mean())
                key = phase + "_" + ("valid" if split == "val" else split)
                assert abs(acc - result[key]) < 1e-6
            assert (path.parent / f"{phase}.pt").stat().st_size > 0
            record[phase + "_row_sum_error"] = error
        assert torch.equal(cached["noisy"][cached["val_index"]], cached["clean"][cached["val_index"]])
        assert torch.equal(cached["noisy"][cached["test_index"]], cached["clean"][cached["test_index"]])
        assert (path.parent / "epochs.jsonl").stat().st_size > 0
        record["status"] = "passed"
        audits.append(record)
    (ROOT / "output_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Audited {len(audits)} completed runs: probabilities, saved metrics, checkpoints, clean validation/test")
    return audits


if __name__ == "__main__":
    main()
