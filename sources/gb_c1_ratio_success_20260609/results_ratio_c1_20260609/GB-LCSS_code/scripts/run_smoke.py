from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.experiment import load_config, make_smoke_graph, run_gblcss


def main() -> None:
    config = load_config(ROOT / "configs" / "main" / "gblcss.yaml")
    config["method"].update({"max_balls": 6, "min_ball_size": 12})
    config["training"].update({"epochs": 12, "device": "cpu"})
    graph, representation = make_smoke_graph(seed=0)
    output = run_gblcss(graph, representation, config, seed=0)
    payload = {
        "status": "ok",
        "nodes": graph.num_nodes,
        "train_nodes": int(graph.train_mask.sum()),
        "num_balls": output.diagnostics["num_balls"],
        "eligible_train_nodes": output.diagnostics["eligible_train_nodes"],
        "test_accuracy": output.result.test_accuracy,
        "inference_inputs": ["X"],
    }
    print(json.dumps(payload, indent=2))
    if payload["num_balls"] < 2 or payload["eligible_train_nodes"] == 0:
        raise SystemExit("smoke test did not exercise local-consensus targets")


if __name__ == "__main__":
    main()
