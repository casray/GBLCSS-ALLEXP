"""Read-only checks of historical evidence; no model training or parameter search."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sources = []


def read_csv(relative):
    path = ROOT / relative
    sources.append(path)
    return pd.read_csv(path)


def paired(df, first, second, value, keys):
    a = df[df.arm == first][keys + [value]]
    b = df[df.arm == second][keys + [value]]
    out = a.merge(b, on=keys, validate="one_to_one", suffixes=("_a", "_b"))
    assert len(out) == len(a) == len(b)
    out["delta"] = out[value + "_a"] - out[value + "_b"]
    return out


count = read_csv("experiments/ball_count_diagnostic_20261015/student_results.csv")
selected = count[(count.dataset.isin(["corafull", "coauthorcs", "squirrel_filtered"]))
                 & count.noise.isin(["uniform", "pair"])
                 & count.arm.isin(["GREET_B1", "GREET_B64", "GREET_B256"])]
count_summary = selected.groupby(["dataset", "noise", "arm"]).test_accuracy.agg(
    ["count", "mean", "std"]
).reset_index()
count_summary[["mean", "std"]] *= 100
count_summary.to_csv(HERE / "historical_count_check.csv", index=False, encoding="utf-8-sig")

prune = read_csv("experiments/evidence_pruning_20261015/student_results.csv")
pdiff = paired(prune[prune.noise != "clean"], "PRUNE", "GREET_B64",
               "test_accuracy", ["dataset", "noise", "seed"])
prune_deltas = (100 * pdiff.groupby("dataset").delta.mean()).to_dict()
targets = read_csv("experiments/evidence_pruning_20261015/target_results.csv")
tdiff = paired(targets[targets.noise != "clean"], "PRUNE", "CF_FINE",
               "target_brier", ["dataset", "noise", "seed"])
assert len(tdiff) == 36
assert int((tdiff.delta < 0).sum()) == 0
assert abs(prune_deltas["corafull"] - (-7.938)) < .001

kd = read_csv("experiments/gb_dist_protocol_seeds_20261007/cell_summary.csv")
assert len(kd) == 10
kd_delta = float(kd.delta_pp_mean.mean())
assert abs(kd_delta - .120901) < .000001
recovery = read_csv("experiments/supervision_recovery_EF_20261012/seed_macro.csv")
recovery_keys = ["B_minus_A_student_pp", "B_minus_C_student_pp", "E_minus_A_student_pp"]
recovery_check = {key: {"mean": float(recovery[key].mean()),
                        "sample_std": float(recovery[key].std(ddof=1))}
                  for key in recovery_keys}
assert abs(recovery_check["E_minus_A_student_pp"]["mean"] - (-.0391)) < .0001

# Algebra checks illustrate a decomposition, not evidence of an effective algorithm.
rng = np.random.default_rng(20261019)
worst_identity = 0.0
max_bound_violation = 0.0
for trial in range(100):
    n, m, d = 30, 5, 7
    assignment = np.r_[np.arange(m), rng.integers(m, size=n-m)]
    S = np.eye(m)[assignment]
    R = S.T / S.sum(axis=0)[:, None]
    P = rng.uniform(size=(n, n))
    P /= P.sum(axis=1, keepdims=True)
    H = rng.normal(size=(n, d))
    Pc = R @ P @ S
    lhs = P @ H - S @ Pc @ R @ H
    rhs = P @ (H - S @ R @ H) + (P @ S - S @ Pc) @ R @ H
    worst_identity = max(worst_identity, float(np.max(np.abs(lhs-rhs))))
    eps_x = float(np.linalg.norm(H-S@R@H, axis=1).max())
    eps_p = float(np.abs(P@S-S@Pc).sum(axis=1).max())
    hmax = float(np.linalg.norm(R@H, axis=1).max())
    for k in [1, 2, 4]:
        error = np.linalg.matrix_power(P, k)@H-S@np.linalg.matrix_power(Pc, k)@R@H
        violation = float(np.linalg.norm(error, axis=1).max() - (eps_x+k*eps_p*hmax))
        max_bound_violation = max(max_bound_violation, violation)
assert worst_identity < 1e-12 and max_bound_violation <= 1e-12

# A one-ball solution makes PS-SPc zero: this score alone is degenerate.
P = np.array([[0,0,.5,.5], [0,0,.5,.5], [.5,.5,0,0], [.5,.5,0,0]])
H = np.array([[1.], [1.], [-1.], [-1.]])
toy = {}
for label, assignment in {"same_role": [0,0,1,1], "mixed": [0,1,0,1], "one_ball": [0,0,0,0]}.items():
    S = np.eye(max(assignment)+1)[assignment]
    R = S.T / S.sum(axis=0)[:,None]
    Pc = R@P@S
    toy[label] = {"operator_error": float(np.linalg.norm(P@S-S@Pc)),
                  "feature_error": float(np.linalg.norm(H-S@R@H)),
                  "one_step_error": float(np.linalg.norm(P@H-S@Pc@R@H)),
                  "coarse_transition": Pc.tolist()}
assert toy["same_role"]["one_step_error"] == 0
assert toy["mixed"]["one_step_error"] > 0
assert toy["one_ball"]["operator_error"] == 0

for rel in ["01_literature_and_research_map.md", "02_experiment_history_and_lessons.md",
            "GB-LCSS_code/src/gblcss/granular_ball.py", "GB-LCSS_code/src/gblcss/targets.py",
            "experiments/ball_count_diagnostic_20261015/config.json",
            "experiments/evidence_pruning_20261015/config.json"]:
    sources.append(ROOT / rel)
sources.append(Path("C:/Users/Admin/Downloads/新建文件 (7).txt"))
manifest = [{"path": str(p), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
result = {"date": "2026-10-19", "type": "historical arithmetic and algebra sanity checks only",
          "new_training": False, "historical_rows_ball_count": len(count),
          "prune_noisy_delta_pp": prune_deltas, "prune_target_pairs": len(tdiff),
          "prune_target_improved_pairs": int((tdiff.delta < 0).sum()),
          "dist_region_delta_pp": kd_delta, "recovery": recovery_check,
          "algebra": {"random_cases": 100, "max_identity_error": worst_identity,
                      "max_bound_violation": max_bound_violation, "toy": toy},
          "sources": manifest, "checks_passed": True,
          "limits": "Does not rerun training, re-audit every historical prediction, or prove novelty."}
(HERE / "evidence_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: result[key] for key in ["checks_passed", "historical_rows_ball_count",
                   "prune_target_improved_pairs", "dist_region_delta_pp", "prune_noisy_delta_pp", "recovery"]},
                 ensure_ascii=False, indent=2))
