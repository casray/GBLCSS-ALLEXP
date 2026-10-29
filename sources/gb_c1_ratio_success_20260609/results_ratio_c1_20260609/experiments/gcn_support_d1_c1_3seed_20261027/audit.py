"""Independent CPU audit of frozen GCN/DIST artifacts; writes the final gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent


def import_absolute(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analysis = import_absolute("gcn_extension_independent_analysis", CODE/"analysis.py")
require = analysis.require
check_freeze = analysis.check_freeze
METHODS = analysis.METHODS


def candidates():
    out = {}
    for support in (4, 8, 16, 32, 64, 128):
        for strength in (.25, .5, 1.):
            out[f"S{support}_L{strength}"] = dict(support=support, kind="uniform", strength=strength)
        for agree, conflict in ((0., 1.), (.25, 1.), (1., .25), (1., 0.)):
            out[f"S{support}_G_A{agree}_C{conflict}"] = dict(
                support=support, kind="group", agree=agree, conflict=conflict)
    return out


def evidence(observed, assignment, classes):
    z, assignment = np.asarray(observed, dtype=np.int64), np.asarray(assignment, dtype=np.int64)
    require(z.ndim == 1 and z.shape == assignment.shape and len(z) > 0 and classes > 1, "Invalid evidence shape")
    require((z >= 0).all() and (z < classes).all(), "Invalid observed labels")
    own = np.eye(classes, dtype=np.float64)[z]
    _, inverse = np.unique(assignment, return_inverse=True)
    total = np.zeros((int(inverse.max())+1, classes), dtype=np.float64)
    np.add.at(total, inverse, own)
    counts = total[inverse]-own
    support = counts.sum(1)
    vote = counts/np.maximum(support, 1)[:, None]
    winner = counts.argmax(1)
    maximum = counts.max(1)
    tied = (counts == maximum[:, None]).sum(1) > 1
    alpha = np.clip((maximum/np.maximum(support, 1)-1/classes)/(1-1/classes), 0, 1)
    alpha[support < 2] = 0
    beta = np.where(tied, 0., .5)
    q = (1-beta[:, None])*vote+beta[:, None]*np.eye(classes)[winner]
    qz = q[np.arange(len(z)), z]
    mass = alpha*(1-qz)
    direction = q.copy()
    direction[np.arange(len(z)), z] = 0
    direction /= np.maximum(1-qz, 1e-30)[:, None]
    return dict(z=z, own=own, counts=counts, support=support, vote=vote, winner=winner,
                tied=tied, alpha=alpha, q=q, m=mass, d=direction,
                native=((1-alpha[:, None])*own+alpha[:, None]*q).astype(np.float32))


def weights(e, spec):
    if spec["kind"] == "uniform":
        return np.full(len(e["z"]), spec["strength"], dtype=np.float64)
    return np.where(e["tied"], 1., np.where(e["winner"] == e["z"], spec["agree"], spec["conflict"]))


def shrink(e, strength):
    strength = np.asarray(strength, dtype=np.float64)
    require(((strength >= 0) & (strength <= 1)).all(), "Invalid target strength")
    mass = e["m"]*strength
    result = ((1-mass[:, None])*e["own"]+mass[:, None]*e["d"]).astype(np.float32)
    require(np.isfinite(result).all() and np.allclose(result.sum(1), 1, atol=4e-7), "Invalid target")
    return result


def rank(row):
    allowed = {"arm", "valid_correct", "valid_total", "valid_nll", "mean_lambda", "support", "best_epoch", "epoch"}
    require(set(row) <= allowed and row["valid_total"] > 0, "Non-validation selector field")
    require(np.isfinite([row["valid_nll"], row["mean_lambda"]]).all(), "Nonfinite selector")
    arm = row["arm"]
    fixed_id = (0 if arm == "CE" else 1 if "_L" in arm else 2, arm)
    return (-row["valid_correct"]/row["valid_total"], row["valid_nll"], row["mean_lambda"],
            -row["support"], row["best_epoch"], fixed_id)


def prefix_score(logs, prefix, arm, support, mean_lambda, total):
    eligible = [r for r in logs if r["epoch"] <= prefix]
    require([r["epoch"] for r in eligible] == list(range(1, prefix+1)), "Incomplete causal prefix")
    # max preserves the earliest row on an accuracy tie; NLL never reselects a checkpoint.
    best = max(eligible, key=lambda r: r["validation_correct"])
    return dict(arm=arm, support=support, mean_lambda=mean_lambda, epoch=prefix,
                best_epoch=best["epoch"], valid_correct=best["validation_correct"], valid_total=total,
                valid_nll=best["validation_nll"])


def survivors(rows, count, prefix):
    require(all(r["epoch"] == prefix for r in rows), "Future score in race")
    require(len({r["arm"] for r in rows}) == len(rows) and len(rows) >= count, "Duplicate/short race")
    chosen = [r["arm"] for r in sorted(rows, key=rank)[:count]]
    if "S32_L1.0" in {r["arm"] for r in rows} and "S32_L1.0" not in chosen:
        chosen[-1] = "S32_L1.0"
    return chosen


def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(str((key, value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def verify_manifest(path, reader):
    manifest = reader.json(path)
    for name, digest in manifest.items():
        require(Path(name).is_absolute(), "Expected absolute source/input path")
        reader.expect(name, digest)
    return manifest


def assert_close(a, b, message, tol=1e-12):
    require(abs(float(a)-float(b)) <= tol, message+f": {a} != {b}")


def equivalent_score(actual, expected):
    require(set(actual) == set(expected), "Prefix score schema mismatch")
    for key in expected:
        if key in ("mean_lambda", "valid_nll"):
            assert_close(actual[key], expected[key], "Prefix score "+key)
        else:
            require(actual[key] == expected[key], "Prefix score "+key)


def split_indices(nodes, seed):
    rng = np.random.RandomState(seed)
    remaining = np.arange(nodes)
    out = []
    for fraction in (.2, .1, .7):
        ids = rng.choice(remaining, int(nodes*fraction), replace=False)
        out.append(ids)
        remaining = np.setdiff1d(remaining, ids)
    return out


def prepare_cell(workspace, dataset, seed, condition, x, y, geo, reader):
    path = analysis.geometry_path(workspace, dataset, seed).parent/f"{condition}.pt"
    pack = reader.tensor(path)
    train, valid = geo["train"].numpy(), geo["valid"].numpy()
    observed, valid_y = pack["observed"].numpy(), pack["valid_y"].numpy()
    require(np.array_equal(valid_y, y[valid]), "Validation labels drift")
    if condition == "clean":
        require(np.array_equal(observed, y[train]), "Clean observed labels drift")
    previous_root = "neighbor_support_ablation_20261022" if dataset in analysis.DATASETS[:3] else "r2_support_5dataset_20261022"
    previous = workspace/"experiments"/previous_root/"inputs"/dataset/f"seed{seed}"/f"{condition}.pt"
    if previous.exists() or condition != "clean":
        oldpack = reader.tensor(previous)
        require(np.array_equal(oldpack["observed"].numpy(), observed), "Historical noise pack drift")
        require(np.array_equal(oldpack["valid_y"].numpy(), valid_y), "Historical validation pack drift")
    classes = int(y.max())+1
    parts = {str(k): a.numpy() for k, a in geo["partitions"].items()}
    parts["LEGACY"] = geo["legacy32"].numpy()
    old_assignment = workspace/"experiments/main_protocol_replay_20260922/matrix"/dataset/"default"/f"seed{seed}.npz"
    reader.hash(old_assignment)
    with np.load(old_assignment) as stored:
        require(np.array_equal(stored["train_index"], train), "OLD64 train order drift")
        parts["OLD64"] = stored["assignment"]
    es = {key: evidence(observed, a[train], classes) for key, a in parts.items()}
    for support in (4, 8, 16, 32, 64, 128):
        require(np.max(np.abs(es[str(support)]["native"]-pack["targets"][support].numpy())) < 3e-7,
                "Historical native target mismatch")
    base = {a: shrink(es[k], 1.) for a, k in (("OLD64", "OLD64"), ("LEGACY_SUPPORT32", "LEGACY"))}
    old_baseline = workspace/"experiments/teacher_support_lcss_20261022/inputs"/dataset/f"seed{seed}"/f"{condition}.pt"
    if old_baseline.exists():
        cached = reader.tensor(old_baseline)
        require(np.array_equal(cached["observed_train"].numpy(), observed), "Cached baseline labels drift")
        for k, ids in (("train_index", train), ("valid_index", valid)):
            require(np.array_equal(cached[k].numpy(), ids), "Cached baseline split drift")
        for arm, key in (("OLD64", "OLD_LCSS"), ("LEGACY_SUPPORT32", "SUPPORT_LCSS")):
            stored = cached["targets"][key].numpy()
            require(np.max(np.abs(base[arm]-stored)) < 3e-7, "Cached target bridge drift")
            base[arm] = stored
    values = dict(x=x.float(), train=geo["train"], valid=geo["valid"], observed=pack["observed"], valid_y=pack["valid_y"])
    return es, base, state_hash(values)


def target_for(arm, es, base, grid):
    if arm == "CE":
        return es["LEGACY"]["own"].astype(np.float32), "LEGACY"
    if arm in base:
        return base[arm], "OLD64" if arm == "OLD64" else "LEGACY"
    spec = grid[arm]
    e = es[str(spec["support"])]
    return shrink(e, weights(e, spec)), str(spec["support"])


def target_diagnostic(dataset, seed, condition, method, ref, es, grid):
    arm, partition = ref["arm"], ref["partition"]
    e = es[partition]
    w = np.zeros(len(e["m"])) if arm == "CE" else (
        np.ones(len(e["m"])) if arm in ("OLD64", "LEGACY_SUPPORT32") else weights(e, grid[arm]))
    effective = e["m"]*w
    return dict(dataset=dataset, seed=seed, condition=condition, method=method,
                historical=ref["historical"], selected_arm=arm, partition=partition,
                **analysis.selected_spec(arm), train_count=len(e["m"]),
                actual_loo_support_mean=float(e["support"].mean()), actual_loo_support_p50=float(np.median(e["support"])),
                actual_loo_support_min=float(e["support"].min()), actual_loo_support_max=float(e["support"].max()),
                support_ge2_coverage=float((e["support"] >= 2).mean()),
                mean_lambda=float(w.mean()), min_lambda=float(w.min()), max_lambda=float(w.max()),
                mean_alpha=float(e["alpha"].mean()), mean_m=float(e["m"].mean()),
                mean_effective_m=float(effective.mean()), modified_rate=float((effective > 1e-8).mean()),
                raw_vote_tie_rate=float(e["tied"].mean()),
                unique_agree_rate=float((~e["tied"] & (e["winner"] == e["z"])).mean()),
                unique_conflict_rate=float((~e["tied"] & (e["winner"] != e["z"])).mean()))


def snapshot_path(root, old, dataset, seed, condition, arm, prefix):
    if analysis.historical(dataset, seed):
        return old/"fits/teacher"/analysis.cell_id(dataset, seed, condition)/arm/"prefixes"/str(prefix)
    return root/"prefix_snapshots"/analysis.cell_id(dataset, seed, condition)/arm/str(prefix)


def audit_snapshot(path, prefix, result, logs, expected_score, reader):
    manifest = reader.json(path/"manifest.json")
    require(manifest["prefix"] == prefix, "Snapshot prefix mismatch")
    reader.expect(path/"resume.pt", manifest["resume_sha256"])
    reader.expect(path/"result.json", manifest["result_sha256"])
    stored = reader.json(path/"result.json")
    require(stored["epoch"] == prefix and stored["best_epoch"] == expected_score["best_epoch"], "Snapshot best drift")
    ck = reader.tensor(path/"resume.pt", objects=True)
    for field in ("seed", "target_hash", "input_hash", "source_sha256", "initial_hash", "initial_rng_hash"):
        require(ck["meta"][field] == result[field] == stored[field], "Snapshot passport "+field)
    require(ck["epoch"] == prefix and ck["best"] == expected_score["valid_correct"] and
            ck["best_epoch"] == expected_score["best_epoch"], "Snapshot progress drift")
    assert_close(ck["best_nll"], expected_score["valid_nll"], "Snapshot NLL")
    require(set(ck["model"]) == set(ck["best_state"]), "Snapshot model schema drift")
    groups = ck["optimizer"]["param_groups"]
    require(len(groups) == 1 and groups[0]["lr"] == .01 and groups[0]["weight_decay"] == .0005, "Snapshot optimizer drift")
    require(bool(ck["optimizer"]["state"]), "Missing optimizer continuation state")
    for state in ck["optimizer"]["state"].values():
        require(int(state["step"]) == prefix, "Optimizer step does not match causal prefix")
    require(set(ck["rng"]) == {"cpu", "gpu", "numpy", "python"}, "Missing continuation RNG state")
    require(ck["rng"]["cpu"].device.type == "cpu" and ck["rng"]["gpu"] is not None,
            "Missing stored RNG (read-only CPU tensors)")
    require(ck["best_f1"] == logs[expected_score["best_epoch"]-1]["validation_macro_f1"], "Snapshot best F1 drift")
    require(all(bool(t.isfinite().all()) for t in ck["model"].values()), "Nonfinite snapshot model")
    require(all(bool(t.isfinite().all()) for t in ck["best_state"].values()), "Nonfinite snapshot best state")
    require(len(logs) >= prefix, "Prefix exceeds trajectory")


def historical_reuse(root, old, refs, reader):
    frozen, oldrefs = check_freeze(old, reader, historical=True)
    for manifest in ("sources.json", "inputs.json"):
        verify_manifest(old/manifest, reader)
    manifest = reader.json(root/"historical_manifest.json")
    reader.expect(old/"selections_frozen.json", manifest["old_selection_freeze_sha256"])
    reader.expect(old/"refs.json", manifest["source_refs_sha256"])
    require(Path(manifest["source"]).resolve() == old.resolve(), "Historical source root drift")
    require(manifest["physical_manifest"] == reader.json(old/"physical_fits.json"), "Historical costs drift")
    for key, ref in oldrefs.items():
        require(refs[key] == dict(ref, historical=True), "Old reference mutated: "+key)
    for dataset in analysis.DATASETS[:3]:
        for condition in analysis.CONDITIONS:
            for stage in ("teacher", "student"):
                oldpath = analysis.selection_path(old, stage, dataset, 0, condition, old=True)
                newpath = analysis.selection_path(root, stage, dataset, 0, condition)
                require(reader.json(newpath) == reader.json(oldpath), "Copied historical selection drift")
    return len(oldrefs), frozen


def audit_race(root, old, dataset, seed, condition, refs, es, reader, log_cache, result_cache):
    grid = candidates()
    oldcell = analysis.historical(dataset, seed)
    racepath = (old if oldcell else root)/"races"/analysis.cell_id(dataset, seed, condition)/"joint.json"
    race = reader.json(racepath)
    require((race["dataset"], race["seed"], race["condition"]) == (dataset, seed, condition), "Race identity drift")
    expected, count_scores, count_snapshots = list(grid), 0, 0
    require([r["prefix"] for r in race["rounds"]] == [25, 75, 200], "Race rounds drift")
    total = next(result_cache[k]["valid_total"] for k in result_cache if k.startswith(analysis.cell_id(dataset, seed, condition)+"/"))
    for rd in race["rounds"]:
        prefix = rd["prefix"]
        arms = expected+(["CE"] if prefix == 200 else [])
        require([r["arm"] for r in rd["scores"]] == arms, "Race candidate membership/order drift")
        scores = []
        for arm, actual in zip(arms, rd["scores"]):
            key = analysis.cell_id(dataset, seed, condition)+"/teacher/"+arm
            spec = grid.get(arm)
            support = spec["support"] if spec else 0
            mean_lambda = float(weights(es[str(support)], spec).mean()) if spec else 0.
            score = prefix_score(log_cache[key], prefix, arm, support, mean_lambda, total)
            equivalent_score(actual, score)
            scores.append(score)
            count_scores += 1
            if prefix < 200:
                snap = snapshot_path(root, old, dataset, seed, condition, arm, prefix)
                audit_snapshot(snap, prefix, result_cache[key], log_cache[key], score, reader)
                count_snapshots += 1
        expected = survivors(scores, 14 if prefix == 25 else 4, prefix) if prefix < 200 else [min(scores, key=rank)["arm"]]
        require(rd["survivors"] == expected, "Race survivors/anchor/rank drift")
    require(race["selected"] == expected[0], "Final race winner drift")
    choice = reader.json(analysis.selection_path(root, "teacher", dataset, seed, condition))
    reader.expect(racepath, choice["joint_sha256"])
    require(choice["choices"]["JOINT"].endswith("/"+expected[0]), "JOINT selection drift")
    fixed_arms = ["CE"]+[arm for arm, spec in grid.items() if spec["support"] == 32]
    require([r["arm"] for r in choice["fixed_scores"]] == fixed_arms, "Fixed control pool drift")
    for arm, actual in zip(fixed_arms, choice["fixed_scores"]):
        spec = grid.get(arm)
        mean_lambda = float(weights(es["32"], spec).mean()) if spec else 0.
        score = prefix_score(log_cache[analysis.cell_id(dataset, seed, condition)+"/teacher/"+arm], 200,
                             arm, 32 if spec else 0, mean_lambda, total)
        equivalent_score(actual, score)
    selected_fixed = min(choice["fixed_scores"], key=rank)["arm"]
    require(choice["choices"]["FIXED_D1"].endswith("/"+selected_fixed), "Fixed D1 selection drift")
    for method, arm in (("CE", "CE"), ("OLD64", "OLD64"), ("LEGACY_SUPPORT32", "LEGACY_SUPPORT32"), ("DEEP_S32", "S32_L1.0")):
        require(choice["choices"][method].endswith("/"+arm), "Control route drift")
    frozen_at = datetime.fromisoformat(race["frozen"])
    # Fixed completion may reuse an already-completed survivor. Otherwise it follows JOINT freeze.
    for arm in fixed_arms[1:]:
        key = analysis.cell_id(dataset, seed, condition)+"/teacher/"+arm
        segments = reader.logs(Path(refs[key]["folder"])/"segments.jsonl")
        if arm not in [r["arm"] for r in race["rounds"][-1]["scores"]]:
            terminal = [s for s in segments if s["end_epoch"] == 200]
            require(len(terminal) == 1 and datetime.fromisoformat(terminal[0]["finished"]) >= frozen_at,
                    "Fixed control completed before JOINT decision freeze")
    return count_scores, count_snapshots


def audit_tables(root, reader):
    selected = analysis.csv_read(root/"selected_results.csv")
    paired = analysis.csv_read(root/"paired_differences.csv")
    gaps = analysis.csv_read(root/"transfer_gaps.csv")
    require(len(selected) == 540 and len(paired) == 540 and len(gaps) == 270, "Table endpoint count drift")
    lookup = {(r["dataset"], int(r["seed"]), r["condition"], r["stage"], r["method"]): r for r in selected}
    require(len(lookup) == 540, "Duplicate/missing selected endpoint")
    for r in paired:
        key = (r["dataset"], int(r["seed"]), r["condition"], r["stage"])
        a, b = lookup[key+(r["method"],)], lookup[key+(r["reference"],)]
        for m in analysis.METRICS:
            assert_close(100*(float(a[m])-float(b[m])), r[m+"_diff_pp"], "Paired difference", 1e-10)
    for r in gaps:
        key = (r["dataset"], int(r["seed"]), r["condition"])
        a, b = lookup[key+("student", r["method"])], lookup[key+("teacher", r["method"])]
        for m, field in (("test_accuracy", "accuracy"), ("test_macro_f1", "macro_f1")):
            assert_close(100*(float(a[m])-float(b[m])), r[f"student_minus_teacher_{field}_pp"], "Transfer gap", 1e-10)
    checks = 0
    for filename, source, ids, metric_names in (
        ("summary.csv", selected, ("dataset", "condition", "stage", "method"), analysis.METRICS),
        ("paired_summary.csv", paired, ("dataset", "condition", "stage", "method", "reference"), tuple(m+"_diff_pp" for m in analysis.METRICS)),
        ("gap_summary.csv", gaps, ("dataset", "condition", "method"), ("student_minus_teacher_accuracy_pp", "student_minus_teacher_macro_f1_pp"))):
        for r in analysis.csv_read(root/filename):
            matched = [s for s in source if all((k == "dataset" and r[k] == "ALL") or
                       (k == "condition" and r[k] == "NOISY_MACRO" and s[k] in analysis.CONDITIONS[1:]) or
                       s[k] == r[k] for k in ids)]
            require(int(r["n_seeds"]) == 3 and int(r["ddof"]) == 1, "Summary seed/SD denominator drift")
            values_by_seed = []
            for seed in range(3):
                cells = [s for s in matched if int(s["seed"]) == seed]
                require(len(cells) == int(r["cells_per_seed"]), "Summary seed cell count drift")
                values_by_seed.append(cells)
            for m in metric_names:
                values = np.asarray([np.mean([float(s[m]) for s in group]) for group in values_by_seed])
                assert_close(values.mean(), r[m+"_mean"], "Independent summary mean", 1e-10)
                assert_close(values.std(ddof=1), r[m+"_std"], "Independent sample SD", 1e-10)
                for seed in range(3):
                    assert_close(values[seed], r[f"{m}_seed{seed}"], "Independent macro seed", 1e-10)
            checks += 1
        reader.hash(root/filename)
    return selected, lookup, checks


def audit(root=None, workspace=analysis.WORKSPACE, threads=4, cpu_tolerance=5e-5):
    config = analysis.read(CODE/"config.json")
    analysis.validate_config(config)
    root, workspace = Path(root or config["runtime"]).resolve(), Path(workspace).resolve()
    old = workspace/"experiments"/config["historical"]
    reader = analysis.Reader()
    require(not (root/"run.lock").exists(), "Fitting lock held")
    require(reader.json(root/"fits_complete.json")["status"] == "FITTING_COMPLETE_TEST_DEFERRED", "Fitting incomplete")
    frozen, refs = check_freeze(root, reader)
    endpoints = analysis.check_choices(root, reader, refs)
    analysis.source_freeze(root, reader)
    done = reader.json(root/"analysis_complete.json")
    require(done["status"] == "ANALYSIS_COMPLETE" and done["selected"] == 540, "Analysis incomplete")
    reader.expect(root/"postprocessing_sources.json", done["source_manifest_sha256"])
    reader.expect(root/"postprocessing_inputs.json", done["input_manifest_sha256"])
    verify_manifest(root/"postprocessing_inputs.json", reader)
    for name, digest in done["outputs"].items():
        reader.expect(relative_input(root, name), digest)
    started = reader.json(root/"testing_started.json")
    reader.expect(root/"selections_frozen.json", started["selection_freeze_sha256"])
    require(datetime.fromisoformat(started["started"]) >= datetime.fromisoformat(frozen["frozen"]), "Testing before freeze")
    sources = verify_manifest(root/"sources.json", reader)
    verify_manifest(root/"inputs.json", reader)
    oldrefs_count, _ = historical_reuse(root, old, refs, reader)
    physical = reader.json(root/"physical_fits.json")
    require(sum(r["kind"] == "teacher" for r in physical.values()) == 1620, "Expected 1620 new physical teacher trajectories")
    require(all(not analysis.historical(r["dataset"], int(r["seed"])) for r in physical.values()), "Historical fitting repeated")
    require(len(refs) == oldrefs_count+len(physical), "Unaccounted logical/physical references")
    for name, r in physical.items():
        folder = relative_input(root, name)
        refkey = analysis.cell_id(r["dataset"], r["seed"], r["condition"])+f"/{r['kind']}/{r['arm']}"
        require(Path(refs[refkey]["folder"]).resolve() == folder and not refs[refkey]["historical"], "Physical/reference mismatch")
    selected, selected_lookup, summary_checks = audit_tables(root, reader)
    import torch
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    torch.set_num_threads(threads)
    model_paths = [Path(p) for p in sources if Path(p).name == "models.py" and Path(p).parent.name == "gblcss"]
    require(len(model_paths) == 1, "Ambiguous frozen original model source")
    models = import_absolute("gcn_extension_original_cpu_models", model_paths[0])
    paths = analysis.data_paths(workspace, reader)
    grid = candidates()
    counts = defaultdict(int)
    reloads, target_diagnostics = [], []
    initial_hashes, rng_hashes = {}, {}
    recalls = analysis.csv_read(root/"per_class_recall.csv")
    recall_lookup = {(r["dataset"], int(r["seed"]), r["condition"], r["stage"], r["method"], int(r["class_id"])): r for r in recalls}
    require(len(recall_lookup) == len(recalls), "Duplicate per-class recall")
    for dataset in analysis.DATASETS:
        x, y, edges = analysis.raw_graph(paths[dataset], reader)
        x, y, edges = x.float(), y.long(), edges.long()
        yn = y.numpy()
        classes = int(y.max())+1
        adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (len(x), len(x))).coalesce()
        adj_hash = state_hash(dict(indices=adj.indices(), values=adj.values()))
        for seed in range(3):
            geo = reader.tensor(analysis.geometry_path(workspace, dataset, seed))
            for key, expected in zip(("train", "valid", "test"), split_indices(len(x), seed)):
                require(np.array_equal(geo[key].numpy(), expected), "Independent ordered split mismatch")
            reader.expect(geo["h_path"], geo["h_sha256"])
            for stage in ("teacher", "student"):
                # Seed only the CPU generator. Allocating discarded MLP precedes original GCN.
                torch.random.default_generator.manual_seed(seed)
                args = (x.shape[1], 64, classes, 2, .5)
                if stage == "teacher":
                    models.MLP(*args)
                    model = models.GCN(*args)
                else:
                    model = models.MLP(*args)
                initial_hashes[dataset, seed, stage] = state_hash(model.state_dict())
                del model
            for condition in analysis.CONDITIONS:
                es, base, input_hash = prepare_cell(workspace, dataset, seed, condition, x, yn, geo, reader)
                cid = analysis.cell_id(dataset, seed, condition)
                cellrefs = {k: v for k, v in refs.items() if k.startswith(cid+"/")}
                require(sum(r["kind"] == "teacher" for r in cellrefs.values()) == 45, "Expected 45 teacher candidates/cell")
                logs_cache, result_cache = {}, {}
                for key, ref in cellrefs.items():
                    folder = Path(ref["folder"])
                    require(ref["historical"] is analysis.historical(dataset, seed), "Ref historical flag drift")
                    expected_root = old if ref["historical"] else root
                    require(folder.resolve().is_relative_to(expected_root.resolve()), "Artifact folder escapes owning runtime")
                    require(Path(ref["target_path"]).resolve().is_relative_to(expected_root.resolve()), "Target escapes owning runtime")
                    reader.expect(folder/"result.json", ref["result_sha256"])
                    result = reader.json(folder/"result.json")
                    result_cache[key] = result
                    for field in ("dataset", "seed", "condition", "arm", "partition"):
                        require(result[field] == ref[field], "Result/reference passport "+field)
                    require(result["epoch"] == ref["epoch"], "Ref epoch drift")
                    require(result["epoch"] in ((25, 75, 200) if ref["kind"] == "teacher" else (200,)), "Unexpected scientific epoch")
                    require(result["status"] == ("complete" if result["epoch"] == 200 else "prefix"), "Epoch/status drift")
                    expected_source = old/"sources.json" if ref["historical"] else root/"sources.json"
                    reader.expect(expected_source, result["source_sha256"])
                    reader.expect(folder/"probabilities.pt", result["probabilities_sha256"])
                    reader.expect(folder/"checkpoint.pt", result["checkpoint_sha256"])
                    cfg = reader.json(folder/"config.json")
                    for field in cfg:
                        require(result[field] == cfg[field], "Fit config/result drift: "+field)
                    require(result["backbone"] == ("GCN" if ref["kind"] == "teacher" else "DIST") and
                            result["checkpoint_format"] == "full" and result["q"] is None and
                            result["encoder_hash"] is None, "Unexpected backbone/warmstart")
                    require(result["optimizer"] == "Adam" and result["lr"] == .01 and result["weight_decay"] == .0005,
                            "Optimizer protocol drift")
                    require(result["selection"] == "first_strict_max_validation_accuracy", "Checkpoint selector drift")
                    require(result["input_hash"] == input_hash, "Training input hash mismatch")
                    require(result["initial_hash"] == ref["initial_hash"] == initial_hashes[dataset, seed, ref["kind"]],
                            "CPU original initialization hash mismatch")
                    rngkey = (dataset, seed, ref["kind"])
                    rng_hashes.setdefault(rngkey, result["initial_rng_hash"])
                    require(result["initial_rng_hash"] == ref["initial_rng_hash"] == rng_hashes[rngkey], "Initial RNG inconsistency")
                    if ref["kind"] == "teacher":
                        require(result["adjacency_hash"] == adj_hash, "Original coalesced COO drift")
                    target_root = old if ref["historical"] else root
                    require((target_root/result["target_ref"]).resolve() == Path(ref["target_path"]).resolve(), "Target path/reference drift")
                    target = reader.tensor(ref["target_path"])
                    require(state_hash(dict(target=target)) == result["target_hash"], "Stored target hash mismatch")
                    arm = ref["arm"] if ref["kind"] == "teacher" else "CE"
                    expected_target, partition = target_for(arm, es, base, grid)
                    require(ref["partition"] == partition, "Target partition mismatch")
                    require(target.shape == expected_target.shape and np.max(np.abs(target.numpy()-expected_target)) < 3e-7,
                            "Independent NumPy target mismatch")
                    logs = reader.logs(folder/"epochs.jsonl")
                    logs_cache[key] = logs
                    require([r["epoch"] for r in logs] == list(range(1, result["epoch"]+1)), "Epoch log continuity drift")
                    best = max(logs, key=lambda r: r["validation_correct"])
                    require(result["best_epoch"] == best["epoch"] and result["valid_correct"] == best["validation_correct"], "Strict max checkpoint drift")
                    assert_close(result["valid_nll"], best["validation_nll"], "Selected NLL")
                    assert_close(result["valid_macro_f1"], best["validation_macro_f1"], "Selected valid F1")
                    assert_close(result["validation"], result["valid_correct"]/result["valid_total"], "Validation ratio")
                    require(result["valid_total"] == len(geo["valid"]), "Validation total drift")
                    if not ref["historical"]:
                        pkey = Path(ref["folder"]).resolve().relative_to(root).as_posix()
                        pfit = physical[pkey]
                        require(pfit["epoch"] == result["epoch"] and pfit["seconds"] == result["seconds"], "Physical cost drift")
                    counts["physical_artifacts"] += 1
                    counts["target_checks"] += 1
                scores, snapshots = audit_race(root, old, dataset, seed, condition, refs, es, reader, logs_cache, result_cache)
                counts["race_checks"] += 1
                counts["prefix_scores"] += scores
                counts["prefix_snapshots"] += snapshots
                teacher_choice = reader.json(analysis.selection_path(root, "teacher", dataset, seed, condition))
                student_choice = reader.json(analysis.selection_path(root, "student", dataset, seed, condition))
                for method, key in teacher_choice["choices"].items():
                    diag = target_diagnostic(dataset, seed, condition, method, refs[key], es, grid)
                    diag.update(ref=key, target_hash=result_cache[key]["target_hash"])
                    target_diagnostics.append(diag)
                probability_cache = {}
                for endpoint in endpoints:
                    if (endpoint["dataset"], endpoint["seed"], endpoint["condition"]) != (dataset, seed, condition):
                        continue
                    stage, method, key = endpoint["stage"], endpoint["method"], endpoint["ref"]
                    ref, result = refs[key], result_cache[key]
                    folder = Path(ref["folder"])
                    if key not in probability_cache:
                        probability_cache[key] = reader.tensor(folder/"probabilities.pt")
                    p = probability_cache[key]
                    row = selected_lookup[dataset, seed, condition, stage, method]
                    require(row["ref"] == key and Path(row["folder"]).resolve() == folder.resolve(), "Selected table reference drift")
                    require(p.shape == (len(x), classes) and bool(torch.isfinite(p).all()) and
                            bool((p >= 0).all()) and bool((p <= 1).all()) and
                            bool(torch.allclose(p.sum(1), torch.ones(len(x)), atol=1e-6, rtol=0)), "Invalid probabilities")
                    te, vi = geo["test"], geo["valid"]
                    pred, truth = p[te].argmax(1).numpy(), yn[te.numpy()]
                    assert_close(accuracy_score(truth, pred), row["test_accuracy"], "Independent test accuracy")
                    assert_close(f1_score(truth, pred, labels=np.arange(classes), average="macro", zero_division=0),
                                 row["test_macro_f1"], "Independent test F1")
                    assert_close(accuracy_score(yn[vi.numpy()], p[vi].argmax(1)), row["validation"], "Independent validation accuracy")
                    valid_f1 = f1_score(yn[vi.numpy()], p[vi].argmax(1), labels=np.arange(classes), average="macro", zero_division=0)
                    assert_close(valid_f1, row["valid_macro_f1"], "Independent validation F1")
                    nll = float(-np.log(np.maximum(p[vi].numpy()[np.arange(len(vi)), yn[vi.numpy()]], np.finfo(np.float32).tiny)).mean())
                    assert_close(nll, row["valid_nll"], "Probability/selected checkpoint NLL", 5e-5)
                    recall = recall_score(truth, pred, labels=np.arange(classes), average=None, zero_division=0)
                    for cl in range(classes):
                        rec = recall_lookup[dataset, seed, condition, stage, method, cl]
                        number = int(np.count_nonzero(truth == cl))
                        require(int(rec["count"]) == number, "Recall class count drift")
                        if number:
                            assert_close(recall[cl], rec["recall"], "Independent class recall")
                        else:
                            require(rec["recall"] == "", "Absent class recall must be empty")
                        counts["recall_checks"] += 1
                    counts["metric_checks"] += 1
                    if stage == "student":
                        tkey = teacher_choice["choices"][method]
                        teacher_result = result_cache[tkey]
                        require(result["teacher_probability_sha256"] == teacher_result["probabilities_sha256"], "Student alias teacher SHA mismatch")
                        require(result["epoch"] == 200 and result["status"] == "complete", "Incomplete student")
                        if not ref["historical"]:
                            require(datetime.fromisoformat(result["finished"]) >= datetime.fromisoformat(analysis.read(root/"teachers_frozen.json")["frozen"]),
                                    "New student predates global teacher freeze")
                        for other in METHODS:
                            if student_choice["choices"][other] == key:
                                other_teacher = result_cache[teacher_choice["choices"][other]]
                                require(other_teacher["probabilities_sha256"] == result["teacher_probability_sha256"], "Unsafe logical student alias")
                    if method != "JOINT":
                        continue
                    ck = reader.tensor(folder/"checkpoint.pt")
                    require(ck["epoch"] == result["best_epoch"] and ck["kind"] == result["backbone"], "Checkpoint identity drift")
                    model = (models.GCN if stage == "teacher" else models.MLP)(x.shape[1], 64, classes, 2, .5)
                    model.load_state_dict(ck["model"], strict=True)
                    model.eval()
                    with torch.no_grad():
                        q = (model(x, adj) if stage == "teacher" else model(x)).softmax(1)
                    error = float((q-p).abs().max())
                    require(error < cpu_tolerance, "CPU reload probability mismatch: "+str(folder)+f" error={error}")
                    require(torch.equal(q[te].argmax(1), p[te].argmax(1)) and torch.equal(q[vi].argmax(1), p[vi].argmax(1)),
                            "CPU reload prediction mismatch")
                    reloads.append(dict(dataset=dataset, seed=seed, condition=condition, stage=stage,
                                        ref=key, max_probability_error=error, test_valid_predictions_exact=True))
                    del model
                print(f"AUDIT {cid}: metrics=12, CPU JOINT reloads=2", flush=True)
    require(counts["metric_checks"] == 540 and len(reloads) == 90 and counts["race_checks"] == 45,
            "Audit coverage incomplete")
    require(counts["recall_checks"] == len(recalls), "Recall coverage incomplete")
    require(len(target_diagnostics) == 270, "Target diagnostic coverage incomplete")
    smoke = reader.json(root/"smoke_complete.json")
    require(smoke["old_initialization_and_two_epoch_bridge_exact"] is True and smoke["trajectories"] <= 6,
            "Original seed0 bridge missing")
    require(all(r["resume_exact"] for r in smoke["checks"] if r["dataset"] == "cora"), "Seed1 continuation smoke missing")
    costs = analysis.csv_read(root/"costs.csv")
    recomputed = analysis.costs(root, old, reader, endpoints)
    require(len(costs) == len(recomputed), "Cost table scope drift")
    cost_lookup = {(r["dataset"], int(r["seed"]), r["condition"]): r for r in costs}
    for r in recomputed:
        recorded = cost_lookup[r["dataset"], r["seed"], r["condition"]]
        for key, value in r.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                assert_close(recorded[key], value, "Independent physical/logical cost", 1e-8)
    reader.verify()
    analysis.save(root/"cpu_audit_inputs.json", reader.inputs)
    analysis.csv_save(root/"target_diagnostics.csv", target_diagnostics)
    audit_report = ["# CPU Audit", "", "Status: passed", "",
                    "- 540 logical endpoint metrics, 90 JOINT teacher/student CPU reloads.",
                    "- 45 causal races, independent NumPy LOO targets and validation-only ranking.",
                    "- Historical nine cells and source/input freeze hashes verified read-only.",
                    "- Paired differences and seed-first macro/sample SD independently recomputed.",
                    "- target_diagnostics.csv: all 270 teacher routes, actual LOO support/coverage,",
                    "  lambda/alpha/raw mass/effective mass/modified rate, selected arm and target hash.",
                    "- requested_support records the requested threshold; CE/OLD64 leave it empty.",
                    "  OLD64 has ball_count_cap=64. actual_loo_support_* records observed LOO vote counts.",
                    "- modified_rate uses effective transferred mass > 1e-8; CE effective mass is zero.",
                    "- mean_alpha and mean_m describe the partition evidence before route strength.",
                    "- costs.csv separates historical reuse, logical students and new physical fits.",
                    "- GPU RNG bytes are checked as stored continuation state; GPU execution remains in fitting smoke evidence."]
    (root/"CPU_AUDIT.md").write_text("\n".join(audit_report)+"\n", encoding="utf-8")
    result = dict(status="passed", finished=analysis.now(), device="cpu", **counts,
                  summary_checks=summary_checks, cpu_reload_count=90, cpu_reloads=reloads,
                  max_cpu_probability_error=max(r["max_probability_error"] for r in reloads),
                  old_reference_checks=oldrefs_count, n_seeds=3, ddof=1,
                  logical_students=270, historical_logical_students=54, new_logical_students=216,
                  new_teacher_trajectories=1620, new_physical_students=sum(r["kind"] == "student" for r in physical.values()),
                  fit_sources_unchanged=True, historical_immutable=True, all_negative_results_retained=True,
                  target_diagnostics_rows=270, target_diagnostics="target_diagnostics.csv",
                  target_diagnostics_sha256=analysis.sha(root/"target_diagnostics.csv"),
                  audit_source_sha256=analysis.sha(__file__), cpu_audit_inputs_sha256=analysis.sha(root/"cpu_audit_inputs.json"))
    analysis.save(root/"audit.json", result)
    analysis.csv_save(root/"cpu_reloads.csv", reloads)
    final = dict(status="COMPLETE_AUDITED", audit="passed", finished=analysis.now(),
                 selection_freeze_sha256=analysis.sha(root/"selections_frozen.json"),
                 analysis_complete_sha256=analysis.sha(root/"analysis_complete.json"), audit_sha256=analysis.sha(root/"audit.json"),
                 logical_students=270, new_teacher_trajectories=1620, new_physical_students=result["new_physical_students"],
                 metric_checks=540, cpu_joint_reloads=90, target_diagnostics="target_diagnostics.csv",
                 target_diagnostics_sha256=analysis.sha(root/"target_diagnostics.csv"))
    analysis.save(root/"complete.json", final)
    if root != CODE:
        for name in ("audit.json", "cpu_reloads.csv", "target_diagnostics.csv", "CPU_AUDIT.md", "complete.json"):
            (CODE/name).write_bytes((root/name).read_bytes())
    print(json.dumps(final, indent=2), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--runtime", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=analysis.WORKSPACE)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        audit(args.root, args.workspace, args.threads)
    except Exception as exc:
        # A previous success marker must never survive a failed re-audit as current success.
        root = Path(args.root or analysis.read(CODE/"config.json")["runtime"])
        if (root/"analysis_complete.json").exists():
            analysis.save(root/"audit_failure.json", dict(status="AUDIT_FAILED", time=analysis.now(), error=str(exc)))
            if (root/"complete.json").exists():
                analysis.save(root/"complete.json", dict(status="AUDIT_FAILED", time=analysis.now(), error=str(exc)))
        raise


if __name__ == "__main__":
    main()
