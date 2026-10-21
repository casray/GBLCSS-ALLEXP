from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
import yaml

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "GB-LCSS_GitHub/src"))
from gblcss.data import GraphData, percent_split_indices, inject_train_label_noise, set_seed
from gblcss.models import GCN, MLP
from gblcss.targets import build_leave_one_out_targets
from gblcss.training import soft_cross_entropy
from adapters import tfr_classes, erase_classes, erase_preprocess, fit_gpu_logistic, probe_probabilities

DATASETS = ["cora", "citeseer", "corafull", "coauthorcs", "amazonpho"]
NOISYGL = PROJECT / "external/NoisyGL"
DATA_FILES = {
    "cora": NOISYGL / "data/Cora/processed/data.pt",
    "citeseer": NOISYGL / "data/CiteSeer/processed/data.pt",
    "corafull": PROJECT / "gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt",
    "coauthorcs": NOISYGL / "data/CS/processed/data.pt",
    "amazonpho": NOISYGL / "data/Photo/processed/data.pt",
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def load_data(dataset, rate):
    stored = torch.load(DATA_FILES[dataset], map_location="cpu", weights_only=False)
    raw = Data.from_dict(stored[0]) if isinstance(stored[0], dict) else stored[0]
    n = raw.x.shape[0]
    indices = percent_split_indices(n, 0)
    masks = []
    for index in indices:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    data = GraphData(dataset, raw.x.float(), raw.edge_index.long(), raw.y.long(),
                     *masks, train_index=torch.tensor(indices[0]),
                     val_index=torch.tensor(indices[1]), test_index=torch.tensor(indices[2]))
    inject_train_label_noise(data, "uniform", rate, 0)
    split_hash = fingerprint(*indices)
    noise_hash = fingerprint(data.y, data.noisy_y)
    reference_path = WORKSPACE / f"experiments/main_protocol_replay_20260922/matrix/{dataset}/default/seed0.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    assert split_hash == reference["split_hash"], "main-experiment split mismatch"
    if rate == .4:
        assert noise_hash == reference["noise_hash"], "main-experiment noise mismatch"
    else:
        rows = []
        for path in (NOISYGL / "runs").glob("mlp*/teacher_u0p6_seeds0_4.csv"):
            with path.open(encoding="utf-8-sig") as handle:
                rows.extend(row for row in csv.DictReader(handle)
                            if row["dataset"] == dataset and row["seed"] == "0")
        if not rows:
            # The citation datasets precede the breadth extension.
            path = NOISYGL / "runs/mlp57_xteacher_loo_cleanup_20260813/xteacher_uniform0p6_seeds0_4.csv"
            with path.open(encoding="utf-8-sig") as handle:
                rows.extend(row for row in csv.DictReader(handle)
                            if row.get("dataset") == dataset and row.get("seed") == "0"
                            and row.get("noise_scope") == "train")
        if not rows:
            raise ValueError(f"no historical noise fingerprint for {dataset} u0.6")
        assert all(row["split_hash"] == split_hash and row["noise_hash"] == noise_hash for row in rows), dataset
    assert torch.equal(data.noisy_y[data.val_mask], data.y[data.val_mask])
    assert torch.equal(data.noisy_y[data.test_mask], data.y[data.test_mask])
    archive = ROOT / "protocol_cache" / f"{dataset}_u{rate}.pt"
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        torch.save(dict(train_index=data.train_index, val_index=data.val_index,
                        test_index=data.test_index, clean=data.y, noisy=data.noisy_y), archive)
    audit = dict(split_hash=split_hash, noise_hash=noise_hash, seed=0,
                 modified_train_count=int((data.noisy_y[data.train_mask] != data.y[data.train_mask]).sum()),
                 modified_val_count=0, modified_test_count=0, features="raw main-experiment attributes",
                 data_file=str(DATA_FILES[dataset]), num_nodes=n, num_classes=data.num_classes)
    return data, audit


def gpu_data(data):
    for key, value in vars(data).items():
        if isinstance(value, torch.Tensor):
            setattr(data, key, value.cuda())
    return data


def metrics(probabilities, data):
    pred = probabilities.argmax(1)
    return dict(train=float((pred[data.train_mask] == data.noisy_y[data.train_mask]).float().mean()),
                valid=float((pred[data.val_mask] == data.y[data.val_mask]).float().mean()),
                test=float((pred[data.test_mask] == data.y[data.test_mask]).float().mean()))


def log_epoch(handle, phase, epoch, loss, valid):
    handle.write(json.dumps(dict(phase=phase, epoch=epoch, loss=float(loss), valid=valid)) + "\n")
    handle.flush()
    if epoch == 1 or epoch % 25 == 0:
        print(f"{phase} epoch={epoch} loss={float(loss):.5f} val={valid:.4f}", flush=True)


def train_reference(data, method, epochs, handle):
    # Preserve the original student-before-teacher initialization order.
    MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    teacher = GCN(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    edges = data.edge_index
    adjacency = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device="cuda"),
                                       (data.num_nodes, data.num_nodes)).coalesce()
    target = F.one_hot(data.noisy_y, data.num_classes).float()
    if method == "gblcss":
        cache = WORKSPACE / f"experiments/main_protocol_replay_20260922/matrix/{data.name}/default/seed0.npz"
        assignment = np.load(cache)["assignment"]
        ref = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8-sig"))
        canonical = np.ascontiguousarray(assignment, dtype="<i8")
        header = f"dtype={canonical.dtype.str};shape={canonical.shape[0]};order=C\n".encode("ascii")
        assert hashlib.sha256(header + canonical.tobytes()).hexdigest() == ref["ball_assignment_hash"]
        target = build_leave_one_out_targets(data.noisy_y, assignment, data.train_index,
                                             data.num_classes, beta=.5, min_train_per_ball=2).soft_target
    optimizer = torch.optim.Adam(teacher.parameters(), lr=.01, weight_decay=.0005)
    best_val, best_epoch, best_state = -1, -1, None
    for epoch in range(1, epochs + 1):
        teacher.train()
        optimizer.zero_grad()
        logits = teacher(data.x, adjacency)
        loss = soft_cross_entropy(logits[data.train_index], target[data.train_index])
        loss.backward()
        optimizer.step()
        teacher.eval()
        with torch.no_grad():
            logits = teacher(data.x, adjacency)
            valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
        if valid >= best_val:
            best_val, best_epoch, best_state = valid, epoch, deepcopy(teacher.state_dict())
        log_epoch(handle, "teacher", epoch, loss.item(), valid)
    teacher.load_state_dict(best_state)
    teacher.eval()
    with torch.no_grad():
        probabilities = teacher(data.x, adjacency).softmax(1)
    return teacher, probabilities, best_epoch, dict(hidden=64, dropout=.5, lr=.01, weight_decay=.0005, epochs=epochs)


def train_tfr(data, epochs, handle):
    conf = yaml.safe_load((ROOT / "vendor/tfr/config/tfr_gcn/tfr_gcn_dblp.yaml").read_text())
    model_conf, train_conf = conf["model"], conf["training"]
    GCNClass, Decoder = tfr_classes()
    model = GCNClass(data.x.shape[1], model_conf["n_hidden"], data.num_classes,
                     n_layers=model_conf["n_layer"], dropout=model_conf["dropout"],
                     norm_info=model_conf["norm_info"], act=model_conf["act"],
                     input_layer=False, output_layer=False).cuda()
    decoder = Decoder(data.num_classes, model_conf["n_hidden"], data.x.shape[1],
                      model_conf["decoder_dropout"], model_conf["att_softmax"]).cuda()
    encoder = torch.nn.Linear(data.x.shape[1], model_conf["n_hidden"]).cuda()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()) + list(encoder.parameters()),
                                 lr=train_conf["lr"], weight_decay=train_conf["weight_decay"])
    loop_edges = data.edge_index
    # Upstream DBLP config has add_self_loop=false. GCN and decoder add their own loops.
    adjacency = torch.sparse_coo_tensor(loop_edges, torch.ones(loop_edges.shape[1], device="cuda"),
                                       (data.num_nodes, data.num_nodes)).coalesce()
    loop_edges = adjacency.indices()
    coefficient, best_val, best_state, best_epoch = 0., -1., None, -1
    def objective(logits, mask, training):
        pseudo = logits.softmax(1)
        dropped = dropout_edge(loop_edges, p=model_conf["beta_dropout_ratio"])[0]
        reconstructed, confidence = decoder(pseudo, dropped)
        selected = confidence.squeeze().argsort(descending=True)[:int(data.num_nodes * model_conf["pseudo_ratio"])]
        if model_conf["use_recon_softmax"]:
            reconstructed = reconstructed.softmax(1)
        recon = F.cross_entropy(reconstructed, data.x)
        pseudo_loss = F.cross_entropy(logits[selected], pseudo[selected].detach())
        supervised = F.cross_entropy(logits[mask], data.noisy_y[mask])
        return (1 - coefficient) * supervised + model_conf["beta"] * recon + coefficient * pseudo_loss
    for epoch in range(1, epochs + 1):
        model.train()
        # Upstream only switches the GCN back to train after validation.
        decoder.train(epoch == 1)
        optimizer.zero_grad()
        logits = model(data.x, adjacency)
        loss = objective(logits, data.train_index, True)
        if not torch.isfinite(loss):
            raise ValueError("non-finite TFR objective")
        coefficient = float((logits[data.train_index].argmax(1) == data.noisy_y[data.train_index]).float().mean())
        loss.backward()
        optimizer.step()
        model.eval()
        decoder.eval()
        with torch.no_grad():
            logits = model(data.x, adjacency)
            objective(logits, data.val_index, False)
            # Consume upstream test-pass augmentation RNG without consulting test labels.
            objective(logits, data.val_index, False)
            valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
        if valid > best_val:
            best_val, best_epoch = valid, epoch
            best_state = deepcopy(model.state_dict())
        log_epoch(handle, "teacher", epoch, loss.item(), valid)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(data.x, adjacency).softmax(1)
    return model, probabilities, best_epoch, dict(author_preset="DBLP", **conf, epochs=epochs,
                                                 graph_self_loops=False, adapter_revision=2,
                                                 decoder_mode="preserved upstream train/eval behavior")


def erase_settings(name):
    presets = {"cora": dict(gam2=2, eps=.05, alpha=.6, beta=.6, T=5),
               "citeseer": dict(gam2=2, eps=.4, alpha=.6, beta=.7, T=4),
               "corafull": dict(gam2=2, eps=.01, alpha=.7, beta=.7, T=2)}
    source = name if name in presets else "cora"
    return dict(**presets[source], author_preset=source, gam1=1, hidden=256, embedding=512,
                heads=8, layers=2, dropout=.5, lr=.001, weight_decay=.0005, patience=150)


def train_erase(data, epochs, handle):
    conf = erase_settings(data.name)
    ModelType, LossClass = erase_classes()
    model = ModelType.GAT.get_model(data.x.shape[1], conf["hidden"], conf["embedding"],
                                    conf["layers"], conf["heads"], conf["dropout"], "cuda",
                                    False, False, False, False).cuda()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"], amsgrad=True)
    labels, adjacency = erase_preprocess(data, conf["alpha"], conf["T"])
    loss_fn = LossClass(conf["gam1"], conf["gam2"], conf["eps"], corafull=True)
    best_val, best_state, best_probe, best_epoch = -1., None, None, -1
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss, _, refined = loss_fn(out, data, adjacency, labels, conf["alpha"], conf["beta"], conf["T"], data.train_mask)
        if not torch.isfinite(loss):
            raise ValueError("non-finite ERASE objective")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        probe = fit_gpu_logistic(out[data.train_mask], refined[data.train_mask])
        probabilities = probe_probabilities(out, probe, data.num_classes)
        valid = float((probabilities[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean())
        if valid > best_val:
            best_val, best_epoch, best_state = valid, epoch, deepcopy(model.state_dict())
            best_probe = tuple(item.clone() for item in probe)
        log_epoch(handle, "teacher", epoch, loss.item(), valid)
        if epoch - best_epoch > conf["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    probabilities = probe_probabilities(out, best_probe, data.num_classes)
    return model, probabilities, best_epoch, dict(**conf, epochs=epochs, executed_epochs=epoch,
                                                 probe_state=[item.cpu() for item in best_probe])


def train_student(data, probabilities, epochs, handle):
    set_seed(0)
    student = MLP(data.x.shape[1], 64, data.num_classes, 2, .5).cuda()
    optimizer = torch.optim.Adam(student.parameters(), lr=.01, weight_decay=.0005)
    best_val, best_epoch, best_state = -1., -1, None
    probabilities = probabilities.detach()
    for epoch in range(1, epochs + 1):
        student.train()
        optimizer.zero_grad()
        loss = soft_cross_entropy(student(data.x), probabilities)
        loss.backward()
        optimizer.step()
        student.eval()
        with torch.no_grad():
            logits = student(data.x)
            valid = float((logits[data.val_index].argmax(1) == data.y[data.val_index]).float().mean())
        if valid > best_val:
            best_val, best_epoch, best_state = valid, epoch, deepcopy(student.state_dict())
        log_epoch(handle, "student", epoch, loss.item(), valid)
    student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        probabilities = student(data.x).softmax(1)
    return student, probabilities, best_epoch


def run_one(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; CPU fallback is forbidden")
    torch.set_num_threads(4)
    folder = ROOT / ("smoke" if args.smoke else "runs") / args.dataset / f"uniform{args.rate}" / args.method
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budget = args.epochs or {"tfr": 500, "erase": 400, "gblcss": 200, "gcn_ce": 200}[args.method]
    student_epochs = args.epochs or 200
    result = dict(dataset=args.dataset, noise_rate=args.rate, method=args.method, seed=0,
                  smoke=args.smoke, device=torch.cuda.get_device_name(0), status="running")
    write_json(folder / "result.json", result)
    try:
        data, audit = load_data(args.dataset, args.rate)
        result.update(audit)
        write_json(folder / "protocol.json", audit)
        data = gpu_data(data)
        set_seed(0)
        torch.cuda.reset_peak_memory_stats()
        with (folder / "epochs.jsonl").open("w", encoding="utf-8") as handle:
            if args.method in {"gblcss", "gcn_ce"}:
                teacher, probs, epoch, settings = train_reference(data, args.method, budget, handle)
            elif args.method == "tfr":
                teacher, probs, epoch, settings = train_tfr(data, budget, handle)
            else:
                teacher, probs, epoch, settings = train_erase(data, budget, handle)
            torch.cuda.synchronize()
            result.update(teacher_seconds=time.perf_counter() - started, teacher_best_epoch=epoch)
            result.update({f"teacher_{key}": value for key, value in metrics(probs, data).items()})
            assert torch.isfinite(probs).all() and (probs.sum(1) - 1).abs().max() < 1e-5
            torch.save(dict(state=teacher.state_dict(), settings=settings), folder / "teacher.pt")
            torch.save(probs.cpu(), folder / "teacher_probabilities.pt")
            if "probe_state" in settings:
                settings = {key: value for key, value in settings.items() if key != "probe_state"}
            write_json(folder / "config.json", dict(teacher=settings, student=dict(hidden=64, layers=2,
                       dropout=.5, lr=.01, weight_decay=.0005, epochs=student_epochs, temperature=1,
                       scope="all nodes", initialization_seed=0, rng_reset=True)))
            begin_student = time.perf_counter()
            student, student_probs, student_epoch = train_student(data, probs, student_epochs, handle)
        torch.cuda.synchronize()
        result.update(student_seconds=time.perf_counter() - begin_student, student_best_epoch=student_epoch,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      wall_seconds=time.perf_counter() - started, status="ok")
        result.update({f"student_{key}": value for key, value in metrics(student_probs, data).items()})
        torch.save(student.state_dict(), folder / "student.pt")
        torch.save(student_probs.cpu(), folder / "student_probabilities.pt")
        write_json(folder / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), wall_seconds=time.perf_counter() - started)
        write_json(folder / "result.json", result)
        raise


def run_grid(args):
    env = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
               pyg=__import__("torch_geometric").__version__, hip=torch.version.hip,
               os=platform.platform(), gpu=torch.cuda.get_device_name(0), threads=4)
    for name in ("erase", "tfr"):
        env[name + "_commit"] = subprocess.check_output(["git", "-C", str(ROOT / "vendor" / name),
                                                       "rev-parse", "HEAD"], text=True).strip()
    env["gblcss_source_hashes"] = {str(path.relative_to(WORKSPACE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (WORKSPACE / "GB-LCSS_GitHub/src/gblcss").glob("*.py")}
    write_json(ROOT / "environment.json", env)
    for dataset in args.datasets.split(","):
        for rate in (.4, .6):
            for method in args.methods.split(","):
                folder = ROOT / "runs" / dataset / f"uniform{rate}" / method
                if (folder / "result.json").exists():
                    result = json.loads((folder / "result.json").read_text())
                    if result["status"] == "ok":
                        continue
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--dataset", dataset,
                           "--rate", str(rate), "--method", method]
                print("START", dataset, rate, method, flush=True)
                with (folder / "console.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                print("DONE", dataset, rate, method, "exit", completed.returncode, flush=True)
                if (folder / "result.json").exists():
                    result = json.loads((folder / "result.json").read_text())
                    print(json.dumps({key: result.get(key) for key in ["status", "teacher_test", "student_test", "wall_seconds"]}), flush=True)
                summarize()


def summarize():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((ROOT / "runs").glob("*/*/*/result.json"))]
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with (ROOT / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default="gcn_ce,gblcss,tfr,erase")
    parser.add_argument("--dataset", choices=DATASETS, default="cora")
    parser.add_argument("--rate", type=float, choices=[.4, .6], default=.4)
    parser.add_argument("--method", choices=["gcn_ce", "gblcss", "tfr", "erase"], default="tfr")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_grid(args) if args.grid else run_one(args)
