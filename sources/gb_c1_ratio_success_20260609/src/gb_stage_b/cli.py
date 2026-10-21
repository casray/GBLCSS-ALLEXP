from __future__ import annotations

import argparse
from pathlib import Path

from gb_stage_b.config import load_config
from gb_stage_b.report import summarize_results
from gb_stage_b.train import rebuild_balls_from_run, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gb-stage-b")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--method")
    run.add_argument("--ssl-objective", dest="ssl_objective")
    run.add_argument("--experiment-group", dest="experiment_group")
    run.add_argument("--variant")
    run.add_argument("--device")
    run.add_argument("--edge-drop-1", type=float, dest="edge_drop_1")
    run.add_argument("--edge-drop-2", type=float, dest="edge_drop_2")
    run.add_argument("--feat-mask-1", type=float, dest="feat_mask_1")
    run.add_argument("--feat-mask-2", type=float, dest="feat_mask_2")
    run.add_argument("--ball-algorithm", dest="ball_algorithm")
    run.add_argument("--gain-split-min-ball-size", type=int, dest="gain_split_min_ball_size")
    run.add_argument("--gain-split-max-depth", type=int, dest="gain_split_max_depth")
    run.add_argument("--gain-split-min-gain", type=float, dest="gain_split_min_gain")
    run.add_argument("--results-dir", dest="results_dir")
    run.add_argument("--seed", type=int)
    run.add_argument("--epochs", type=int)
    run.add_argument("--warmup-epochs", type=int, dest="warmup_epochs")
    run.add_argument("--noise-rate", type=float, dest="noise_rate")
    run.add_argument("--phase2-epochs", type=int, dest="phase2_epochs")
    run.add_argument("--phase2-lr", type=float, dest="phase2_lr")
    run.add_argument("--lambda-intra", type=float, dest="lambda_intra")
    run.add_argument("--lambda-inter", type=float, dest="lambda_inter")
    run.add_argument("--lambda-sce", type=float, dest="lambda_sce")
    run.add_argument("--phase2-ball-refresh", type=int, dest="phase2_ball_refresh")
    run.add_argument("--phase2-label-smoothing", type=float, dest="phase2_label_smoothing")
    run.add_argument("--gate-tau-cov", type=int, dest="gate_tau_cov")
    run.add_argument("--gate-tau-conf", type=float, dest="gate_tau_conf")
    run.add_argument("--gate-alpha", type=float, dest="gate_alpha")
    run.add_argument("--weighted-k-ref", type=int, dest="weighted_k_ref")
    run.add_argument("--weighted-floor", type=float, dest="weighted_floor")
    run.add_argument("--phase2-supervised-loss", dest="phase2_supervised_loss")
    run.add_argument("--phase2-sce-alpha", type=float, dest="phase2_sce_alpha")
    run.add_argument("--phase2-sce-beta", type=float, dest="phase2_sce_beta")
    run.add_argument("--split-mode", dest="split_mode")
    run.add_argument("--train-ratio", type=float, dest="train_ratio")
    run.add_argument("--val-ratio", type=float, dest="val_ratio")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--results", required=True)
    rebuild = sub.add_parser("rebuild-balls")
    rebuild.add_argument("--run-dir", required=True)
    rebuild.add_argument("--min-ball-size", type=int, required=True)
    rebuild.add_argument("--max-depth", type=int, required=True)
    rebuild.add_argument("--min-split-gain", type=float, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(
            args.config,
            overrides={
                "dataset": args.dataset,
                "method": args.method,
                "ssl_objective": args.ssl_objective,
                "experiment_group": args.experiment_group,
                "variant": args.variant,
                "device": args.device,
                "edge_drop_1": args.edge_drop_1,
                "edge_drop_2": args.edge_drop_2,
                "feat_mask_1": args.feat_mask_1,
                "feat_mask_2": args.feat_mask_2,
                "ball_algorithm": args.ball_algorithm,
                "gain_split_min_ball_size": args.gain_split_min_ball_size,
                "gain_split_max_depth": args.gain_split_max_depth,
                "gain_split_min_gain": args.gain_split_min_gain,
                "results_dir": args.results_dir,
                "seed": args.seed,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "noise_rate": args.noise_rate,
                "phase2_epochs": args.phase2_epochs,
                "phase2_lr": args.phase2_lr,
                "lambda_intra": args.lambda_intra,
                "lambda_inter": args.lambda_inter,
                "lambda_sce": args.lambda_sce,
                "phase2_ball_refresh": args.phase2_ball_refresh,
                "phase2_label_smoothing": args.phase2_label_smoothing,
                "gate_tau_cov": args.gate_tau_cov,
                "gate_tau_conf": args.gate_tau_conf,
                "gate_alpha": args.gate_alpha,
                "weighted_k_ref": args.weighted_k_ref,
                "weighted_floor": args.weighted_floor,
                "phase2_supervised_loss": args.phase2_supervised_loss,
                "phase2_sce_alpha": args.phase2_sce_alpha,
                "phase2_sce_beta": args.phase2_sce_beta,
                "split_mode": args.split_mode,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": None if args.train_ratio is None or args.val_ratio is None else 1.0 - args.train_ratio - args.val_ratio,
            },
        )
        out = run_experiment(cfg)
        print(f"wrote run artifacts to {out}")
    elif args.command == "summarize":
        out = summarize_results(Path(args.results))
        print(f"wrote summary to {out}")
    elif args.command == "rebuild-balls":
        out = rebuild_balls_from_run(
            args.run_dir,
            min_ball_size=args.min_ball_size,
            max_depth=args.max_depth,
            min_split_gain=args.min_split_gain,
        )
        print(f"wrote ball ablation metrics to {out}")


if __name__ == "__main__":
    main()
