import csv
import os
import time
from copy import deepcopy

import nni
import torch
import torch.nn.functional as F

from predictor.Base_Predictor import Predictor
from predictor.FrozenGraceGB_GCN_Predictor import concat_l2_normalized_embeddings
from predictor.module.GNNs import GCN
from utils.frozen_grace_targets import load_frozen_embeddings, partition_purity_diagnostics
from utils.gbselect import build_gbselect_selection, gbselect_diagnostics
from utils.ordinary_gb import build_ordinary_granular_balls, build_random_size_matched_assignment


class _GBSelectBase(Predictor):
    selection_mode = "intersection"
    random_control = False

    def method_init(self, conf, data):
        self.dataset_name = data.name
        embedding_path = self._conf_value("embedding_path", None)
        if embedding_path is None:
            raise ValueError("gb.embedding_path is required for GBSelect")
        self.embeddings = load_frozen_embeddings(embedding_path, expected_nodes=self.n_nodes)
        self.feats = concat_l2_normalized_embeddings(self.feats, self.embeddings).to(self.device)
        in_channels = int(self.feats.shape[1])
        conf.model["n_feat"] = in_channels
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=conf.model["n_hidden"],
            out_channels=conf.model["n_classes"],
            n_layers=conf.model["n_layer"],
            dropout=conf.model["dropout"],
            norm_info=conf.model["norm_info"],
            act=conf.model["act"],
            input_layer=conf.model["input_layer"],
            output_layer=conf.model["output_layer"],
        ).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        self.selected_train_mask = None
        self.gbselect_diagnostics = {}
        self.best_epoch = -1

    def _gb_conf(self):
        return getattr(self.conf, "gb", {})

    def _conf_value(self, key, default):
        gb_conf = self._gb_conf()
        if isinstance(gb_conf, dict):
            value = gb_conf.get(key, default)
        else:
            value = getattr(gb_conf, key, default)
        return default if value is None else value

    def _warmup_probs(self):
        model = GCN(
            in_channels=int(self.feats.shape[1]),
            hidden_channels=self.conf.model["n_hidden"],
            out_channels=self.n_classes,
            n_layers=self.conf.model["n_layer"],
            dropout=self.conf.model["dropout"],
            norm_info=self.conf.model["norm_info"],
            act=self.conf.model["act"],
            input_layer=self.conf.model["input_layer"],
            output_layer=self.conf.model["output_layer"],
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.conf.training["lr"],
            weight_decay=self.conf.training["weight_decay"],
        )
        warmup_epochs = int(self._conf_value("warmup_epochs", 10))
        window_start = int(self._conf_value("warmup_window_start", 5))
        window_end = int(self._conf_value("warmup_window_end", 10))
        snapshots = []
        for epoch in range(1, warmup_epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(self.feats, self.adj)
            loss = F.cross_entropy(logits[self.train_mask], self.noisy_label[self.train_mask].long())
            loss.backward()
            optimizer.step()
            if window_start <= epoch <= window_end:
                model.eval()
                with torch.no_grad():
                    snapshots.append(F.softmax(model(self.feats, self.adj), dim=1).detach())
        if not snapshots:
            raise ValueError("GBSelect warmup window produced no snapshots")
        return torch.stack(snapshots, dim=0).mean(dim=0)

    def _make_selected_train_mask(self):
        seed = int(getattr(self, "seed", torch.initial_seed() % (2 ** 32 - 1)))
        gb_result = build_ordinary_granular_balls(
            self.embeddings,
            max_balls=self._conf_value("max_balls", 64),
            min_ball_size=self._conf_value("min_ball_size", 32),
            split_improvement=self._conf_value("split_improvement", 0.05),
            random_state=seed,
        )
        assignment = gb_result.assignment
        if self.random_control:
            assignment = build_random_size_matched_assignment(assignment, random_state=seed)
        warmup_probs = self._warmup_probs()
        node_idx = torch.arange(int(self.noisy_label.shape[0]), device=self.device)
        s_gcn = 1.0 - warmup_probs[node_idx, self.noisy_label.long()].to(self.device)
        selected, s_ball = build_gbselect_selection(
            noisy_label=self.noisy_label,
            assignment=assignment,
            train_mask=self.train_mask,
            n_classes=self.n_classes,
            s_gcn=s_gcn,
            keep_ratio=self._conf_value("keep_ratio", 0.6),
            mode=self.selection_mode,
            min_train_per_ball=self._conf_value("min_train_per_ball", 2),
            clean_label=self.clean_label,
        )
        if int(selected.sum().item()) == 0:
            raise ValueError("GBSelect selected no train nodes")
        diag = gbselect_diagnostics(selected, self.noisy_label, self.clean_label, self.train_mask)
        purity = partition_purity_diagnostics(
            assignment=assignment,
            noisy_label=self.noisy_label,
            clean_label=self.clean_label,
            train_mask=self.train_mask,
            n_classes=self.n_classes,
            min_train_per_ball=self._conf_value("min_train_per_ball", 2),
        )
        self.selected_train_mask = selected.to(self.device)
        self.gb_assignment = assignment
        self.gbselect_diagnostics = {
            "selection_mode": self.selection_mode,
            "random_control": int(self.random_control),
            "keep_ratio": float(self._conf_value("keep_ratio", 0.6)),
            "warmup_epochs": int(self._conf_value("warmup_epochs", 10)),
            "warmup_window_start": int(self._conf_value("warmup_window_start", 5)),
            "warmup_window_end": int(self._conf_value("warmup_window_end", 10)),
            "num_balls": int(gb_result.summary["num_balls"]),
            "mean_ball_size": float(gb_result.summary["mean_ball_size"]),
            "mean_s_ball_selected": float(s_ball[self.selected_train_mask].mean().item()),
            "mean_s_gcn_selected": float(s_gcn[self.selected_train_mask].mean().item()),
            "selected_count": diag.selected_count,
            "selected_ratio_train": diag.selected_ratio_train,
            "selected_noise_rate": diag.selected_noise_rate,
            "selected_clean_count": diag.selected_clean_count,
            "selected_noisy_count": diag.selected_noisy_count,
            **purity,
        }
        if self.conf.training["debug"]:
            print("GBSelect diagnostics:", self.gbselect_diagnostics)

    def _append_diagnostics(self):
        if not self.gbselect_diagnostics:
            return
        os.makedirs("./runs", exist_ok=True)
        path = "./runs/gbselect_diagnostics_20260616.csv"
        row = {
            "method": getattr(self, "method_name", self.conf.model.get("method", "gbselect")),
            "dataset": getattr(self, "dataset_name", ""),
            "seed": getattr(self, "seed", ""),
            "noise_type": getattr(self, "noise_type", ""),
            "noise_rate": getattr(self, "noise_rate", ""),
            "train": self.result.get("train", ""),
            "valid": self.result.get("valid", ""),
            "test": self.result.get("test", ""),
            "best_epoch": self.best_epoch,
            "total_time": self.total_time,
            **self.gbselect_diagnostics,
        }
        fieldnames = [
            "method", "dataset", "seed", "noise_type", "noise_rate", "train", "valid", "test",
            "best_epoch", "total_time", "selection_mode", "random_control", "keep_ratio",
            "warmup_epochs", "warmup_window_start", "warmup_window_end", "selected_count",
            "selected_ratio_train", "selected_noise_rate", "selected_clean_count", "selected_noisy_count",
            "num_balls", "mean_ball_size", "mean_s_ball_selected", "mean_s_gcn_selected",
            "mean_train_per_ball", "zero_train_balls", "noisy_train_purity_weighted",
            "true_train_purity_weighted", "true_all_purity_weighted",
        ]
        exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        self._make_selected_train_mask()
        for epoch in range(self.conf.training["n_epochs"]):
            improve = ""
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            output, loss_train, acc_train = self.get_prediction(
                self.feats, self.adj, self.noisy_label, self.selected_train_mask
            )
            loss_train.backward()
            self.optim.step()

            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = "*"
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result["valid"] = acc_val
                self.result["train"] = acc_train
                self.best_epoch = epoch + 1
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training["debug"]:
                nni.report_intermediate_result(acc_val)
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve
                    )
                )

        if self.total_time < 0:
            self.total_time = time.time() - self.start_time
        loss_test, acc_test = self.test(self.test_mask)
        self.result["test"] = acc_test
        self._append_diagnostics()
        if self.conf.training["debug"]:
            print("Optimization Finished!")
            print("Time(s): {:.4f}".format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result


class gbselect_Predictor(_GBSelectBase):
    selection_mode = "intersection"
    random_control = False


class gbselect_gcn_only_Predictor(_GBSelectBase):
    selection_mode = "gcn_only"
    random_control = False


class gbselect_shuffled_Predictor(_GBSelectBase):
    selection_mode = "intersection"
    random_control = True
