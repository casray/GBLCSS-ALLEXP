from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}


def _plain(value: object) -> str:
    return re.sub(r"\*\*", "", str(value)).strip()


def _formatted(frame: pd.DataFrame, noise_rate: float) -> dict[tuple[str, str], str]:
    selected = frame[(frame.noise_type == "uniform") & (frame.noise_rate == noise_rate)]
    output: dict[tuple[str, str], str] = {}
    for (dataset, method), group in selected.groupby(["dataset", "method"]):
        value = f"{100 * group.test_accuracy.mean():.2f}±{100 * group.test_accuracy.std(ddof=1):.2f}"
        output[(DATASETS[dataset], method)] = value
    return output


def _check_table(path: Path, values: dict[tuple[str, str], str], method_aliases: dict[str, str] | None = None) -> None:
    table = pd.read_csv(path)
    method_column = table.columns[0]
    aliases = method_aliases or {}
    mismatches = []
    for _, row in table.iterrows():
        reported_method = str(row[method_column])
        method = aliases.get(reported_method, reported_method)
        for dataset in DATASETS.values():
            expected = _plain(row[dataset])
            actual = values.get((dataset, method))
            if actual != expected:
                mismatches.append((reported_method, dataset, expected, actual))
    if mismatches:
        raise AssertionError(f"{path.name}: aggregate mismatches: {mismatches[:5]}")
    print(f"{path.name}: {len(table) * len(DATASETS)} cells verified")


def _check_validation_table() -> None:
    raw = pd.read_csv(ROOT / "results" / "ablations" / "validation_protocol_per_seed.csv")
    if len(raw) != 180:
        raise AssertionError(f"Table C requires 180 seed-wise rows, found {len(raw)}")
    noisy_scope = raw.loc[raw.validation_protocol == "noisy", "noise_scope"]
    clean_scope = raw.loc[raw.validation_protocol == "clean", "noise_scope"]
    if not noisy_scope.eq("train_val").all():
        raise AssertionError("noisy-validation rows must use noise_scope=train_val")
    if not clean_scope.eq("train").all():
        raise AssertionError("clean-validation rows must use noise_scope=train")
    values = {}
    for (protocol, dataset, method), group in raw.groupby(["validation_protocol", "dataset", "method"]):
        key = ("Clean validation labels" if protocol == "clean" else "Noisy validation labels", DATASETS[dataset], method)
        values[key] = f"{100 * group.test_accuracy.mean():.2f}±{100 * group.test_accuracy.std(ddof=1):.2f}"
    aliases = {
        "GSDN (protocol adaptation)": "GSDN",
        "PosteL (protocol adaptation)": "PosteL",
        "Leave-one-out random size-matched grouping": "Random size-matched grouping",
    }
    table = pd.read_csv(ROOT / "results" / "paper_tables" / "tablec_1.csv")
    mismatches = []
    for _, row in table.iterrows():
        method = aliases.get(row["Method"], row["Method"])
        actual = values.get((row["Validation protocol"], row["Dataset"], method))
        if actual != _plain(row["Test accuracy"]):
            mismatches.append((row["Validation protocol"], row["Dataset"], method, row["Test accuracy"], actual))
    if mismatches:
        raise AssertionError(f"tablec_1.csv: aggregate mismatches: {mismatches[:5]}")
    print(f"tablec_1.csv: {len(table)} cells verified")


def main() -> None:
    table2 = pd.read_csv(ROOT / "results" / "main_tables" / "table2_per_seed.csv")
    table3 = pd.read_csv(ROOT / "results" / "main_tables" / "table3_per_seed.csv")
    table4 = pd.read_csv(ROOT / "results" / "distillation" / "table4_per_seed.csv")
    for frame, label in [(table2, "Table 3"), (table3, "Table 4"), (table4, "Table 8")]:
        group = ["dataset", "noise_type", "noise_rate", "method"]
        if not frame.groupby(group).seed.nunique().eq(5).all():
            raise AssertionError(f"{label} does not contain five seeds per setting")
        if frame.duplicated(group + ['seed']).any():
            raise AssertionError(f'{label} contains duplicated seed records')

    _check_table(ROOT / "results" / "paper_tables" / "table2.csv", _formatted(table2, 0.4))
    _check_table(ROOT / "results" / "paper_tables" / "table3.csv", _formatted(table3, 0.4))
    aliases = {
        "Basic GB-LCSS (without distillation)": "Basic GB-LCSS",
        "GCN(X)+CE teacher (GLNN-style)": "GCN(X)+CE teacher",
    }
    _check_table(ROOT / "results" / "paper_tables" / "table4a.csv", _formatted(table4, 0.4), aliases)
    _check_table(ROOT / "results" / "paper_tables" / "table4b.csv", _formatted(table4, 0.6), aliases)
    _check_validation_table()

    snapshots = list((ROOT / "results" / "paper_tables").glob("*.csv"))
    if len(snapshots) != 22:
        raise AssertionError(f"expected 22 manuscript table snapshots, found {len(snapshots)}")
    print("reported result verification: ok")


if __name__ == "__main__":
    main()
