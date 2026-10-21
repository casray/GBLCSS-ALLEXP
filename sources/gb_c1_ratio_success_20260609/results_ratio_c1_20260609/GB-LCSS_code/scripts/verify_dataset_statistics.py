from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import load_public_dataset, unique_undirected_edge_statistics


DATASET_NAMES = {
    "Amazon-Computers": "AmazonComputers",
    "Amazon-Photo": "AmazonPhoto",
    "Attributed-Wiki": "AttributedWiki",
    "CiteSeer": "CiteSeer",
    "CiteSeerFull": "CiteSeerFull",
    "Coauthor-CS": "CoauthorCS",
    "Coauthor-Physics": "CoauthorPhysics",
    "Cora": "Cora",
    "CoraFull": "CoraFull",
    "Cora_ML": "CoraML",
    "PubMed": "PubMed",
}


def _integer(value: object) -> int:
    return int(str(value).replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the dataset statistics reported in Tables 1 and I.1.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated manuscript dataset names, or 'all'.",
    )
    args = parser.parse_args()
    selected = None if args.datasets == "all" else {
        item.strip().lower() for item in args.datasets.split(",") if item.strip()
    }

    tables = [
        pd.read_csv(ROOT / "results" / "paper_tables" / "table1.csv"),
        pd.read_csv(ROOT / "results" / "paper_tables" / "tablei_1.csv"),
    ]
    mismatches: list[tuple[object, ...]] = []
    checked = 0
    for table in tables:
        for _, row in table.iterrows():
            dataset = str(row["Dataset"])
            if selected is not None and dataset.lower() not in selected:
                continue
            graph = load_public_dataset(DATASET_NAMES[dataset], args.data_root, seed=0)
            edges, homophily = unique_undirected_edge_statistics(graph.edge_index, graph.y)
            expected_edges = _integer(row["Undirected edges"])
            expected_homophily = float(row["Edge homophily"])
            if edges != expected_edges or round(homophily, 3) != expected_homophily:
                mismatches.append((dataset, expected_edges, edges, expected_homophily, homophily))
            checked += 1
    if mismatches:
        raise AssertionError(f"dataset-statistic mismatches: {mismatches}")
    print(f"dataset statistic verification: {checked} dataset(s) matched")


if __name__ == "__main__":
    main()
