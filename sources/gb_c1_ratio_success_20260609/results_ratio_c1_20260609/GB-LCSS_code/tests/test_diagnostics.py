from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gblcss.benchmark import structure_storage_bytes


def test_retained_storage_matches_cora_figure_convention():
    assert structure_storage_bytes('granular_ball', 2708, 541, 128, 52, 10) == 48288
    assert structure_storage_bytes('exact_knn', 2708, 541, 128, 52, 10) == 43280

