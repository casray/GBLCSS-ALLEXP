"""Use a frozen copy of v2 helpers when deployed; never change v2 source."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEGACY = HERE/'legacy'
if not LEGACY.is_dir():
    LEGACY = HERE.parent/'budget_gb_coarsening_20261019'/'v2'
sys.path.insert(0, str(LEGACY))
if (HERE/'support').is_dir():
    sys.path.insert(0, str(HERE/'support'))

from run_arm import protocol_labels, sparse_tensor, set_seed
from mechanisms import build_targets, random_size_matched_assignment
from runtime import Checkpoints, Stage, array_hash

