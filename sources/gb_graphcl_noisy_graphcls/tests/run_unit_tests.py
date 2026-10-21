from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TEST_MODULES = [
    "test_data",
    "test_granular_ball",
    "test_reliability",
    "test_selection",
    "test_reweight",
    "test_smoothing",
]


def main():
    passed = 0
    failed = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            test_id = f"{module_name}.{name}"
            try:
                fn()
                print(f"PASS {test_id}")
                passed += 1
            except Exception:
                print(f"FAIL {test_id}")
                traceback.print_exc()
                failed += 1
    print(f"RESULT passed={passed} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
