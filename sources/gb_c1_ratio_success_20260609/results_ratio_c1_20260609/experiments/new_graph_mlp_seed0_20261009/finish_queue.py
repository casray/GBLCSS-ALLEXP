"""Wait for the existing GSSC grid, then finish the authorized GPU queue."""
import argparse
import subprocess
import sys

from common import ROOT


def launch(arguments, logfile=None):
    command = [sys.executable, str(ROOT / "run_screen.py"), *arguments]
    print("RUN", " ".join(command), flush=True)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                    f"Wait-Process -Id {args.wait_pid} -ErrorAction SilentlyContinue"], check=False)
    for dataset in ["citeseer", "corafull", "coauthorcs", "amazonpho"]:
        if not (ROOT / "runs" / dataset / "gssc" / "result.json").exists():
            raise RuntimeError(f"Existing GSSC grid did not finish {dataset}; inspect original logs")
    launch(["--method", "gssc", "--dataset", "cora"], ROOT / "runs/cora/gssc/console.log")
    launch(["--method", "tined", "--dataset", "cora", "--smoke"], ROOT / "smoke/cora/tined/console.log")
    launch(["--grid", "--methods", "tined"])
    subprocess.run([sys.executable, str(ROOT / "report_screen.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
