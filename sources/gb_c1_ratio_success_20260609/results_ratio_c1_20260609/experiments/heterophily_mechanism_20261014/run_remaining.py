"""Execute the preregistered Roman pipeline serially; stop on any failure."""
from pathlib import Path
from datetime import datetime
import hashlib
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def now():
    return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume a failed pipeline, retaining prior attempts")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    status_file = ROOT / "pipeline_status.json"
    previous = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else None
    if previous and (not args.resume or previous["status"] != "failed"):
        raise RuntimeError("Pipeline already has a status file; inspect it before an explicit restart")
    stages = []
    for representation, action, epochs in (("graphacl_portable", "train-graphacl", 50),
                                          ("grace", "train-grace", 1000)):
        output = ROOT / "representations" / "Roman-empire" / representation
        stages.extend([
            (representation + ":representation", "prepare_representations.py",
             ["--action", action, "--epochs", str(epochs), "--device", "cuda", "--output", str(output)]),
            (representation + ":diagnostic", "diagnostic_v2.py",
             ["--datasets", "roman", "--representation", representation, "--h-path", str(output / "H.pt"),
              "--seeds", "0", "1", "2", "3", "4", "--device", "cuda"]),
            (representation + ":students", "run_students.py",
             ["--datasets", "roman", "--representation", representation, "--seeds", "0", "--device", "cuda"]),
        ])
    stages.extend([("audit", "audit_diagnostic.py", ["--device", "cuda"]),
                   ("report", "analyze_results.py", [])])
    state = dict(status="running", started_at=now(), completed_at=None, stages=[],
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    done = set()
    if previous:
        state["prior_attempt"] = previous
        done = {r["name"] for r in previous["stages"] if r["status"] == "completed"}
        done.update(previous.get("reused_completed_stages", []))
        state["reused_completed_stages"] = sorted(done)
        for label in done:
            if label.endswith(":representation"):
                output = ROOT / "representations" / "Roman-empire" / label.split(":")[0]
                summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
                digest = hashlib.sha256((output / "H.pt").read_bytes()).hexdigest()
                if digest != summary["representation_sha256"]:
                    raise RuntimeError("Completed representation hash changed")

    def save():
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_file)

    save()
    with (ROOT / "pipeline_raw.log").open("a" if previous else "x", encoding="utf-8") as log:
        try:
            for label, script, args in stages:
                if label in done:
                    continue
                command = [sys.executable, "-X", "utf8", "-u", str(ROOT / script), *args]
                record = dict(name=label, status="running", started_at=now(), command=command)
                state["stages"].append(record)
                state["current_stage"] = label
                save()
                print("START " + label, flush=True)
                log.write("\nSTART " + now() + " " + label + "\n")
                log.flush()
                with subprocess.Popen(command, cwd=WORKSPACE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                      errors="replace") as child:
                    record["pid"] = child.pid
                    save()
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
                record.update(return_code=code, completed_at=now(), status="completed" if code == 0 else "failed")
                save()
                if code:
                    raise RuntimeError(f"Stage {label} failed with exit code {code}")
            state.update(status="completed", completed_at=now(), current_stage=None)
        except BaseException as exc:
            state.update(status="failed", completed_at=now(), error=repr(exc))
            save()
            raise
        save()
        print("ALL PREREGISTERED STAGES COMPLETED", flush=True)


if __name__ == "__main__":
    main()
