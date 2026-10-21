import ast
import json

from common import DATASETS, ROOT, load_data, write_json


def main():
    repo = ROOT / "vendor/dgd"
    definitions = set()
    calls = set()
    for path in repo.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.add(node.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
    assert "dtw_dsasm" in calls and "dtw_dsasm" not in definitions
    params = json.loads((repo / "param/best_param.json").read_text())
    alpha = params["cora"]["GCN"]["MLP"]["alpha"]
    assert alpha == 18
    student_source = (repo / "train_student.py").read_text(encoding="utf-8-sig")
    assert "(1-param['alpha']) * loss_t" in student_source
    evidence = dict(status="blocked", reason="missing verified DGD fusion/reward/objective implementation",
                    missing_function="dtw_dsasm", author_alpha=alpha,
                    coded_kl_coefficient=1-alpha, note="DGD_NOTES.md",
                    trained=False, accuracy=None)
    for name in DATASETS:
        _, audit = load_data(name, "cpu")
        folder = ROOT / "runs" / name / "dgd"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "blocked.json", {**evidence, **audit})
    write_json(ROOT / "dgd_preflight.json", evidence)
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
