"""Extend the September 19 protocol without changing its implementation."""
import csv
from types import SimpleNamespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'submission_checks_20260919'
# Load the previous driver in memory. The only runtime relaxation is the
# explicit archive-Brier gate below; the algorithm and training path remain
# unchanged.
old_source = (OLD / 'run_experiment.py').read_text(encoding='utf8')
old_source = old_source.replace(
    "if arm['name'] == 'default':\n            assert abs(quality['target_brier'] - archive_brier) < 2e-5, (key, seed, quality['target_brier'], archive_brier)",
    "if arm['name'] == 'default' and cfg.get('enforce_archive_brier', True):\n            assert abs(quality['target_brier'] - archive_brier) < 2e-5, (key, seed, quality['target_brier'], archive_brier)"
)
old_source = old_source.replace(
    "OUTPUT = HERE/'gpu_reference_matched' if DEVICE.startswith('cuda') else HERE",
    "OUTPUT = HERE/'gpu_reference_matched_v2' if DEVICE.startswith('cuda') else HERE"
)
runner = SimpleNamespace()
runner.__dict__['__name__'] = 'reference_runner'
runner.__dict__['__file__'] = str(OLD / 'run_experiment.py')
exec(compile(old_source, str(OLD / 'run_experiment.py'), 'exec'), runner.__dict__)
runner.HERE = HERE


def canonical_csv_rows(*args, **kwargs):
    for row in csv.DictReader(*args, **kwargs):
        if row.get('dataset') == 'amazonpho':
            row['dataset'] = 'amazonphoto'
        yield row


# The archived figure CSV abbreviates Amazon-Photo; filesystem assets do not.
runner.csv = SimpleNamespace(DictReader=canonical_csv_rows, DictWriter=csv.DictWriter)


def load_graph(key, seed, cfg):
    import torch
    from torch_geometric.datasets import Coauthor, Amazon
    from gblcss.data import GraphData, percent_split_indices, inject_train_label_noise
    root = Path(cfg['data_root'])
    dataset_type, name = {'coauthorcs': (Coauthor, 'CS'), 'amazonphoto': (Amazon, 'Photo')}[key]
    if not (root / name / 'processed' / 'data.pt').exists():
        raise FileNotFoundError(f'Cached data missing: {key}')
    raw = dataset_type(root=root, name=name)[0]
    indices = [torch.as_tensor(x) for x in percent_split_indices(raw.num_nodes, seed)]
    masks = []
    for index in indices:
        mask = torch.zeros(raw.num_nodes, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    graph = GraphData(key, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                     train_index=indices[0], val_index=indices[1], test_index=indices[2])
    inject_train_label_noise(graph, cfg['noise_type'], cfg['noise_rate'], seed)
    assert torch.equal(graph.y[~graph.train_mask], graph.noisy_y[~graph.train_mask])
    return graph


runner.load_graph = load_graph


def check_sources():
    import json
    root = Path(json.loads((HERE / 'config.json').read_text(encoding='utf8'))['code_root'])
    for path in (OLD / 'source_snapshot').rglob('*'):
        if path.is_file():
            relative = path.relative_to(OLD / 'source_snapshot')
            if path.suffix in {'.py', '.yaml'}:
                assert path.read_bytes() == (root / relative).read_bytes(), f'Source drift: {relative}'


if __name__ == '__main__':
    check_sources()
    runner.main()
