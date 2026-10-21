"""Repeat two GPU arms in reverse order to check seed isolation."""
import json

import numpy as np
import torch

from run_experiment import HERE, load_graph, snapshot, write_json


def main():
    cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
    _, source = snapshot(cfg)
    from gblcss.data import set_seed
    from gblcss.experiment import load_config
    from gblcss.training import train_x_only_mlp
    base = load_config(source/'configs/main/gblcss.yaml')
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    graph = load_graph('cora',0,cfg)
    out = HERE/'gpu_reference_matched'
    rows = []
    for arm in ['beta_0', 'default']:
        path = out/'runs/cora/seed0'/arm
        stored = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        targets = torch.from_numpy(np.load(path.with_suffix('.npz'))['targets'])
        previous = np.load(path.with_name(path.name+'_predictions.npz'))['logits']
        set_seed(0)
        result = train_x_only_mlp(graph,targets,**base['model'],
                                 learning_rate=base['training']['learning_rate'],
                                 weight_decay=base['training']['weight_decay'],
                                 epochs=cfg['epochs'],device='cuda:0')
        same_logits = np.array_equal(result.logits.numpy(), previous)
        assert same_logits and result.test_accuracy == stored['test_accuracy']
        rows.append(dict(arm=arm,identical_logits=same_logits,test_accuracy=result.test_accuracy,
                         best_epoch=result.best_epoch,device='cuda:0'))
    write_json(out/'training_replay_check.json', rows)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
