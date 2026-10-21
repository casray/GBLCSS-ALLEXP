from pathlib import Path
import importlib.util
import numpy as np
import torch


def module():
    path=Path(__file__).with_name('run_students.py')
    assert path.exists(), 'student runner not yet implemented'
    spec=importlib.util.spec_from_file_location('student_runner',path)
    out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out)
    return out


def test_only_training_targets_are_replaced():
    m=module()
    noisy=torch.tensor([0,1,0,1,0])
    train=torch.tensor([3,0])
    local=np.array([[.2,.8],[.6,.4]],dtype=np.float32)
    target=m.full_targets(noisy,train,2,local)
    torch.testing.assert_close(target[train],torch.tensor(local))
    torch.testing.assert_close(target[[1,2,4]],torch.nn.functional.one_hot(noisy,2).float()[[1,2,4]])


def test_pilot_summary_does_not_invent_standard_deviation():
    m=module()
    rows=[dict(dataset='toy',representation='grace',noise='uniform',arm='CE',seed=0,test_accuracy=.4)]
    summary=m.aggregate(rows)
    assert summary[0]['n']==1
    assert summary[0]['test_std'] is None
    rows.append({**rows[0],'seed':1,'test_accuracy':.6})
    summary=m.aggregate(rows)
    assert summary[0]['n']==2
    assert np.isclose(summary[0]['test_mean'],.5)
