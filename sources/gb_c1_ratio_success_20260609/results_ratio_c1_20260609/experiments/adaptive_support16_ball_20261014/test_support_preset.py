from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'GB-LCSS_code/src'))
from gblcss.granular_ball_adaptive import build_adaptive_granular_balls


def test_support16_caps_count_and_meets_leave_one_out_variance_limit():
    angles=np.arange(183)*2*np.pi/183
    h=np.column_stack([np.cos(angles),np.sin(angles)])
    train=np.random.RandomState(8).choice(183,87,replace=False)
    with threadpool_limits(limits=4):
        result=build_adaptive_granular_balls(h,train_index=train,min_train_per_ball=16)
    assert result.summary['requested_target']==14
    assert result.summary['feasible_target']==5
    assert result.summary['num_balls']<=5
    assert all(ball['train_count']>=16 for ball in result.balls)
    assert max(1/(2*np.sqrt(ball['train_count']-1)) for ball in result.balls)<=.13


def test_support_constraint_uses_membership_not_training_index_order():
    h=np.random.RandomState(2).normal(size=(256,12))
    train=np.random.RandomState(8).choice(256,128,replace=False)
    with threadpool_limits(limits=4):
        first=build_adaptive_granular_balls(h,train_index=train,min_train_per_ball=16,random_state=3)
        second=build_adaptive_granular_balls(h,train_index=train[::-1],min_train_per_ball=16,random_state=3)
    np.testing.assert_array_equal(first.assignment,second.assignment)
