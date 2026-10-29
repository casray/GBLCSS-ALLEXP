import torch

def tasks(config):
    return [dict(dataset=ds,seed=seed,features=f,targets=t,condition=c)
            for ds,pair in config['pairs'].items() for seed in config['seeds']
            for f in pair for t in pair if f!=t for c in config['conditions']]

def validate_target(target,expected,train_count,classes):
    if target.shape!=(train_count,classes):raise ValueError('Target shape mismatch')
    if not torch.isfinite(target).all() or (target<0).any():raise ValueError('Invalid target probabilities')
    if not torch.allclose(target.sum(1),torch.ones(train_count),atol=1e-6,rtol=0):raise ValueError('Invalid target normalization')
    if not torch.equal(target,expected):raise ValueError('Target source mismatch')

def factorial_effects(aa,ab,ba,bb):
    return dict(target_B_minus_A_on_A=ab-aa,target_B_minus_A_on_B=bb-ba,
                feature_B_minus_A_on_A_target=ba-aa,feature_B_minus_A_on_B_target=bb-ab,
                interaction=bb-ba-ab+aa)
