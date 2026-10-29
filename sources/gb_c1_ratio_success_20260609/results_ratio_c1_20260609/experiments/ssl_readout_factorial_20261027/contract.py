"""Registered cells, paired factorial effects and validation-only decision gate."""
import statistics as st

def tasks(config):
    return [dict(dataset=d,seed=s,scale=scale,initialization=init,condition=c,head=h)
            for d in config['datasets'] for s in config['seeds'] for scale,init in config['new_cells']
            for c in config['conditions'] for h in config['heads']]

def effects(raw_legacy,raw_unified,unit_legacy,unit_unified):
    a,b,c,d=raw_legacy,raw_unified,unit_legacy,unit_unified
    return dict(normalization_legacy=c-a,normalization_unified=d-b,
                initialization_raw=b-a,initialization_unit=d-c,
                normalization_main=((c-a)+(d-b))/2,initialization_main=((b-a)+(d-c))/2,
                interaction=d-b-c+a,diagonal=d-a)

def eligible(noisy_seeds,clean_seeds):
    return len(noisy_seeds)==len(clean_seeds)==2 and all(v>0 for v in noisy_seeds) and st.mean(noisy_seeds)>=.5 and st.mean(clean_seeds)>=-.5
