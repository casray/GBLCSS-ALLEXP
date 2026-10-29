import statistics as st

def tasks(cfg):
 return [dict(dataset=d,seed=s,scale='unit',condition=c,head=h) for d in cfg['new_datasets'] for s in cfg['seeds'] for c in cfg['conditions'] for h in cfg['heads']]

def noisy_gate(values):return len(values)==2 and all(v>0 for v in values) and st.mean(values)>=.5
