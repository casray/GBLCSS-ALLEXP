"""Record borrowed method sources and historical partition inputs before external fits."""
from pathlib import Path
import run as r

def main():
    paths={Path(r.f.__file__),Path(r.f.r2.__file__),Path(r.st.__file__)}
    paths.update((r.ROOT.parents[1]/'GB-LCSS_code/src/gblcss').glob('*.py'))
    for directory in ['simmlp_tined_audit_seed0_20261009','new_graph_mlp_seed0_20261009','representation_supervision_20261021']:
        paths.update((r.EXP/directory).glob('*.py'))
    for method in ['simmlp_tined_audit_seed0_20261009/vendor/simmlp','new_graph_mlp_seed0_20261009/vendor/gssc']:
        base=r.EXP/method
        for pattern in ['*.py','*.json']:paths.update(base.rglob(pattern))
    for name in r.CFG['datasets']:
        for seed in r.CFG['seeds']:paths.add(r.EXP/'main_protocol_replay_20260922/matrix'/name/'default'/f'seed{seed}.npz')
    r.save(r.ROOT/'additional_sources.json',dict(timing='R2 underway; before external fits',sha256={str(p):r.sha(p) for p in paths if p.is_file()}))
if __name__=='__main__':main()
