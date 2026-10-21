"""Label-free, post-training perturbation scale diagnostics on fixed nodes."""
from pathlib import Path
import json
import sys
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_experiment as run
from pingda import build_generators


def main():
    torch.set_num_threads(4)
    rows=[]
    for p in sorted((ROOT/'runs/pretrain').glob('*/seed*/*/complete.json')):
        arm=p.parent.name
        if arm=='grace':continue
        dataset=p.parents[2].name;seed=int(p.parents[1].name[4:])
        graph=run.load_graph(dataset,seed)
        x=run._row_normalize_features(graph.x)
        # The diagnostic subset is fixed and uses no labels or split membership.
        ids=torch.randperm(len(x),generator=torch.Generator().manual_seed(13000))[:512]
        edge_gen,feat_gen=build_generators(x.shape[1],'cpu')
        state=torch.load(p.parent/'checkpoint.pt',map_location='cpu',weights_only=True)
        edge_gen.load_state_dict(state['edge_generator']);feat_gen.load_state_dict(state['feature_generator'])
        with torch.no_grad():
            mu,scale=feat_gen(x[ids]);signal_rms=x[ids].square().mean().sqrt()
            expected_noise_rms=(mu.square()+scale.square()).mean().sqrt()
            sample_edges=graph.edge_index[:,:min(8192,graph.edge_index.shape[1])]
            logits=edge_gen(x,sample_edges)
            row=dict(dataset=dataset,seed=seed,arm=arm,sampled_nodes=len(ids),
                feature_signal_rms=float(signal_rms),feature_mean_shift_rms=float(mu.square().mean().sqrt()),
                feature_scale_rms=float(scale.square().mean().sqrt()),expected_noise_rms=float(expected_noise_rms),
                noise_signal_rms_ratio=float(expected_noise_rms/signal_rms.clamp_min(1e-12)),
                edge_logit_mean=float(logits.mean()),edge_sigmoid_logit_mean=float(logits.sigmoid().mean()),
                scope='posthoc without labels; sigmoid(logit) is not expected relaxed random gate')
        rows.append(row)
    if rows:
        run.csv_write(ROOT/'runs/generator_diagnostics.csv',rows)
        run.write(ROOT/'runs/generator_diagnostics.json',rows)
    print(json.dumps(rows,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
