"""Read-only mechanism checks and paired plots after the frozen experiment."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c


PAIR_LIST=[('PRUNE','GREET_B64'),('PRUNE_ZERO','GREET_B64'),('PRUNE_ALL','GREET_B64'),('CF_FINE','GREET_B256'),
           ('PRUNE','CF_FINE'),('PRUNE_ZERO','PRUNE'),('PRUNE_ALL','PRUNE')]


def main():
    assert c.read(c.ROOT/'audit.json')['status']=='passed'
    source={str(p.resolve()):c.sha(p) for p in [Path(__file__),c.ROOT/'audit.json',c.ROOT/'target_results.csv',c.ROOT/'student_results.csv']}
    target=pd.read_csv(c.ROOT/'target_results.csv');student=pd.read_csv(c.ROOT/'student_results.csv')
    assert len(target)==len(student)==540
    mechanism=[];deltas=[];cut_capacity=[]
    for name in c.CONFIG['datasets']:
        for seed in c.CONFIG['seeds']:
            tree=c.load_tree(name,seed)
            for noise in c.CONFIG['conditions']:
                folder=c.case_path(name,seed,noise);decisions=c.read(folder/'decisions.json')
                for decision in decisions:
                    scores=decision['scores']
                    def capacity(node):
                        keep=1 if scores[str(node)]['valid'] else -10**8
                        if node in tree.children:keep=max(keep,sum(capacity(child) for child in tree.children[node]))
                        return keep
                    possible=capacity(0)
                    cut_capacity.append(dict(dataset=name,seed=seed,noise=noise,query_fold=decision['query_fold'],orientation=decision['orientation'],
                        max_feasible_disjoint_regions=max(0,possible),prune_regions=len(decision['regions']),zero_regions=len(decision['regions_zero']),
                        scores_use_labels_for_validity=False,not_a_trained_additional_control=True))
                for arm in c.CONFIG['arms']:
                    p=folder/(arm+'.npz');source[str(p.resolve())]=c.sha(p)
                    with np.load(p) as arrays:
                        region=arrays['component_region'];alpha=arrays['component_alpha']
                        mechanism.append(dict(dataset=name,seed=seed,noise=noise,arm=arm,
                            root_query_fraction=float((region==0).mean()),ce_fallback_fraction=float((region<0).mean()),
                            component_alpha_mean=float(alpha.mean()),
                            vote_support_mean=float(arrays['component_support'].mean())))
                for frame,metric in [(target,'target_brier'),(student,'test_accuracy'),(student,'test_macro_f1')]:
                    values=frame[(frame.dataset==name)&(frame.seed==seed)&(frame.noise==noise)].set_index('arm')[metric]
                    for left,right in PAIR_LIST:
                        deltas.append(dict(dataset=name,seed=seed,noise=noise,metric=metric,left=left,right=right,
                            delta=float(values[left]-values[right])*(1 if metric=='target_brier' else 100)))
    out=c.ROOT/'posthoc';out.mkdir(exist_ok=True)
    cap=pd.DataFrame(cut_capacity);cap.to_csv(out/'feasible_cut_capacity.csv',index=False,encoding='utf-8-sig')
    cap.groupby(['dataset','noise'])[['max_feasible_disjoint_regions','prune_regions','zero_regions']].agg(['mean','std']).to_csv(out/'cut_capacity_summary.csv',encoding='utf-8-sig')
    m=pd.DataFrame(mechanism);m.to_csv(out/'query_region_diagnostics.csv',index=False,encoding='utf-8-sig')
    ms=m.groupby(['dataset','noise','arm'])[['root_query_fraction','ce_fallback_fraction','component_alpha_mean','vote_support_mean']].agg(['mean','std'])
    ms.to_csv(out/'region_summary.csv',encoding='utf-8-sig')
    p=pd.DataFrame(deltas);p.to_csv(out/'paired_mechanism_deltas.csv',index=False,encoding='utf-8-sig')
    p.groupby(['dataset','noise','metric','left','right']).delta.agg(['count','mean','std']).to_csv(out/'paired_summary.csv',encoding='utf-8-sig')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    names=['Actor','Chameleon-f','Squirrel-f','Roman','CoraFull','CS'];plots=[]
    for metric in ['test_accuracy','target_brier']:
        for noise in c.CONFIG['conditions']:
            fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
            fig.subplots_adjust(left=.12,right=.97,bottom=.13,top=.88,hspace=.42,wspace=.30)
            for ax,(left,right) in zip(axes.flat,PAIR_LIST[:4]):
                ax.axvline(0,color='#777777',lw=1,ls='--',zorder=1)
                for i,name in enumerate(c.CONFIG['datasets']):
                    v=p[(p.dataset==name)&(p.noise==noise)&(p.metric==metric)&(p.left==left)&(p.right==right)].sort_values('seed').delta.to_numpy()
                    assert len(v)==3
                    ax.scatter(v,i+np.array([-.10,0,.10]),s=22,color='#777777',alpha=.65,zorder=2)
                    ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='D',ms=5,color='#0072B2',capsize=3,zorder=3)
                ax.set_yticks(range(6),names);ax.set_ylim(5.5,-.5)
                ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
                ax.set_title(f'{left} - {right}',fontsize=11)
                ax.set_xlabel('Accuracy difference (percentage points)' if metric=='test_accuracy' else 'Target Brier difference')
                # Auto locators propose ticks outside view limits; those labels
                # are not drawn, but have misleading off-canvas extents.
                low,high=ax.get_xlim();ticks=ax.get_xticks()
                ax.set_xticks(ticks[(ticks>=low)&(ticks<=high)]);ax.set_xlim(low,high)
            fig.suptitle(f'{noise}: paired differences across protocol seeds',fontsize=15)
            fig.text(.12,.035,'Gray points: three paired protocol seeds. Blue diamond/bar: mean and sample SD (not CI).\nAccuracy: positive is better. Brier: negative is better. One frozen SSL seed; aliases are not extra runs.',fontsize=10)
            fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
            outside=[]
            for ax in axes.flat:
                for obj in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
                    if not obj.get_visible() or not obj.get_text():continue
                    bounds=obj.get_window_extent(renderer)
                    if bounds.x0<0 or bounds.y0<0 or bounds.x1>width or bounds.y1>height:outside.append(obj.get_text())
            assert not outside,outside
            stem=out/f'{metric}_{noise}'
            for ext in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=180,facecolor='white')
            plt.close(fig);plots.append(dict(metric=metric,noise=noise,png=str(stem.with_suffix('.png').resolve()),layout_check='passed',visual_review='pending'))
    c.write(out/'manifest.json',dict(status='passed',type='post-hoc descriptive analysis only',new_models_trained=0,
        sources=source,mechanism_rows=len(m),paired_rows=len(p),cut_capacity_rows=len(cap),plot_groups=plots,clean_label_usage='metrics only',not_used_for_training=True,
        comparator_limit='CF_FINE may use overlapping ancestor-backoff neighborhoods; DP cuts are disjoint. Their difference is not solely the scoring function. Capacity is count-only, not a trained control.'))
    print(json.dumps({'status':'passed','mechanism_rows':len(m),'paired_rows':len(p),'figures':len(plots)}))


if __name__=='__main__':main()
