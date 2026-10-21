from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'figures' / 'source_data'
OUT = ROOT / 'figures' / 'revised_20260914'
WORK = ROOT / '.drawio-tmp' / 'submission_figures_20260914'
DATASETS = ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho']
LABELS = ['Cora', 'CiteSeer', 'CoraFull', 'Coauthor-CS', 'Amazon-Photo']
SETTINGS = [('uniform', .2), ('uniform', .4), ('uniform', .6), ('pair', .4), ('instance', .4)]
TICKS = ['Uniform\n0.2', 'Uniform\n0.4', 'Uniform\n0.6', 'Pair\n0.4', 'Instance\n0.4']
C = dict(blue='#1578B7', orange='#E39A00', green='#169C76', pink='#C9799A',
         gray='#58595B', light='#AAB2B9', ink='#26323D', grid='#E6EBEF')


def setup():
    OUT.mkdir(exist_ok=True, parents=True)
    WORK.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9.5,
        'axes.titleweight': 'bold', 'axes.linewidth': .75,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.frameon': False, 'legend.fontsize': 7.5,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'savefig.facecolor': 'white', 'text.color': C['ink'],
        'axes.labelcolor': C['ink']})


def save(fig, stem):
    fig.canvas.draw()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{extension}', dpi=600, facecolor='white')
    fig.savefig(WORK / f'{stem}_preview.png', dpi=180, facecolor='white')
    plt.close(fig)


def ordered_matrix(df, field, dataset_col='dataset', names=DATASETS):
    result = np.empty((5, 5))
    for i, name in enumerate(names):
        for j, (noise, rate) in enumerate(SETTINGS):
            row = df[(df[dataset_col] == name) & (df.noise_type == noise) & np.isclose(df.noise_rate, rate)]
            assert len(row) == 1, (name, noise, rate, len(row))
            result[i, j] = row.iloc[0][field]
    return result


def heatmap_axes(ax):
    ax.set_xticks(range(5), TICKS)
    ax.set_yticks(range(5), LABELS)
    ax.tick_params(length=0, pad=6)
    ax.set_xlim(-.5, 4.5)
    ax.set_ylim(4.5, -.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig2(df):
    matrix = ordered_matrix(df, 'gain_pp', names=LABELS)
    assert (matrix > 0).sum() == 24
    fig = plt.figure(figsize=(7.4, 3.7))
    ax = fig.add_axes([.18, .19, .70, .68])
    cax = fig.add_axes([.907, .19, .020, .68])
    cmap = LinearSegmentedColormap.from_list('gain', ['#C95A49', '#F5F4F2', '#2F6B9A'])
    im = ax.pcolormesh(np.arange(6)-.5, np.arange(6)-.5, matrix, cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=6), edgecolors='white', linewidth=1.4)
    heatmap_axes(ax)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if value >= 3.7 else C['ink'])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1, 2, 3, 4, 5, 6])
    cb.set_label('Accuracy gain (percentage points)', fontsize=8)
    cb.outline.set_linewidth(.6)
    fig.suptitle('Accuracy gain over the strongest noisy-label MLP baseline', y=.962,
                 fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Noise setting', labelpad=9)
    save(fig, 'fig2_gain_heatmap_v2')


def fig3(df):
    assert len(df) == 60 and df['n'].eq(5).all()
    styles = {
        'GB-LCSS': (C['blue'], 'o', '-', 1.8),
        'JoCoR': (C['orange'], 's', '-', 1.35),
        'MLP+CE': (C['gray'], '^', (0, (4, 2)), 1.25),
        'Random grouping': (C['light'], 'o', (0, (2, 2)), 1.15),
    }
    limits = [((28,66), [30,40,50,60]), ((27,67), [30,40,50,60]),
              ((18,46), [20,30,40]), ((76,94), [80,85,90]), ((54,90), [60,70,80,90])]
    fig, axs = plt.subplots(1, 5, figsize=(7.4, 2.8))
    for ax, ds, label, (ylim, yticks) in zip(axs, DATASETS, LABELS, limits):
        for k, (method, (color, marker, line, lw)) in enumerate(styles.items()):
            rows = df[(df.dataset == ds) & (df.method_label == method)].sort_values('noise_rate')
            assert len(rows) == 3
            x, y, sd = [rows[c].to_numpy(float) for c in ['noise_rate','mean_accuracy_pct','std_accuracy_pct']]
            ax.fill_between(x, y-sd, y+sd, color=color, alpha=.13 if k == 0 else .08, linewidth=0)
            ax.plot(x, y, color=color, marker=marker, linestyle=line, lw=lw,
                    markersize=3.9, markeredgewidth=.45,
                    markeredgecolor=color, markerfacecolor='white' if k == 3 else color,
                    label=method, zorder=5-k)
        ax.set_ylim(ylim); ax.set_yticks(yticks)
        ax.set_xticks([.2,.4,.6], ['0.2','0.4','0.6'])
        ax.set_xlim(.18,.62)
        ax.set_title(label, fontsize=8.7, pad=9)
        ax.grid(axis='y', color=C['grid'], linewidth=.6)
        ax.tick_params(length=2.8, pad=3)
    axs[0].set_ylabel('Test accuracy (%)', fontsize=8)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53,.995), ncol=4,
               handlelength=2.6, columnspacing=1.8)
    fig.supxlabel('Uniform label-noise rate', y=.038, fontsize=8)
    fig.subplots_adjust(left=.066, right=.992, top=.735, bottom=.21, wspace=.40)
    save(fig, 'fig3_uniform_noise_curves_v3')


def fig4(df):
    assert len(df) == 25 and df.seeds.eq(5).all()
    colors = dict(zip(DATASETS, [C['blue'], C['orange'], C['green'], C['pink'], '#8F969C']))
    markers = {'uniform':'o', 'pair':'s', 'instance':'^'}
    fig = plt.figure(figsize=(7.4, 3.8))
    ax = fig.add_axes([.088,.205,.349,.665])
    heat = fig.add_axes([.596,.205,.330,.665])
    cax = fig.add_axes([.942,.205,.016,.665])
    ax.plot([.08,1.3],[.08,1.3], color='#9EA8B1', lw=.9, linestyle=(0,(4,3)))
    for ds in DATASETS:
        for noise, marker in markers.items():
            rows = df[(df.dataset == ds)&(df.noise_type == noise)]
            ax.scatter(rows.noisy_onehot_brier_mean, rows.soft_target_brier_mean,
                       c=colors[ds], marker=marker, s=30 if noise!='instance' else 36,
                       edgecolors='white', linewidths=.65, zorder=3)
    ax.set(xlim=(.08,1.3), ylim=(.08,1.3))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([.2,.4,.6,.8,1,1.2]); ax.set_yticks([.2,.4,.6,.8,1,1.2])
    ax.set_xlabel('Observed one-hot target Brier score', fontsize=7.8, labelpad=6)
    ax.set_ylabel('GB-LCSS target Brier score', fontsize=7.8, labelpad=6)
    ax.set_title('Target quality', fontsize=10, pad=10)
    ax.grid(color=C['grid'], linewidth=.65)
    handles=[Line2D([],[],color=colors[d], marker='o',ls='',markersize=4.6,label=l) for d,l in zip(DATASETS,LABELS)]
    legend=ax.legend(handles=handles,loc='upper left',fontsize=7,handletextpad=.3,labelspacing=.25,borderpad=.25)
    ax.add_artist(legend)
    ax.legend(handles=[Line2D([],[],color='#7F8992',marker=m,ls='',markersize=4.5,label=n.title()) for n,m in markers.items()],
              loc='lower right',fontsize=7,handletextpad=.3,labelspacing=.3,borderpad=.25)
    matrix=ordered_matrix(df,'brier_gain_mean')
    cmap=LinearSegmentedColormap.from_list('brier',['#F5F7F8','#BFD6E6','#2F6B9A'])
    im=heat.pcolormesh(np.arange(6)-.5,np.arange(6)-.5,matrix,cmap=cmap,vmin=0,vmax=.52,
                     edgecolors='white',linewidth=1.2)
    heatmap_axes(heat)
    heat.tick_params(axis='x',labelsize=7.4)
    heat.tick_params(axis='y',labelsize=7.4,pad=5)
    heat.set_title('Mean Brier-score reduction',fontsize=9.5,pad=10)
    heat.set_xlabel('Noise setting',labelpad=8)
    for (i,j),v in np.ndenumerate(matrix):
        heat.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8.3,color='white' if v>=.34 else C['ink'])
    cb=fig.colorbar(im,cax=cax,ticks=[0,.1,.2,.3,.4,.5])
    cb.outline.set_linewidth(.6)
    fig.text(.025,.922,'a',fontweight='bold',fontsize=11)
    fig.text(.503,.922,'b',fontweight='bold',fontsize=11)
    save(fig,'fig4_target_quality_v6')


def fig5(df):
    assert len(df)==15 and df.seed.eq(0).all() and df.seeds.eq(5).all()
    styles={'GB partition':('#2F6B9A','o',-.16), 'Exact kNN':('#D97706','s',0),
            'k-means partition':('#118A6A','^',.16)}
    fig,axs=plt.subplots(1,3,figsize=(7.4,3.15))
    x=np.arange(5)
    for method,(color,marker,offset) in styles.items():
        rows=df[df.method_label==method].set_index('dataset').loc[DATASETS]
        for ax,field in zip(axs[:2],['end_to_end_structure_seconds','structure_storage_kib']):
            ax.scatter(x+offset,rows[field],color=color,marker=marker,s=32,
                       edgecolors='white',linewidths=.55,label=method,zorder=3)
        if method!='GB partition':
            axs[2].scatter(x+(-.09 if method=='Exact kNN' else .09),rows.accuracy_delta_vs_gb_pp,
                           color=color,marker=marker,s=32,edgecolors='white',linewidths=.55,zorder=3)
    axs[0].set(ylim=(0,15.5),yticks=[0,5,10,15],ylabel='Construction time (s)')
    axs[0].set_title('Grouping construction time',fontsize=8.8,pad=10)
    axs[1].set_yscale('log',base=2)
    axs[1].set_ylim(24,2600)
    ticks=[32,64,128,256,512,1024,2048]
    axs[1].set_yticks(ticks,[str(t) for t in ticks]);axs[1].set_ylabel('Stored structure (KiB; log scale)')
    axs[1].set_title('Grouping storage',fontsize=9.3,pad=10)
    axs[2].axhspan(-1,1,color='#D7E7F3',alpha=.45,zorder=0)
    axs[2].axhline(0,color='#7D8994',lw=.8,linestyle=(0,(4,3)))
    axs[2].set(ylim=(-1.35,3.55),yticks=[-1,0,1,2,3],ylabel='Accuracy difference vs GB (pp)')
    axs[2].set_title('Accuracy relative to GB',fontsize=9.3,pad=10)
    actual=float(df[(df.dataset=='corafull')&(df.method_label=='Exact kNN')].accuracy_delta_vs_gb_pp.iloc[0])
    axs[2].annotate(f'+{actual:.2f}',xy=(1.91,actual),xytext=(2.27,2.68),fontsize=7.5,
                    color='#D97706',arrowprops={'arrowstyle':'-','color':'#D97706','lw':.7})
    labels=LABELS
    for i,ax in enumerate(axs):
        ax.set_xticks(x,labels,rotation=40,ha='right',rotation_mode='anchor')
        ax.tick_params(axis='x',labelsize=7,pad=6,length=3)
        ax.tick_params(axis='y',labelsize=7.5,length=3)
        ax.set_xlim(-.42,4.42);ax.grid(axis='y',color=C['grid'],lw=.65)
        ax.text(-.21,1.085,chr(97+i),transform=ax.transAxes,fontsize=11,fontweight='bold')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.995),ncol=3,
               fontsize=8,handletextpad=.45,columnspacing=2.2)
    fig.subplots_adjust(left=.065,right=.988,top=.75,bottom=.265,wspace=.54)
    save(fig,'fig5_grouping_profile_v3')


def main():
    setup()
    files=['fig2_gblcss_gain_heatmap_source.csv','fig3_uniform_noise_rate_curves_source.csv',
           'fig4_target_quality_source_v2.csv','fig5_grouping_profile_source.csv']
    before={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    for plot,name in zip([fig2,fig3,fig4,fig5],files):
        plot(pd.read_csv(SOURCE/name))
        print('Completed',plot.__name__)
    after={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    assert before==after
    (WORK/'data_checks.json').write_text(json.dumps({'unchanged_source_data':True,'sources':before},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
