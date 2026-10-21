"""Actual-count response curves with seed traces and explicitly labelled sample SD."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {'GRACE': '#0072B2', 'GREET': '#D55E00'}
STYLES = {'GRACE': ('-', 'o'), 'GREET': ('--', 's')}
METRICS = [('test_accuracy', 'Test accuracy (%)', 100),
           ('test_macro_f1', 'Test macro F1 (%)', 100),
           ('target_brier', 'Target Brier score', 1),
           ('mean_support', 'Mean other training votes', 1),
           ('positive_clean_margin_fraction', 'Positive clean vote margin fraction', 1),
           ('normalized_sse', 'Mean squared within-ball distance', 1)]


def collapse_repeated_partitions(frame, metric):
    """A reached cap is one seed observation, regardless of requested budgets."""
    frame = frame[frame.kind == 'B'].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors='coerce')
    frame = frame.sort_values('requested_balls')
    keys = ['dataset', 'seed', 'noise', 'space', 'actual_balls']
    for key, part in frame.groupby(keys, dropna=False):
        values = part[metric].dropna().to_numpy(float)
        if len(values) and not np.allclose(values, values[0], atol=2e-12, rtol=0):
            raise ValueError(f'same-count nested aliases disagree for {metric}: {key}')
    return frame.drop_duplicates(keys)


def audit_layout(fig):
    """Deterministic text clipping check; parent also reviews rendered PNGs."""
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); boundary = fig.bbox
    outside = []; hidden_tick_labels=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            low,high=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<low or tick.get_loc()>high:
                    hidden_tick_labels.update([id(tick.label1),id(tick.label2)])
    for artist in fig.findobj():
        if id(artist) in hidden_tick_labels or not hasattr(artist, 'get_text') or not artist.get_visible() or not artist.get_text():
            continue
        # Hidden outside ticks are legitimate; only visible axes ticks matter.
        if hasattr(artist, 'axes') and artist.axes is not None and not artist.axes.get_visible():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < boundary.x0-2 or bbox.x1 > boundary.x1+2 or bbox.y0 < boundary.y0-2 or bbox.y1 > boundary.y1+2:
            outside.append(artist.get_text())
    return sorted(set(outside))


def aggregate_snapshots(frame,metric):
    """Pair protocol seeds by the same planned snapshot, including reached caps."""
    aggregate=frame.groupby('requested_balls')[metric].agg(['mean','std','count'])
    counts=frame.groupby('requested_balls').actual_balls.agg(mean_actual='mean',min_actual='min',max_actual='max')
    return aggregate.join(counts).sort_index()


def make_plots(root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    import data_tools as dt

    root = Path(root) if root is not None else dt.ROOT
    targets = pd.read_csv(root/'target_results.csv', keep_default_na=False)
    students = pd.read_csv(root/'student_results.csv', keep_default_na=False)
    keys = ['dataset', 'seed', 'noise', 'arm']
    extra = [metric for metric in ['test_accuracy', 'test_macro_f1'] if metric not in targets]
    frame = targets.merge(students[keys+extra], on=keys, validate='one_to_one')
    frame.actual_balls = pd.to_numeric(frame.actual_balls)
    frame.requested_balls = pd.to_numeric(frame.requested_balls)
    datasets = dt.CONFIG['datasets']; spaces = dt.CONFIG['spaces']
    node_counts = {name: dt.load_case(name, dt.CONFIG['protocol_seeds'][0], 'clean')[0].num_nodes for name in datasets}
    output = root/'figures'; output.mkdir(exist_ok=True)
    profile = dict(rows=len(frame), datasets=datasets,
                   duplicate_protocol_arm_rows=int(frame.duplicated(keys).sum()),
                   statistical_unit='one protocol seed per space/condition/actual partition',
                   uncertainty='sample SD (ddof=1), not a CI', metrics={})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.dpi': 300})
    manifest = []; ncols = min(3, len(datasets)); nrows = math.ceil(len(datasets)/ncols)
    for metric, ylabel, scale in METRICS:
        if metric not in frame:
            raise ValueError(f'required curve metric missing: {metric}')
        profile['metrics'][metric] = dict(missing=int(pd.to_numeric(frame[metric], errors='coerce').isna().sum()))
        compact = collapse_repeated_partitions(frame, metric)
        for noise in dt.CONFIG['conditions']:
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 3.5*nrows+1.1), squeeze=False)
            fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.81, wspace=.30, hspace=.65)
            visible = []
            for ax, name in zip(axes.flat, datasets):
                visible.append(ax)
                panel = compact[(compact.dataset == name)&(compact.noise == noise)]
                all_points = []
                for space in spaces:
                    color = COLORS[space]; style, marker = STYLES[space]
                    part = panel[panel.space == space].sort_values('actual_balls')
                    for seed, seed_part in part.groupby('seed'):
                        valid = seed_part.dropna(subset=[metric])
                        x = valid.actual_balls.to_numpy(float); y = valid[metric].to_numpy(float)*scale
                        ax.plot(x, y, linestyle=style, marker=marker, ms=2.5, lw=.65, color=color, alpha=.32, zorder=2)
                        all_points.extend(x.tolist())
                    aggregate = aggregate_snapshots(part,metric)
                    if len(aggregate):
                        x = aggregate['mean_actual'].to_numpy(float); y = aggregate['mean'].to_numpy(float)*scale
                        sd = aggregate['std'].to_numpy(float)*scale
                        ax.plot(x, y, linestyle=style, marker=marker, ms=4, lw=1.7, color=color, zorder=4)
                        has_sd = np.isfinite(sd)
                        xerr=np.stack([x-aggregate['min_actual'].to_numpy(float),aggregate['max_actual'].to_numpy(float)-x])
                        ax.errorbar(x[has_sd], y[has_sd], yerr=sd[has_sd], xerr=xerr[:,has_sd], fmt='none', color=color,
                                    elinewidth=.9, capsize=2, alpha=.8, zorder=3)
                    anchors = frame[(frame.dataset == name)&(frame.noise == noise)&(frame.space == space)&(frame.kind == 'OLD64')]
                    anchor_y = pd.to_numeric(anchors[metric], errors='coerce')*scale
                    valid = anchor_y.notna()
                    ax.scatter(anchors.loc[valid, 'actual_balls'], anchor_y[valid], marker='D', s=24,
                               facecolors='none', edgecolors=color, linewidths=.9, zorder=5)
                    all_points.extend(anchors.loc[valid, 'actual_balls'].tolist())
                # CE has no ball count; display its mean as a horizontal reference.
                if metric in ['test_accuracy', 'test_macro_f1', 'target_brier']:
                    ce = frame[(frame.dataset == name)&(frame.noise == noise)&(frame.arm == 'CE')]
                    ce_y = pd.to_numeric(ce[metric], errors='coerce')*scale
                    if ce_y.notna().any():
                        ax.axhline(ce_y.mean(), color='.4', lw=.9, ls=':', zorder=1)
                reference = math.ceil(math.sqrt(node_counts[name]))
                ax.axvline(reference, color='.6', lw=.8, ls=(0, (4, 3)), zorder=0)
                ax.text(reference, 1.025, f'ceil(sqrt(N))={reference}', transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=7, color='.35')
                ax.set_xscale('log', base=2)
                # Nearby achieved caps (e.g. 16/20) are points, not overlapping ticks.
                ticks = [1, 8, 16, 32, 64, 128, 256]
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
                ax.set_xlim(.8, max([reference, 256]+all_points)*1.3)
                ax.set_xlabel('Actual balls (mean for thick line; log2)')
                ax.set_ylabel(ylabel); ax.set_title(name.replace('_', ' '), loc='left', pad=25)
                ax.grid(axis='both', color='.9', linewidth=.55)
                if metric == 'mean_support':
                    ax.set_yscale('log')
                    ax.set_ylabel('Mean other training votes (log scale)')
            for ax in axes.flat[len(datasets):]:
                ax.set_visible(False)
            handles = [Line2D([0], [0], color=COLORS[s], ls=STYLES[s][0], marker=STYLES[s][1], lw=1.7, ms=4, label=s) for s in spaces]
            handles += [Line2D([0], [0], color='.4', marker='D', mfc='none', ls='none', label='Historical OLD64 (seed points)'),
                        Line2D([0], [0], color='.4', lw=.9, ls=':', label='CE mean'),
                        Line2D([0], [0], color='.6', lw=.8, ls=(0, (4, 3)), label='sqrt(N) requested reference')]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .965), ncol=3, frameon=False)
            title = {'clean': 'Clean labels', 'uniform': 'Uniform label noise 0.4', 'pair': 'Pair label noise 0.4'}.get(noise, noise)
            fig.suptitle(title, fontsize=12, y=.995)
            fig.text(.075, .035, 'Thin lines: individual seeds. Thick lines: paired seed means at planned snapshots (x = mean achieved count).\n'
                     'Vertical bars: sample SD (ddof=1), not CI. Horizontal bars: achieved-count range. Cap aliases counted once.\n'
                     'OLD64 uses the historical generator; only B-curve points share the fixed trajectory constraints.', fontsize=8, va='bottom')
            clipped = audit_layout(fig)
            # Cosmetic tick labels outside current data limits are not drawn by axes.
            clipped = [item for item in clipped if item not in {str(tick) for tick in ticks}]
            if clipped:
                plt.close(fig)
                raise RuntimeError(f'clipped figure text for {metric}/{noise}: {clipped}')
            base = output/f'{metric}_{noise}'
            for extension in ['png', 'svg', 'pdf']:
                fig.savefig(base.with_suffix('.'+extension), facecolor='white')
            # A grayscale preview is a perceptual check of redundant line/marker codes.
            if metric == 'test_accuracy':
                from PIL import Image
                with Image.open(base.with_suffix('.png')) as preview:
                    preview.convert('L').save(output/f'{metric}_{noise}_grayscale.png')
            manifest.append(dict(metric=metric, condition=noise, png=str(base.with_suffix('.png').resolve()),
                                 layout_check='passed', visual_review='pending parent inspection'))
            plt.close(fig)
    (output/'data_profile.json').write_text(json.dumps(profile, indent=2)+'\n', encoding='utf-8')
    (output/'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    make_plots()
