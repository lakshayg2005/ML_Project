"""Aggregate runs into ablation tables (mean ± std over seeds) and produce figures for the report.
Usage: python aggregate.py [--runs ../runs] [--ref cacg_utility]"""
import argparse
import sys
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ['baseline', 'baseline_fixattn', 'softmax_fixed', 'cacg_noloss', 'cacg_entropy_sop', 'cacg_utility',
         'cacg_utility_T025']
LABEL = {'baseline': 'ECERC (reproduced)', 'baseline_fixattn': 'ECERC + empty-history attention fix',
         'softmax_fixed': '+ cause softmax (masked), fixed τ',
         'cacg_noloss': '+ dynamic τ (no L_gate)', 'cacg_entropy_sop': '+ dynamic τ + L_gate (SOP Eq.3 entropy)',
         'cacg_utility': '+ dynamic τ + L_gate (utility, T=1)', 'cacg_utility_T025': '+ dynamic τ + L_gate (utility, T=0.25)'}
COLS = [('acc', 'Acc'), ('wf1', 'wF1'), ('mf1', 'mF1'), ('ece', 'ECE↓'), ('ece_ts', 'ECE+TS↓'), ('nll', 'NLL↓'),
        ('overconf_err', 'ConfErr↓')]


def load(runs, ds):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(runs, ds, '*_seed*.json'))):
        r = json.load(open(f))
        groups[r['args']['name']].append(r)
    return groups


def fmt(vals, digits=2):
    vals = np.asarray(vals, dtype=float)
    return f'{vals.mean():.{digits}f} ± {vals.std():.{digits}f}' if len(vals) > 1 else f'{vals.mean():.{digits}f}'


def table(groups, names, ds):
    names_present = [n for n in ORDER if n in groups] + [n for n in groups if n not in ORDER]
    lines = [f'### {ds} (test, mean ± std over seeds)', '',
             '| Model | n | valid wF1 | ' + ' | '.join(c for _, c in COLS) + ' | ECE+CA-TS↓ | H(g) | corr(τ, ctx) |',
             '|---' * (len(COLS) + 6) + '|']
    for n in names_present:
        rs = groups[n]
        row = [LABEL.get(n, n), str(len(rs)), fmt([r['valid']['wf1'] for r in rs])]
        for k, _ in COLS:
            row.append(fmt([r['test'][k] for r in rs], 3 if k == 'nll' else 2))
        row.append(fmt([r['posthoc']['cats']['ece'] for r in rs]) if all('posthoc' in r for r in rs) else '–')
        g =[r['test'].get('gate') for r in rs]
        row.append(fmt([x['gate_entropy_mean'] for x in g], 3) if all(g) else '–')
        row.append(fmt([x.get('corr_tau_ctx', 0) for x in g], 2) if all(g) and n.startswith('cacg') else '–')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', f'Per-class F1 ({ds}):', '',
              '| Model | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 1) + '|']
    for n in names_present:
        rs = groups[n]
        lines.append(f'| {LABEL.get(n, n)} | ' + ' | '.join(fmt([r['test']['per_class_f1'][c] for r in rs], 1)
                                                         for c in names) + ' |')
    return '\n'.join(lines)


def reliability_plot(groups, ds, names_to_plot, out):
    fig, axes = plt.subplots(1, len(names_to_plot), figsize=(4 * len(names_to_plot), 4), squeeze=False)
    for ax, n in zip(axes[0], names_to_plot):
        rs = groups[n]
        r = rs[0]  # first seed
        bins = [b for b in r['test']['reliability'] if b[1] is not None]
        centers = [b[0] for b in bins]
        ax.bar(centers, [b[1] for b in bins], width=1 / 15, edgecolor='k', alpha=0.7, label='accuracy')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.plot(centers, [b[2] for b in bins], 'r.-', label='confidence')
        ax.set_title(f"{LABEL.get(n, n)}\nECE={r['test']['ece']:.2f}", fontsize=9)
        ax.set_xlabel('confidence'); ax.set_ylabel('accuracy'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def gate_behaviour_plot(runs, ds, name, out):
    files = sorted(glob.glob(os.path.join(runs, ds, f'{name}_seed*_test_outputs.npz')))
    if not files:
        return None
    z = np.load(files[0])
    g, tau, ctx, avail = z['g'], z['tau'], z['ctx'], z['avail']
    ent = -(g * np.log(np.clip(g, 1e-12, None))).sum(1)
    buckets = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 1000)]
    lab = [f'{a}' if a == b else (f'{a}-{b}' if b < 1000 else f'>{a - 1}') for a, b in buckets]
    idx = [(ctx >= a) & (ctx <= b) for a, b in buckets]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(lab, [tau[m].mean() if m.any() else 0 for m in idx]); axes[0].set_title('mean τ_i by context length')
    axes[1].bar(lab, [ent[m].mean() if m.any() else 0 for m in idx]); axes[1].set_title('mean gate entropy H(g_i)')
    causes = ['self-contagion', 'cross-emotion', 'self-event', 'cross-event']
    bottom = np.zeros(len(buckets))
    for k, c in enumerate(causes):
        v = np.array([g[m, k].mean() if m.any() else 0 for m in idx])
        axes[2].bar(lab, v, bottom=bottom, label=c); bottom += v
    axes[2].set_title('mean cause weight g_i'); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('# prior utterances in dialogue')
    fig.suptitle(f'{ds}: {LABEL.get(name, name)}', fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    pred, y, conf = z['lp'].argmax(1), z['y'], np.exp(z['lp']).max(1)
    corr = pred == y
    return {'entropy_correct': float(ent[corr].mean()), 'entropy_wrong': float(ent[~corr].mean()),
            'tau_correct': float(tau[corr].mean()), 'tau_wrong': float(tau[~corr].mean()),
            'corr_tau_conf': float(np.corrcoef(tau, conf)[0, 1]) if tau.std() > 0 else 0.0}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
    p.add_argument('--ref', default='cacg_utility')
    a = p.parse_args()
    fig_dir = os.path.join(a.runs, 'figures'); os.makedirs(fig_dir, exist_ok=True)
    md = []
    for ds, names in [('IEMOCAP', ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']),
                      ('MELD', ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang'])]:
        groups = load(a.runs, ds)
        if not groups:
            continue
        md.append(table(groups, names, ds))
        plot_names = [n for n in ['baseline', 'softmax_fixed', a.ref] if n in groups]
        reliability_plot(groups, ds, plot_names, os.path.join(fig_dir, f'{ds}_reliability.png'))
        for n in [x for x in groups if x.startswith('cacg')]:
            s = gate_behaviour_plot(a.runs, ds, n, os.path.join(fig_dir, f'{ds}_{n}_gate.png'))
            if s:
                md.append(f'\n{ds} {n} (seed 1st): ' + ', '.join(f'{k}={v:.3f}' for k, v in s.items()))
        md.append('')
    text = '\n'.join(md)
    open(os.path.join(a.runs, 'RESULTS.md'), 'w', encoding='utf-8').write(text)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
