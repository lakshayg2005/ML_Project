"""Post-hoc analyses on saved test outputs (no retraining):
  1. Seed ensemble: average the class probabilities of the 3 seeds of each variant
     (raw, and after each member's own validation-fitted temperature).
  2. Selective prediction: keep only the most confident X% of utterances (coverage) and report accuracy;
     AURC = area under the risk-coverage curve (mean error over all coverages, lower = better confidence ranking).
Usage: python ensemble_coverage.py   -> runs/RESULTS_ensemble_coverage.md, runs/figures/selective_accuracy.png"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import ece_score  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')
VARIANTS = ['baseline', 'softmax_fixed', 'cacg_noloss', 'cacg_entropy_sop', 'cacg_utility']
LABEL = {'baseline': 'ECERC baseline', 'softmax_fixed': 'Cause softmax, fixed τ', 'cacg_noloss': 'CACG, dynamic τ',
         'cacg_entropy_sop': 'CACG + SOP entropy loss', 'cacg_utility': 'CACG + utility loss'}
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.5]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(ds, name):
    out = []
    for jf in sorted(glob.glob(os.path.join(RUNS, ds, f'{name}_seed*.json'))):
        r = json.load(open(jf))
        z = np.load(jf.replace('.json', '_test_outputs.npz'))
        out.append((z['lp'], z['y'], r['posthoc']['ts_T']))
    return out


def scores(p, y):
    pred = p.argmax(1)
    return {'acc': 100 * accuracy_score(y, pred), 'wf1': 100 * f1_score(y, pred, average='weighted'),
            'mf1': 100 * f1_score(y, pred, average='macro'), 'ece': 100 * ece_score(p, y),
            'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())}


def selective(p, y):
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    order = np.argsort(-conf, kind='stable')
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    at = {c: 100 * cum_acc[max(int(round(c * len(y))) - 1, 0)] for c in COVERAGES}
    aurc = 100 * float((1 - cum_acc).mean())
    return at, aurc, cum_acc


def main():
    md, curves = [], {}
    for ds in ['IEMOCAP', 'MELD']:
        md += [f'## {ds}', '',
               '### 1. Seed ensemble (3 seeds averaged) vs single model (mean of 3 seeds)', '',
               '| Variant | Model | Acc | wF1 | mF1 | ECE raw | ECE + TS | NLL + TS |', '|---|---|---|---|---|---|---|---|']
        sel_rows = []
        for v in VARIANTS:
            runs = load(ds, v)
            if not runs:
                continue
            y = runs[0][1]
            singles_raw = [scores(np.exp(lp), y) for lp, _, _ in runs]
            singles_ts = [scores(softmax(lp / T), y) for lp, _, T in runs]
            ens_raw = scores(np.mean([np.exp(lp) for lp, _, _ in runs], 0), y)
            p_ens_ts = np.mean([softmax(lp / T) for lp, _, T in runs], 0)
            ens_ts = scores(p_ens_ts, y)
            m = lambda k, L: np.mean([s[k] for s in L])  # noqa: E731
            md.append(f"| {LABEL[v]} | single | {m('acc', singles_raw):.2f} | {m('wf1', singles_raw):.2f} | "
                      f"{m('mf1', singles_raw):.2f} | {m('ece', singles_raw):.2f} | {m('ece', singles_ts):.2f} | "
                      f"{m('nll', singles_ts):.3f} |")
            md.append(f"| | **ensemble** | **{ens_raw['acc']:.2f}** | **{ens_raw['wf1']:.2f}** | **{ens_raw['mf1']:.2f}** | "
                      f"{ens_raw['ece']:.2f} | {ens_ts['ece']:.2f} | {ens_ts['nll']:.3f} |")
            # selective prediction: single models (mean over seeds) and the TS ensemble
            s_at = [selective(softmax(lp / T), y) for lp, _, T in runs]
            e_at, e_aurc, e_curve = selective(p_ens_ts, y)
            sel_rows.append((v, {c: np.mean([a[0][c] for a in s_at]) for c in COVERAGES},
                             np.mean([a[1] for a in s_at]), e_at, e_aurc))
            curves[(ds, v)] = e_curve
        md += ['', '### 2. Selective prediction: accuracy on the most confident X% of utterances', '',
               '| Variant | Model | ' + ' | '.join(f'{int(c * 100)}%' for c in COVERAGES) + ' | AURC ↓ |',
               '|---|---|' + '---|' * (len(COVERAGES) + 1)]
        for v, s_at, s_aurc, e_at, e_aurc in sel_rows:
            md.append(f'| {LABEL[v]} | single | ' + ' | '.join(f'{s_at[c]:.1f}' for c in COVERAGES) + f' | {s_aurc:.2f} |')
            md.append('| | ensemble | ' + ' | '.join(f'{e_at[c]:.1f}' for c in COVERAGES) + f' | {e_aurc:.2f} |')
        md.append('')

    # figure: selective accuracy, baseline vs best CACG variant, ensemble + TS
    best = {'IEMOCAP': 'cacg_noloss', 'MELD': 'cacg_utility'}
    colors = {'baseline': '#2a78d6', 'cacg': '#eb6834'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ds in zip(axes, ['IEMOCAP', 'MELD']):
        for key, v in [('baseline', 'baseline'), ('cacg', best[ds])]:
            c = curves[(ds, v)]
            cov = np.arange(1, len(c) + 1) / len(c) * 100
            m = cov >= 20
            ax.plot(cov[m], 100 * c[m], color=colors[key], lw=2, label=LABEL[v])
        ax.set_title(f'{ds} (3-seed ensemble, temperature-scaled)', fontsize=10, loc='left', color='#0b0b0b')
        ax.set_xlabel('coverage: % of utterances answered', fontsize=9, color='#52514e')
        ax.set_ylabel('accuracy on answered (%)', fontsize=9, color='#52514e')
        ax.set_xlim(100, 20)
        ax.grid(axis='y', color='#e6e5e0', lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#b5b4ad')
        ax.tick_params(colors='#52514e', labelsize=8.5)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.patch.set_facecolor('#fcfcfb')
    fig.suptitle('Answering only the most confident half lifts accuracy from about 70% to about 85%\n'
                 'Baseline and CACG rank their confidence almost equally', fontsize=10.5, x=0.01, ha='left', color='#0b0b0b')
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_png = os.path.join(RUNS, 'figures', 'selective_accuracy.png')
    fig.savefig(out_png, dpi=170, facecolor=fig.get_facecolor())
    text = '# Seed ensemble and selective prediction (test set, no retraining)\n\n' + '\n'.join(md)
    open(os.path.join(RUNS, 'RESULTS_ensemble_coverage.md'), 'w', encoding='utf-8').write(text)
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
    print('figure:', out_png)


if __name__ == '__main__':
    main()
