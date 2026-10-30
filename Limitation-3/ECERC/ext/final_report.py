"""Combined ablation of the merged model (ECERC + L1 + L2 + L3) on both datasets -> runs/RESULTS_merged.{md,json}.

Every number is computed from the saved valid/test outputs of runs/<DATASET>/<name>_seed<s>_{valid,test}_outputs.npz:
  * single runs: mean +- std over seeds; temperature T fitted on that run's validation outputs (L3 post-hoc step);
  * ensemble: mean of the members' temperature-scaled probabilities;
  * L2 LLM fusion (MELD only, Keshav's cached Qwen2.5-32B scores): log p = (1 - w) log p_model + w log p_LLM, with w
    chosen on validation macro-F1 and T re-fitted on the fused validation scores. The test set picks nothing.
Usage: python final_report.py"""
import json
import os
import sys

import numpy as np
from scipy.special import log_softmax, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import classification_metrics, ece_score, fit_temperature  # noqa: E402
from train import DATASET_CFG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
LLM_DIR = os.path.join(HERE, '..', '..', '..', 'Limitation-2', 'results', 'run_kaggle_zeroshot_1', 'outputs_zeroshot')

VARIANTS = {
    'IEMOCAP': [('ECERC (retrained)', 'baseline'),
                ('+ L3 CACG', 'cacg_noloss'),
                ('+ L1 VAD + soft labels (IS10 audio)', 'is10_l1'),
                ('+ L1 WavLM audio', 'wavlm'),
                ('+ L1 full (WavLM + VAD + soft)', 'wavlm_l1'),
                ('+ L1 WavLM + L3', 'wavlm_cacg'),
                ('+ L1 full + L3', 'wavlm_l1_cacg')],
    'MELD': [('ECERC (retrained, release loss γ=1)', 'baseline_g1'),
             ('+ L3 CACG', 'cacg_utility_g1'),
             ('γ=0 (not the release loss): ECERC', 'baseline'),
             ('γ=0: + L3 CACG (utility loss)', 'cacg_utility'),
             ('γ=0: + L3 CACG (no gate loss)', 'cacg_noloss')],
}


FINAL = {('IEMOCAP', 'wavlm_cacg', ''), ('MELD', 'cacg_utility_g1', ' + L2 LLM fusion')}


def softmax(lp):
    return np.exp(lp - logsumexp(lp, 1, keepdims=True))


def seeds_of(ds, name):
    d = os.path.join(RUNS, ds)
    out = []
    for f in os.listdir(d):
        if f.startswith(name + '_seed') and f.endswith('_test_outputs.npz'):
            s = f[len(name) + 5:-len('_test_outputs.npz')]
            if s.isdigit() and os.path.exists(os.path.join(d, f'{name}_seed{s}_valid_outputs.npz')):
                out.append(int(s))
    return sorted(out)


def load(ds, name, s):
    v = np.load(os.path.join(RUNS, ds, f'{name}_seed{s}_valid_outputs.npz'))
    t = np.load(os.path.join(RUNS, ds, f'{name}_seed{s}_test_outputs.npz'))
    return v['lp'].astype(np.float64), v['y'], t['lp'].astype(np.float64), t['y']


def llm_scores():
    cfg = json.load(open(os.path.join(LLM_DIR, 'frozen_config_zeroshot.json')))
    log_q, log_prior = np.array(cfg['log_q_dev']), np.array(cfg['log_prior_train'])
    out = {}
    for split in ['dev', 'test']:
        z = np.load(os.path.join(LLM_DIR, 'dumps', f'{split}_llm.npz'))
        out[split] = (log_softmax(z['logp'] - log_q + log_prior, axis=1), z['y'])
    return out, cfg['w_grid']


def score(lp_valid, y_valid, lp_test, y_test, names):
    """Metrics of raw test scores, plus ECE after a temperature fitted on validation."""
    m = classification_metrics(log_softmax(lp_test, axis=1), y_test, names)
    m['valid_wf1'] = classification_metrics(log_softmax(lp_valid, axis=1), y_valid, names)['wf1']
    T = fit_temperature(lp_valid, y_valid)
    m['ece_ts'], m['T'] = 100 * ece_score(softmax(lp_test / T), y_test), T
    return m


def fuse(lp_valid, y_valid, lp_test, y_test, llm, w_grid, names):
    (ld, yd), (lt, yt) = llm['dev'], llm['test']
    assert (yd == y_valid).all() and (yt == y_test).all(), 'utterance order differs from the L2 dumps'
    mf1 = lambda w: classification_metrics(log_softmax((1 - w) * lp_valid + w * ld, axis=1), y_valid, names)['mf1']
    w = max(w_grid, key=mf1)
    m = score((1 - w) * lp_valid + w * ld, y_valid, (1 - w) * lp_test + w * lt, yt, names)
    m['w'] = w
    return m


def pair_errors(cm):
    cm = np.array(cm)
    return {'hap_exc': int(cm[0, 4] + cm[4, 0]), 'ang_fru': int(cm[3, 5] + cm[5, 3])}


def summarize(ds, name, llm=None, w_grid=None):
    names = DATASET_CFG[ds]['target_names']
    seeds = seeds_of(ds, name)
    if not seeds:
        return None
    runs = [load(ds, name, s) for s in seeds]
    per = [score(*r, names) for r in runs]
    out = {'seeds': seeds, 'single': per}
    # ensemble of temperature-scaled members (IEMOCAP's validation order differs per seed, so no joint fit)
    p_ens = np.mean([softmax(r[2] / m['T']) for r, m in zip(runs, per)], 0)
    out['ensemble'] = classification_metrics(np.log(p_ens), runs[0][3], names)
    if llm is not None:
        out['single_fused'] = [fuse(*r, llm, w_grid, names) for r in runs]
        lv = np.log(np.mean([softmax(r[0]) for r in runs], 0))
        lt = np.log(np.mean([softmax(r[2]) for r in runs], 0))
        out['ensemble_fused'] = fuse(lv, runs[0][1], lt, runs[0][3], llm, w_grid, names)
    return out


def ms(xs):
    xs = np.asarray(xs, float)
    return f'{xs.mean():.2f} ± {xs.std():.2f}' if len(xs) > 1 else f'{xs.mean():.2f}'


def row(label, n, ms_list, ds, cms):
    cells = [label, str(n)] + ms_list
    if ds == 'IEMOCAP':
        pe = [pair_errors(c) for c in cms]
        he, af = [p['hap_exc'] for p in pe], [p['ang_fru'] for p in pe]
        cells += [f'{np.mean(he):.1f}', f'{np.mean(af):.1f}', f'{np.mean(he) + np.mean(af):.1f}']
    return '| ' + ' | '.join(cells) + ' |'


def main():
    llm, w_grid = llm_scores()
    report, lines = {}, ['# Merged model: ECERC + L1 + L2 + L3', '',
                         'Generated by `ext/final_report.py` from the saved run outputs. ECE in %, 15 bins; '
                         '"TS" = after temperature scaling fitted on validation. Ensemble = mean of the '
                         'temperature-scaled members (its ECE is already calibrated).', '']
    for ds in ['IEMOCAP', 'MELD']:
        names = DATASET_CFG[ds]['target_names']
        extra = ['Hap↔Exc', 'Ang↔Fru', 'Σ pairs'] if ds == 'IEMOCAP' else ['F1 fear', 'F1 disgust']
        head = ['Variant', 'seeds', 'valid wF1', 'acc', 'wF1', 'mF1', 'ECE raw', 'ECE TS'] + extra
        lines += [f'## {ds}', '', '### Single runs (mean ± std over seeds)', '',
                  '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        ens_lines = ['', '### Seed ensembles', '', '| ' + ' | '.join(['Variant', 'members', 'acc', 'wF1', 'mF1', 'ECE']
                                                                   + extra) + ' |', '|' + '---|' * (6 + len(extra))]
        report[ds] = {}
        for label, name in VARIANTS[ds]:
            r = summarize(ds, name, llm if ds == 'MELD' else None, w_grid)
            if r is None:
                continue
            report[ds][name] = r
            for suffix, single, ens in [('', r['single'], r['ensemble'])] + (
                    [(' + L2 LLM fusion', r['single_fused'], r['ensemble_fused'])] if ds == 'MELD' else []):
                label_s = label + suffix + (' **(final)**' if (ds, name, suffix) in FINAL else '')
                vals = [ms([m[k] for m in single]) for k in ['valid_wf1', 'acc', 'wf1', 'mf1', 'ece', 'ece_ts']]
                if ds == 'MELD':
                    vals_x = [ms([m['per_class_f1'][c] for m in single]) for c in ['fea', 'dis']]
                    lines.append('| ' + ' | '.join([label_s, str(len(single))] + vals + vals_x) + ' |')
                else:
                    lines.append(row(label_s, len(single), vals, ds, [m['confusion'] for m in single]))
                e = [f"{ens[k]:.2f}" for k in ['acc', 'wf1', 'mf1']]
                e.append(f"{ens.get('ece_ts', ens['ece']):.2f}")
                if ds == 'IEMOCAP':
                    p = pair_errors(ens['confusion'])
                    e += [str(p['hap_exc']), str(p['ang_fru']), str(p['hap_exc'] + p['ang_fru'])]
                else:
                    e += [f"{ens['per_class_f1'][c]:.2f}" for c in ['fea', 'dis']]
                ens_lines.append('| ' + ' | '.join([label_s, str(len(single))] + e) + ' |')
        lines += ens_lines + ['']
    with open(os.path.join(RUNS, 'RESULTS_merged.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(RUNS, 'RESULTS_merged.json'), 'w') as f:
        json.dump(report, f, indent=1, default=float)
    sys.stdout.reconfigure(encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
