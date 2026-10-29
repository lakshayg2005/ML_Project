"""Limitation 2: fuse trained ECERC variants with the cached zero-shot LLM scores (MELD only).

    fused = (1 - w) * log p_model + w * log p~_LLM,   p~_LLM = label-free calibrated LLM (Limitation-2/MERGING.md)

Two stages, so every choice is made on dev before the test outputs are read:

    python fuse_llm.py freeze --variants baseline,cacg_none --seeds 0,1,2
        reads only *_valid_outputs.npz; chooses w per variant (highest mean dev macro-F1 over seeds, ties -> smaller w)
        and a temperature per run on the fused dev scores; writes runs/MELD/fusion_freeze.json.
    python fuse_llm.py test
        reads the freeze file and the *_test_outputs.npz once; writes fusion_results.json / .md. Refuses to run twice.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import classification_metrics, ece_score, fit_temperature  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(HERE, '..', '..', 'Limitation-2', 'results', 'run_kaggle_zeroshot_1', 'outputs_zeroshot')
NAMES = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def llm_scores(llm_dir, split):
    """Calibrated LLM log-probs: divide out the LLM's mean dev prediction, multiply in the train prior."""
    F = json.load(open(os.path.join(llm_dir, 'frozen_config_zeroshot.json')))
    z = np.load(os.path.join(llm_dir, 'dumps', f'{"dev" if split == "valid" else "test"}_llm.npz'))
    s = z['logp'] - np.array(F['log_q_dev']) + np.array(F['log_prior_train'])
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True), z['y'], F['w_grid']


def model_scores(run_dir, name, seed, split, y_ref):
    z = np.load(os.path.join(run_dir, f'{name}_seed{seed}_{split}_outputs.npz'))
    if not np.array_equal(z['y'], y_ref):
        raise SystemExit(f'{name} seed {seed} {split}: utterance order differs from the LLM scores (release order)')
    return z['lp']


def fuse(model_lp, llm_lp, w):
    s = (1 - w) * model_lp + w * llm_lp
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def macro_f1(y, s, C=len(NAMES)):
    p = s.argmax(1)
    return 100 * np.mean([2 * ((p == c) & (y == c)).sum() / max(1, (p == c).sum() + (y == c).sum()) for c in range(C)])


def scaled(lp, T):
    s = lp / T
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def stage_freeze(a, run_dir, freeze_path):
    if os.path.exists(freeze_path):
        raise SystemExit(f'{freeze_path} exists; the choices are already frozen')
    llm_dev, y_dev, w_grid = llm_scores(a.llm_dir, 'valid')
    out = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'seeds': a.seeds, 'variants': {},
           'llm_files_sha256': {f: sha256(os.path.join(a.llm_dir, f)) for f in
                                ('frozen_config_zeroshot.json', 'dumps/dev_llm.npz', 'dumps/test_llm.npz')},
           'rule': 'w: highest mean dev macro-F1 over seeds (ties -> smaller w); T: fitted per run on fused dev'}
    for v in a.variants:
        dev = {s: model_scores(run_dir, v, s, 'valid', y_dev) for s in a.seeds}
        mean_f1 = [np.mean([macro_f1(y_dev, fuse(dev[s], llm_dev, w)) for s in a.seeds]) for w in w_grid]
        w = w_grid[int(np.argmax(mean_f1))]
        out['variants'][v] = {
            'w': w, 'dev_mf1_alone': mean_f1[0], 'dev_mf1_fused': max(mean_f1),
            'T_alone': {str(s): fit_temperature(dev[s], y_dev) for s in a.seeds},
            'T_fused': {str(s): fit_temperature(fuse(dev[s], llm_dev, w), y_dev) for s in a.seeds}}
        print(f'{v:28s} w = {w:.2f}  dev macro-F1 {mean_f1[0]:.2f} -> {max(mean_f1):.2f}')
    with open(freeze_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('frozen:', freeze_path)


def summarise(rows):
    keys = ['acc', 'wf1', 'mf1', 'ece', 'ece_ts'] + [f'f1_{n}' for n in NAMES]
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.)
            for k in keys}


def evaluate(lp, y, T):
    m = classification_metrics(lp, y, NAMES)
    row = {k: m[k] for k in ('acc', 'wf1', 'mf1', 'ece')}
    row.update({f'f1_{n}': v for n, v in m['per_class_f1'].items()})
    row['ece_ts'] = 100 * ece_score(np.exp(scaled(lp, T)), y)
    return row


def stage_test(a, run_dir, freeze_path):
    res_path = os.path.join(run_dir, 'fusion_results.json')
    if not os.path.exists(freeze_path):
        raise SystemExit('run the freeze stage first')
    if os.path.exists(res_path):
        raise SystemExit(f'{res_path} exists; the test set has already been scored')
    Fz = json.load(open(freeze_path))
    for f, h in Fz['llm_files_sha256'].items():
        if sha256(os.path.join(a.llm_dir, f)) != h:
            raise SystemExit(f'{f} changed since the freeze')
    llm_test, y_test, _ = llm_scores(a.llm_dir, 'test')
    out = {'freeze': Fz, 'rows': {}}
    for v, cfg in Fz['variants'].items():
        alone, fused = [], []
        for s in Fz['seeds']:
            lp = model_scores(run_dir, v, s, 'test', y_test)
            alone.append(evaluate(lp, y_test, cfg['T_alone'][str(s)]))
            fused.append(evaluate(fuse(lp, llm_test, cfg['w']), y_test, cfg['T_fused'][str(s)]))
        out['rows'][v] = {'w': cfg['w'], 'per_seed': alone, 'mean_std': summarise(alone)}
        out['rows'][f'{v} + LLM'] = {'w': cfg['w'], 'per_seed': fused, 'mean_std': summarise(fused)}
    with open(res_path, 'w') as f:
        json.dump(out, f, indent=1)
    head = '| Row | Acc | wF1 | mF1 | ' + ' | '.join(f'F1 {n}' for n in NAMES) + ' | ECE | ECE+TS |'
    lines = [f'# MELD test, mean ± std over seeds {Fz["seeds"]}', '', head, '|' + '---|' * (len(NAMES) + 6)]
    for r, d in out['rows'].items():
        ms = d['mean_std']
        cells = [f'{ms[k][0]:.2f} ± {ms[k][1]:.2f}' for k in ('acc', 'wf1', 'mf1')]
        cells += [f'{ms[f"f1_{n}"][0]:.1f}' for n in NAMES] + [f'{ms[k][0]:.2f}' for k in ('ece', 'ece_ts')]
        lines.append(f'| {r} | ' + ' | '.join(cells) + ' |')
    with open(res_path.replace('.json', '.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'test'])
    p.add_argument('--variants', default='baseline,cacg_none')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
    p.add_argument('--llm_dir', default=LLM_DIR)
    a = p.parse_args()
    a.variants = a.variants.split(',')
    a.seeds = [int(s) for s in a.seeds.split(',')]
    run_dir = os.path.join(a.runs, 'MELD')
    freeze_path = os.path.join(run_dir, 'fusion_freeze.json')
    (stage_freeze if a.stage == 'freeze' else stage_test)(a, run_dir, freeze_path)


if __name__ == '__main__':
    main()
