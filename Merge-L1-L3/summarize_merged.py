"""
Aggregate result_*.json from train_merged.py: mean +- std per config, and paired (same-seed) comparison vs a
reference config. Pair errors are also split by annotator agreement ("clear" = every evaluator voted the gold label).

    python summarize_merged.py ./merged_run [CTRL] [--data_dir ../ECERC/data/iemocap]
"""
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('out_dir')
p.add_argument('ref', nargs='?', default='CTRL')
p.add_argument('--data_dir', default='../ECERC/data/iemocap')
args = p.parse_args()

# test utterances in loader order, and which of them every annotator agreed on
videoIDs, _, videoLabels, _, _, _, _, _, testVid = pickle.load(
    open(os.path.join(args.data_dir, 'IEMOCAP_features.pkl'), 'rb'), encoding='latin1')
aux = pickle.load(open(os.path.join(args.data_dir, 'iemocap_aux_labels.pkl'), 'rb'))
uids = [u for v in testVid for u in videoIDs[v]]
gold = np.array([y for v in testVid for y in videoLabels[v]])
clear = np.array([u in aux and aux[u]['votes'][g] >= 0.999 for u, g in zip(uids, gold)])


def pair_errors(pred, a, b, m):
    return int((((gold == a) & (pred == b)) | ((gold == b) & (pred == a)))[m].sum())


runs = defaultdict(dict)
for f in glob.glob(os.path.join(args.out_dir, 'result_*.json')):
    r = json.load(open(f))
    d = np.load(os.path.join(args.out_dir, 'preds_{}.npz'.format(r['run'])))
    assert (d['labels'] == gold).all(), 'test order mismatch'
    r['test_clear_hap_exc'] = pair_errors(d['preds'], 0, 4, clear)
    r['test_clear_ang_fru'] = pair_errors(d['preds'], 3, 5, clear)
    runs[r['tag']][r['seed']] = r

keys = ['valid_wf1', 'test_acc', 'test_wf1', 'test_mf1', 'test_hap_exc', 'test_ang_fru',
        'test_clear_hap_exc', 'test_clear_ang_fru', 'test_ece', 'test_ece_ts']
print('{:16s} {:>3s} '.format('config', 'n') + ' '.join('{:>13s}'.format(k.replace('test_', '')) for k in keys))
for tag in sorted(runs, key=lambda t: (t != args.ref, t)):
    rs = list(runs[tag].values())
    row = '{:16s} {:3d} '.format(tag, len(rs))
    for k in keys:
        x = np.array([r[k] for r in rs], dtype=float)
        row += ' {:>13s}'.format('{:.2f}+-{:.2f}'.format(x.mean(), x.std()))
    print(row)

if args.ref in runs:
    print('\nPaired vs {} (same seeds): mean diff [95% bootstrap CI], seeds where config is better'.format(args.ref))
    rng = np.random.default_rng(0)
    for tag in sorted(runs):
        if tag == args.ref:
            continue
        seeds = sorted(set(runs[tag]) & set(runs[args.ref]))
        if not seeds:
            continue
        line = '{:16s} (n={:2d})'.format(tag, len(seeds))
        for k, higher_better in [('test_wf1', True), ('test_mf1', True), ('test_hap_exc', False),
                                 ('test_ang_fru', False), ('test_clear_ang_fru', False), ('test_ece_ts', False)]:
            d = np.array([runs[tag][s][k] - runs[args.ref][s][k] for s in seeds], dtype=float)
            boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            better = int(((d > 0) if higher_better else (d < 0)).sum())
            line += ' | {} {:+.2f} [{:+.2f},{:+.2f}] {}/{}'.format(k.replace('test_', ''), d.mean(), lo, hi,
                                                                     better, len(d))
        print(line)
