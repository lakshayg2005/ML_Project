"""
Aggregate result_*.json from train_aux.py: mean +- std per config, and paired (same-seed) comparison vs CTRL.
    python summarize_aux.py ./aux_run [CTRL]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else './aux_run'
ref = sys.argv[2] if len(sys.argv) > 2 else 'CTRL'

runs = defaultdict(dict)
for f in glob.glob(os.path.join(out_dir, 'result_*.json')):
    r = json.load(open(f))
    runs[r['tag']][r['seed']] = r

keys = ['valid_wf1', 'test_acc', 'test_wf1', 'test_mf1', 'test_hap_exc', 'test_ang_fru']
print('{:14s} {:>3s} '.format('config', 'n') + ' '.join('{:>14s}'.format(k.replace('test_', '')) for k in keys)
      + '   arousal_r')
for tag in sorted(runs, key=lambda t: (t != ref, t)):
    rs = list(runs[tag].values())
    row = '{:14s} {:3d} '.format(tag, len(rs))
    for k in keys:
        x = np.array([r[k] for r in rs], dtype=float)
        row += ' {:>14s}'.format('{:.2f}+-{:.2f}'.format(x.mean(), x.std()))
    ar = [r['test_vad_r'][1] for r in rs if r.get('test_vad_r')]
    row += '   {:.3f}'.format(np.mean(ar)) if ar else ''
    print(row)

if ref in runs:
    print('\nPaired vs {} (same seeds): mean diff [95% bootstrap CI], seeds where config is better'.format(ref))
    rng = np.random.default_rng(0)
    for tag in sorted(runs):
        if tag == ref:
            continue
        seeds = sorted(set(runs[tag]) & set(runs[ref]))
        if not seeds:
            continue
        line = '{:14s} (n={:2d})'.format(tag, len(seeds))
        for k, higher_better in [('test_wf1', True), ('test_mf1', True), ('test_hap_exc', False), ('test_ang_fru', False)]:
            d = np.array([runs[tag][s][k] - runs[ref][s][k] for s in seeds], dtype=float)
            boots = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            better = int(((d > 0) if higher_better else (d < 0)).sum())
            line += ' | {} {:+.2f} [{:+.2f},{:+.2f}] {}/{}'.format(k.replace('test_', ''), d.mean(), lo, hi,
                                                                     better, len(d))
        print(line)
