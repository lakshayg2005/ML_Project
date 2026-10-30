"""Re-evaluate saved checkpoints (runs/<DATASET>/<name>_seed<s>.pt) and write their valid/test outputs, for runs
trained before train.py saved validation outputs.
Usage: python dump_outputs.py --dataset MELD --names baseline cacg_utility --seeds 0 1 2"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_loaders  # noqa: E402
from losses import build_task_loss  # noqa: E402
from train import DATASET_CFG, build_model, run_epoch  # noqa: E402

DEFAULTS = dict(audio='default', focal_gamma=0.0, lambda_vad=0.0, soft_beta=0.0)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['IEMOCAP', 'MELD'], required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--runs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = DATASET_CFG[a.dataset]
    for name in a.names:
        for s in a.seeds:
            tag = os.path.join(a.runs, a.dataset, f'{name}_seed{s}')
            if os.path.exists(tag + '_valid_outputs.npz'):
                continue
            args = SimpleNamespace(**{**DEFAULTS, **json.load(open(tag + '.json'))['args']})
            args.data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')  # folder moved since
            model = build_model(args, cfg).to(device)
            model.load_state_dict(torch.load(tag + '.pt', map_location=device))
            task_loss = build_task_loss(args, torch.FloatTensor(cfg['class_weights']).to(device))
            _, valid, test = get_loaders(a.dataset, args.data_root, cfg['batch_size'], audio=args.audio)
            for split, loader in [('valid', valid), ('test', test)]:
                res = run_epoch(model, loader, device, args, task_loss)
                np.savez_compressed(f'{tag}_{split}_outputs.npz',
                                    **{k: v for k, v in res.items() if k != 'losses' and v is not None})
            print('dumped', tag, flush=True)
