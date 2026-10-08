"""Post-hoc calibration of trained runs (backbone frozen, fitted on the validation split only).

  * TS     -- global temperature scaling (Guo et al., 2017): one scalar T.
  * CA-TS  -- cause-aware temperature scaling (CACG output stage): per-utterance temperature
              log T_i = b + w^T z_i, with z_i the standardized CACG evidence-quality descriptor h_i
              (modality-gate strength, context sizes, inter-cause cosine similarities) plus, for gated
              models, the gate entropy H(g_i) and the predicted gate temperature tau_i.
              L2-regularised, fitted by LBFGS on validation NLL. Never changes the argmax -> accuracy/F1 unchanged.

Usage: python calibrate.py --dataset IEMOCAP   (updates every run json in runs/<dataset> with a 'posthoc' block)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
from data import get_loaders  # noqa: E402
from metrics import ece_score, reliability_bins  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    lp, y, z = [], [], []
    for data in loader:
        U_e, sem, qmask, umask, seq_lengths, labels = T.unpack(data, device)
        out = model(U_e, sem, qmask, umask, seq_lengths)
        feats = [out['desc']]
        if out['gate'] is not None:
            g = out['gate']['g'][out['valid']]
            ent = -(g * g.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            feats += [ent, out['gate']['tau'][out['valid']].unsqueeze(-1)]
        lp.append(out['log_prob'].cpu()); y.append(labels.cpu()); z.append(torch.cat(feats, -1).cpu())
    return torch.cat(lp).double(), torch.cat(y), torch.cat(z).double()


def fit_ts(lp, y):
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=300)

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lp / log_t.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return log_t.detach()


def fit_cats(lp, y, z, l2=1e-2):
    mu, sd = z.mean(0), z.std(0).clamp_min(1e-6)
    zs = (z - mu) / sd
    w = torch.zeros(z.size(1), dtype=torch.float64, requires_grad=True)
    b = fit_ts(lp, y).clone().requires_grad_(True)  # warm start at global TS
    opt = torch.optim.LBFGS([w, b], lr=0.1, max_iter=500)

    def closure():
        opt.zero_grad()
        log_t = (b + zs @ w).unsqueeze(-1)
        loss = F.cross_entropy(lp / log_t.exp(), y) + l2 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return lambda lp_, z_: lp_ / (b.detach() + ((z_ - mu) / sd) @ w.detach()).exp().unsqueeze(-1), w.detach()


def scores(logits, y):
    p = F.softmax(logits, -1).numpy()
    return {'ece': 100 * ece_score(p, y.numpy()), 'nll': float(F.cross_entropy(logits, y)),
            'reliability': reliability_bins(p, y.numpy())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--runs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
    ap.add_argument('--l2', type=float, default=1e-2)
    a = ap.parse_args()
    cfg = T.DATASET_CFG[a.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, valid_loader, test_loader = get_loaders(a.dataset, os.path.join(a.runs, '..', 'data'), cfg['batch_size'])
    for jf in sorted(glob.glob(os.path.join(a.runs, a.dataset, '*_seed*.json'))):
        r = json.load(open(jf))
        args = argparse.Namespace(**r['args'])
        for k, v in [('aux_heads', False), ('fix_empty_attention', 0), ('mask_unavailable', 1)]:
            if not hasattr(args, k):
                setattr(args, k, v)
        model = T.build_model(args, cfg).to(device)
        model.load_state_dict(torch.load(jf.replace('.json', '.pt'), map_location=device))
        vlp, vy, vz = collect(model, valid_loader, device)
        tlp, ty, tz = collect(model, test_loader, device)
        log_t = fit_ts(vlp, vy)
        cats, w = fit_cats(vlp, vy, vz, a.l2)
        r['posthoc'] = {'raw': scores(tlp, ty), 'ts': scores(tlp / log_t.exp(), ty), 'cats': scores(cats(tlp, tz), ty),
                        'ts_T': float(log_t.exp()), 'cats_w': w.tolist()}
        json.dump(r, open(jf, 'w'), indent=1)
        ph = r['posthoc']
        print(f"{os.path.basename(jf):40s} ECE raw {ph['raw']['ece']:.2f} TS {ph['ts']['ece']:.2f} CA-TS {ph['cats']['ece']:.2f} | "
              f"NLL raw {ph['raw']['nll']:.3f} TS {ph['ts']['nll']:.3f} CA-TS {ph['cats']['nll']:.3f}", flush=True)


if __name__ == '__main__':
    main()
