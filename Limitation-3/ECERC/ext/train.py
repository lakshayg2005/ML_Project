"""Unified train / evaluate entry point for ECERC (+ CACG).

Examples
  python train.py --dataset IEMOCAP --eval_ckpt ../IEMOCAP/ECERC_MODEL.pkl      # released checkpoint
  python train.py --dataset IEMOCAP --gate baseline --seed 2007                  # retrain the baseline
  python train.py --dataset IEMOCAP --gate cacg --gate_loss utility --aux_heads  # CACG
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import get_loaders  # noqa: E402
from cacg import gate_losses  # noqa: E402
from losses import build_task_loss, contrastive_loss  # noqa: E402
from metrics import classification_metrics, ece_score, fit_temperature, gate_metrics, reliability_bins  # noqa: E402
from model import ECERC  # noqa: E402

DATASET_CFG = {
    # values taken from the official IEMOCAP/train.py and MELD/train.py
    'IEMOCAP': dict(n_classes=6, d_a=1582, d_v=342, lr=1e-4, l2=2e-4, batch_size=64, epochs=200, patience=50,
                    class_weight=True, seed=2007, pos_scale=1.0, dropout1=0.5, dropout2=0.5,
                    target_names=['hap', 'sad', 'neu', 'ang', 'exc', 'fru'],
                    class_weights=[1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305,
                                   1 / 0.140051123, 1 / 0.24875555]),
    'MELD': dict(n_classes=7, d_a=300, d_v=342, lr=1e-5, l2=2e-4, batch_size=32, epochs=40, patience=20,
                 class_weight=False, seed=0, pos_scale=0.0, dropout1=0.0, dropout2=0.2,
                 target_names=['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang'],
                 class_weights=[1 / 0.469506857, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.073096002,
                                1 / 0.168368836, 1 / 0.026334987, 1 / 0.117230814]),
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unpack(data, device):
    emo, sem, audio, vision, qmask, umask, label = [d.to(device) for d in data[:-1]]
    U_e = torch.cat([emo, audio, vision], dim=-1)
    seq_lengths = umask.sum(1).long().tolist()
    labels = label[umask.bool()]
    return U_e, sem, qmask, umask, seq_lengths, labels


def run_epoch(model, loader, device, args, task_loss=None, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, n_batches = {'task': 0., 'gate': 0., 'aux': 0., 'con': 0.}, 0
    keep = {'lp': [], 'y': [], 'g': [], 'tau': [], 'ctx': [], 'avail': []}
    with torch.set_grad_enabled(train):
        for data in loader:
            U_e, sem, qmask, umask, seq_lengths, labels = unpack(data, device)
            out = model(U_e, sem, qmask, umask, seq_lengths)
            l_task = task_loss(out['log_prob'], labels)
            l_gate = l_aux = l_con = out['log_prob'].new_zeros(())
            if out['gate'] is not None:
                l_gate, l_aux = gate_losses(out['gate'], labels, out['valid'], args.gate_loss, args.utility_T)
            if args.lambda_contrast > 0:
                l_con = contrastive_loss(out['fused'], labels)
            loss = l_task + args.lambda_gate * l_gate + args.lambda_aux * l_aux + args.lambda_contrast * l_con
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for k, v in zip(tot, [l_task, l_gate, l_aux, l_con]):
                tot[k] += float(v)
            n_batches += 1
            keep['lp'].append(out['log_prob'].detach().cpu())
            keep['y'].append(labels.cpu())
            keep['ctx'].append(out['n_ctx'].cpu())
            if out['gate'] is not None:
                v = out['valid']
                keep['g'].append(out['gate']['g'][v].detach().cpu())
                keep['tau'].append(out['gate']['tau'][v].detach().cpu())
                keep['avail'].append(out['gate']['avail'][v].cpu())
    res = {k: (torch.cat(v).numpy() if v else None) for k, v in keep.items()}
    res['losses'] = {k: v / max(n_batches, 1) for k, v in tot.items()}
    return res


def evaluate(res, names):
    m = classification_metrics(res['lp'], res['y'], names)
    m['losses'] = res['losses']
    if res['g'] is not None:
        m['gate'] = gate_metrics(res['g'], res['tau'], res['ctx'], res['avail'])
    return m


def build_model(args, cfg):
    gate_kwargs = None
    if args.gate != 'baseline':
        gate_kwargs = dict(tau_min=args.tau_min, tau_max=args.tau_max, fixed_tau=args.fixed_tau,
                           mask_unavailable=args.mask_unavailable, aux_heads=args.aux_heads)
    return ECERC(d_t=1024, d_a=cfg['d_a'], d_v=cfg['d_v'], base_layer=args.base_layer, hidden_size=128,
                 n_classes=cfg['n_classes'], pos_scale=cfg['pos_scale'], dropout1=cfg['dropout1'],
                 dropout2=cfg['dropout2'], gate=args.gate, gate_kwargs=gate_kwargs,
                 fix_empty_attention=args.fix_empty_attention)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['IEMOCAP', 'MELD'], required=True)
    p.add_argument('--data_root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    p.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
    p.add_argument('--name', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--base_layer', type=int, default=1)
    p.add_argument('--eval_ckpt', default=None, help='only evaluate this state_dict')
    # objective
    p.add_argument('--task_loss', default='ce')
    p.add_argument('--class_weight', type=int, default=None)
    p.add_argument('--lambda_contrast', type=float, default=0.0)
    # CACG
    p.add_argument('--gate', choices=['baseline', 'softmax', 'cacg'], default='baseline')
    p.add_argument('--gate_loss', choices=['none', 'entropy', 'utility'], default='none')
    p.add_argument('--lambda_gate', type=float, default=0.1)
    p.add_argument('--lambda_aux', type=float, default=0.0)
    p.add_argument('--aux_heads', action='store_true')
    p.add_argument('--utility_T', type=float, default=1.0)
    p.add_argument('--tau_min', type=float, default=0.25)
    p.add_argument('--tau_max', type=float, default=4.0)
    p.add_argument('--fixed_tau', type=float, default=1.0)
    p.add_argument('--mask_unavailable', type=int, default=1)
    p.add_argument('--fix_empty_attention', type=int, default=0)
    args = p.parse_args()

    cfg = DATASET_CFG[args.dataset]
    for k in ['seed', 'epochs', 'patience', 'lr', 'class_weight']:
        if getattr(args, k) is None:
            setattr(args, k, cfg[k])
    args.name = args.name or f'{args.gate}' + (f'_{args.gate_loss}' if args.gate != 'baseline' else '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = cfg['target_names']
    class_weights = torch.FloatTensor(cfg['class_weights']).to(device)

    seed_everything(args.seed)
    model = build_model(args, cfg).to(device)
    task_loss = build_task_loss(args, class_weights)
    train_loader, valid_loader, test_loader = get_loaders(args.dataset, args.data_root, cfg['batch_size'])

    if args.eval_ckpt:
        model.load_state_dict(torch.load(args.eval_ckpt, map_location=device), strict=args.gate == 'baseline')
        test = evaluate(run_epoch(model, test_loader, device, args, task_loss), names)
        print(json.dumps({k: v for k, v in test.items() if k != 'confusion'}, indent=1))
        print('confusion:', test['confusion'])
        return

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg['l2'])
    run_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(run_dir, exist_ok=True)
    tag = f'{args.name}_seed{args.seed}'
    best_f1, best_epoch, best_loss, patience, patience2, best_state = -1, -1, None, 0, 0, None
    history = []
    for e in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, args, task_loss, optimizer)
        va = evaluate(run_epoch(model, valid_loader, device, args, task_loss), names)
        history.append({'epoch': e, 'train_losses': tr['losses'], 'valid_wf1': va['wf1'], 'valid_acc': va['acc']})
        if va['wf1'] > best_f1:
            best_f1, best_epoch, patience = va['wf1'], e, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        vloss = va['losses']['task']
        if best_loss is None or vloss < best_loss:
            best_loss, patience2 = vloss, 0
        else:
            patience2 += 1
        g = va.get('gate', {})
        print(f"[{tag}] ep {e:3d} train_task {tr['losses']['task']:.4f} gate {tr['losses']['gate']:.4f} "
              f"aux {tr['losses']['aux']:.4f} | valid wF1 {va['wf1']:.2f} acc {va['acc']:.2f} "
              + (f"H(g) {g['gate_entropy_mean']:.3f} tau {g['tau_mean']:.2f}±{g['tau_std']:.2f} " if g else '')
              + f"| {time.time() - t0:.1f}s", flush=True)
        if patience >= args.patience and patience2 >= args.patience:
            break

    model.load_state_dict(best_state)
    va_res = run_epoch(model, valid_loader, device, args, task_loss)
    te_res = run_epoch(model, test_loader, device, args, task_loss)
    test, valid = evaluate(te_res, names), evaluate(va_res, names)
    T = fit_temperature(va_res['lp'], va_res['y'])
    test['ece_ts'] = 100 * ece_score(np.exp(te_res['lp'] / T) / np.exp(te_res['lp'] / T).sum(1, keepdims=True), te_res['y'])
    test['ts_temperature'] = T
    test['reliability'] = reliability_bins(np.exp(te_res['lp']), te_res['y'])
    result = {'args': vars(args), 'best_epoch': best_epoch, 'valid': valid, 'test': test, 'history': history}
    with open(os.path.join(run_dir, tag + '.json'), 'w') as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(run_dir, tag + '_test_outputs.npz'), **{k: v for k, v in te_res.items()
                                                                             if k != 'losses' and v is not None})
    torch.save(best_state, os.path.join(run_dir, tag + '.pt'))
    print(f"[{tag}] DONE best_epoch {best_epoch} | TEST acc {test['acc']:.2f} wF1 {test['wf1']:.2f} "
          f"mF1 {test['mf1']:.2f} ECE {test['ece']:.2f} (TS {test['ece_ts']:.2f}) NLL {test['nll']:.3f}"
          + (f" H(g) {test['gate']['gate_entropy_mean']:.3f}" if 'gate' in test else ''))


if __name__ == '__main__':
    main()
