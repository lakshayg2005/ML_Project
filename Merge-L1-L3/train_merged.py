"""
ECERC on IEMOCAP with Limitation 1 (WavLM audio + auxiliary VAD supervision, Divyansh)
merged with Limitation 3 (Confidence-Aware Cause Gating + temperature scaling, Lakshay).

    L = w_y * CE(p, (1-beta)*onehot(y) + beta*votes)          task loss (beta=0: authors' class-weighted NLL)
      + lambda_vad  * MSE(head(R), zscore(VAD))                L1: VAD head on the fused (cause-gated) feature R
      + lambda_gate * L_gate + lambda_aux * L_aux              L3: optional CACG gate loss (off on IEMOCAP)

After training, one temperature T is fitted on the validation split (Lakshay's fit_temperature) and
test ECE is reported raw and after temperature scaling. Predictions (and accuracy / F1) do not change.

    --gate baseline                      original ECERC (bit-identical to the authors' model.py)
    --gate cacg --gate_loss none         CACG with dynamic tau, no gate loss (Lakshay's IEMOCAP setting)
    --audio_feat iemocap_wavlm.pkl       WavLM-large audio instead of IS10
    --lambda_vad 1.0                     VAD supervision (Round 2 choice)
"""
import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from cacg import gate_losses
from dataloader_aux import IEMOCAPAuxDataset
from metrics import ece_score, fit_temperature, gate_metrics
from model import ECERC

warnings.filterwarnings("ignore")

TARGET_NAMES = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
# perform 1/weight according to the weight of each label in training data (authors' values)
CLASS_WEIGHTS = [1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089, 1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555]


def seed_everything(seed=2007):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_loaders(data_dir, batch_size, valid_rate=0.1, audio_file=None):
    trainset = IEMOCAPAuxDataset(data_dir, train=True, audio_file=audio_file)
    testset = IEMOCAPAuxDataset(data_dir, train=False, audio_file=audio_file)
    idx = list(range(len(trainset)))
    split = int(valid_rate * len(trainset))  # same unshuffled split as the authors
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[split:]),
                              collate_fn=trainset.collate_fn)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[:split]),
                              collate_fn=trainset.collate_fn)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn)

    # VAD normalisation statistics from the training split only
    vad = np.concatenate([trainset[i][7].numpy()[trainset[i][8].numpy() > 0] for i in idx[split:]])
    return train_loader, valid_loader, test_loader, vad.mean(0), vad.std(0), trainset[0][2].shape[-1]


def flat(x, seq_lengths):
    # batch-major, same order as the model's out['log_prob'] = log_softmax(...)[umask.bool()]
    return torch.cat([x[j][:seq_lengths[j]] for j in range(len(seq_lengths))])


def run_epoch(model, vad_head, loader, args, class_w, vad_mu, vad_sd, vad_dims, optimizer=None):
    train = optimizer is not None
    model.train(train); vad_head.train(train)
    stats = {'ce': [], 'vad': [], 'gate': [], 'aux': []}
    preds, labels, probs, vad_pred, vad_true = [], [], [], [], []
    keep = {'g': [], 'tau': [], 'ctx': [], 'avail': []}

    for data in loader:
        if train:
            optimizer.zero_grad()
        tensors = [d.to(args.device) for d in data[:-1]]
        emo, sem, audio, vision, qmask, umask, label2, vad2, vmask2, soft2 = tensors
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        emo = torch.cat([emo, audio, vision], dim=-1)

        out = model(emo, sem, qmask, umask, seq_lengths)
        log_prob, feats = out['log_prob'], out['fused']
        label, vad, vmask, soft = (flat(t, seq_lengths) for t in (label2, vad2, vmask2, soft2))

        w = class_w[label]
        # hard-label NLL with the authors' class weights: used for early stopping in every config
        ce_hard = (w * -log_prob.gather(1, label[:, None]).squeeze(1)).mean()
        loss = ce_hard
        if train and args.soft_beta > 0:
            target = (1 - args.soft_beta) * torch.nn.functional.one_hot(label, 6).float() + args.soft_beta * soft
            loss = (w * -(target * log_prob).sum(1)).mean()

        if args.lambda_vad > 0:
            vz = (vad - vad_mu) / vad_sd
            vp = vad_head(feats)
            m = vmask > 0
            vad_loss = ((vp[m][:, vad_dims] - vz[m][:, vad_dims]) ** 2).mean() if m.any() else feats.sum() * 0
            stats['vad'].append(vad_loss.item())
            if train:
                loss = loss + args.lambda_vad * vad_loss
            vad_pred.append(vp[m].detach().cpu().numpy()); vad_true.append(vz[m].cpu().numpy())

        if out['gate'] is not None:
            l_gate, l_aux = gate_losses(out['gate'], label, out['valid'], args.gate_loss, args.utility_T)
            stats['gate'].append(float(l_gate)); stats['aux'].append(float(l_aux))
            if train:
                loss = loss + args.lambda_gate * l_gate + args.lambda_aux * l_aux
            v = out['valid']
            keep['g'].append(out['gate']['g'][v].detach().cpu().numpy())
            keep['tau'].append(out['gate']['tau'][v].detach().cpu().numpy())
            keep['avail'].append(out['gate']['avail'][v].cpu().numpy())
            keep['ctx'].append(out['n_ctx'].cpu().numpy())

        if train:
            loss.backward()
            optimizer.step()

        stats['ce'].append(ce_hard.item())
        preds.append(log_prob.argmax(1).cpu().numpy())
        labels.append(label.cpu().numpy())
        probs.append(log_prob.detach().exp().cpu().numpy())

    labels, preds, probs = np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)
    cm = metrics.confusion_matrix(labels, preds, labels=list(range(6)))
    out = {'loss': float(np.mean(stats['ce'])),
           'vad_mse': float(np.mean(stats['vad'])) if stats['vad'] else float('nan'),
           'gate_loss': float(np.mean(stats['gate'])) if stats['gate'] else float('nan'),
           'acc': accuracy_score(labels, preds) * 100,
           'wf1': f1_score(labels, preds, average='weighted') * 100,
           'mf1': f1_score(labels, preds, average='macro') * 100,
           'hap_exc': int(cm[0, 4] + cm[4, 0]), 'ang_fru': int(cm[3, 5] + cm[5, 3]),
           'cm': cm, 'labels': labels, 'preds': preds, 'probs': probs}
    if vad_pred:
        vp, vt = np.concatenate(vad_pred), np.concatenate(vad_true)
        out['vad_r'] = [float(np.corrcoef(vp[:, k], vt[:, k])[0, 1]) for k in range(3)]  # V, A, D Pearson r
    if keep['g']:
        out['gate'] = gate_metrics(*(np.concatenate(keep[k]) for k in ['g', 'tau', 'ctx', 'avail']))
    return out


def scaled(probs, T):
    z = np.log(np.clip(probs, 1e-12, None)) / T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def nll(probs, labels):
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../ECERC/data/iemocap')
    parser.add_argument('--output_dir', type=str, default='./merged_run')
    parser.add_argument('--tag', type=str, default='CTRL')
    parser.add_argument('--seed', type=int, default=2007)
    parser.add_argument('--base_layer', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--valid_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.0002)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    # Limitation 1
    parser.add_argument('--lambda_vad', type=float, default=0.0, help='weight of the auxiliary VAD regression loss')
    parser.add_argument('--vad_dims', type=str, default='vad', help="which dims to supervise: any subset of 'vad'")
    parser.add_argument('--soft_beta', type=float, default=0.0,
                        help='0 = hard labels, 1 = pure annotator-vote distribution')
    parser.add_argument('--audio_feat', type=str, default=None,
                        help='path to a pkl from extract_wavlm.py; default = the original IS10 audio features')
    # Limitation 3 (names and defaults as in Lakshay's ext/train.py)
    parser.add_argument('--gate', choices=['baseline', 'softmax', 'cacg'], default='baseline')
    parser.add_argument('--gate_loss', choices=['none', 'entropy', 'utility'], default='none')
    parser.add_argument('--lambda_gate', type=float, default=0.1)
    parser.add_argument('--lambda_aux', type=float, default=0.0)
    parser.add_argument('--aux_heads', action='store_true')
    parser.add_argument('--utility_T', type=float, default=1.0)
    parser.add_argument('--tau_min', type=float, default=0.25)
    parser.add_argument('--tau_max', type=float, default=4.0)
    parser.add_argument('--fixed_tau', type=float, default=1.0)
    parser.add_argument('--mask_unavailable', type=int, default=1)
    parser.add_argument('--fix_empty_attention', type=int, default=0)
    parser.add_argument('--save_ckpt', action='store_true', default=False)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = '{}_s{}'.format(args.tag, args.seed)
    result_path = os.path.join(args.output_dir, 'result_{}.json'.format(run_name))
    if os.path.exists(result_path):
        print('Skip, already done:', result_path)
        raise SystemExit(0)

    train_loader, valid_loader, test_loader, vad_mu, vad_sd, D_audio = get_loaders(
        args.data_dir, args.batch_size, args.valid_rate, args.audio_feat)

    seed_everything(args.seed)
    D_text, D_visual, hidden_size = 1024, 342, 128   # D_audio: 1582 for IS10, 1024 for WavLM-large
    gate_kwargs = None
    if args.gate != 'baseline':
        gate_kwargs = dict(tau_min=args.tau_min, tau_max=args.tau_max, fixed_tau=args.fixed_tau,
                           mask_unavailable=bool(args.mask_unavailable), aux_heads=args.aux_heads)
    # IEMOCAP release settings: pos_scale 1.0, dropout 0.5 / 0.5
    model = ECERC(d_t=D_text, d_a=D_audio, d_v=D_visual, base_layer=args.base_layer, hidden_size=hidden_size,
                  n_classes=6, pos_scale=1.0, dropout1=0.5, dropout2=0.5, gate=args.gate, gate_kwargs=gate_kwargs,
                  fix_empty_attention=bool(args.fix_empty_attention)).to(args.device)
    vad_head = nn.Linear(hidden_size * 3 * 4, 3).to(args.device)   # on the fused 4-cause feature R (1536-d)
    vad_mu = torch.tensor(vad_mu, device=args.device)
    vad_sd = torch.tensor(vad_sd, device=args.device)
    vad_dims = [i for i, c in enumerate('vad') if c in args.vad_dims]
    class_w = torch.tensor(CLASS_WEIGHTS, device=args.device)

    params = list(model.parameters()) + (list(vad_head.parameters()) if args.lambda_vad > 0 else [])
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.l2)
    print('{} | device {} | ECERC params {} | gate {} ({}) | audio dim {} | lambda_vad {} ({}) | soft_beta {}'.format(
        run_name, args.device, sum(p.numel() for p in model.parameters()), args.gate, args.gate_loss, D_audio,
        args.lambda_vad, args.vad_dims, args.soft_beta))

    best, best_valid_f1, best_valid_loss, patience, patience2 = None, -1, None, 0, 0
    t0 = time.time()
    for e in range(args.epochs):
        start = time.time()
        tr = run_epoch(model, vad_head, train_loader, args, class_w, vad_mu, vad_sd, vad_dims, optimizer)
        with torch.no_grad():
            va = run_epoch(model, vad_head, valid_loader, args, class_w, vad_mu, vad_sd, vad_dims)
            te = run_epoch(model, vad_head, test_loader, args, class_w, vad_mu, vad_sd, vad_dims)

        # same model-selection / early-stopping rule as the authors' train.py
        if e == 0 or va['wf1'] > best_valid_f1:
            best_valid_f1, patience = va['wf1'], 0
            best = {'epoch': e, 'valid': va, 'test': te}
            if args.save_ckpt:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'ECERC_{}.pkl'.format(run_name)))
                if args.lambda_vad > 0:
                    torch.save(vad_head.state_dict(), os.path.join(args.output_dir, 'VADhead_{}.pkl'.format(run_name)))
        else:
            patience += 1
        if best_valid_loss is None or va['loss'] < best_valid_loss:
            best_valid_loss, patience2 = va['loss'], 0
        else:
            patience2 += 1

        g = va.get('gate')
        print('epoch {:3d} | train loss {:.4f} f1 {:.2f} vad {:.3f} | valid f1 {:.2f} | test acc {:.2f} f1 {:.2f} '
              'hap<->exc {} ang<->fru {} {}| {:.1f}s'.format(
                  e, tr['loss'], tr['wf1'], tr['vad_mse'], va['wf1'], te['acc'], te['wf1'], te['hap_exc'],
                  te['ang_fru'], 'tau {:.2f}+-{:.2f} '.format(g['tau_mean'], g['tau_std']) if g else '',
                  time.time() - start), flush=True)
        if patience >= args.patience and patience2 >= args.patience:
            break

    te, va = best['test'], best['valid']
    # Limitation 3: post-hoc temperature scaling, fitted on the validation split only
    T = fit_temperature(np.log(np.clip(va['probs'], 1e-12, None)), va['labels'])
    te_ts = scaled(te['probs'], T)
    summary = {k: v for k, v in vars(args).items()}
    summary.update({'run': run_name, 'best_epoch': best['epoch'], 'minutes': (time.time() - t0) / 60,
                    'valid_wf1': va['wf1'], 'valid_mf1': va['mf1'],
                    'test_acc': te['acc'], 'test_wf1': te['wf1'], 'test_mf1': te['mf1'],
                    'test_hap_exc': te['hap_exc'], 'test_ang_fru': te['ang_fru'],
                    'test_ece': 100 * ece_score(te['probs'], te['labels']),
                    'test_ece_ts': 100 * ece_score(te_ts, te['labels']),
                    'test_nll': nll(te['probs'], te['labels']), 'test_nll_ts': nll(te_ts, te['labels']),
                    'ts_temperature': T, 'test_gate': te.get('gate'),
                    'test_vad_r': te.get('vad_r'), 'test_cm': te['cm'].tolist(),
                    'test_report': metrics.classification_report(te['labels'], te['preds'],
                                                                 target_names=TARGET_NAMES, digits=4)})
    np.savez(os.path.join(args.output_dir, 'preds_{}.npz'.format(run_name)),
             labels=te['labels'], preds=te['preds'], probs=te['probs'], probs_ts=te_ts,
             valid_labels=va['labels'], valid_probs=va['probs'])
    with open(result_path, 'w') as f:
        json.dump(summary, f, indent=1)
    print(summary['test_report'])
    print(te['cm'])
    print('BEST epoch {} | test acc {:.2f} wF1 {:.2f} mF1 {:.2f} | hap<->exc {} ang<->fru {} | '
          'ECE {:.2f} -> TS {:.2f} (T={:.2f}) | VAD r {}'.format(
              best['epoch'], te['acc'], te['wf1'], te['mf1'], te['hap_exc'], te['ang_fru'],
              summary['test_ece'], summary['test_ece_ts'], T, te.get('vad_r')))
