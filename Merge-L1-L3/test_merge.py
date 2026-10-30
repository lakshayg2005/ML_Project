"""
Sanity checks for the merged code (CPU is fine, ~1 min):
  1. With the same seed, the merged model (gate='baseline') initialises exactly like the authors' model
     (model_con.py) and gives the same outputs in train mode (same dropout masks). So CTRL / W_VAD runs of the
     merged code should reproduce the earlier train_aux.py runs on the same hardware.
  2. CACG with the availability mask off equals the baseline at initialisation (Lakshay's check).
  3. One training step of CACG + WavLM + VAD runs, and the gate, the tau network and the VAD head all get gradients.

    python test_merge.py [--data_dir ../ECERC/data/iemocap] [--audio_feat ../ECERC/data/iemocap/iemocap_wavlm.pkl]
"""
import argparse
import importlib.util
import os

import torch
import torch.nn as nn

from model import ECERC
from train_merged import get_loaders, flat

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='../ECERC/data/iemocap')
p.add_argument('--audio_feat', default='../ECERC/data/iemocap/iemocap_wavlm.pkl')
p.add_argument('--authors_model', default='../ECERC/IEMOCAP/model_con.py')
args = p.parse_args()


def batch(loader):
    emo, sem, audio, vision, qmask, umask, label2, vad2, vmask2, soft2 = next(iter(loader))[:-1]
    lens = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
    return torch.cat([emo, audio, vision], -1), sem, qmask, umask, lens, label2, vad2, vmask2


def build(d_a, seed, **kw):
    torch.manual_seed(seed)
    return ECERC(d_t=1024, d_a=d_a, d_v=342, hidden_size=128, n_classes=6, pos_scale=1.0, dropout1=0.5,
                 dropout2=0.5, **kw)


_, _, test_loader, *_ = get_loaders(args.data_dir, 16)
U_e, sem, qmask, umask, lens, label2, _, _ = batch(test_loader)

# 1. merged baseline == authors' model
if os.path.exists(args.authors_model):
    spec = importlib.util.spec_from_file_location('model_con', args.authors_model)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.manual_seed(1)
    off = mod.ECERC(None, d_t=1024, d_a=1582, d_v=342, base_layer=1, input_size=1024 + 1582 + 342, hidden_size=128,
                    n_speakers=2, n_classes=6, cuda_flag=False)
    ext = build(1582, 1)
    sd_off, sd_ext = off.state_dict(), ext.state_dict()
    assert sd_off.keys() == sd_ext.keys(), 'parameter names differ'
    init_diff = max((sd_off[k].float() - sd_ext[k].float()).abs().max().item() for k in sd_off)
    off.train(); ext.train()
    torch.manual_seed(5); a = off(U_e, sem, qmask, umask, lens)
    torch.manual_seed(5); b = ext(U_e, sem, qmask, umask, lens)['log_prob']
    print('1. init max diff {:.1e} | train-mode output max diff {:.1e}'.format(init_diff, (a - b).abs().max().item()))
    assert init_diff == 0 and torch.allclose(a, b, atol=1e-5)
else:
    print('1. skipped (authors model not found at {})'.format(args.authors_model))

# 2. CACG with the mask off == baseline at initialisation
base = build(1582, 2).eval()
cacg = build(1582, 2, gate='cacg', gate_kwargs=dict(mask_unavailable=False)).eval()
cacg.load_state_dict(base.state_dict(), strict=False)
with torch.no_grad():
    d2 = (base(U_e, sem, qmask, umask, lens)['log_prob'] - cacg(U_e, sem, qmask, umask, lens)['log_prob']).abs().max()
print('2. CACG (mask off) vs baseline at init: max diff {:.1e}'.format(d2.item()))
assert d2 < 1e-5

# 3. one training step of the merged model on WavLM features
if os.path.exists(args.audio_feat):
    tr, _, _, mu, sd, D_audio = get_loaders(args.data_dir, 16, audio_file=args.audio_feat)
    U_e, sem, qmask, umask, lens, label2, vad2, vmask2 = batch(tr)
    model = build(D_audio, 3, gate='cacg').train()
    head = nn.Linear(1536, 3)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-4)
    label, vad, vmask = (flat(t, lens) for t in (label2, vad2, vmask2))
    m = vmask > 0
    # two steps: the cause scores start at 0, so tau only gets a gradient once the scorer has moved
    for _ in range(2):
        opt.zero_grad()
        out = model(U_e, sem, qmask, umask, lens)
        loss = nn.functional.nll_loss(out['log_prob'], label) + \
            ((head(out['fused'])[m] - (vad[m] - torch.tensor(mu)) / torch.tensor(sd)) ** 2).mean()
        loss.backward()
        opt.step()
    grads = {n: p.grad is not None and p.grad.abs().sum().item() > 0 for n, p in
             [('cause gate scorer', model.cause_gate.scorer.weight),
              ('tau network', model.cause_gate.tau_net[-1].weight), ('VAD head', head.weight)]}
    print('3. audio dim {} | log_prob {} fused {} | loss {:.3f} | gradients: {}'.format(
        D_audio, tuple(out['log_prob'].shape), tuple(out['fused'].shape), loss.item(), grads))
    assert out['fused'].shape[0] == label.shape[0] and all(grads.values())
else:
    print('3. skipped (WavLM features not found at {})'.format(args.audio_feat))
print('OK')
