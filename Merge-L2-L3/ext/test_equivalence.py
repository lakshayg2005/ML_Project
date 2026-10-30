"""Checks that ext/model.py (gate='baseline') reproduces the official model.py bit-for-bit,
and that CACG with a uniform gate at init leaves the output unchanged."""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import ECERC as ExtECERC  # noqa: E402


def load_official(dataset):
    spec = importlib.util.spec_from_file_location(f'official_{dataset}', os.path.join(HERE, '..', dataset, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ECERC


def check(dataset, d_a, d_v, n_cls, P, pos_scale, dr1, dr2):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    off = load_official(dataset)(None, d_t=1024, d_a=d_a, d_v=d_v, base_layer=1, input_size=1024 + d_a + d_v,
                                 hidden_size=128, n_speakers=P, n_classes=n_cls, cuda_flag=dev.type == 'cuda').to(dev).eval()
    ext = ExtECERC(1024, d_a, d_v, n_classes=n_cls, pos_scale=pos_scale, dropout1=dr1, dropout2=dr2).to(dev).eval()
    ext.load_state_dict(off.state_dict())  # strict: parameter names identical
    cacg = ExtECERC(1024, d_a, d_v, n_classes=n_cls, pos_scale=pos_scale, dropout1=dr1, dropout2=dr2, gate='cacg',
                    gate_kwargs=dict(mask_unavailable=False)).to(dev).eval()
    cacg.load_state_dict(off.state_dict(), strict=False)

    lens = [12, 7, 3]
    L, B = max(lens), len(lens)
    U_e = torch.randn(L, B, 1024 + d_a + d_v, device=dev)
    U_s = torch.randn(L, B, 1024, device=dev)
    spk = torch.randint(0, P, (L, B), device=dev)
    qmask = torch.nn.functional.one_hot(spk, P).float()
    umask = torch.zeros(B, L, device=dev)
    for j, n in enumerate(lens):
        umask[j, :n] = 1
        U_e[n:, j] = 0; U_s[n:, j] = 0; qmask[n:, j] = 0
    with torch.no_grad():
        a = off(U_e, U_s, qmask, umask, lens)
        b = ext(U_e, U_s, qmask, umask, lens)['log_prob']
        c = cacg(U_e, U_s, qmask, umask, lens)['log_prob']
    print(f'{dataset}: max|official - ext| = {(a - b).abs().max():.2e}, max|official - cacg@init| = {(a - c).abs().max():.2e}')
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)


if __name__ == '__main__':
    check('IEMOCAP', 1582, 342, 6, 2, 1.0, 0.5, 0.5)
    check('MELD', 300, 342, 7, 9, 0.0, 0.0, 0.2)
    print('OK')
