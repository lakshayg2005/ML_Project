"""Replay demo of the merged model (ECERC + L1 + L2 + L3) on the IEMOCAP and MELD test conversations.

At start-up the final checkpoints are run on every test dialogue (CPU, ~1 min), then a local page shows, per
utterance: the predicted emotion, its calibrated confidence (L3 temperature scaling), a flag on the least confident
predictions, the CACG cause weights (L3), and on MELD the model / LLM / fused probabilities (L2).
No extra packages: standard library http.server + the project's torch/numpy.

    python ui/app.py [--port 8000]      then open http://localhost:8000
"""
import argparse
import json
import os
import pickle
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from scipy.special import log_softmax, softmax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'ext'))
from torch.utils.data import DataLoader  # noqa: E402

from data import ConvDataset, get_loaders  # noqa: E402
from final_report import llm_scores, seeds_of  # noqa: E402
from losses import build_task_loss  # noqa: E402
from metrics import classification_metrics, fit_temperature  # noqa: E402
from train import DATASET_CFG, build_model, run_epoch  # noqa: E402

DATA, RUNS = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'runs')
DEFAULTS = dict(audio='default', focal_gamma=0.0, lambda_vad=0.0, soft_beta=0.0)
# final model per dataset (5-seed / 3-seed ablation in runs/RESULTS_merged.md); all trained seeds are ensembled.
# IEMOCAP drops L1's VAD loss + soft labels: they cost ~0.5 wF1 and temperature scaling already calibrates.
FINAL = {
    'IEMOCAP': dict(name='wavlm_cacg', llm=False,
                    parts=['ECERC', 'L1: WavLM audio features',
                           'L3: CACG cause gate + temperature scaling']),
    'MELD': dict(name='cacg_utility_g1', llm=True,
                 parts=['ECERC', 'L2: zero-shot LLM fusion (Qwen2.5-32B, cached scores)',
                        'L3: CACG cause gate (utility loss) + temperature scaling']),
}
CAUSES = ['self-contagion', 'cross-emotion', 'self-event', 'cross-event']
FLAG_COVERAGE = 0.8  # flag the 20% least confident predictions (threshold set on validation)
CP1252 = str.maketrans({'\x91': "'", '\x92': "'", '\x93': '"', '\x94': '"', '\x96': '-', '\x97': '-', '\x85': '...'})


def load_text():
    """Speaker names and transcripts of the test dialogues, cached in data/ui_text.pkl."""
    path = os.path.join(DATA, 'ui_text.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    out = {}
    d = pickle.load(open(os.path.join(DATA, 'iemocap', 'IEMOCAP_features.pkl'), 'rb'), encoding='latin1')
    out['IEMOCAP'] = {str(v): ([{'M': 'Male', 'F': 'Female'}[s] for s in d[1][v]], list(d[6][v])) for v in d[8]}
    d = pickle.load(open(os.path.join(DATA, 'meld', 'MELD_features.pkl'), 'rb'), encoding='latin1')
    out['MELD'] = {str(v): (['Speaker ' + 'ABCDEFGHI'[int(np.argmax(s))] for s in d[1][v]],
                            [t.translate(CP1252) for t in d[6][v]]) for v in d[8]}
    pickle.dump(out, open(path, 'wb'))
    return out


def run_member(ds, name, seed, device):
    tag = os.path.join(RUNS, ds, f'{name}_seed{seed}')
    args = SimpleNamespace(**{**DEFAULTS, **json.load(open(tag + '.json'))['args']})
    cfg = DATASET_CFG[ds]
    model = build_model(args, cfg).to(device)
    model.load_state_dict(torch.load(tag + '.pt', map_location=device))
    _, valid, test = get_loaders(ds, DATA, cfg['batch_size'], audio=args.audio)
    if ds == 'IEMOCAP':  # the release samples validation in random order; fix it so the members line up
        valid = DataLoader(valid.dataset, batch_size=cfg['batch_size'], sampler=sorted(valid.sampler.indices),
                           collate_fn=ConvDataset.collate_fn)
    loss = build_task_loss(args, torch.FloatTensor(cfg['class_weights']))
    res, v = run_epoch(model, test, device, args, loss), run_epoch(model, valid, device, args, loss)
    return res, v['lp'].astype(np.float64), v['y'], test.dataset.keys


def build(ds, device):
    f, cfg = FINAL[ds], DATASET_CFG[ds]
    names = cfg['target_names']
    members = [run_member(ds, f['name'], s, device) for s in seeds_of(ds, f['name'])]
    keys, y = members[0][3], members[0][0]['y']
    T = [fit_temperature(lv, yv) for _, lv, yv, _ in members]
    out = {'model_probs': np.mean([softmax(m[0]['lp'] / t, 1) for m, t in zip(members, T)], 0)}
    p_valid = np.mean([softmax(lv / t, 1) for (_, lv, _, _), t in zip(members, T)], 0) if not f['llm'] else None
    if f['llm']:  # L2: fuse the (raw) ensemble with the LLM; w and T chosen on validation
        llm, w_grid = llm_scores()
        (ld, yd), (lt, yt) = llm['dev'], llm['test']
        assert (yt == y).all() and (yd == members[0][2]).all(), 'utterance order differs from the L2 dumps'
        lv = np.log(np.mean([softmax(m[1], 1) for m in members], 0))
        lte = np.log(np.mean([softmax(m[0]['lp'], 1) for m in members], 0))
        w = max(w_grid, key=lambda w: classification_metrics(log_softmax((1 - w) * lv + w * ld, 1), yd, names)['mf1'])
        tf = fit_temperature((1 - w) * lv + w * ld, yd)
        out.update(llm_probs=softmax(lt, 1), fused_w=w,
                   probs=softmax(((1 - w) * lte + w * lt) / tf, 1))
        p_valid = softmax(((1 - w) * lv + w * ld) / tf, 1)
    else:
        out['probs'] = out['model_probs']
    thr = float(np.quantile(p_valid.max(1), 1 - FLAG_COVERAGE))
    g = np.mean([m[0]['g'] for m in members], 0) if members[0][0]['g'] is not None else None
    tau = np.mean([m[0]['tau'] for m in members], 0) if g is not None else None
    avail = members[0][0]['avail'] if g is not None else None

    text = load_text()[ds]
    pred = out['probs'].argmax(1)
    conf = out['probs'].max(1)
    dialogues, start = {}, 0
    for k in keys:
        spk, sent = text[str(k)]
        n = len(sent)
        utts = []
        for i in range(n):
            j = start + i
            u = {'speaker': spk[i], 'text': sent[i], 'label': names[y[j]], 'pred': names[pred[j]],
                 'conf': float(conf[j]), 'flag': bool(conf[j] < thr),
                 'probs': [float(x) for x in out['probs'][j]]}
            if g is not None:
                u['causes'] = [float(x) for x in g[j]]
                u['avail'] = [bool(x) for x in avail[j]]
                u['tau'] = float(tau[j])
            if f['llm']:
                u['model_probs'] = [float(x) for x in out['model_probs'][j]]
                u['llm_probs'] = [float(x) for x in out['llm_probs'][j]]
            utts.append(u)
        dialogues[str(k)] = {'id': str(k), 'n': n, 'acc': float(np.mean(pred[start:start + n] == y[start:start + n])),
                             'utts': utts}
        start += n
    assert start == len(y)
    m = classification_metrics(np.log(out['probs']), y, names)
    flagged = conf < thr
    summary = {'acc': m['acc'], 'wf1': m['wf1'], 'mf1': m['mf1'], 'ece': m['ece'],
               'flag_share': 100 * float(flagged.mean()),
               'acc_kept': 100 * float((pred == y)[~flagged].mean()), 'acc_flagged': 100 * float((pred == y)[flagged].mean()),
               'threshold': thr, 'members': len(members), 'fused_w': out.get('fused_w')}
    return {'names': names, 'causes': CAUSES if g is not None else None, 'parts': f['parts'], 'summary': summary,
            'order': [str(k) for k in keys], 'dialogues': dialogues}


class Handler(BaseHTTPRequestHandler):
    state = {}

    def _send(self, body, ctype):
        b = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        s = self.state
        if url.path == '/':
            return self._send(open(os.path.join(HERE, 'index.html'), encoding='utf-8').read(), 'text/html; charset=utf-8')
        if url.path == '/api/meta':
            return self._send(json.dumps({ds: {k: v for k, v in s[ds].items() if k != 'dialogues'}
                                          | {'list': [{'id': d, 'n': s[ds]['dialogues'][d]['n'],
                                                       'acc': s[ds]['dialogues'][d]['acc']} for d in s[ds]['order']]}
                                          for ds in s}), 'application/json')
        if url.path == '/api/dialogue' and q.get('ds') in s and q.get('id') in s[q['ds']]['dialogues']:
            return self._send(json.dumps(s[q['ds']]['dialogues'][q['id']]), 'application/json')
        self.send_error(404)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8000)
    a = p.parse_args()
    device = torch.device('cpu')
    for ds in FINAL:
        print(f'running the final {ds} model on the test set ...', flush=True)
        Handler.state[ds] = build(ds, device)
        sm = Handler.state[ds]['summary']
        print(f"  {ds}: acc {sm['acc']:.2f} wF1 {sm['wf1']:.2f} mF1 {sm['mf1']:.2f} ECE {sm['ece']:.2f}", flush=True)
    print(f'open http://localhost:{a.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
