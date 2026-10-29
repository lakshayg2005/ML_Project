"""
Extract one WavLM vector per IEMOCAP utterance, to replace the IS10 OpenSMILE audio features.

For every utterance: run WavLM, average each layer's output over time.
The layer to keep is chosen with a linear probe trained on the training split and scored on the
validation split (same 10% split as train.py), so the test set plays no part in the choice.
Features are z-scored with training-split statistics.

    python extract_wavlm.py --wav_dir ../data/iemocap_wav --data_dir ../data/iemocap \
                            --out ../data/iemocap/iemocap_wavlm.pkl

Output pkl: {'features': {dialogue_id: float32 array (n_utterances, dim)}, 'meta': {...}}
"""
import argparse
import os
import pickle
import time
import warnings

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')
SR = 16000


def load_wav(path, max_sec):
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    return x[:int(max_sec * SR)]


def probe(Xtr, ytr, Xev, yev, pairs=False):
    """6-class macro-F1, plus balanced accuracy on hap-vs-exc and ang-vs-fru when pairs=True."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced'))
    clf.fit(Xtr, ytr)
    out = {'mF1': f1_score(yev, clf.predict(Xev), average='macro') * 100}
    if pairs:
        for a, b, name in [(0, 4, 'hap/exc'), (3, 5, 'ang/fru')]:
            mt, me = np.isin(ytr, [a, b]), np.isin(yev, [a, b])
            c = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced'))
            c.fit(Xtr[mt], ytr[mt] == a)
            out[name] = balanced_accuracy_score(yev[me] == a, c.predict(Xev[me])) * 100
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_dir', required=True, help='flat folder of <utterance_id>.wav (from collect_wavs.py)')
    p.add_argument('--data_dir', default='../data/iemocap', help='folder with IEMOCAP_features.pkl')
    p.add_argument('--out', default='../data/iemocap/iemocap_wavlm.pkl')
    p.add_argument('--model', default='microsoft/wavlm-large')
    p.add_argument('--max_sec', type=float, default=30.0, help='crop longer utterances (GPU memory)')
    p.add_argument('--layer', default='auto', help="'auto' (probe on validation), 'avg' (mean of all layers) or an index")
    p.add_argument('--valid_rate', type=float, default=0.1)
    args = p.parse_args()

    from transformers import AutoFeatureExtractor, AutoModel

    videoIDs, _, videoLabels, _, videoAudio, _, _, trainVid, testVid = pickle.load(
        open(os.path.join(args.data_dir, 'IEMOCAP_features.pkl'), 'rb'), encoding='latin1')
    trainVid, testVid = [x for x in trainVid], [x for x in testVid]   # same order as dataloader.py
    split = int(args.valid_rate * len(trainVid))
    parts = {'valid': trainVid[:split], 'train': trainVid[split:], 'test': testVid}
    all_vids = trainVid + testVid
    uids = [u for v in all_vids for u in videoIDs[v]]
    missing = [u for u in uids if not os.path.exists(os.path.join(args.wav_dir, u + '.wav'))]
    if missing:
        raise FileNotFoundError('{} wav files missing in {}, e.g. {}'.format(len(missing), args.wav_dir, missing[:3]))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    n_layers, dim = model.config.num_hidden_layers + 1, model.config.hidden_size
    print('{} on {}: {} hidden states x {} dims, {} utterances'.format(args.model, device, n_layers, dim, len(uids)))

    feats = np.zeros((len(uids), n_layers, dim), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for i, u in enumerate(uids):
            x = load_wav(os.path.join(args.wav_dir, u + '.wav'), args.max_sec)
            inp = fe(x, sampling_rate=SR, return_tensors='pt').input_values.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                hs = model(inp, output_hidden_states=True).hidden_states
            feats[i] = torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16)
            if (i + 1) % 500 == 0:
                print('  {}/{} utterances, {:.1f} min'.format(i + 1, len(uids), (time.time() - t0) / 60), flush=True)

    # row indices per split
    row, pos = {}, 0
    for v in all_vids:
        row[v] = list(range(pos, pos + len(videoIDs[v])))
        pos += len(videoIDs[v])
    idx = {k: [r for v in vs for r in row[v]] for k, vs in parts.items()}
    labels = np.array([y for v in all_vids for y in videoLabels[v]])
    is10 = np.concatenate([np.array(videoAudio[v]) for v in all_vids])

    def layer_feats(choice):
        return feats.astype(np.float32).mean(1) if choice == 'avg' else feats[:, int(choice)].astype(np.float32)

    # choose the layer on VALIDATION only
    if args.layer == 'auto':
        scores = {}
        for choice in ['avg'] + [str(l) for l in range(n_layers)]:
            X = layer_feats(choice)
            scores[choice] = probe(X[idx['train']], labels[idx['train']], X[idx['valid']], labels[idx['valid']])['mF1']
            print('  probe layer {:>3s}: validation macro-F1 {:.1f}'.format(choice, scores[choice]), flush=True)
        choice = max(scores, key=scores.get)
    else:
        choice, scores = args.layer, {}
    X = layer_feats(choice)
    print('Chosen layer:', choice)

    # report-only diagnostic: audio-only probes, IS10 vs WavLM, trained on train, scored on test
    for name, F in [('IS10', is10), ('WavLM', X)]:
        r = probe(F[idx['train']], labels[idx['train']], F[idx['test']], labels[idx['test']], pairs=True)
        print('Audio-only test probe {:6s}: 6-class macro-F1 {:.1f} | hap/exc bal-acc {:.1f} | ang/fru bal-acc {:.1f}'.format(
            name, r['mF1'], r['hap/exc'], r['ang/fru']))

    mu, sd = X[idx['train']].mean(0), X[idx['train']].std(0) + 1e-6
    X = (X - mu) / sd
    out = {'features': {v: X[row[v]].astype(np.float32) for v in all_vids},
           'meta': {'model': args.model, 'layer': choice, 'dim': dim, 'probe_valid_mF1': scores, 'max_sec': args.max_sec}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print('Saved', args.out, '| {:.1f} min total'.format((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
