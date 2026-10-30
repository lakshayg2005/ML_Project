"""One-time build of the Limitation 1 inputs for IEMOCAP (Divyansh's files), aligned per dialogue:
  * WavLM-large audio features (layer 20, 1024-d)  <- iemocap_wavlm.pkl      (replaces OpenSMILE IS10)
  * VAD targets (valence, activation, dominance)   <- iemocap_aux_labels.pkl (auxiliary regression)
  * annotator vote distributions over the 6 classes <- iemocap_aux_labels.pkl (soft labels)
Both pickles go in data/iemocap/. Output: data/iemocap/extras.pkl
Usage: python prepare_extras.py [--data_root ../data]"""
import argparse
import os
import pickle

import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    a = p.parse_args()
    root = os.path.join(a.data_root, 'iemocap')
    with open(os.path.join(root, 'IEMOCAP_features.pkl'), 'rb') as f:
        video_ids, _, labels, *_ = pickle.load(f, encoding='latin1')
    with open(os.path.join(root, 'iemocap_aux_labels.pkl'), 'rb') as f:
        aux = pickle.load(f)
    with open(os.path.join(root, 'iemocap_wavlm.pkl'), 'rb') as f:
        wavlm = pickle.load(f)

    out = {'vad': {}, 'vmask': {}, 'votes': {}, 'wavlm': {}, 'wavlm_meta': wavlm['meta']}
    missing = 0
    for vid, uids in video_ids.items():
        n = len(uids)
        vad, vmask, votes = np.zeros((n, 3), np.float32), np.zeros(n, np.float32), np.zeros((n, 6), np.float32)
        for i, (uid, y) in enumerate(zip(uids, labels[vid])):
            rec = aux.get(uid)
            if rec is None:
                missing += 1
                votes[i, y] = 1.  # no annotation: fall back to the hard label
            else:
                vad[i], vmask[i], votes[i] = rec['vad'], 1., rec['votes']
        w = np.asarray(wavlm['features'][vid], np.float32)
        assert w.shape[0] == n, (vid, w.shape, n)
        out['vad'][vid], out['vmask'][vid], out['votes'][vid], out['wavlm'][vid] = vad, vmask, votes, w
    print(f'{len(out["vad"])} dialogues | utterances without aux labels: {missing} | wavlm dim {w.shape[1]}')
    with open(os.path.join(root, 'extras.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print('saved', os.path.join(root, 'extras.pkl'))
