"""One-time conversion of the released feature pickles into compact float32 caches.
The originals store Python float lists and need >8 GB RAM to load together; the cache is ~10x smaller
and loads in seconds. Usage: python prepare_data.py [--data_root ../data]"""
import argparse
import gc
import os
import pickle

import numpy as np


def load(path):
    print('loading', path, flush=True)
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def f32(d, keys):
    return {k: np.asarray(d[k], dtype=np.float32) for k in keys}


def iemocap(root):
    d = load(os.path.join(root, 'iemocap', 'IEMOCAP_features.pkl'))
    _, spk, labels, _, audio, visual, _, train_vid, test_vid = d
    keys = list(train_vid) + list(test_vid)
    out = {'train': list(train_vid), 'test': list(test_vid),
           'speakers': {k: np.array([[1, 0] if x == 'M' else [0, 1] for x in spk[k]], np.float32) for k in keys},
           'labels': {k: np.asarray(labels[k], np.int64) for k in keys},
           'audio': f32(audio, keys), 'visual': f32(visual, keys)}
    del d; gc.collect()
    d = load(os.path.join(root, 'iemocap', 'iemocap_emotion_features_roberta.pkl'))
    out['emo'] = f32(d[2], keys)
    del d; gc.collect()
    d = load(os.path.join(root, 'iemocap', 'iemocap_emotion_semantic_features_roberta.pkl'))
    out['sem'] = f32(d[6], keys)
    del d; gc.collect()
    return out


def meld(root):
    d = load(os.path.join(root, 'meld', 'meld_emotion_semantic_features_roberta.pkl'))
    spk, labels, emo, sem, audio, visual, train_ids, test_ids, valid_ids = d[0], d[1], d[3], d[7], d[11], d[12], d[14], d[15], d[16]
    keys = list(train_ids) + list(test_ids) + list(valid_ids)
    out = {'train': list(train_ids), 'test': list(test_ids), 'valid': list(valid_ids),
           'speakers': f32(spk, keys), 'labels': {k: np.asarray(labels[k], np.int64) for k in keys},
           'emo': f32(emo, keys), 'sem': f32(sem, keys), 'audio': f32(audio, keys), 'visual': f32(visual, keys)}
    del d; gc.collect()
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    p.add_argument('--only', choices=['IEMOCAP', 'MELD'], default=None)
    a = p.parse_args()
    for name, fn in [('IEMOCAP', iemocap), ('MELD', meld)]:
        if a.only and a.only != name:
            continue
        dst = os.path.join(a.data_root, name.lower(), 'cache.pkl')
        with open(dst, 'wb') as f:
            pickle.dump(fn(a.data_root), f, protocol=4)
        print('wrote', dst, os.path.getsize(dst) // 2 ** 20, 'MB', flush=True)
        gc.collect()
