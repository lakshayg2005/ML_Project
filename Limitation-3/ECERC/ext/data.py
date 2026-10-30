"""Dataset loaders replicating the official IEMOCAP/MELD dataloaders and splits.
Reads the compact cache written by prepare_data.py (same arrays as the release pickles, as float32), plus the
optional Limitation 1 extras (WavLM audio, VAD, annotator votes) written by prepare_extras.py."""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

_CACHE = {}


def _load(data_root, dataset):
    path = os.path.join(data_root, dataset.lower(), 'cache.pkl')
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} missing -- run `python prepare_data.py` first')
        with open(path, 'rb') as f:
            _CACHE[path] = pickle.load(f)
    return _CACHE[path]


def _load_extras(data_root, dataset):
    """Limitation 1 inputs from prepare_extras.py (IEMOCAP only); None when absent."""
    path = os.path.join(data_root, dataset.lower(), 'extras.pkl')
    if path not in _CACHE:
        _CACHE[path] = None
        if os.path.exists(path):
            with open(path, 'rb') as f:
                _CACHE[path] = pickle.load(f)
    return _CACHE[path]


class ConvDataset(Dataset):
    """Item layout: (emo_text, sem_text, slot_a, slot_v, speakers, umask, labels, vad, vad_mask, votes, vid).
    The first 7 match the release. MELD release puts visual (342-d) in the audio slot and audio (300-d) in the
    vision slot; kept as-is so that the released checkpoint loads and numbers are comparable.
    audio='wavlm' swaps the IS10 audio for WavLM (Limitation 1). Without extras, vad/vad_mask are zeros and
    votes is the one-hot label."""

    def __init__(self, data_root, dataset, split, audio='default', n_classes=None):
        self.d = _load(data_root, dataset)
        self.x = _load_extras(data_root, dataset)
        self.keys = list(self.d[split])
        self.slots = ('audio', 'visual') if dataset == 'IEMOCAP' else ('visual', 'audio')
        self.n_classes = n_classes or (6 if dataset == 'IEMOCAP' else 7)
        if audio == 'wavlm' and self.x is None:
            raise FileNotFoundError('--audio wavlm needs extras.pkl -- run `python prepare_extras.py` first')
        self.audio = self.x['wavlm'] if audio == 'wavlm' else self.d[self.slots[0]]

    def __getitem__(self, index):
        vid, d = self.keys[index], self.d
        y = d['labels'][vid]
        n = len(y)
        if self.x is not None:
            vad, vmask, votes = self.x['vad'][vid], self.x['vmask'][vid], self.x['votes'][vid]
        else:
            vad, vmask = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
            votes = np.eye(self.n_classes, dtype=np.float32)[y]
        return (torch.from_numpy(d['emo'][vid]), torch.from_numpy(d['sem'][vid]),
                torch.from_numpy(self.audio[vid]), torch.from_numpy(d[self.slots[1]][vid]),
                torch.from_numpy(d['speakers'][vid]), torch.ones(n), torch.from_numpy(y),
                torch.from_numpy(vad), torch.from_numpy(vmask), torch.from_numpy(votes), vid)

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def collate_fn(data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 5 else pad_sequence(dat[i], True) if i < 10 else dat[i].tolist() for i in dat]


def get_loaders(dataset, data_root, batch_size, valid_rate=0.1, audio='default'):
    cf = ConvDataset.collate_fn
    if dataset == 'IEMOCAP':
        trainset = ConvDataset(data_root, dataset, 'train', audio)
        split = int(valid_rate * len(trainset))
        idx = list(range(len(trainset)))  # release: first 10% of training dialogues = validation (no shuffle)
        train = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[split:]), collate_fn=cf)
        valid = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[:split]), collate_fn=cf)
    else:
        train = DataLoader(ConvDataset(data_root, dataset, 'train', audio), batch_size=batch_size, shuffle=True,
                           collate_fn=cf)
        valid = DataLoader(ConvDataset(data_root, dataset, 'valid', audio), batch_size=batch_size, collate_fn=cf)
    test = DataLoader(ConvDataset(data_root, dataset, 'test', audio), batch_size=batch_size, collate_fn=cf)
    return train, valid, test


def vad_stats(dataset, data_root, valid_rate=0.1):
    """Mean / std of the VAD targets over the training dialogues (validation split excluded); None without extras."""
    x = _load_extras(data_root, dataset)
    if x is None:
        return None
    keys = list(_load(data_root, dataset)['train'])
    keys = keys[int(valid_rate * len(keys)):] if dataset == 'IEMOCAP' else keys
    v = np.concatenate([x['vad'][k][x['vmask'][k] > 0] for k in keys])
    return v.mean(0), v.std(0)
