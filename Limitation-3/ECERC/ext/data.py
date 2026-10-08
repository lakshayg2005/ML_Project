"""Dataset loaders replicating the official IEMOCAP/MELD dataloaders and splits.
Reads the compact cache written by prepare_data.py (same arrays as the release pickles, as float32)."""
import os
import pickle

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


class ConvDataset(Dataset):
    """Item layout matches the release: (emo_text, sem_text, slot_a, slot_v, speakers, umask, labels, vid).
    MELD release puts visual (342-d) in the audio slot and audio (300-d) in the vision slot; kept as-is so that
    the released checkpoint loads and numbers are comparable."""

    def __init__(self, data_root, dataset, split):
        self.d = _load(data_root, dataset)
        self.keys = list(self.d[split])
        self.slots = ('audio', 'visual') if dataset == 'IEMOCAP' else ('visual', 'audio')

    def __getitem__(self, index):
        vid, d = self.keys[index], self.d
        lab = torch.from_numpy(d['labels'][vid])
        return (torch.from_numpy(d['emo'][vid]), torch.from_numpy(d['sem'][vid]),
                torch.from_numpy(d[self.slots[0]][vid]), torch.from_numpy(d[self.slots[1]][vid]),
                torch.from_numpy(d['speakers'][vid]), torch.ones(len(lab)), lab, vid)

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def collate_fn(data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 5 else pad_sequence(dat[i], True) if i < 7 else dat[i].tolist() for i in dat]


def get_loaders(dataset, data_root, batch_size, valid_rate=0.1):
    cf = ConvDataset.collate_fn
    if dataset == 'IEMOCAP':
        trainset = ConvDataset(data_root, dataset, 'train')
        split = int(valid_rate * len(trainset))
        idx = list(range(len(trainset)))  # release: first 10% of training dialogues = validation (no shuffle)
        train = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[split:]), collate_fn=cf)
        valid = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(idx[:split]), collate_fn=cf)
    else:
        train = DataLoader(ConvDataset(data_root, dataset, 'train'), batch_size=batch_size, shuffle=True, collate_fn=cf)
        valid = DataLoader(ConvDataset(data_root, dataset, 'valid'), batch_size=batch_size, collate_fn=cf)
    test = DataLoader(ConvDataset(data_root, dataset, 'test'), batch_size=batch_size, collate_fn=cf)
    return train, valid, test
