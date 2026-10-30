import os
import pickle

import numpy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class IEMOCAPAuxDataset(Dataset):
    """
    Same inputs as dataloader.IEMOCAPRobertaDataset, plus per-utterance auxiliary targets:
    VAD (valence, activation, dominance), a VAD-present mask and the annotator vote distribution.
    label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    """

    def __init__(self, data_dir='../data/iemocap', aux_file='iemocap_aux_labels.pkl', train=True, audio_file=None):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(os.path.join(data_dir, 'IEMOCAP_features.pkl'), 'rb'), encoding='latin1')

        _, _, self.roberta1, _, _, _, \
        _, _, _, _ = pickle.load(open(os.path.join(data_dir, 'iemocap_emotion_features_roberta.pkl'), 'rb'), encoding='latin1')

        _, _, _, _, _, _, \
        self.sroberta1, _, _, _, \
        _, _, _, _, _, _ = pickle.load(open(os.path.join(data_dir, 'iemocap_emotion_semantic_features_roberta.pkl'), 'rb'),
                                       encoding='latin1')

        self.aux = pickle.load(open(os.path.join(data_dir, aux_file), 'rb'))

        if audio_file:  # e.g. WavLM features from extract_wavlm.py, replacing the IS10 audio features
            self.videoAudio = pickle.load(open(audio_file, 'rb'))['features']

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        labels = self.videoLabels[vid]
        vad, vmask, soft = [], [], []
        for uid, y in zip(self.videoIDs[vid], labels):
            rec = self.aux.get(uid)
            if rec is None:
                vad.append([0., 0., 0.]); vmask.append(0.)
                onehot = [0.] * 6; onehot[y] = 1.; soft.append(onehot)
            else:
                vad.append(rec['vad'].tolist()); vmask.append(1.)
                soft.append(rec['votes'].tolist())
        return torch.FloatTensor(numpy.array(self.roberta1[vid])), \
               torch.FloatTensor(numpy.array(self.sroberta1[vid])), \
               torch.FloatTensor(numpy.array(self.videoAudio[vid])), \
               torch.FloatTensor(numpy.array(self.videoVisual[vid])), \
               torch.FloatTensor(numpy.array([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]])), \
               torch.FloatTensor(numpy.array([1] * len(labels))), \
               torch.LongTensor(numpy.array(labels)), \
               torch.FloatTensor(numpy.array(vad)), \
               torch.FloatTensor(numpy.array(vmask)), \
               torch.FloatTensor(numpy.array(soft)), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        cols = list(zip(*data))
        # first 5 (features, qmask) are time-major like the original loader; the rest are batch-first
        return [pad_sequence(c) if i < 5 else pad_sequence(c, True) if i < 10 else list(c) for i, c in enumerate(cols)]
