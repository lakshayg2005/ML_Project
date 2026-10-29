"""
Build auxiliary labels (VAD + annotator vote distributions) for IEMOCAP from the raw
EmoEvaluation files, aligned to the utterance IDs in IEMOCAP_features.pkl.

Run ONCE (locally or on Kaggle), then upload the output pkl with the other features:
    python build_aux_labels.py --iemocap_root /path/to/IEMOCAP_full_release \
                               --features ../data/iemocap/IEMOCAP_features.pkl \
                               --out ../data/iemocap/iemocap_aux_labels.pkl

Output: dict {utterance_id: {'vad': np.float32[3] (valence, activation, dominance; 1-5 scale),
                             'votes': np.float32[6] (vote distribution over hap sad neu ang exc fru),
                             'n_votes': float}}
"""
import argparse
import glob
import os
import pickle
import re
from collections import Counter

import numpy as np

# label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
CLASSES = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
VOTE_PREFIX = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}  # matched on lower-cased prefix

HEADER = re.compile(r'^\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s+(\S+)\s+(\S+)\s+\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')


def vote_to_class(word):
    w = word.strip().lower()
    for prefix, idx in VOTE_PREFIX.items():
        if w.startswith(prefix):
            return idx
    return None  # surprise / fear / disgust / other -> ignored


def parse_file(path, include_self):
    out = {}
    cur = None
    with open(path, encoding='latin1') as f:
        for line in f:
            line = line.rstrip('\n')
            m = HEADER.match(line)
            if m:
                uid = m.group(3)
                cur = {'gold': m.group(4),
                       'vad': np.array([float(m.group(5)), float(m.group(6)), float(m.group(7))], dtype=np.float32),
                       'raw_votes': []}
                out[uid] = cur
                continue
            if cur is None or not line.startswith('C-'):
                continue
            # e.g. "C-E2:\tFrustration; Anger;\t()"   (C-E* = evaluators, C-F*/C-M* = actor self-report)
            who, _, rest = line.partition(':')
            if not include_self and not who.startswith('C-E'):
                continue
            words = [w for w in rest.split('(')[0].split(';') if w.strip()]
            cls = [c for c in (vote_to_class(w) for w in words) if c is not None]
            if words:
                # one vote per annotator, split evenly over the categories they ticked
                cur['raw_votes'].append((cls, len(words)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iemocap_root', required=True, help='folder containing Session1..Session5')
    p.add_argument('--features', default='../data/iemocap/IEMOCAP_features.pkl')
    p.add_argument('--out', default='../data/iemocap/iemocap_aux_labels.pkl')
    p.add_argument('--include_self', action='store_true', help='also count the actors\' self-assessments')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.iemocap_root, 'Session*', 'dialog', 'EmoEvaluation', '*.txt')))
    if not files:
        raise FileNotFoundError('No EmoEvaluation/*.txt files under ' + args.iemocap_root)
    parsed = {}
    for fp in files:
        parsed.update(parse_file(fp, args.include_self))
    print('Parsed {} utterances from {} files'.format(len(parsed), len(files)))

    videoIDs, _, videoLabels, *_rest = pickle.load(open(args.features, 'rb'), encoding='latin1')
    trainVid, testVid = _rest[-2], _rest[-1]

    aux, missing, gold_mismatch, no_votes, argmax_disagree = {}, [], 0, 0, 0
    for vid in list(trainVid) + list(testVid):
        for uid, y in zip(videoIDs[vid], videoLabels[vid]):
            if uid not in parsed:
                missing.append(uid)
                continue
            rec = parsed[uid]
            if rec['gold'] != CLASSES[y]:
                gold_mismatch += 1
            votes = np.zeros(6, dtype=np.float32)
            for cls, n_words in rec['raw_votes']:
                for c in cls:
                    votes[c] += 1.0 / n_words
            n = float(votes.sum())
            if n == 0:
                no_votes += 1
                votes[y] = 1.0
            else:
                votes /= n
                argmax_disagree += int(votes.argmax() != y)
            aux[uid] = {'vad': rec['vad'], 'votes': votes, 'n_votes': n}

    print('Aligned {} utterances | missing: {} | header-vs-pkl label mismatch: {} | '
          'no usable votes (one-hot fallback): {} | vote argmax != gold: {}'.format(
              len(aux), len(missing), gold_mismatch, no_votes, argmax_disagree))
    if missing:
        print('First missing IDs:', missing[:10])

    # How ambiguous are the confusable pairs according to the annotators?
    for a, b in [(0, 4), (3, 5)]:
        part = [r['votes'][b] if y == a else r['votes'][a]
                for vid in trainVid for uid, y in zip(videoIDs[vid], videoLabels[vid])
                if uid in aux and y in (a, b) for r in [aux[uid]]]
        part = np.array(part)
        print('{}/{}: mean vote mass on partner class {:.3f}, utterances with any partner vote {:.0f}%'.format(
            CLASSES[a], CLASSES[b], part.mean(), 100 * (part > 0).mean()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(aux, f)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
