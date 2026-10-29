"""
Collect only the utterance .wav files ECERC uses (the 7,433 utterances in IEMOCAP_features.pkl)
from the official IEMOCAP release, so you upload ~1 GB instead of the full ~12 GB archive.

Source files in the release:  Session{1-5}/sentences/wav/<dialogue>/<utterance>.wav
Output layout:                <out>/<utterance>.wav   (flat folder)

    # from the archive (.tar.gz is streamed, never fully unpacked)
    python collect_wavs.py --archive ~/Downloads/IEMOCAP_full_release.tar.gz --out ../data/iemocap_wav
    # or from an already-extracted release folder
    python collect_wavs.py --iemocap_root /path/to/IEMOCAP_full_release --out ../data/iemocap_wav

Then zip the output folder and upload it to Kaggle as a private dataset.
"""
import argparse
import glob
import os
import pickle
import re
import shutil
import tarfile
import zipfile

WAV = re.compile(r'(?:^|/)Session[1-5]/sentences/wav/[^/]+/([^/]+)\.wav$')


def wanted_ids(features):
    videoIDs, _, _, _, _, _, _, trainVid, testVid = pickle.load(open(features, 'rb'), encoding='latin1')
    return {uid for vid in list(trainVid) + list(testVid) for uid in videoIDs[vid]}


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--archive', help='IEMOCAP_full_release .tar.gz / .tar / .zip')
    src.add_argument('--iemocap_root', help='extracted folder containing Session1..Session5')
    p.add_argument('--features', default='../data/iemocap/IEMOCAP_features.pkl')
    p.add_argument('--out', default='../data/iemocap_wav')
    args = p.parse_args()

    ids = wanted_ids(args.features)
    os.makedirs(args.out, exist_ok=True)
    found = set()

    def take(name, open_fn):
        m = WAV.search(name.replace('\\', '/'))
        if not m or m.group(1) not in ids or os.path.basename(name).startswith('._'):
            return
        with open_fn() as src_f, open(os.path.join(args.out, m.group(1) + '.wav'), 'wb') as dst:
            shutil.copyfileobj(src_f, dst)
        found.add(m.group(1))
        if len(found) % 500 == 0:
            print('  collected', len(found), flush=True)

    if args.iemocap_root:
        for path in glob.glob(os.path.join(args.iemocap_root, 'Session*', 'sentences', 'wav', '*', '*.wav')):
            take(path, lambda path=path: open(path, 'rb'))
    elif zipfile.is_zipfile(args.archive):
        with zipfile.ZipFile(args.archive) as z:
            for info in z.infolist():
                take(info.filename, lambda info=info: z.open(info))
    else:
        with tarfile.open(os.path.expanduser(args.archive), mode='r|*') as tar:
            for member in tar:
                if member.isfile():
                    take(member.name, lambda member=member: tar.extractfile(member))

    missing = sorted(ids - found)
    print('Collected {} / {} wav files into {}'.format(len(found), len(ids), os.path.abspath(args.out)))
    if missing:
        print('MISSING {} files, e.g. {}'.format(len(missing), missing[:5]))


if __name__ == '__main__':
    main()
