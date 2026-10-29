"""
Extract only Session*/dialog/EmoEvaluation/*.txt from the IEMOCAP release.

IEMOCAP is licensed: use the download link USC SAIL sends you after approval
(https://sail.usc.edu/iemocap/release_form.php), or the archive you already downloaded.

    # stream from the official link: the full ~12 GB archive is never written to disk
    python fetch_emoevaluation.py --url "<link from USC email>" --out ../data/IEMOCAP_full_release

    # or from a local archive (.tar.gz / .tgz / .tar / .zip)
    python fetch_emoevaluation.py --archive ~/Downloads/IEMOCAP_full_release.tar.gz --out ../data/IEMOCAP_full_release

Then:
    python build_aux_labels.py --iemocap_root ../data/IEMOCAP_full_release
"""
import argparse
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

# .../Session3/dialog/EmoEvaluation/Ses03F_impro01.txt  (top level only, not Categorical/ etc.)
WANTED = re.compile(r'(?:^|/)(Session[1-5])/dialog/EmoEvaluation/([^/]+\.txt)$')


def target(name, out_dir):
    m = WANTED.search(name.replace('\\', '/'))
    if not m or m.group(2).startswith('._'):      # skip macOS AppleDouble files
        return None
    return os.path.join(out_dir, m.group(1), 'dialog', 'EmoEvaluation', m.group(2))


def save(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        shutil.copyfileobj(src, f)


def from_tar(fileobj, out_dir):
    n = 0
    # 'r|*' = sequential streaming read, auto-detects gzip/bz2/xz/plain tar
    with tarfile.open(fileobj=fileobj, mode='r|*') as tar:
        for member in tar:
            dst = target(member.name, out_dir) if member.isfile() else None
            if dst:
                save(tar.extractfile(member), dst)
                n += 1
                if n % 25 == 0:
                    print('  extracted', n, 'files ...', flush=True)
    return n


def from_zip(path, out_dir):
    n = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            dst = target(info.filename, out_dir)
            if dst and not info.is_dir():
                with z.open(info) as src:
                    save(src, dst)
                n += 1
    return n


class Progress:
    """File-like wrapper that prints download progress."""

    def __init__(self, raw, total):
        self.raw, self.total, self.read_bytes, self.next_report = raw, total, 0, 0

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.read_bytes += len(chunk)
        if self.read_bytes >= self.next_report:
            pct = ' ({:.0f}%)'.format(100 * self.read_bytes / self.total) if self.total else ''
            print('  downloaded {:.2f} GB{}'.format(self.read_bytes / 1e9, pct), flush=True)
            self.next_report += 500 * 1024 ** 2
        return chunk


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--url', help='official IEMOCAP download link')
    src.add_argument('--archive', help='local IEMOCAP_full_release archive')
    p.add_argument('--out', default='../data/IEMOCAP_full_release')
    args = p.parse_args()

    if args.archive:
        path = os.path.expanduser(args.archive)
        n = from_zip(path, args.out) if zipfile.is_zipfile(path) else from_tar(open(path, 'rb'), args.out)
    else:
        req = urllib.request.Request(args.url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req)
        total = int(resp.headers.get('Content-Length') or 0)
        if '.zip' in args.url.lower().split('?')[0]:
            # zip needs random access, so it has to be downloaded to a temp file first
            with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
                shutil.copyfileobj(Progress(resp, total), tmp, 1024 ** 2)
                tmp.flush()
                n = from_zip(tmp.name, args.out)
        else:
            n = from_tar(Progress(resp, total), args.out)

    print('Extracted {} EmoEvaluation files to {}'.format(n, os.path.abspath(args.out)))
    if n != 151:
        print('WARNING: expected 151 files (one per dialogue). Check the archive/link.')


if __name__ == '__main__':
    main()
