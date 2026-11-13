#!/usr/bin/env python3
"""Rebuild the raw-frame manifest for a local copy of the dataset.

Usage:
  python build_manifest.py --data-root /path/to/accident_v26 --out raw_images_manifest.csv.gz
  python build_manifest.py --data-root /path/to/accident_v26 --sha256

The manifest contains no pixels: split, relative path, byte size, mtime, and optionally
a per-file sha256. It lets users verify that their local raw copy matches the released
annotation stems.
"""
import argparse, csv, gzip, hashlib, os


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default='raw_images_manifest.csv.gz')
    ap.add_argument('--sha256', action='store_true')
    args = ap.parse_args()

    header = ['split', 'relative_path', 'bytes', 'mtime_epoch'] + (['sha256'] if args.sha256 else [])
    n = 0
    with gzip.open(args.out, 'wt', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for split in ('train', 'val'):
            d = os.path.join(args.data_root, 'images', split)
            if not os.path.isdir(d):
                continue
            for name in os.scandir(d):
                if not name.is_file():
                    continue
                st = name.stat()
                row = [split, 'images/%s/%s' % (split, name.name), st.st_size, int(st.st_mtime)]
                if args.sha256:
                    row.append(sha256(name.path))
                w.writerow(row)
                n += 1
    print('wrote %s (%d rows)' % (args.out, n))


if __name__ == '__main__':
    main()
