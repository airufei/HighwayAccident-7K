# Manifests

- `raw_images_manifest.csv.gz` — 435,440 rows, one per raw frame in `images/train`
  (346,048) and `images/val` (89,392): split, relative path, byte size, mtime. No pixels.
- `dataset_inventory.json` — file counts and byte sizes per top-level subset of the
  research tree (images 87.5 GiB, meta 3.5 GiB, external_data 1.15 GiB, etc.).
- `build_manifest.py` — regenerate the manifest for a local copy; add `--sha256` for
  per-file checksums (slow but exact).

Stems in the annotation tables are the filename without `.jpg`, e.g.
`images/val/tv17c30a31_0000001450.jpg` → stem `tv17c30a31_0000001450`.
