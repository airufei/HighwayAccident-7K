# Consolidated Annotation Tables

Generated from the per-frame files without changing any label value.

| File | Rows (approx.) | Content |
|---|---|---|
| `scene_cause_train.csv.gz` | 346,048 | scene id + 9 cause bits + validity |
| `scene_cause_val.csv.gz` | 89,392 | same |
| `type_train.csv.gz` | 346,048 | type id (-1 = unlabeled) |
| `type_val.csv.gz` | 89,392 | same |
| `detection_train.csv.gz` | 4,156,605 boxes | stem, class, cx, cy, w, h |
| `detection_val.csv.gz` | 1,030,926 boxes | same |
| `lane_train.jsonl.gz` | 14,338 | lane JSON objects |
| `lane_val.jsonl.gz` | 2,239 | same |

`stem` matches `manifests/raw_images_manifest.csv.gz`. The image manifest contains no
pixels; it maps each stem to its split, byte size, and mtime in the local raw copy.

Note on scene ids: the raw files use 0 = ignore, 1 = normal, 2 = pre_crash, 3 = crash.
The manuscript's three-class task uses classes 1/2/3; id 0 is excluded from supervision.
