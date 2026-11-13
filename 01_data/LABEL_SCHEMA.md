# Label Schema

## Directory contract

```
<DATA_ROOT>/
  images/{train,val}/<stem>.jpg
  labels/{train,val}/<stem>.txt          # detection boxes
  scene_labels/{train,val}/<stem>.txt    # scene + optional cause
  type_labels/{train,val}/<stem>.txt     # type
  lane_labels/{train,val}/<stem>.json    # lane polylines
```

## Detection (`labels/`)

One box per line, YOLO format:

```
class cx cy w h
```

`class` in {0 car, 1 person, 2 cyclist}; coordinates normalized to [0,1]. Empty files
mean "no box".

## Scene and cause (`scene_labels/`)

```
<scene id>
<cause bits>        # optional second line
```

- Line 1: 0 ignore, 1 normal, 2 pre_crash, 3 crash.
- Optional line 2: nine cause bits, either comma-separated (`0,1,0,...`)
  or a 16-bit integer mask where bit *i* corresponds to cause *i*. A missing second
  line means "no cause annotation" (not all-zero). The release preserves second-line
  presence even for ignored frames; the training/scoring gate additionally requires
  an eligible pre-crash/crash scene.

## Type (`type_labels/`)

A single integer: -1 unlabeled, 0 rear_end, 1 side_impact, 2 rollover,
3 pedestrian_hit, 4 head_on, 5 fixed_object, 6 other.

## Lane (`lane_labels/`)

JSON object with `stem`, `lines` (list of polylines), `boundary_none` flags, and optional
geometry/semantic notes.

## Consolidated tables

| File | Columns |
|---|---|
| `annotations/scene_cause_{split}.csv.gz` | stem, scene, cause_valid, cause_tailgating … cause_low_visibility |
| `annotations/type_{split}.csv.gz` | stem, type |
| `annotations/detection_{split}.csv.gz` | stem, cls, cx, cy, w, h |
| `annotations/lane_{split}.jsonl.gz` | one JSON object per line |

The machine-readable version of this schema is `schema/label_schema.json`.
