# HighwayAccident-7K

Research companion to **Temporal modeling for video-based crash detection and
collision-type classification**. This release contains multi-source reannotations,
training and model code, inference checkpoints, and a CPU-only evaluation pipeline.
The manuscript is not yet assigned a publication DOI.

## What can be reproduced

The released score streams reproduce the final checkpoint's scene/type/factor metrics,
frozen-feature control results, two-record exclusion analysis, source inventories,
video/event-cluster bootstrap intervals, and recorded DADA/CCD video aggregations.
They require no original images and no GPU.

```bash
python -m pip install -r requirements-evaluation.txt
python 03_evaluation/reproduce.py --check
python 05_figures/render_all.py
```

The first command installs the numerical dependencies; the second independently
recomputes and checks the released numeric references. Results are written to
`04_results/recomputed/results.json`. Figure generation writes under `05_figures/`.
See [evaluation details](03_evaluation/README.md) for endpoints and table coverage.

## Contents

| Directory | Contents |
|---|---|
| `01_data/` | Consolidated labels, label schema, annotation protocol, source and split manifests |
| `02_training/` | Training/model/evaluation implementation and recorded configuration documentation |
| `03_evaluation/` | Portable metrics, per-clip score streams, frozen-head outputs and verified references |
| `04_results/` | Printed bilingual table values and the scope of reproducible results |
| `05_figures/` | Numeric plots and diagrams; third-party video crops are excluded |
| `models/` | Epoch-10 final and epoch-8 inference exports with tensor identity manifests |
| `tools/` | Annotation restoration, checkpoint export and release verification |

## Recorded results

| Final-checkpoint scene endpoint | Eligible clips | Balanced accuracy | Macro-F1 | Normal-clip FAR |
|---|---:|---:|---:|---:|
| Original validation inventory | 4,866 | 77.61% | 72.27% | 11.90% |
| Two flagged records excluded | 4,864 | 77.60% | 72.26% | 11.90% |

Collision-type accuracy on the original inventory is 87.02% on 1,980 scorable labels
and 92.14% on 1,870 labels in the five supervised classes. The retained non-CARLA
subset is reported separately, including its weaker type/factor performance; pooled
mixed-source accuracy is not a real-world deployment guarantee.

The frame inventory contains 346,048 training and 89,392 validation images. It combines
CARLA-generated material, TUMTraf, authorized real footage, CCD and sampled video.
Public source data are reannotated under the study label schema. Read
[data provenance](01_data/DATA_SOURCES.md) before interpreting the collection name,
source families or access conditions.

## Labels and original media

No raw videos, images, faces, license plates or video crops are redistributed.
Original media must be obtained under the relevant providers' terms. The released
metadata and annotations do not grant access to third-party pixels. Some authorized
and generated inputs are not supplied, so this release does **not** claim one-command
reconstruction of the entire training corpus.

To restore labels into a **new empty directory**, first try a small sample:

```bash
python tools/restore_labels.py --split val --limit 32 --output restored_labels
```

Omit `--limit` and use a different empty output directory for a complete restoration.
Read [training reproduction](02_training/REPRODUCE.md) before attempting GPU training.

## Checkpoints

Models are stored with Git LFS. Install Git LFS and run `git lfs pull` to download them.
CPU-only score reproduction does not need the model files. The exports preserve all
886 model tensors and remove optimizer/scheduler/private run metadata. Epoch 10 is
the primary-model checkpoint; epoch 8 underlies the archived external and visualization
analyses. Their trained feature pools differ and must not be interchanged.

## Licensing and citation

Software and accompanying model exports: [AGPL-3.0](LICENSE).
The authors' data/document contributions: [CC BY-NC 4.0](LICENSE-DATA.md).
Third-party rights are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
See `CITATION.cff` for repository citation metadata. No Zenodo DOI has been assigned
to this version; the repository must not be cited as an already archived DOI deposit.
