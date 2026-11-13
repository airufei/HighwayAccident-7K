# Reproducing results and running the model

## 1. Verified result reproduction

For paper metrics, first use the root numerical requirements and run
`python 03_evaluation/reproduce.py --check`. No GPU, Torch, weights or source images
are needed. This is the clean-directory path verified for this release.

## 2. Model evaluation with legally obtained images

Install the recorded training stack from `environment/requirements.txt` in a separate
compatible Torch/CUDA environment. Its historical pinned stack has not been freshly
installed on all supported operating systems by this release procedure.

Restore the per-frame label trees with `tools/restore_labels.py` into an empty
directory, obtain the corresponding source images under their provider terms, and
assemble `images/{train,val}` beside those labels. The dataset YAML `path:` must
refer to that real dataset root.

From `02_training/`, for example:

```bash
python code/eval_acceptance.py --checkpoint ../models/temporal_final_epoch10.pt --data-root /absolute/path/to/accident_v26 --data-yaml config/accident_detect.yaml --model-yaml config/yolo26m_detect_cls_fixed.yaml --clip-len 16 --device 0 --output-dir ../04_results/model_eval_final
```

Use `temporal_best_epoch8.pt` for the separately recorded epoch-8 analyses.
`--weights` is not an argument to this evaluation entry point. Download Git LFS
objects before loading a checkpoint; a small LFS pointer is not a Torch checkpoint.

## 3. Training entry point

Use explicit `--epochs`, `--lr0`, `--clip-len`, seed and predecessor checkpoint paths.
`run_train.py` defaults to 20 epochs and passes this to the trainer; omitting the
flag does not activate the phase-specific documented epoch counts.

The temporal phase settings are 15/8/10 epochs and 5e-4/3e-4/1e-4. A phase-3c
continuation from an appropriately established predecessor would use:

```bash
python code/run_train.py --phase 3c --name reproduced_p3c --epochs 10 --lr0 0.0001 --clip-len 16 --seed 0 --chain-from /absolute/path/to/phase3b_checkpoint.pt --data-root /absolute/path/to/accident_v26
```

This example is not a substitute for the complete historical upstream chain. The
original training gate also expects `meta/quality_report.json` or
`meta/data_quality_report.json`; those original human-QA scoring artifacts are not
part of this public release. Do not manufacture a passing score. The historical
`--skip-quality-gate` option exists but is not silently enabled by these instructions.

## 4. Scope

The full authorized raw pool, source-frame retrieval transformations, all simulator
assets/seeds and every historical training predecessor are not supplied. Full
from-scratch retraining has not been demonstrated from this repository alone.
Inference exports preserve the tensors but omit optimizer/scheduler/scaler state,
so they are intended for inference rather than exact optimizer-state resumption.
Read the evaluation README for which supplementary tables are archival summaries.
