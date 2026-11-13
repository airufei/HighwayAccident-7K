# Inference checkpoints

Git LFS files:

- `temporal_final_epoch10.pt`: primary final checkpoint.
- `temporal_best_epoch8.pt`: checkpoint used in separate external/video analyses.

Each `.identity.json` stores original/export SHA-256 values, epoch, size and the
886-tensor equality check. Optimizer/scheduler/scaler/private run fields were removed.
All model tensors, shapes and dtypes are unchanged. These exports do not support
exact optimizer-state resumption. Their trained FrameFeaturePool states differ.

Use `git lfs pull` after cloning. To verify the released files, run
`python tools/verify_release.py` from the repository root. The numerical metric
pipeline can run without downloading weights.
