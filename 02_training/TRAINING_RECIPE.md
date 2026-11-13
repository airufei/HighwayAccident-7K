# Recorded training configuration

The code is the historical model/training implementation accompanying the evaluated
checkpoints. The numeric evaluation release does not initiate training.

| Item | Recorded setting |
|---|---|
| Model input | 640 x 640 images; 16 sampled frames per clip |
| Clip advance in training/evaluation | 8 sampled frames |
| Temporal phases | 3a: 15 epochs; 3b: 8 epochs; 3c: 10 epochs |
| Temporal learning rates | 5e-4, 3e-4, 1e-4 |
| Optimizer | AdamW; weight decay 0.15; OneCycle scheduling |
| Batch | 4 clips x gradient accumulation 4 = 16 clips/step |
| Primary seed | 0 |
| Temporal dropout | 0.5 |
| Final scene loss weight | approximately 5.243 |
| Type loss weights | 1.2, 1.0, 1.0 |
| Factor loss weights | 2.0, 1.5, 1.0 |
| Active types in the final phase | 0,1,4,5,6 |
| Primary checkpoint | final epoch 10 |
| Separate validation-selected checkpoint | epoch 8 |

Detector initialization and refinement precede the temporal chain. The manuscript
records initial 40-epoch detector training and a final 20-epoch upstream refinement;
that last refinement is not the entire historical upstream training sequence.
Earlier p1/p2/p3 argument snapshots are retained under `config/archived_snapshots/`
as legacy implementation references and are not claimed to be the exact final-run
arguments. Do not use their 60/15/20-epoch defaults as replacements for the temporal
settings above.

At inference, the primary classifier receives the shared padding type input and
does not mask inactive type outputs. The frozen-head control evaluation has a
separate archived masked result; the released five/seven-class comparison is unmasked.
Synthetic samples are not restricted to a consistency-only loss.

Checkpoint selection, runtime, video aggregation and output definitions must remain
checkpoint-specific. The video visualization recipe (step 1, overlap averaging,
two-pass type conditioning) differs from the primary clip-level scoring recipe.
