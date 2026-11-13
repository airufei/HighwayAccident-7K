# Score-based evaluation

Run from any directory:

```bash
python /path/to/HighwayAccident-7K/03_evaluation/reproduce.py --check
```

Inputs are ordinary gzip-compressed JSONL and NPZ arrays. NPZ identifiers are stored
as Unicode arrays and loaded with `allow_pickle=False`; cached feature vectors are
not required for metric reproduction and are not included.

## Endpoints

- Scene classification uses the frame output at the last non-ignore label in each
  16-frame record. Ignore label 0 is excluded. FAR counts normal clips predicted as
  either pre-crash or crash; it is not merged false alarms per hour.
- The control script's archived scene summaries use the last-valid-frame output.
  Actual `clip_scene` results are recomputed separately. Last/mean/middle names
  identify auxiliary clip-loss aggregation; the two readouts are not interchangeable.
- Main type P/R/AP use scorable labels. Missing labels do not become negative examples.
  Correct-type yield uses the separate type-candidate denominator. AP groups tied
  scores at a common threshold; archived row-order rank AP is retained for comparison.
- The five/seven-class GRU comparison applies no inference-time type mask. The
  archived masked-control results are labeled separately.
- Main-cluster intervals use 2,000 bootstrap draws, seed 0, resampling video IDs or
  parsed event IDs with all their eligible records. External video aggregation uses
  1,000 draws, seed 0. Operational event keys are not certified independent incidents.
- DADA "hit anywhere" is distinct from a hit in a crash-labeled window. Nominal
  hourly false-window counts use assumed sampling rates (DADA 12.5/s, CCD 3/s), not
  measured continuous-stream exposure or merged alarm events.

The archived external score streams are epoch 8. The primary validation score stream
is epoch 10. Uncompressed score SHA-256 identities are in `data/score_identity.json`.

## Reproduced and archival tables

Recomputed from per-record data: final/control rows of Table 3; Table 4 type metrics;
Table 5 factor metrics; A.2 exclusion controls; A.3 type-supervision controls;
A.4 final/source clustered intervals; E.2 video aggregation; and source/split counts.

The earlier-system and epoch-8 reference rows, training-configuration paired tests,
additional-seed summaries, runtime and retrospective-localization summaries are
archived table values, not newly reconstructed from unavailable paired prediction or
timing logs by this command. Printed values of all 16 tables are in `04_results/`.
Do not infer full retraining, event-matched deployment validation or remeasurement of
runtime from a successful score-based check.
