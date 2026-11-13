# Annotation Protocol

## Scene labels (S1)

- Three annotators label each frame independently as normal / pre_crash / crash
  (raw encoding 1/2/3; 0 marks ignored frames).
- Disagreements go to an expert arbiter. The acceptance threshold is **>= 95%**
  agreement on the audited sample.
- Frames whose source video identity cannot be verified are marked ignore (0) rather
  than normal, so that uncertain frames do not pollute the negative class.

## Type labels (S2)

- Two annotators label crash-positive frames with one of seven types; disagreements go
  to an arbiter. The acceptance threshold is **>= 90%**.
- Frames without a crash are stored as -1 (unlabeled) and are excluded from the type
  loss.
- Long-tail classes (rollover, pedestrian_hit) are kept even when rare; they are not
  merged into "other".

## Cause labels (S3, counterfactual interrogation)

- Annotators answer, for each candidate cause, the question: *"If this factor were
  absent, would the accident still have happened?"*
- Only causes that survive the counterfactual question are marked positive. This cuts
  the "correlation implies causation" contamination at the label source.
- Each positive cause carries an evidence layer (behavioral / environmental) and, where
  available, a counterfactual pair used by the CCS evaluation.
- Nine bits are defined; two bits (pedestrian_highway, distracted_driving) have zero
  positives in this release and are never activated by the curriculum.

## Synthetic clips

- Labels are derived from simulation ground truth through four derivation gates:
  camera-pose reconstruction, lane-geometry verification, dual-window speed
  calibration, and a final consistency gate.
- Synthetic labels enter only the consistency-regularization term, never the supervised
  cross-entropy.

## Quality control

- Same-source neighboring clips are grouped into the same split; pHash near-duplicate
  scanning flags duplicates (documented rate < 0.02%).
- An independent correction layer (`corrections/`) overrides labels at load time; every
  correction keeps its own audit trail.
- Lane annotations cover 16,577 frames for the auxiliary lane-crossing task.

## Released tables

The per-frame files are consolidated into CSV/JSONL tables (see
`annotations/README.md`). The consolidated tables are byte-for-byte derived from the
per-frame files; no label values were changed during consolidation.
