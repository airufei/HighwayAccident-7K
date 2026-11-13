# HighwayAccident-7K dataset card

## Tasks and labels

- Scene: 0 ignore, 1 normal, 2 pre-crash, 3 crash. Scene 0 is excluded from scoring.
- Type: -1 unscorable; 0 rear-end, 1 side impact, 2 rollover, 3 pedestrian collision,
  4 head-on, 5 fixed-object collision, 6 other.
- Contributing factors: nine schema bits; seven supervised bits (0,1,2,3,5,6,8).
  These are annotated/predicted factors, not experimentally identified causal effects.
- Detection: car/person/cyclist boxes. Lane polylines support an auxiliary task.

The primary final training phase supervises types 0,1,4,5,6. The main evaluation
uses unmasked type logits and separately reports scorable-label and active-subset
accuracy. Training-loss masks and inference masks are different operations.

## Inventory and metric eligibility

See `DATA_SOURCES.md` and machine-readable statistics for the original image inventory.
Scene/type tables contain one row for each of 346,048 train and 89,392 val images.
Lane labels cover 16,577 frames. The original final-checkpoint scene evaluation has
4,866 eligible records; type evaluation has 1,980 labels and factor evaluation 1,596
active records. The two-record exclusion analysis changes these to 4,864 / 1,979 /
1,595, respectively. Active-type support stays 1,870.

## Intended use and generalization

Research on video crash detection and collision-type classification, with auxiliary
factor prediction and retrospective review. Report results by recording source and
class, as well as on the pooled collection. Most training images are synthetic;
pooled type accuracy is substantially stronger than the retained non-CARLA subset.
Rollover and pedestrian collision are outside the final five-class supervision set.
Source-specific performance is heterogeneous. Do not represent internal validation
or retrospective temporal localization as certified online deployment performance.

Video and parsed-event cluster intervals are provided. The parsed event groups are
operational identifiers, not verified independent accident identities. DADA/CCD
recorded external diagnostics use epoch 8. CCD also occurs in the assembled dataset;
its dataset label alone does not establish source independence.

## Access and rights

This repository redistributes labels/metadata and model outputs, not source images
or videos. Source materials retain their providers' terms. The annotation protocol
describes the study procedure; it is not a new verification of every source license
or every frame. See `LICENSES_AND_ACCESS.md` and the root license notices.
