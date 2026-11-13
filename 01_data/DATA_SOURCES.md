# Source inventory and provenance

HighwayAccident-7K is a multi-source research collection with study-specific
reannotations. It is not a wholly original recording collection. The name preserves
the early bookkeeping scale and is not a certified count of independent accidents.

| Source family | Nature | Training images | Validation images |
|---|---|---:|---:|
| synthetic | CARLA-generated scenes | 226,383 | 64,213 |
| tumtraf | Public real footage | 74,222 | 16,141 |
| realv | Real footage with study annotations | 24,955 | 3,421 |
| real | Real CCTV/dashcam material, mixed provenance | 14,421 | 3,370 |
| ccd | Public CCD material | 3,280 | 875 |
| video_sampling | Sampled real video with study/model-assisted annotations | 2,787 | 1,372 |
| Total | | 346,048 | 89,392 |

Public sources identified for the study include CCD, TUM Traffic Accid3nD, and
TUMTraf-A9. Family names are inventory groupings, not one-to-one dataset identities.
The manifests preserve the supplied family labels and parsed source/video/event IDs.
They do not add unverified original download identifiers or source timestamps.
See `THIRD_PARTY_NOTICES.md` at repository root for provider links.

## Distinct evaluation universes

- Frame inventory: 435,440 images, including derived tiles.
- Primary score stream: 5,556 records, of which 4,866 have valid scene labels.
- Original type population: 1,980 scorable labels; 1,870 in the five supervised classes.
- Archived type-candidate denominator: 2,323, including 343 unscorable labels.
- Frozen-feature controls: 18,232 training windows; 5,556 validation records.
  Their cached features came from epoch 8, not the final checkpoint.
- Final exclusion analysis: 4,864 valid scene clips after two candidate records are
  removed. The associated manifest excludes 13 stored validation images.

These populations are not interchangeable. Synthetic image share is not the same
as synthetic clip share. Synthetic labels participate in the reported supervised
training; they must not be described as consistency-only supervision.

## Content independence and access

Parsed train/validation event IDs have no intersection. Two representative-frame
hash candidates are separately marked in `manifests/evaluation_exclusions.json`.
ID separation and the two-record sensitivity analysis do not certify zero content
overlap throughout the original media pool.

No source media are redistributed. Public provider terms and specific authorization
agreements govern access. The authors' annotation license does not grant access to
source videos or override third-party rights. Source-frame time indices, all
simulator assets/seeds and the complete authorized media pool are not supplied;
the full raw training corpus cannot be reconstructed solely from this repository.
