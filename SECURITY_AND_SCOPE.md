# Release scope

Included: research labels/metadata, model code, inference-only checkpoint exports,
record-level predictions and numeric reproduction tools. Excluded: original media,
third-party video crops, credentials/service secrets, private governance records,
optimizer states and unrelated model weights.

The public numerical path loads NPZ files with pickle disabled. Torch inference
exports contain tensors and simple metadata; obtain checkpoints only from trusted,
hash-verified releases. Historical training utilities have broader serialization
assumptions and must not be exposed as an unauthenticated public service.

File hashes and a release verification command accompany the repository. Static
checks and numerical reproduction are not a claim of complete security auditing,
full raw-data provenance certification or full from-scratch training reproduction.
