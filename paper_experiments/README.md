# Paper experiment archive

This directory contains the compact experimental evidence used in the
K-ProtoDiff journal manuscript. It is designed to be versioned in Git and
audited without copying datasets, environments, generated arrays, or model
checkpoints into the repository.

## Completion snapshot

- Final K-ProtoDiff-J runs: 30 three-seed dataset runs.
- Main-result records: 226 JSON records.
- Ablation study: 92/92 quality records and 44/44 completed training variants.
- Sampling-efficiency records: 40.
- Training-efficiency records: 20.
- Qualitative temporal-structure records: 36.
- Downstream-utility records: 45.
- Failed planned ablation jobs: 0.
- Fourteen GT-GAN jobs were explicitly cancelled and are documented; they are
  not presented as completed experiments.

## Directory map

- `tables/`: manuscript-ready result tables.
- `records/`: per-seed and per-condition machine-readable results.
- `figures/`: qualitative and temporal-structure figures.
- `configs/`: final, baseline, tuned, and ablation YAML files.
- `protocols/`: selection, evaluation, cancellation, and environment records.
- `scripts/`: exact launch, evaluation, table-building, and archiving scripts.
- `environment/`: dependency specifications.
- `weights/`: released-weight manifest. The large weight files remain in the
  GitHub release `journal-v1-weights`.
- `MANIFEST.json`: SHA-256 and byte size for every archived payload file.

## Reproduction boundary

Run the scripts from the repository root so they can import the model and data
utilities. Dataset files and checkpoints are intentionally excluded from this
folder. Their acquisition paths, seeds, selected configurations, and weight
checksums are preserved here. All paper tables retain three-decimal formatting;
main results use three seeds, while the predefined ablation and efficiency
studies follow their documented single-seed protocol.
