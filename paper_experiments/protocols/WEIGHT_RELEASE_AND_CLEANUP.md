# Weight release and cleanup policy

The release contains one archive per dataset. Every archive provides all three
formal seeds, the frozen tuned configuration, and the matching metric records.
This avoids selecting a lucky test seed. Asset and checkpoint SHA-256 hashes are
recorded in `WEIGHTS_MANIFEST.json`.

Cleanup occurs in two guarded stages. Completed search/confirmation weights and
old numbered journal checkpoints are removable immediately. Final sample arrays
and superseded formal weights are removed only after all tuned records exist and
the GitHub Release assets have been checked by name and byte size. Baseline and
ablation artifacts are excluded while those experiment queues remain active.
