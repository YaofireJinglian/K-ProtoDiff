# Validation-selected main-result refresh

The extended search uses only the validation split and fixed seeds 2031, 2032,
and 2033. One per-dataset configuration is frozen before formal evaluation.

Formal evaluation uses the original full-data protocol and seeds 2026, 2027,
and 2028. Search training-step budgets never replace the full training budgets in
`Config/journal`. A dataset replaces its `K-ProtoDiff-J` row inputs only after all
three formal records exist. Existing records are preserved under
`OUTPUT/main_results_records_pre_tuning` before promotion.

An unchanged frozen candidate may reuse the original formal record. A candidate
with unchanged training settings may reuse the original full checkpoint and is
resampled with the frozen sampler. A candidate that changes training settings is
trained from a fresh initialization for every formal seed.

Formal test metrics are reported but are never fed back into configuration
selection.

New main, tuned-main, and ablation completion records are monitored by
`scripts/watch_result_git_sync.py`. Each detected completion triggers a compact
snapshot commit and GitHub push; completions observed in the same polling window
are committed together to avoid conflicting concurrent pushes.
