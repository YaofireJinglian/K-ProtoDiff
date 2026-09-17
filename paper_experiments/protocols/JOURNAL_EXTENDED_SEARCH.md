# Additional paired-seed exploratory experiments

Authorized after the user requested additional experiments following the warning
against choosing lucky seeds or tuning on previously observed test results.

The ten dataset YAML plans are in `Config/journal_search_extended/`. Seeds are
fixed in advance at 2031, 2032 and 2033 for every candidate and every dataset.
No seed is dropped based on its score. Models start from fresh initialization;
the original 2026/2027/2028 runs remain unchanged and separately reported.

There are five training configurations per dataset: original parameters;
warmup target LR 0.0003 with prototype loss weight 0.01; EMA decay 0.999;
the alternative L1/L2 loss; and longer training. The first four use 6000
optimizer steps; the longer-training configuration uses 12000. Each trained
checkpoint gets three sampling settings: 50/100/200 maximum denoiser evaluations,
with reflection strength 0.1/0.05/0.02. The longer configurations receive more
compute, so these are not equal-compute comparisons. Architecture dimensions,
layer counts, prototype counts/scales and adaptive-reflection architecture stay
fixed. No postprocessing, clipping change or metric definition change is added.

Budget: 10 datasets × 5 training configurations × 3 paired seeds = 150 training
runs; three samplers each yield 450 sampling/evaluation combinations. Existing
jobs keep running. New jobs are appended to the existing watchdog queue and run
only when GPUs become available. No unrelated process is stopped.

Data partitioning and training-only normalization are unchanged from the earlier
search. Only training and validation blocks are used by these new jobs. All five
metric families use the common evaluator with matched evaluation initialization.
Selection waits until every planned candidate has all three seeds, then ranks
their full-precision three-seed mean metrics. Segment-DTW contributes one metric
family rather than three votes. Tables report three-decimal means ± sample SD.
There is no per-metric candidate selection or per-seed replacement.

Runtime specs are frozen under `Config/journal_search_extended_seed2031/` and the
corresponding 2032/2033 directories. Each run's artifacts are in the matching
`checkpoints/journal_search_extended_seed<seed>/<dataset>/<trial>/` directory.
Aggregated validation tables, all individual seed records and the selected
candidate YAML go in `checkpoints/journal_search_extended/<dataset>/`.

This is explicitly EXPLORATORY: prior test results have already been observed.
The planner never reads those results and does not trigger new automatic test-
set selection or independent-test claims. New independent validation data or a
clearly disclosed exploratory reporting protocol is needed before stronger
publication claims. Main tables and earlier confirmation results are untouched.
