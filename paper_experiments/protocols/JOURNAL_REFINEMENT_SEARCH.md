# Validation-only parameter refinement

The completed extended search fixed one validation winner per dataset using
seeds 2031/2032/2033. This follow-up uses fresh, prospectively fixed seeds
2041/2042/2043 and never reads test metrics for selection.

For each dataset, it compares the incumbent training configuration with
prototype-loss weights 0.005 and 0.02. Sampling compares the incumbent with
150 evaluations at reflection strength 0.03 and 300 evaluations at strength
0.01. Duplicate sampler settings are removed. Architecture dimensions,
prototype counts and prototype scales remain fixed.

Selection uses the mean rank across C-FID, KL, DS, PS and the mean of the three
segment-DTW metrics. Every candidate must complete all three seeds. These are
exploratory validation results and do not directly replace the main table.
