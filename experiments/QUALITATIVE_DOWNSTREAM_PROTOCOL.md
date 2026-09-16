# Qualitative, temporal-structure, and downstream-utility protocol

## Scope and pre-registered case selection

This study uses **Stocks, Exchange, and fMRI**. They were fixed before running
the analyses because K-ProtoDiff-J is ranked first on the predictive score for
all three datasets in `MAIN_RESULTS.md`. They are therefore reported as
**advantage case studies**, not as evidence that every qualitative property is
best on every dataset.

No dataset, metric, channel, window, method, or seed may be removed after the
results are produced.

## Reproducibility controls

- Generator seeds: 2026, 2027, and 2028.
- The qualitative figures use seed 2026 only.
- The displayed channel is selected using variance in real data only.
- The displayed real and generated windows use a fixed deterministic index.
- Displayed trajectories are standardized within each fixed window so the
  panel compares temporal shape rather than the unrelated absolute level of
  unpaired real and generated windows. Numerical metrics use the original
  common zero-to-one scale.
- All numerical structure results use the same number of windows per method.
- Every generated array is converted to the common zero-to-one scale before
  evaluation.
- Values in tables are mean plus/minus sample standard deviation across three
  generator seeds and are rounded to three decimals.

## Temporal-structure analysis

The following lower-is-better distances are evaluated against real data:

- short-lag autocorrelation distance;
- normalized power-spectrum distance;
- cross-variable correlation distance;
- local trend distance.

For fMRI, the cross-variable correlation comparison is also shown as a
functional-connectivity heatmap. The figures additionally include a
standardized fixed example trajectory and aggregate autocorrelation and
spectrum curves.

## Downstream task (item 5)

The task predicts the final multivariate observation in each 24-step window
from its first 23 observations. A fixed lightweight GRU is trained separately
on real or synthetic windows. The real test set comprises five fixed temporal
blocks distributed across the full timeline and totaling about 15 percent of
the windows. For the real-data reference, every training window within 23
positions of a held-out block is removed. This prevents overlapping-window
leakage without confounding the comparison with a single end-of-series shift.

All methods use the same training-set size, optimizer, early-stopping rule, and
three predictor seeds. MAE and RMSE are reported; lower is better. The real-data
training result is an upper-reference condition rather than a generative model.

The generators used by the main paper were trained under its full-data
generation protocol. Consequently, this experiment measures downstream
utility under that protocol; it is not claimed as strict unseen-period
generalization by the generator. A strict forecasting claim would require
retraining every generator only on the early chronological partition.
