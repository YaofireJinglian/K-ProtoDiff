# Baseline cancellation record

On 2026-09-16 the remaining GT-GAN reproductions were cancelled by the user.
The two active jobs had retained a full CPU core while producing no new log
record for many hours and reporting zero GPU utilization. All already completed
baseline records remain in the main table. Missing GT-GAN seeds remain missing;
they are not imputed and are not reported as completed.

The scheduler uses the terminal state `skipped_by_user`, which prevents these
jobs from being relaunched and allows the queued ablation study to proceed.
The remaining ablations are sharded by dataset over GPUs 0–3: Stocks, EEG,
Traffic, and fMRI respectively.
