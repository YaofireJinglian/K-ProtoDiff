# Architecture-preserving journal hyperparameter screening

This is prospective exploratory tuning, not a claim that every metric will
become rank 1. Existing publication tables and model artifacts are unchanged.
The shared evaluator itself notes that parity with the conference paper's
KL / segment-DTW implementations has not been established. Rankings against
published values therefore remain provisional, even when a number is lower.

Each dataset has a dedicated specification under `Config/journal_search/`.
The first screening round uses seed 2026, 3000 optimizer steps per training
configuration, and at most 1024 fixed validation windows. Network dimensions,
layer counts, number/scales of prototypes, and sampling architecture stay fixed.
Three training configurations (control, lower warmup target learning rate,
lower prototype-loss weight) each receive three sampling configurations
(50/100/200 steps and 0.1/0.05/0.02 reflection strength). All candidates use
equal training-step budgets and paired random seeds. Longer sampling receives
more compute, and must not be presented as an equal-speed comparison.

Raw chronological data are divided 70/15/15 before constructing windows.
No window straddles a boundary. Only the first block is used for training,
outlier-cleaning statistics (EEG), and scaling. The second is validation.
The final 15% is sealed from screening and configuration selection. This new
split has not been used to train these search checkpoints. It is NOT globally
unseen data: earlier all-data journal/baseline models used the full source.
Accordingly, do not compare a new held-out result to those old models as if
they shared a held-out protocol; a consistent confirmatory comparison would
require fresh baseline training on the same split.

EEG retains the existing loader's class-segment filtering and channel-major
flatten/reshape convention rather than silently changing the input layout.
Cleaning and normalization statistics are refit on its training block only;
interpolation stays inside each block. This split-specific preprocessing differs
from the original all-data preprocessing and is recorded as such.

All five metric families use the shared metric code. `eval_seeded.py` additionally
fixes NumPy, Python and PyTorch initialization, including TS2Vec. DS/PS seeds
are supplied explicitly. Validation metric fitting is part of the validation
procedure, not generator training. Three DTW segment lengths are averaged into
one category so DTW does not receive triple weight. Selection minimizes the
equal-weight mean validation rank across C-FID, KL, DS, PS and segment-DTW.
No per-metric model/seed selection or rounded-value ranking is allowed.

Search outputs: `checkpoints/journal_search/<dataset>/<training_trial>/`.
Checkpoints include weights, EMA, optimizer, scheduler, sampling order/cursor,
and random states. Model checkpoints are saved every 500 optimizer steps.
Each dataset's `selection.json`, `candidate.yaml`, `VALIDATION_RESULTS.md` are
written only after every planned candidate has completed. Any failure remains
visible in worker logs and prevents a complete selection for that dataset.
Smoke tests are isolated in `OUTPUT/journal_search_smoke/`.

The emitted candidate is a screening recommendation only. Full-budget training,
three-seed confirmation, and a consistent evaluation-protocol audit remain
required before replacing any publication results. The current queue does not
silently launch or publish that next stage. Displayed screening values use three
decimals, explicitly without a fabricated three-seed standard deviation.

Run `bash scripts/launch_journal_search.sh`. Workers on GPU 6/7 run each dataset's
training and validation sequentially; existing baselines on GPUs 0–5 are untouched.

## Authorized unattended continuation (2026-09-08)

The user subsequently requested continued improvement and real-time monitoring.
`scripts/journal_watchdog.py`, launched in the `kpd_journal_watchdog` tmux session,
now polls every 60 seconds. It writes `EXPERIMENT_STATUS.md` plus machine-readable
`OUTPUT/journal_monitor/status.json`. This is an actual server process, not a
promise that the chat assistant will remain continuously awake. It survives
client disconnects while the server/session remains running, not a server reboot.

The unattended search budget is deliberately finite and prospective: the original
3000-step round, a second 6000-step validation round using the incumbent plus
nearby learning-rate/prototype/reflection settings, then frozen-choice full-budget
three-seed confirmation. Second-round specs are emitted once per dataset under
`Config/journal_search_round2/`. No architecture parameter is altered. Compatible
same-seed checkpoints are resumed; changed-training-parameter trials start fresh.
This allocates more training steps to promising candidates and is not an equal-
compute comparison with arbitrary external baselines.

After round two finishes, its validation winner is frozen in immutable per-dataset
specs under `Config/journal_confirmation_seed2026/` (and 2027/2028). Both the
original parameter control and the selected parameter setting train only on the
70% training block, at the original dataset's full optimizer-step budget. They
are evaluated on the previously sealed last 15% (up to 1024 fixed windows).
The original EEG preprocessing convention is still retained. If model parameters
are identical but only sampling differs, trained control weights are reused with
the selected sampler. These are not independently trained duplicate models.
Otherwise each configuration gets its own three seed runs.

Confirmation scores never enter `select()` (the function rejects test-split
selection). The scheduler does not create a third search round based on test
results. It writes `JOURNAL_CONFIRMATION_RESULTS.md`, with three decimals and
three-seed mean/sample standard deviation; it does not overwrite either main
publication table or claim rank 1 against old all-data/published results.

Free GPUs are detected from both NVIDIA process ownership and known tmux workers.
Active baselines/search jobs are never killed. As baseline cards become free,
they can process ready second-round/confirmation jobs. Recoverable new jobs
receive at most two retries after their initial launch, with five-minute backoff.
TimeGAN's tested resume path and the initial search queues also receive at most
two recovery launches. GT-GAN failures are alerted rather than silently restarting
without optimizer state. Logs stale for an hour are flagged, not automatically
killed. Failure alerts remain in the status document for user review.

When this budget completes, monitoring continues but no unbounded parameter
search is launched. Persistent failures pause the affected branch. Extending the
budget or changing the architecture/evaluation protocol requires a new decision.
No notifications are sent to external accounts; the status files and logs are the
durable handoff. Restart monitoring with `bash scripts/launch_journal_watchdog.sh`.
