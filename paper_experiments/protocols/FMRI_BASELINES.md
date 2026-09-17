# fMRI baseline reproduction

Configuration: `Config/baselines/fmri.yaml`. All adapters use the same 9977
normalized windows (24 time steps, 50 channels, sim4) as the journal fMRI run.
Seeds: 2026, 2027, 2028. All five metrics run in the shared `.venv` evaluation
environment, and three-seed mean/sample standard deviation update the tables.

Official sources already checked out under `baselines/`:

- TimeGAN: `8f6181c`. Preserve three training phases and default GRU parameters.
  TensorFlow v1 operations run via compat.v1, and contrib dense layers via
  tf-slim. Graph seed is set after graph reset; generation is batched.
- TimeVAE: `9914576`. Use official timeVAE defaults and early stopping.
  Fix the callback's monitored key to `loss`, the actual train_step return key,
  and initialize with the requested seed instead of the helper's fixed 123.
- GT-GAN: `360c79e`. Preserve the regular-Energy reference training settings for
  the new multivariate dataset, replacing only the input data/dimension and seed.
  Use official CDE spline coefficients and model/training code. Disable internal
  repeated t-SNE/evaluation; run batched final generation and shared evaluation.
  Unused torchaudio/sktime loader imports are removed.
- TimeVQVAE: `39f98e9`. Use the official two-stage VQVAE/MaskGIT modules and
  config, with 50 input channels and length 24. Standardize each channel using
  training statistics and invert after generation. Single dummy class with
  unconditional generation. Run 20000/40000 steps with warmup/cosine learning
  rate. This current source checkout was smoke-tested directly with the shared
  Python 3.11/PyTorch 2.7 environment; its published package declares newer
  minimum dependencies, so this is a tested source-level compatibility setup.

Checkpoints and generated arrays: `checkpoints/<method>/seed<seed>/`.
Logs: `OUTPUT/venv_logs/<method>_<seed>.log`.
Smoke results are isolated under `OUTPUT/smoke_remaining/` and never enter tables.

TimeGAN/TimeVAE use an isolated `.venv-tf` when launched through
`scripts/run_tf_baseline.sh`; legacy Keras is enabled only for training.
Shared evaluation explicitly clears that environment setting.

The stock TensorFlow 2.20 GPU wheel detected the RTX 5090 but failed the model
smoke tests with CUDA_ERROR_INVALID_HANDLE. The isolated GPU environment uses
the downstream [sm_120 TensorFlow build](https://github.com/Syraxius/TensorflowDockerBuilder/releases/tag/v2.21.0-sm120-cuda128-py310-py313),
with the artifact SHA256 pinned in `requirements-tf-gpu.txt`. This is a community
binary build, not an upstream TensorFlow wheel. CUDA libraries are taken from
the two local venvs; the isolated cuDNN 9.8 takes precedence without modifying
the PyTorch runtime. Shared scientific evaluations still use `.venv`.

GPU smoke tests for both TimeGAN (all three phases) and TimeVAE (fit/save/sample)
passed on the sm_120 build. The CUDA nvcc wheel supplies libdevice; the launcher
sets XLA_FLAGS to that local directory. Initial GPU layout: GT-GAN on 0/1/2,
TimeVQVAE on 3/4/5, three TimeGAN seeds sharing 6, three TimeVAE seeds sharing 7.
Launch/check duplicate sessions with `bash scripts/launch_remaining_fmri.sh`.

## Scheduling revision, 2026-09-08

After TimeVQVAE completed, move the three TimeGAN seeds to dedicated GPUs 3/4/5.
Existing GT-GAN jobs continue on 0/1/2, and existing TimeVAE jobs finish on 7.
Future TimeVAE launches use 6/7. This is independent-experiment parallelism,
not eight-GPU distributed training of one seed; these adapters do not implement
distributed gradient synchronization.

With user authorization, increase TimeGAN batch size from 128 to 256 for all
three seeds, retaining 50000 optimizer steps per phase, all three phases, and
the original learning rates. This doubles nominal examples processed per step;
it is not an equal-sample-budget comparison with the original batch size.
Short isolated 50-step-per-phase GPU checks passed at both batch sizes, including
generation; joint-phase elapsed times were 35.315s (128) and 32.539s (256),
including graph initialization and checkpoint overhead. These short checks are
not a rigorous throughput benchmark or evidence of improved model quality.

The original running TimeGAN processes had no intermediate checkpoint support;
moving them requires restarting their unfinished embedding phase. Preserve the
old logs under `timegan_<seed>.pre_dedicated_20260908.log`; do not merge those
abandoned runs into scientific results. New runs save TensorFlow model/optimizer,
phase, next iteration and NumPy RNG state every 1000 steps and at phase boundaries,
and on SIGTERM/SIGINT at the next completed training step. Resume validates the
configuration. GPU kernel nondeterminism may still prevent bitwise equivalence.
New records include `run_config` metadata. Completed model runs are not retrained.

### Runtime profiling and XLA

After TimeVAE completed, GPUs 6/7 were used for isolated TimeGAN profiling,
while all six remaining formal training jobs continued. Each profiling run
uses the full shared training array, seed 2026 and batch size 256, with 100 or
200 steps per phase. Timing excludes the first ten steps and checkpoint saves;
generated outputs are checked for expected shape and finite values. Profiling
outputs are isolated from scientific result tables.

Measured seconds per joint-training step: default 4/2 intra/inter-op threads
0.604937 (200-step test); 1/1 threads 0.667654 (100-step test); XLA with 1/1
threads 0.521068; XLA with 4/2 threads 0.353971 (both 100-step tests). These are
short early-training measurements, not an identical-convergence or long-run
speedup claim. The conditional discriminator update frequency can change later.

Enable TensorFlow automatic XLA clustering using `--tf-xla` with the default
4/2 threads for the three formal TimeGAN runs. Stop through the existing
SIGTERM checkpoint handler, then resume from saved model, optimizer, phase,
iteration and NumPy RNG state. No batch-size, learning-rate, or iteration-budget
change accompanies this runtime adjustment. Floating-point execution may differ;
bitwise identity is not claimed. Shared evaluation clears `TF_XLA_FLAGS` and
continues to use the unchanged common metric environment.
