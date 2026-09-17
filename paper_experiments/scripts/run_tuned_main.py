"""Run full-protocol, three-seed confirmation of frozen validation winners."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promote_tuned_main import promote


PYTHON = ROOT / '.venv-cu130' / 'bin' / 'python'
LABELS = {
    'etth': 'ETTh', 'electricity': 'Electricity', 'energy': 'Energy',
    'traffic': 'Traffic', 'weather': 'Weather', 'illness': 'Illness',
    'exchange': 'Exchange', 'stocks': 'Stocks', 'eeg': 'EEG', 'fmri': 'fMRI',
}
METRIC_MAP = {
    'Context-FID': 'C-FID', 'KL': 'KL', 'DS_mean': 'DS', 'PS_mean': 'PS',
    'Segment-DTW-L6': 'Seg-DTW-L6', 'Segment-DTW-L8': 'Seg-DTW-L8',
    'Segment-DTW-L10': 'Seg-DTW-L10',
}


def run_logged(command, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('Running', ' '.join(map(str, command)), flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND: ' + ' '.join(map(str, command)) + '\n')
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def parse_metrics(path):
    values = dict(line.split(':', 1) for line in path.read_text(encoding='utf-8').splitlines())
    result = {target: float(values[source]) for source, target in METRIC_MAP.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(f'Nonfinite formal metrics in {path}')
    return result


def copy_control_record(dataset, seed, metadata):
    source = ROOT / 'OUTPUT' / 'main_results_records' / f'{dataset}__k_protodiff_j__seed{seed}.json'
    if not source.exists():
        raise FileNotFoundError(f'Cannot reuse missing formal control record: {source}')
    record = json.loads(source.read_text(encoding='utf-8'))
    record.update(provenance='formal_control_reuse', frozen_validation_choice=metadata['candidate'])
    destination = ROOT / 'OUTPUT' / 'tuned_main_records' / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, destination)


def run_job(dataset, seed, gpu, manifest):
    metadata = manifest['datasets'][dataset]
    if metadata['reuse_original_record']:
        copy_control_record(dataset, seed, metadata)
        promote(dataset)
        print(f'{dataset} seed{seed}: exact formal control reused', flush=True)
        return
    config_path = ROOT / 'Config' / 'journal_tuned' / f'{dataset}.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    params = config['dataloader']['train_dataset']['params']
    data_name, window = params.get('name', dataset), params['window']
    run_name = f'journal_tuned_{dataset}_seed{seed}'
    if metadata['reuse_original_checkpoint']:
        checkpoint_base = ROOT / 'checkpoints' / 'journal' / f'Checkpoints_journal_{dataset}_seed{seed}'
    else:
        checkpoint_base = ROOT / 'checkpoints' / 'journal_tuned' / f'Checkpoints_journal_tuned_{dataset}_seed{seed}'
    checkpoint = Path(f'{checkpoint_base}_{window}') / 'checkpoint-10.pt'
    output_root = ROOT / 'OUTPUT' / 'formal_tuned'
    output_dir = output_root / run_name
    fake = output_dir / f'ddpm_fake_{run_name}.npy'
    generated_real = output_dir / 'samples' / f'{data_name}_norm_truth_{window}_train.npy'
    original_real = (ROOT / 'OUTPUT' / 'formal' / f'journal_{dataset}_seed{seed}' /
                     'samples' / f'{data_name}_norm_truth_{window}_train.npy')
    metrics_dir = ROOT / 'OUTPUT' / 'tuned_main_metrics' / f'{dataset}_seed{seed}'
    metrics = metrics_dir / 'metrics.txt'
    log_root = ROOT / 'OUTPUT' / 'tuned_main_logs' / f'gpu{gpu}'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    common = [PYTHON, '-u', ROOT / 'run.py', '--name', run_name,
              '--config_file', config_path, '--output', output_root,
              '--gpu', '0', '--seed', str(seed)]
    if not checkpoint.exists():
        if metadata['reuse_original_checkpoint']:
            raise FileNotFoundError(f'Expected reusable full checkpoint: {checkpoint}')
        run_logged(common + ['--train', 'solver.results_folder', checkpoint_base],
                   log_root / f'{run_name}_train.log', env)
    if not fake.exists():
        run_logged(common + ['--milestone', '10', 'solver.results_folder', checkpoint_base],
                   log_root / f'{run_name}_sample.log', env)
    real = original_real if original_real.exists() else generated_real
    if not fake.exists() or not real.exists():
        raise FileNotFoundError(f'Missing formal samples for {dataset} seed{seed}')
    if not metrics.exists():
        eval_env = env.copy()
        for key in ['TF_USE_LEGACY_KERAS', 'REQUIRE_TF_GPU', 'XLA_FLAGS', 'TF_XLA_FLAGS']:
            eval_env.pop(key, None)
        run_logged([ROOT / '.venv' / 'bin' / 'python', '-u', ROOT / 'scripts' / 'eval_seeded.py',
                    '--root', metrics_dir, '--ori_path', real, '--fake_path', fake,
                    '--seed', str(seed), '--n_iter', '1'],
                   log_root / f'{run_name}_metrics.log', eval_env)
    record = {
        'dataset': LABELS[dataset], 'method': 'K-ProtoDiff-J', 'seed': seed,
        'provenance': 'formal_tuned_confirmation',
        'selection_split': manifest['selection_split'],
        'selection_seeds': manifest['selection_seeds'],
        'frozen_validation_choice': metadata['candidate'],
        'config_sha256': metadata['config_sha256'],
        'metrics': parse_metrics(metrics),
    }
    destination = ROOT / 'OUTPUT' / 'tuned_main_records' / f'{dataset}__k_protodiff_j__seed{seed}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, destination)
    promote(dataset)
    print(f'{dataset} seed{seed}: formal tuned record complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('jobs', nargs='+', help='dataset:seed')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    manifest_path = ROOT / 'Config' / 'journal_tuned' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures = []
    for job in args.jobs:
        dataset, raw_seed = job.split(':', 1)
        try:
            run_job(dataset, int(raw_seed), args.gpu, manifest)
        except Exception as error:
            failures.append((job, repr(error)))
            print(f'{job}: FAILED {error!r}', flush=True)
    if failures:
        raise RuntimeError(f'Tuned main queue failures: {failures}')


if __name__ == '__main__':
    main()
