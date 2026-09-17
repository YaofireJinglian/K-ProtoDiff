"""Atomically promote complete, frozen three-seed candidates into the main table."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [2026, 2027, 2028]


def promote(dataset, root=ROOT):
    source = root / 'OUTPUT' / 'tuned_main_records'
    main = root / 'OUTPUT' / 'main_results_records'
    backup = root / 'OUTPUT' / 'main_results_records_pre_tuning'
    candidates = [source / f'{dataset}__k_protodiff_j__seed{seed}.json' for seed in SEEDS]
    if not all(path.exists() for path in candidates):
        return False
    records = [json.loads(path.read_text(encoding='utf-8')) for path in candidates]
    if [record['seed'] for record in records] != SEEDS:
        raise ValueError(f'{dataset}: candidate records have invalid seeds')
    if any(record.get('provenance') not in {'formal_tuned_confirmation', 'formal_control_reuse'}
           for record in records):
        raise ValueError(f'{dataset}: candidate records do not have formal provenance')
    main.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'OUTPUT' / '.main-results.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for source_path in candidates:
            target = main / source_path.name
            old = backup / source_path.name
            if target.exists() and not old.exists():
                shutil.copy2(target, old)
            temporary = target.with_suffix('.json.tmp')
            shutil.copy2(source_path, temporary)
            os.replace(temporary, target)
        python = root / '.venv' / 'bin' / 'python'
        for builder in ['build_main_results_table.py', 'build_segment_dtw_table.py']:
            subprocess.run([python, root / 'scripts' / builder], cwd=root, check=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+')
    args = parser.parse_args()
    for dataset in args.datasets:
        print(dataset, 'promoted' if promote(dataset) else 'waiting for 3/3')


if __name__ == '__main__':
    main()
