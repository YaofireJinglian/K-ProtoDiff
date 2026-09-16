"""Export a source/results snapshot without datasets, weights or nested Git repos."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError('Destination must be empty')
    files = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    for name in files:
        if not name:
            continue
        rel = Path(name)
        src = ROOT / rel
        if (rel.parts[0] in {'baselines', 'OUTPUT', 'checkpoints'}
                or '__pycache__' in rel.parts or src.is_symlink()
                or src.suffix.lower() in {'.pdf', '.pyc', '.npy', '.pt', '.pth'}
                or not src.is_file()):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    manifest = []
    patchdir = dest / 'baseline_patches'
    patchdir.mkdir()
    for repo in sorted((ROOT / 'baselines').iterdir()):
        if not (repo / '.git').exists():
            continue
        entry = {'name': repo.name,
                 'url': git('remote', 'get-url', 'origin', cwd=repo).decode().strip(),
                 'revision': git('rev-parse', 'HEAD', cwd=repo).decode().strip()}
        patch = git('diff', '--binary', 'HEAD', cwd=repo)
        if patch:
            entry['patch'] = f'baseline_patches/{repo.name}.patch'
            (dest / entry['patch']).write_bytes(patch)
        manifest.append(entry)
    (dest / 'baselines.lock.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (ROOT / 'OUTPUT/.main-results.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        shutil.copytree(ROOT / 'OUTPUT/main_results_records', dest / 'results/main_results_records')
        compact_output = {
            'training_efficiency': ROOT / 'OUTPUT/training_efficiency',
            'sampling_efficiency': ROOT / 'OUTPUT/efficiency_ablation',
            'tuned_main_records': ROOT / 'OUTPUT/tuned_main_records',
        }
        for name, source in compact_output.items():
            if source.exists():
                shutil.copytree(source, dest / 'results' / name)
        cleanup_reports = sorted((ROOT / 'OUTPUT').glob('cleanup_*_report.json'))
        if cleanup_reports:
            report_dir = dest / 'results' / 'cleanup_reports'
            report_dir.mkdir(parents=True, exist_ok=True)
            for source in cleanup_reports:
                shutil.copy2(source, report_dir / source.name)
        qualitative = ROOT / 'OUTPUT/qualitative_downstream'
        if qualitative.exists():
            qualitative_dest = dest / 'results/qualitative_downstream'
            for subdirectory in ['figures', 'records']:
                source = qualitative / subdirectory
                if source.exists():
                    shutil.copytree(source, qualitative_dest / subdirectory)
            summary = qualitative / 'summary.json'
            if summary.exists():
                qualitative_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, qualitative_dest / summary.name)
            for metadata in qualitative.glob('generated/*/k_protodiff_j/seed*/metadata.json'):
                relative = metadata.relative_to(qualitative)
                target = qualitative_dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata, target)
        for name in ['MAIN_RESULTS.md', 'SEGMENT_DTW_RESULTS.md']:
            shutil.copy2(ROOT / name, dest / name)
        qualitative_markdown = dest / 'QUALITATIVE_DOWNSTREAM_RESULTS.md'
        if qualitative_markdown.exists():
            text = qualitative_markdown.read_text()
            text = text.replace('OUTPUT/qualitative_downstream/',
                                'results/qualitative_downstream/')
            qualitative_markdown.write_text(text)
    # Preserve compact, auditable search/ablation metadata. Generated arrays,
    # model checkpoints and logs remain excluded because they are too large for
    # a normal GitHub repository.
    artifact_names = {
        'record.json', 'selection.json', 'complete.json', 'sampling.json',
        'metrics.txt', 'split.json', 'config.yaml', 'candidate.yaml',
        'VALIDATION_RESULTS.md', 'status.json', 'queue_status.json',
    }
    artifact_root = dest / 'results' / 'experiment_artifacts'
    for source in (ROOT / 'checkpoints').rglob('*'):
        if (source.is_file() and source.name in artifact_names
                and any(part.startswith(('journal_search', 'journal_confirmation',
                                         'journal_ablation'))
                        for part in source.relative_to(ROOT / 'checkpoints').parts)):
            relative = source.relative_to(ROOT / 'checkpoints')
            target = artifact_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    notice = f'''# KPD-Journal

Journal extension: multiscale key prototypes and adaptive fast reflection sampling.

## Experiment snapshot

Exported at {timestamp}. Experiments are still running: consult
[queue status](BASELINE_UNCERTAINTY_STATUS.md), [main results](MAIN_RESULTS.md),
[segment DTW](SEGMENT_DTW_RESULTS.md), [qualitative/downstream results](QUALITATIVE_DOWNSTREAM_RESULTS.md),
and [confirmation results](JOURNAL_CONFIRMATION_RESULTS.md).
Incomplete or failed runs are not reported as completed. Tables retain published
baseline values until three local seeds are available. Highlighting compares
displayed values only, not statistical significance or cross-protocol SOTA.
The conference claims in the original README below are not journal-result claims.

## Restore and reproduce

```bash
python scripts/restore_baselines.py
# Configure the environments using scripts/setup_venv.sh and scripts/setup_cu130.sh.
# See experiments/CUDA13_MIGRATION.md for the separate TensorFlow environment.
# Download datasets using scripts/download_paper_datasets.py and scripts/download_fmri_dataset.py.
python scripts/build_main_results_table.py --input-dir results/main_results_records
python scripts/build_segment_dtw_table.py --input-dir results/main_results_records
```

Per-dataset configurations are in `Config/baselines/` and `Config/journal/`;
experiment protocols are in `experiments/`. Some configurations and provenance
records retain the original server paths and must be adapted on a different host.
Baseline repositories are restored at pinned commits with local patches; their
upstream licenses continue to apply. No datasets, checkpoints, environments,
credentials, generated sample arrays, or live scheduler logs are included.
Compact experiment data are under `results/`: per-seed metrics, efficiency
measurements, validation-search records and ablation metadata. This is a
snapshot, not an automatic synchronization of subsequent experiments.

---

'''
    readme = dest / 'README.md'
    readme.write_text(notice + readme.read_text())
    (dest / 'RELEASE_SNAPSHOT.json').write_text(json.dumps({
        'exported_at': timestamp,
        'source_commit': git('rev-parse', 'HEAD').decode().strip(),
        'includes_working_tree_changes': True,
        'main_result_records': len(list((dest / 'results/main_results_records').glob('*.json'))),
        'compact_experiment_artifacts': len(list((dest / 'results').rglob('*.*'))),
        'experiments_complete': False,
    }, indent=2) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
