"""Publish the frozen three-seed journal checkpoints as GitHub Release assets."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['fmri', 'energy', 'traffic', 'electricity', 'etth',
            'weather', 'illness', 'exchange', 'stocks', 'eeg']
SEEDS = [2026, 2027, 2028]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def selected_checkpoint(root, dataset, seed, metadata):
    config = root / 'Config' / 'journal_tuned' / f'{dataset}.yaml'
    import yaml
    window = yaml.safe_load(config.read_text())['dataloader']['train_dataset']['params']['window']
    if metadata['reuse_original_checkpoint']:
        base = root / 'checkpoints' / 'journal' / f'Checkpoints_journal_{dataset}_seed{seed}_{window}'
    else:
        base = root / 'checkpoints' / 'journal_tuned' / f'Checkpoints_journal_tuned_{dataset}_seed{seed}_{window}'
    return base / 'checkpoint-10.pt'


def require_records(root):
    for dataset in DATASETS:
        for seed in SEEDS:
            record = root / 'OUTPUT' / 'tuned_main_records' / f'{dataset}__k_protodiff_j__seed{seed}.json'
            if not record.exists():
                raise RuntimeError(f'Formal tuned result is not complete: {dataset}:{seed}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', default='YaofireJinglian/KPD-Journal')
    parser.add_argument('--tag', default='journal-v1-weights')
    args = parser.parse_args()
    require_records(ROOT)
    manifest_config = json.loads((ROOT / 'Config/journal_tuned/manifest.json').read_text())
    gh = ROOT / '.tools' / 'gh-apt' / 'root' / 'usr' / 'bin' / 'gh'
    with tempfile.TemporaryDirectory(prefix='kpd-best-weights-') as folder:
        output = Path(folder)
        assets, files = [], []
        for dataset in DATASETS:
            metadata = manifest_config['datasets'][dataset]
            asset = output / f'kpd-journal-{dataset}-3seeds.tar'
            members = []
            with tarfile.open(asset, 'w') as archive:
                config = ROOT / 'Config/journal_tuned' / f'{dataset}.yaml'
                archive.add(config, arcname=f'{dataset}/config.yaml')
                for seed in SEEDS:
                    checkpoint = selected_checkpoint(ROOT, dataset, seed, metadata)
                    if not checkpoint.exists():
                        raise FileNotFoundError(f'Missing final selected checkpoint: {checkpoint}')
                    record = (ROOT / 'OUTPUT/tuned_main_records' /
                              f'{dataset}__k_protodiff_j__seed{seed}.json')
                    archive.add(checkpoint, arcname=f'{dataset}/seed{seed}/checkpoint-10.pt')
                    archive.add(record, arcname=f'{dataset}/seed{seed}/metrics.json')
                    members.append({'seed': seed, 'checkpoint_sha256': digest(checkpoint),
                                    'checkpoint_bytes': checkpoint.stat().st_size})
            assets.append({'dataset': dataset, 'asset': asset.name,
                           'bytes': asset.stat().st_size, 'sha256': digest(asset),
                           'candidate': metadata['candidate'], 'members': members})
            files.append(asset)
        manifest = {
            'release_tag': args.tag, 'repository': args.repository,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'selection_split': 'validation', 'formal_seeds': SEEDS,
            'note': 'All three formal seeds are provided; no test-seed cherry-picking.',
            'assets': assets,
        }
        manifest_path = output / 'WEIGHTS_MANIFEST.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        files.append(manifest_path)
        exists = subprocess.run([gh, 'release', 'view', args.tag, '--repo', args.repository],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if not exists:
            subprocess.run([gh, 'release', 'create', args.tag, '--repo', args.repository,
                            '--target', 'main', '--title', 'KPD-Journal best formal weights',
                            '--notes', ('Frozen validation-selected journal checkpoints for all ten '
                                        'datasets and formal seeds 2026/2027/2028.')], check=True)
        subprocess.run([gh, 'release', 'upload', args.tag, '--repo', args.repository,
                        '--clobber', *files], check=True)
        remote = json.loads(subprocess.check_output(
            [gh, 'release', 'view', args.tag, '--repo', args.repository,
             '--json', 'url,assets'], text=True))
        remote_sizes = {item['name']: item['size'] for item in remote['assets']}
        for path in files:
            if remote_sizes.get(path.name) != path.stat().st_size:
                raise RuntimeError(f'Uploaded asset size mismatch: {path.name}')
        manifest['release_url'] = remote['url']
        (ROOT / 'WEIGHTS_MANIFEST.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        print(remote['url'])


if __name__ == '__main__':
    main()
