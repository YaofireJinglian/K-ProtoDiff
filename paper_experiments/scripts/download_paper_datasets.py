"""Download and normalize the nine public datasets used in the AAAI paper."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


AUTOFORMER_ROOT = (
    'https://huggingface.co/datasets/AutonLab/Timeseries-PILE/'
    'resolve/main/forecasting/autoformer'
)

DIRECT_DATASETS = {
    'ETTh.csv': (
        'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
        'main/ETT-small/ETTh1.csv'
    ),
    'electricity.csv': f'{AUTOFORMER_ROOT}/electricity.csv?download=true',
    'traffic.csv': f'{AUTOFORMER_ROOT}/traffic.csv?download=true',
    'weather.csv': f'{AUTOFORMER_ROOT}/weather.csv?download=true',
    'national_illness.csv': f'{AUTOFORMER_ROOT}/national_illness.csv?download=true',
    'exchange_rate.csv': f'{AUTOFORMER_ROOT}/exchange_rate.csv?download=true',
    'stock_data.csv': (
        'https://raw.githubusercontent.com/jsyoon0823/TimeGAN/'
        'master/data/stock_data.csv'
    ),
}

ARCHIVES = {
    'energy.zip': (
        'https://archive.ics.uci.edu/static/public/374/'
        'appliances+energy+prediction.zip'
    ),
    'eeg.zip': (
        'https://archive.ics.uci.edu/static/public/264/'
        'eeg+eye+state.zip'
    ),
}


def download(url, destination, force=False):
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f'Using existing {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'K-ProtoDiff-research/1.0'})
    print(f'Downloading {destination.name} from {url}')
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def extract_archives(root, archive_root, force=False):
    energy_target = root / 'energy_data.csv'
    if force or not energy_target.exists():
        with zipfile.ZipFile(archive_root / 'energy.zip') as archive:
            member = next(name for name in archive.namelist() if name.endswith('energydata_complete.csv'))
            with archive.open(member) as source:
                frame = pd.read_csv(source)
        # The model consumes the 28 numeric attributes; the timestamp is not a feature.
        frame = frame.select_dtypes(include='number')
        frame.to_csv(energy_target, index=False)

    eeg_target = root / 'EEG_Eye_State.arff'
    if force or not eeg_target.exists():
        with zipfile.ZipFile(archive_root / 'eeg.zip') as archive:
            member = next(name for name in archive.namelist() if name.lower().endswith('.arff'))
            with archive.open(member) as source, eeg_target.open('wb') as output:
                shutil.copyfileobj(source, output)


def describe_csv(path):
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include='number')
    return {
        'rows': len(frame),
        'columns': len(frame.columns),
        'numeric_columns': len(numeric.columns),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='Data/datasets')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    archive_root = root / '.downloads'
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in DIRECT_DATASETS.items():
        download(url, root / filename, force=args.force)
    for filename, url in ARCHIVES.items():
        download(url, archive_root / filename, force=args.force)
    extract_archives(root, archive_root, force=args.force)

    expected = list(DIRECT_DATASETS) + ['energy_data.csv', 'EEG_Eye_State.arff']
    manifest = {}
    for filename in expected:
        path = root / filename
        entry = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if path.suffix == '.csv':
            entry.update(describe_csv(path))
        manifest[filename] = entry
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(f'Dataset manifest saved to {manifest_path}')


if __name__ == '__main__':
    main()
