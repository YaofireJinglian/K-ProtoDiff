"""Download and verify the fMRI benchmark released with Diffusion-TS."""

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

import gdown
from scipy import io


FILE_ID = '11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6'
ARCHIVE_SHA256 = '87fb0640a99bdf32599c1828cf536bf7dc457abe03eb9ebec49783b474aad47d'
EXPECTED = {
    'sim1.mat': ((10000, 5), '7201bbd41f796b86965cbb4a5f83b02845c9fea00baa987c96a4d3a07b28e19f'),
    'sim2.mat': ((10000, 10), 'bb2c03fffa8ecd33803c6897c1fd94cb751feff8b2a4e47c92e0e4e374c262c6'),
    'sim3.mat': ((10000, 15), '50e42e322d0c22994d5293ce30235967b3ee5bd2e1826e65c319f6bf268bed34'),
    'sim4.mat': ((10000, 50), 'ea9e41cf9aceb00bbb887e11a447ba9965401c20557507ca914e20521e67a0b8'),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locate_member(archive, filename):
    matches = [name for name in archive.namelist() if Path(name).name == filename]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {filename} in archive, found {matches}')
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('Data/datasets'))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output_dir = args.data_root / 'fMRI'
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='kprotodiff-fmri-') as temp_dir:
        archive_path = Path(temp_dir) / 'diffusion_ts_datasets.zip'
        print('Downloading the official Diffusion-TS dataset archive...')
        result = gdown.download(id=FILE_ID, output=str(archive_path), quiet=False)
        if result is None or sha256(archive_path) != ARCHIVE_SHA256:
            raise RuntimeError('Downloaded archive failed SHA-256 verification')

        with zipfile.ZipFile(archive_path) as archive:
            for filename, (shape, expected_hash) in EXPECTED.items():
                destination = output_dir / filename
                if not destination.exists() or args.force:
                    member = locate_member(archive, filename)
                    with archive.open(member) as source, destination.open('wb') as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                if sha256(destination) != expected_hash:
                    raise RuntimeError(f'{destination} failed SHA-256 verification')
                actual_shape = io.loadmat(destination)['ts'].shape
                if actual_shape != shape:
                    raise RuntimeError(f'{destination}: expected {shape}, got {actual_shape}')
                print(f'Verified {destination}: ts{actual_shape}')


if __name__ == '__main__':
    main()
