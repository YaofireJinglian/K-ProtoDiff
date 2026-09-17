"""Freeze validation-selected settings into full-budget journal configurations."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.journal_search import apply_options, architecture


DATASETS = [
    'fmri', 'energy', 'traffic', 'electricity', 'etth',
    'weather', 'illness', 'exchange', 'stocks', 'eeg',
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root=ROOT):
    destination = root / 'Config' / 'journal_tuned'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        'selection_split': 'validation',
        'selection_seeds': [2031, 2032, 2033],
        'formal_seeds': [2026, 2027, 2028],
        'promotion_rule': 'promote a frozen per-dataset candidate only after all three formal runs complete',
        'datasets': {},
    }
    for dataset in DATASETS:
        base_path = root / 'Config' / 'journal' / f'{dataset}.yaml'
        selection_path = root / 'checkpoints' / 'journal_search_extended' / dataset / 'selection.json'
        if not selection_path.exists():
            raise FileNotFoundError(f'Missing completed validation selection: {selection_path}')
        base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        selected = selection['selected']
        records = selected.get('records') or []
        if len(records) != 3:
            raise ValueError(f'{dataset}: expected a three-seed validation selection')
        exemplar = records[0]
        if [record['seed'] for record in records] != manifest['selection_seeds']:
            raise ValueError(f'{dataset}: unexpected validation seeds')
        if any(record.get('split') != 'validation' for record in records):
            raise ValueError(f'{dataset}: configuration selection must be validation-only')
        overrides = exemplar['train_overrides']
        sampler = exemplar['sampler_params']
        if any(record['train_overrides'] != overrides or record['sampler_params'] != sampler
               for record in records):
            raise ValueError(f'{dataset}: selected records disagree on the frozen configuration')
        tuned = apply_options(base, overrides)
        tuned['model']['params'].update(sampler)
        # Search budgets are screening-only. Formal runs retain the full original
        # training budget and checkpoint cadence from Config/journal.
        tuned['solver']['max_epochs'] = base['solver']['max_epochs']
        tuned['solver']['save_cycle'] = base['solver']['save_cycle']
        tuned['solver']['results_folder'] = f'./checkpoints/journal_tuned/Checkpoints_journal_tuned_{dataset}'
        if architecture(tuned) != architecture(base):
            raise ValueError(f'{dataset}: frozen candidate changed architecture')
        config_path = destination / f'{dataset}.yaml'
        rendered = yaml.safe_dump(tuned, sort_keys=False)
        if config_path.exists() and config_path.read_text(encoding='utf-8') != rendered:
            raise ValueError(f'Refusing to mutate an existing frozen config: {config_path}')
        config_path.write_text(rendered, encoding='utf-8')
        exact_control = not overrides and sampler == {
            'adaptive_sampling_timesteps': base['model']['params']['adaptive_sampling_timesteps'],
            'reflection_strength': base['model']['params']['reflection_strength'],
        }
        manifest['datasets'][dataset] = {
            'candidate': selected['candidate'],
            'validation_mean_rank': selected['mean_validation_rank'],
            'train_overrides': overrides,
            'sampler_params': sampler,
            'full_training_steps': base['solver']['max_epochs'],
            'screen_training_steps': exemplar['train_steps'],
            'reuse_original_checkpoint': not bool(overrides),
            'reuse_original_record': exact_control,
            'base_config_sha256': sha256(base_path),
            'selection_sha256': sha256(selection_path),
            'config_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
        }
    manifest_path = destination / 'manifest.json'
    rendered_manifest = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    if manifest_path.exists() and manifest_path.read_text(encoding='utf-8') != rendered_manifest:
        raise ValueError(f'Refusing to mutate an existing frozen manifest: {manifest_path}')
    manifest_path.write_text(rendered_manifest, encoding='utf-8')
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(prepare())


if __name__ == '__main__':
    main()
