"""Build the journal main-results table from per-run JSON records.

Each input JSON must contain:
{
  "dataset": "ETTh",
  "method": "K-ProtoDiff-J",
  "seed": 2026,
  "metrics": {"C-FID": 0.01, "KL": 0.01, "DS": 0.01, "PS": 0.01}
}

The script only reports mean +/- standard deviation after enough independent
seeds exist. This prevents a single run from being presented as a variance.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from published_paper_results import PUBLISHED_MAIN_RESULTS


DATASETS = [
    'ETTh', 'Electricity', 'Energy', 'Traffic', 'Weather',
    'Illness', 'Exchange', 'Stocks', 'EEG', 'fMRI',
]

METHODS = [
    'Diffusion-TS',
    'PaD-TS',
    'TimeGAN',
    'GT-GAN',
    'TimeVAE',
    'TimeVQVAE',
    'K-ProtoDiff',
    'K-ProtoDiff-J',
]

METRICS = ['C-FID', 'KL', 'DS', 'PS']

RANKING_NOTE = '> 所有指标越小越好；按表中三位小数均值排名，最优加粗、第二个不同数值加下划线，同值并列；OOM 和待运行不参与。标注仅为当前表内数值比较，不代表跨来源公平比较或显著性结论。'


def highlight_rankings(lines):
    """Rank displayed means within each metric/dataset, preserving uncertainties."""
    groups = defaultdict(list)
    for index, line in enumerate(lines):
        if not line.startswith('| '):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != len(DATASETS) + 2 or cells[1] not in METHODS:
            continue
        groups[cells[0]].append((index, cells))
    for rows in groups.values():
        for column in range(2, len(DATASETS) + 2):
            candidates = []
            for index, cells in rows:
                match = re.match(r'^([0-9]+(?:\.[0-9]+)?)(?:\s|$)', cells[column])
                if match:
                    candidates.append((float(match.group(1)), cells))
            ranks = sorted({value for value, _ in candidates})[:2]
            for value, cells in candidates:
                if value == ranks[0]:
                    cells[column] = '**' + cells[column] + '**'
                elif len(ranks) > 1 and value == ranks[1]:
                    cells[column] = '<u>' + cells[column] + '</u>'
        for index, cells in rows:
            lines[index] = '| ' + ' | '.join(cells) + ' |'
    return lines


def load_records(input_dir):
    records = []
    for path in sorted(input_dir.rglob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        required = {'dataset', 'method', 'seed', 'metrics'}
        missing = required - set(record)
        if missing:
            raise ValueError(f'{path} is missing fields: {sorted(missing)}')
        if record['dataset'] not in DATASETS:
            raise ValueError(f'{path}: unknown dataset {record["dataset"]!r}')
        if record['method'] not in METHODS:
            raise ValueError(f'{path}: unknown method {record["method"]!r}')
        records.append(record)
    return records


def aggregate(records, metrics=METRICS):
    values = defaultdict(list)
    seen_runs = set()
    for record in records:
        run_key = (record['dataset'], record['method'], record['seed'])
        if run_key in seen_runs:
            raise ValueError(f'duplicate run: {run_key}')
        seen_runs.add(run_key)
        for metric, value in record['metrics'].items():
            if metric in metrics and value is not None:
                if not np.isfinite(float(value)):
                    raise ValueError(f'nonfinite metric: {run_key} {metric}')
                values[(metric, record['method'], record['dataset'])].append(
                    float(value)
                )
    return values


def format_cell(samples, min_seeds):
    if len(samples) < min_seeds:
        return f'待运行（{len(samples)}/{min_seeds}）'
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    return f'{mean:.3f} ± {std:.3f}'


def build_markdown(values, min_seeds):
    lines = [
        '# Journal Main Results',
        '',
        f'> 本地复现至少 {min_seeds} 个独立种子后报告均值 ± 样本标准差，三位小数。',
        RANKING_NOTE,
        '> † 为本次三种子复现，均值和标准差一起替换旧论文值；未标 † 的原有九数据集基线暂保留论文值（单值表示原文未给标准差，正在补跑）。',
        '> 本次使用统一重建的评测流程，不宣称与会议论文指标实现完全一致；表中仍混有论文值和本地复现值。',
        '',
        '| Metric | Method | ' + ' | '.join(DATASETS) + ' |',
        '|---|---|' + '|'.join(['---:'] * len(DATASETS)) + '|',
    ]
    for metric in METRICS:
        for method in METHODS:
            cells = []
            for dataset in DATASETS:
                published = PUBLISHED_MAIN_RESULTS.get(metric, {}).get(method, {}).get(dataset)
                samples = values.get((metric, method, dataset), [])
                if published is not None and len(samples) >= min_seeds:
                    cells.append(format_cell(samples, min_seeds) + ' †')
                elif published is not None:
                    cells.append(published)
                elif method != 'K-ProtoDiff-J' and dataset == 'fMRI' and not values.get((metric, method, dataset)):
                    cells.append('待复现（0/3）')
                else:
                    cells.append(
                        format_cell(values.get((metric, method, dataset), []), min_seeds)
                    )
            lines.append(f'| {metric} | {method} | ' + ' | '.join(cells) + ' |')
    lines.extend([
        '',
        '方法说明：',
        '',
        '- `K-ProtoDiff`：会议版模型。',
        '- `K-ProtoDiff-J`：多尺度关键原型学习与自适应快速反射采样的完整期刊版。',
        '- 多尺度原型和快速反射的独立版本只进入消融实验，不进入本主表。',
        '',
    ])
    return '\n'.join(highlight_rankings(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='OUTPUT/main_results_records')
    parser.add_argument('--output', default='MAIN_RESULTS.md')
    parser.add_argument('--min-seeds', type=int, default=3)
    args = parser.parse_args()

    if args.min_seeds < 2:
        raise ValueError('--min-seeds must be at least 2 to compute a standard deviation')
    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    values = aggregate(load_records(input_dir))
    output = Path(args.output)
    output.write_text(build_markdown(values, args.min_seeds), encoding='utf-8')
    print(f'Wrote {output} from {len(list(input_dir.rglob("*.json")))} run records')


if __name__ == '__main__':
    main()
