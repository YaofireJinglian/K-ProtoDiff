"""Build the paper-style Segment-wise DTW table from run records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_main_results_table import DATASETS, METHODS, aggregate, load_records, highlight_rankings, RANKING_NOTE
from published_paper_results import PUBLISHED_SEGMENT_DTW_RESULTS


METRICS = ['Seg-DTW-L6', 'Seg-DTW-L8', 'Seg-DTW-L10']


def build(input_dir, min_seeds):
    values = aggregate(load_records(input_dir), METRICS)

    lines = [
        '# Journal Segment-wise DTW Results',
        '',
        f'> 固定步幅为 4；每项至少 {min_seeds} 个独立种子后报告均值 ± 标准差，统一保留三位小数。',
        '> † 为本次三种子复现（均值和标准差一起更新）；其余原有九数据集基线暂保留论文公布值，正在补跑标准差。',
        '> 本次统一重建的评测实现不保证与会议论文完全一致；表中仍混有论文值和本地复现值。',
        RANKING_NOTE,
        '',
        '| Segment length | Method | ' + ' | '.join(DATASETS) + ' |',
        '|---:|---|' + '|'.join(['---:'] * len(DATASETS)) + '|',
    ]
    for metric in METRICS:
        length = metric.rsplit('L', 1)[1]
        for method in METHODS:
            cells = []
            for dataset in DATASETS:
                published = PUBLISHED_SEGMENT_DTW_RESULTS.get(metric, {}).get(method, {}).get(dataset)
                samples = values.get((metric, method, dataset), [])
                if published is not None and len(samples) >= min_seeds:
                    cells.append(f'{np.mean(samples):.3f} ± {np.std(samples, ddof=1):.3f} †')
                    continue
                if published is not None:
                    cells.append(published)
                    continue
                if method != 'K-ProtoDiff-J' and dataset == 'fMRI' and not values.get((metric, method, dataset)):
                    cells.append('待复现（0/3）')
                    continue
                samples = values.get((metric, method, dataset), [])
                if len(samples) < min_seeds:
                    cells.append(f'待运行（{len(samples)}/{min_seeds}）')
                else:
                    cells.append(
                        f'{np.mean(samples):.3f} ± {np.std(samples, ddof=1):.3f}'
                    )
            lines.append(f'| {length} | {method} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(highlight_rankings(lines)) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path('OUTPUT/main_results_records'))
    parser.add_argument('--output', type=Path, default=Path('SEGMENT_DTW_RESULTS.md'))
    parser.add_argument('--min-seeds', type=int, default=3)
    args = parser.parse_args()
    args.output.write_text(build(args.input_dir, args.min_seeds), encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
