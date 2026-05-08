"""Aggregate evaluation script for K-ProtoDiff.

Wraps the metrics referenced in the README:
- Context-FID
- KL Divergence
- Discriminative Score (DS)
- Predictive Score (PS)
- Segment-wise DTW

NOTE: this is a minimal reference implementation reconstructed from the
scaffolding of Diffusion-TS (ICLR 2024) plus standard metric implementations.
The exact KL / segment-wise DTW formulations used in the K-ProtoDiff paper may
differ slightly; verify against the paper before reporting numbers.
"""
import os
import argparse
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from Utils.context_fid import Context_FID
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics


def kl_divergence(ori, fake, n_bins=20, eps=1e-10):
    ori_flat = ori.reshape(-1)
    fake_flat = fake.reshape(-1)
    lo = min(ori_flat.min(), fake_flat.min())
    hi = max(ori_flat.max(), fake_flat.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(ori_flat, bins=bins, density=True)
    q, _ = np.histogram(fake_flat, bins=bins, density=True)
    p = p / (p.sum() + eps) + eps
    q = q / (q.sum() + eps) + eps
    return float(np.sum(p * np.log(p / q)))


def segment_wise_dtw(ori, fake, segment_len=8):
    n = min(len(ori), len(fake))
    distances = []
    for i in range(n):
        x, y = ori[i], fake[i]
        T = x.shape[0]
        seg_dists = []
        for s in range(0, T - segment_len + 1, segment_len):
            xs = x[s:s + segment_len]
            ys = y[s:s + segment_len]
            d, _ = fastdtw(xs, ys, dist=euclidean)
            seg_dists.append(d)
        if seg_dists:
            distances.append(np.mean(seg_dists))
    return float(np.mean(distances)) if distances else float('nan')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='output root for this run')
    parser.add_argument('--ori_path', type=str, required=True, help='path to .npy of real samples')
    parser.add_argument('--fake_path', type=str, required=True, help='path to .npy of generated samples')
    parser.add_argument('--n_iter', type=int, default=5, help='repetitions for DS / PS')
    parser.add_argument('--skip', nargs='*', default=[], help='metric names to skip: cfid, kl, ds, ps, dtw')
    return parser.parse_args()


def main():
    args = parse_args()
    ori = np.load(args.ori_path)
    fake = np.load(args.fake_path)
    n = min(len(ori), len(fake))
    ori, fake = ori[:n], fake[:n]
    print(f'ori shape: {ori.shape}, fake shape: {fake.shape}')

    results = {}
    if 'cfid' not in args.skip:
        results['Context-FID'] = Context_FID(ori, fake)
    if 'kl' not in args.skip:
        results['KL'] = kl_divergence(ori, fake)
    if 'ds' not in args.skip:
        ds = [discriminative_score_metrics(ori, fake)[0] for _ in range(args.n_iter)]
        results['DS_mean'] = float(np.mean(ds))
        results['DS_std'] = float(np.std(ds))
    if 'ps' not in args.skip:
        ps = [predictive_score_metrics(ori, fake) for _ in range(args.n_iter)]
        results['PS_mean'] = float(np.mean(ps))
        results['PS_std'] = float(np.std(ps))
    if 'dtw' not in args.skip:
        results['Segment-DTW'] = segment_wise_dtw(ori, fake)

    os.makedirs(args.root, exist_ok=True)
    out = os.path.join(args.root, 'metrics.txt')
    with open(out, 'w') as f:
        for k, v in results.items():
            line = f'{k}: {v:.6f}'
            print(line)
            f.write(line + '\n')
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
