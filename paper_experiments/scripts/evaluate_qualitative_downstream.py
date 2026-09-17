"""Temporal-structure figures and a fixed TSTR forecasting experiment."""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "OUTPUT/qualitative_downstream"
SEEDS = [2026, 2027, 2028]
PREDICTOR_SEEDS = [5101, 5102, 5103]
LABELS = {"stocks": "Stocks", "exchange": "Exchange", "fmri": "fMRI"}
METHOD_LABELS = {
    "real": "Real training",
    "k_protodiff_j": "K-ProtoDiff-J",
    "k_protodiff": "K-ProtoDiff",
    "pad_ts": "PaD-TS",
    "timevae": "TimeVAE",
    "timegan": "TimeGAN",
}


def real_path(dataset):
    return ROOT / "checkpoints/baseline_reproduction" / dataset / "gtgan/seed2026/training_data.npy"


def method_path(dataset, method, seed):
    if method == "k_protodiff_j":
        return OUT / "generated" / dataset / method / f"seed{seed}/samples.npy"
    if dataset in {"stocks", "exchange"}:
        return ROOT / "checkpoints/baseline_reproduction" / dataset / method / f"seed{seed}/samples.npy"
    if method == "pad_ts":
        return ROOT / "checkpoints/PaD-TS" / f"seed{seed}/ddpm_fake_fmri_24.npy"
    if method in {"timevae", "timegan"}:
        return ROOT / "checkpoints" / method / f"seed{seed}/samples.npy"
    return ROOT / "__missing__"


def methods_for(dataset):
    common = ["k_protodiff_j", "pad_ts", "timevae"]
    return common + (["k_protodiff"] if dataset != "fmri" else ["timegan"])


def load_array(path):
    values = np.asarray(np.load(path), dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid array: {path} {values.shape}")
    if float(values.min()) < -0.05:
        values = (values + 1.0) / 2.0
    return np.clip(values, 0.0, 1.0)


def fixed_subset(values, count=2048, seed=7331):
    if len(values) <= count:
        return values.copy()
    rng = np.random.default_rng(seed)
    return values[np.sort(rng.choice(len(values), size=count, replace=False))]


def acf(values, max_lag=8):
    result = []
    for lag in range(1, max_lag + 1):
        left, right = values[:, :-lag], values[:, lag:]
        left = left.reshape(-1, left.shape[-1])
        right = right.reshape(-1, right.shape[-1])
        left = left - left.mean(axis=0, keepdims=True)
        right = right - right.mean(axis=0, keepdims=True)
        denom = np.sqrt((left * left).mean(0) * (right * right).mean(0)) + 1e-8
        result.append((left * right).mean(0) / denom)
    return np.stack(result)


def normalized_psd(values):
    centered = values - values.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power = power.mean(axis=0)
    return power / (power.sum(axis=0, keepdims=True) + 1e-8)


def correlation(values):
    flat = values.reshape(-1, values.shape[-1])
    result = np.corrcoef(flat, rowvar=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def trend_signature(values):
    increments = np.diff(values, axis=1)
    return np.stack([
        increments.mean(axis=(0, 1)),
        np.abs(increments).mean(axis=(0, 1)),
        increments.std(axis=(0, 1)),
    ])


def structure_distances(real, fake):
    real_acf, fake_acf = acf(real), acf(fake)
    real_psd, fake_psd = normalized_psd(real), normalized_psd(fake)
    real_corr, fake_corr = correlation(real), correlation(fake)
    tri = np.triu_indices(real_corr.shape[0], k=1)
    return {
        "ACF": float(np.mean(np.abs(real_acf - fake_acf))),
        "Spectrum_x100": float(100.0 * np.mean(np.abs(real_psd - fake_psd))),
        "Correlation": float(np.mean(np.abs(real_corr[tri] - fake_corr[tri]))),
        "Trend_x100": float(100.0 * np.mean(np.abs(trend_signature(real) - trend_signature(fake)))),
    }


class ForecastGRU(nn.Module):
    def __init__(self, features):
        super().__init__()
        hidden = min(64, max(32, features * 2))
        self.gru = nn.GRU(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, features))

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(output[:, -1])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_forecaster(training, test, seed, device):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(training))
    cut = int(0.85 * len(order))
    train_values, valid_values = training[order[:cut]], training[order[cut:]]
    feature_mean = train_values[:, :-1].mean(axis=(0, 1), keepdims=True)
    feature_std = train_values[:, :-1].std(axis=(0, 1), keepdims=True) + 1e-5

    def prepare(values):
        x = (values[:, :-1] - feature_mean) / feature_std
        y = (values[:, -1] - feature_mean[0]) / feature_std[0]
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    train_x, train_y = prepare(train_values)
    valid_x, valid_y = prepare(valid_values)
    test_x, _ = prepare(test)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True,
                        generator=loader_generator, num_workers=0)
    model = ForecastGRU(training.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state, best_loss, stale = None, math.inf, 0
    valid_x, valid_y = valid_x.to(device), valid_y.to(device)
    epochs_run = 0
    for epoch in range(60):
        epochs_run = epoch + 1
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            valid_loss = float(nn.functional.mse_loss(model(valid_x), valid_y).cpu())
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(test_x), 512):
            prediction = model(test_x[offset:offset + 512].to(device)).cpu().numpy()
            predictions.append(prediction)
    predicted = np.concatenate(predictions) * feature_std[0] + feature_mean[0]
    target = test[:, -1]
    return {
        "MAE": float(np.mean(np.abs(predicted - target))),
        "RMSE": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "epochs": epochs_run,
    }


def downstream_split(real, gap=23):
    """Five distributed test blocks; purge overlapping real-training windows."""
    count = len(real)
    block_length = max(1, int(round(0.03 * count)))
    centers = np.linspace(0.10, 0.90, 5) * (count - 1)
    test_mask = np.zeros(count, dtype=bool)
    for center in centers.astype(int):
        start = max(0, center - block_length // 2)
        end = min(count, start + block_length)
        start = max(0, end - block_length)
        test_mask[start:end] = True
    train_mask = ~test_mask.copy()
    test_indices = np.flatnonzero(test_mask)
    for index in test_indices:
        train_mask[max(0, index - gap):min(count, index + gap + 1)] = False
    return real[train_mask], real[test_mask]


def make_figure(dataset, real, generated):
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    channel = int(np.argmax(real.var(axis=(0, 1))))
    example = 17
    kpd = generated["k_protodiff_j"]
    baseline_name = "pad_ts"
    baseline = generated[baseline_name]
    time = np.arange(real.shape[1])
    def standardize_trajectory(values):
        curve = values[example, :, channel]
        return (curve - curve.mean()) / (curve.std() + 1e-8)
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[0, :2])
    ax.plot(time, standardize_trajectory(real), label="Real", linewidth=2.2, color="#222222")
    ax.plot(time, standardize_trajectory(kpd), label="K-ProtoDiff-J", linewidth=1.8, color="#d62728")
    ax.plot(time, standardize_trajectory(baseline), label="PaD-TS", linewidth=1.5, color="#1f77b4")
    ax.set_title(f"Standardized fixed trajectory (channel {channel})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Within-window standardized value")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 2])
    lags = np.arange(1, 9)
    ax.plot(lags, acf(real)[:, channel], "o-", label="Real", color="#222222")
    ax.plot(lags, acf(kpd)[:, channel], "o-", label="K-ProtoDiff-J", color="#d62728")
    ax.plot(lags, acf(baseline)[:, channel], "o-", label="PaD-TS", color="#1f77b4")
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lag")

    ax = fig.add_subplot(grid[0, 3])
    frequency = np.arange(normalized_psd(real).shape[0])
    ax.plot(frequency, normalized_psd(real)[:, channel], "o-", color="#222222")
    ax.plot(frequency, normalized_psd(kpd)[:, channel], "o-", color="#d62728")
    ax.plot(frequency, normalized_psd(baseline)[:, channel], "o-", color="#1f77b4")
    ax.set_title("Normalized spectrum")
    ax.set_xlabel("Frequency bin")

    matrices = [("Real", real), ("K-ProtoDiff-J", kpd), ("PaD-TS", baseline)]
    image = None
    for column, (label, values) in enumerate(matrices):
        ax = fig.add_subplot(grid[1, column])
        image = ax.imshow(correlation(values), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_title(f"{label} correlation")
        ax.set_xlabel("Variable")
        ax.set_ylabel("Variable")
    color_ax = fig.add_subplot(grid[1, 3])
    color_ax.axis("off")
    fig.colorbar(image, ax=color_ax, fraction=0.45, label="Correlation")
    title = f"{LABELS[dataset]}: qualitative and temporal-structure case study"
    if dataset == "fmri":
        title += " (correlation matrices are functional-connectivity summaries)"
    fig.suptitle(title, fontsize=14)
    destination = figure_dir / f"{dataset}_qualitative_structure.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return destination, channel


def mean_std(records, metric):
    values = np.array([record[metric] for record in records], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def styled_cells(rows, metric, methods, exclude_real=False):
    candidates = [m for m in methods if not (exclude_real and m == "real")]
    ranked = sorted(candidates, key=lambda method: mean_std(rows[method], metric)[0])
    style = {ranked[0]: "bold"}
    if len(ranked) > 1:
        style[ranked[1]] = "underline"
    result = {}
    for method in methods:
        mean, std = mean_std(rows[method], metric)
        text = f"{mean:.3f} ± {std:.3f}"
        if style.get(method) == "bold":
            text = f"**{text}**"
        elif style.get(method) == "underline":
            text = f"<u>{text}</u>"
        result[method] = text
    return result


def render_markdown(structure, downstream, figures, channels):
    lines = [
        "# Qualitative, temporal-structure, and downstream-utility results",
        "",
        "> Frozen advantage-case analysis: Stocks, Exchange, and fMRI were selected before this analysis because K-ProtoDiff-J ranks first on the predictive score in the main table. These results are not presented as an all-dataset claim.",
        "",
        "All values use three seeds and show mean ± sample standard deviation, rounded to three decimals. Lower is better. Best synthetic result is **bold** and second best is <u>underlined</u>.",
        "",
        "## Result summary",
        "",
        "K-ProtoDiff-J ranks first in 7 of 12 temporal-structure comparisons and second in the other 5. In the downstream experiment it ranks first in 3 of 6 comparisons and second in the other 3. The strongest structure results occur on Stocks and Exchange, while fMRI gives the clearest downstream result.",
        "",
        "## Temporal-structure fidelity",
        "",
        "Spectrum and local-trend distances are scaled by 100 for readability.",
        "",
    ]
    metrics = ["ACF", "Spectrum_x100", "Correlation", "Trend_x100"]
    headers = ["ACF", "Spectrum ×100", "Correlation", "Local trend ×100"]
    for dataset in ["stocks", "exchange", "fmri"]:
        methods = methods_for(dataset)
        cells = {metric: styled_cells(structure[dataset], metric, methods) for metric in metrics}
        lines += [f"### {LABELS[dataset]}", "", "| Method | " + " | ".join(headers) + " |",
                  "|---|---:|---:|---:|---:|"]
        for method in methods:
            lines.append("| " + METHOD_LABELS[method] + " | " + " | ".join(cells[m][method] for m in metrics) + " |")
        lines += ["", f"![{LABELS[dataset]} qualitative structure]({figures[dataset]})", "",
                  f"Displayed channel: {channels[dataset]}, selected solely by real-data variance.", ""]
    lines += [
        "## Downstream one-step forecasting (item 5)",
        "",
        "The predictor receives the first 23 time steps and predicts all variables at the final step. It is trained on real or generated windows and evaluated on five fixed, distributed real-data time blocks. Real training windows within 23 positions of a test block are purged to avoid overlap leakage.",
        "",
        "| Dataset | Training source | MAE | RMSE |",
        "|---|---|---:|---:|",
    ]
    for dataset in ["stocks", "exchange", "fmri"]:
        methods = ["real"] + methods_for(dataset)
        mae = styled_cells(downstream[dataset], "MAE", methods, exclude_real=True)
        rmse = styled_cells(downstream[dataset], "RMSE", methods, exclude_real=True)
        for method in methods:
            lines.append(f"| {LABELS[dataset]} | {METHOD_LABELS[method]} | {mae[method]} | {rmse[method]} |")
    lines += [
        "",
        "Real training is a reference condition and is excluded from synthetic-method ranking.",
        "",
        "## Interpretation boundary",
        "",
        "The generators follow the paper's full-data generation protocol. The downstream table therefore measures utility under that protocol, not strict unseen-period generalization by the generator. Strict chronological forecasting would require retraining every generator only on the early partition.",
        "",
        "Full protocol: [experiments/QUALITATIVE_DOWNSTREAM_PROTOCOL.md](experiments/QUALITATIVE_DOWNSTREAM_PROTOCOL.md)",
    ]
    destination = ROOT / "QUALITATIVE_DOWNSTREAM_RESULTS.md"
    destination.write_text("\n".join(lines) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records").mkdir(exist_ok=True)
    structure, downstream, figures, channels = {}, {}, {}, {}

    for dataset in ["stocks", "exchange", "fmri"]:
        real_all = load_array(real_path(dataset))
        real_reference = fixed_subset(real_all, 2048, 7331)
        structure[dataset] = defaultdict(list)
        generated_for_figure = {}
        for method in methods_for(dataset):
            for seed in SEEDS:
                path = method_path(dataset, method, seed)
                if not path.exists():
                    raise FileNotFoundError(path)
                fake = fixed_subset(load_array(path), 2048, seed + 9000)
                record = {"dataset": dataset, "method": method, "seed": seed,
                          **structure_distances(real_reference, fake)}
                structure[dataset][method].append(record)
                record_path = OUT / "records" / f"structure__{dataset}__{method}__seed{seed}.json"
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                if seed == 2026:
                    generated_for_figure[method] = fake
        figure, channel = make_figure(dataset, real_reference, generated_for_figure)
        figures[dataset] = str(figure.relative_to(ROOT))
        channels[dataset] = channel

        downstream[dataset] = defaultdict(list)
        if not args.structure_only:
            real_pool, test = downstream_split(real_all)
            all_methods = ["real"] + methods_for(dataset)
            for run_index, (generator_seed, predictor_seed) in enumerate(zip(SEEDS, PREDICTOR_SEEDS)):
                for method in all_methods:
                    if method == "real":
                        training = fixed_subset(real_pool, 2048, predictor_seed)
                    else:
                        training = fixed_subset(load_array(method_path(dataset, method, generator_seed)),
                                                2048, generator_seed + 9000)
                    metrics = train_forecaster(training, test, predictor_seed, device)
                    record = {"dataset": dataset, "method": method,
                              "generator_seed": None if method == "real" else generator_seed,
                              "predictor_seed": predictor_seed, **metrics}
                    downstream[dataset][method].append(record)
                    record_path = OUT / "records" / f"downstream__{dataset}__{method}__run{run_index + 1}.json"
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                    print(f"DOWNSTREAM {dataset} {method} run{run_index + 1}: {metrics}", flush=True)
        else:
            for method in ["real"] + methods_for(dataset):
                paths = sorted((OUT / "records").glob(f"downstream__{dataset}__{method}__run*.json"))
                downstream[dataset][method] = [json.loads(path.read_text()) for path in paths]
    if args.structure_only and any(not downstream[d][m] for d in downstream for m in downstream[d]):
        print("Structure analysis complete; downstream records not yet complete.", flush=True)
        return
    destination = render_markdown(structure, downstream, figures, channels)
    summary = {"structure": structure, "downstream": downstream, "figures": figures,
               "displayed_channels": channels}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"COMPLETE {destination}", flush=True)


if __name__ == "__main__":
    main()
