"""Create publication-ready analysis figures from archived experiment records."""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Stocks", "EEG", "Traffic", "fMRI"]
DATASET_KEYS = {name: name.lower() for name in DATASETS}
BUDGETS = [10, 20, 50, 100, 200]
PRIMARY_METRICS = ["C-FID", "KL", "DS", "PS"]
ALL_DATASETS = [
    "ETTh", "Electricity", "Energy", "Traffic", "Weather",
    "Illness", "Exchange", "Stocks", "EEG", "fMRI",
]
ABLATION_VARIANTS = ["point", "point24", "scale4", "scale8", "scale16", "scale4_8", "full"]
VARIANT_LABELS = [
    "Point (8)", "Point (24)", "Scale 4", "Scale 8",
    "Scale 16", "Scales 4+8", "Multiscale 4+8+16",
]
ABLATION_METRICS = [
    "Context-FID", "KL", "DS_mean", "PS_mean",
    "Segment-DTW-L6", "Segment-DTW-L8", "Segment-DTW-L10",
]


def load_ccfa_palette():
    """Reuse the CCFA plotting palette when the skill package is installed."""
    resource = (Path.home() / ".codex/skills/ccf-visual-composer/resources/python")
    if resource.is_dir():
        sys.path.insert(0, str(resource))
        from ccfa_plot_recipes import PALETTES
        return PALETTES
    return {
        "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
        "ccfa": ["#1F6F8B", "#D2673D", "#6657A8", "#2A9D8F", "#B58B2A", "#BA4C5E", "#477AA6", "#5D6977"],
    }


PALETTES = load_ccfa_palette()
BLUE = PALETTES["okabe_ito"][4]
VERMILLION = PALETTES["okabe_ito"][5]
GREEN = PALETTES["okabe_ito"][2]
GOLD = PALETTES["okabe_ito"][0]
INK = "#20252B"
MUTED = "#68717C"
GRID = "#D8DEE5"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.edgecolor": "#7D8792",
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        path = output / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 320
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            # Matplotlib leaves spaces at the ends of multiline path data.
            # Normalizing them keeps generated artifacts clean in Git diffs.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def load_record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pareto_frontier(points):
    """Indices not dominated when both time and distance are minimized."""
    result = []
    for index, (seconds, distance) in enumerate(points):
        dominated = any(
            other_seconds <= seconds and other_distance <= distance
            and (other_seconds < seconds or other_distance < distance)
            for other_index, (other_seconds, other_distance) in enumerate(points)
            if other_index != index
        )
        if not dominated:
            result.append(index)
    return result


def plot_speed_quality(archive, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.085, top=0.82,
                        wspace=0.25, hspace=0.42)
    all_handles = []
    for panel, (dataset, ax) in enumerate(zip(DATASETS, axes.flat)):
        dataset_key = DATASET_KEYS[dataset]
        combined = []
        for mode, color, marker, label in [
            ("ddim", BLUE, "o", "DDIM"),
            ("adaptive", VERMILLION, "D", "Adaptive reflection"),
        ]:
            seconds, distances = [], []
            for budget in BUDGETS:
                record = load_record(
                    archive / f"records/ablation/{dataset_key}/full/{mode}{budget}/record.json")
                metric = record["metrics"]
                segment_dtw = np.mean([
                    metric["Segment-DTW-L6"], metric["Segment-DTW-L8"], metric["Segment-DTW-L10"]
                ])
                seconds.append(record["seconds"])
                distances.append(segment_dtw)
                combined.append((record["seconds"], segment_dtw, mode, budget))
            ax.plot(seconds, distances, color=color, marker=marker, markersize=5.0,
                    linewidth=1.55, label=label, zorder=3)
            for x, y, budget in zip(seconds, distances, BUDGETS):
                ax.annotate(str(budget), (x, y), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.7, color=color, weight="bold")
        frontier = pareto_frontier([(x, y) for x, y, _, _ in combined])
        for index in frontier:
            x, y, mode, _ = combined[index]
            marker = "D" if mode == "adaptive" else "o"
            ax.scatter([x], [y], s=58, facecolors="none", edgecolors=INK,
                       linewidths=0.9, marker=marker, zorder=4)
        ax.set_xscale("log")
        ax.grid(True, which="major", axis="both")
        ax.set_title(f"({chr(97 + panel)}) {dataset}", loc="left", weight="bold")
        ax.set_xlabel("Sampling time (s, log scale)")
        ax.set_ylabel("Mean segment DTW")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    all_handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=1.6, label="DDIM"),
        Line2D([0], [0], color=VERMILLION, marker="D", linewidth=1.6, label="Adaptive reflection"),
        Line2D([0], [0], color=INK, marker="o", markerfacecolor="none", linestyle="none",
               label="Pareto-efficient"),
    ]
    fig.legend(handles=all_handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False)
    fig.suptitle("Sampling speed–temporal fidelity trade-off", y=0.985,
                 fontsize=11.5, weight="bold")
    save_figure(fig, output, "sampling_speed_quality_pareto")


def plot_multiscale_ablation(archive, output):
    matrix = np.empty((len(ABLATION_VARIANTS), len(DATASETS)), dtype=float)
    for column, dataset in enumerate(DATASETS):
        dataset_key = DATASET_KEYS[dataset]
        values = []
        for variant in ABLATION_VARIANTS:
            sampler = "legacy_full" if variant in {"point", "point24"} else "adaptive50"
            record = load_record(
                archive / f"records/ablation/{dataset_key}/{variant}/{sampler}/record.json")
            values.append([record["metrics"][metric] for metric in ABLATION_METRICS])
        values = np.asarray(values)
        ranks = np.stack([
            rankdata(values[:, metric_index], method="average")
            for metric_index in range(values.shape[1])
        ], axis=1)
        matrix[:, column] = ranks.mean(axis=1)
    full_matrix = np.column_stack([matrix, matrix.mean(axis=1)])
    columns = DATASETS + ["Overall"]
    fig, ax = plt.subplots(figsize=(6.9, 3.85), constrained_layout=True)
    image = ax.imshow(full_matrix, cmap="YlGn_r", vmin=1, vmax=7, aspect="auto")
    for row in range(full_matrix.shape[0]):
        for column in range(full_matrix.shape[1]):
            value = full_matrix[row, column]
            color = "white" if value < 2.7 else INK
            is_best = value == np.min(full_matrix[:, column])
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7.6, color=color, weight="bold" if is_best else "normal")
            if is_best:
                ax.scatter(column + 0.38, row - 0.34, s=12, marker="o",
                           facecolor=color, edgecolor="none", zorder=4)
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(VARIANT_LABELS)), labels=VARIANT_LABELS)
    ax.tick_params(length=0)
    ax.axvline(len(DATASETS) - 0.5, color="white", linewidth=2.5)
    ax.add_patch(Rectangle((-0.49, len(VARIANT_LABELS) - 1.49),
                           len(columns) - 0.02, 0.98, fill=False,
                           edgecolor=VERMILLION, linewidth=1.7))
    ax.set_title("Multiscale prototype ablation", loc="left", fontsize=11.5, weight="bold", pad=30)
    ax.text(0.0, 1.015, "Mean rank across seven fidelity metrics; lower is better",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color=MUTED)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    colorbar.set_label("Mean rank")
    colorbar.ax.tick_params(labelsize=7)
    save_figure(fig, output, "multiscale_prototype_ablation_heatmap")


def parse_main_table(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    values = {metric: {dataset: [] for dataset in ALL_DATASETS} for metric in PRIMARY_METRICS}
    journal = {metric: {} for metric in PRIMARY_METRICS}
    for line in lines:
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(ALL_DATASETS) + 2 or cells[0] not in PRIMARY_METRICS:
            continue
        metric, method = cells[:2]
        for dataset, cell in zip(ALL_DATASETS, cells[2:]):
            clean = re.sub(r"\*\*|</?u>|†", "", cell).strip()
            match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", clean)
            if match:
                value = float(match.group(1))
                values[metric][dataset].append(value)
                if method == "K-ProtoDiff-J":
                    journal[metric][dataset] = value
    ranks = np.empty((len(PRIMARY_METRICS), len(ALL_DATASETS)), dtype=int)
    for row, metric in enumerate(PRIMARY_METRICS):
        for column, dataset in enumerate(ALL_DATASETS):
            available = sorted(set(values[metric][dataset]))
            ranks[row, column] = available.index(journal[metric][dataset]) + 1
    return ranks


def plot_main_rank_profile(archive, output):
    ranks = parse_main_table(archive / "tables/MAIN_RESULTS.md")
    categories = np.where(ranks == 1, 0, np.where(ranks == 2, 1, 2))
    cmap = ListedColormap([GREEN, GOLD, "#E5E8EC"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.35, 3.2))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.74)
    ax.imshow(categories, cmap=cmap, norm=norm, aspect="auto")
    for row in range(ranks.shape[0]):
        for column in range(ranks.shape[1]):
            rank = ranks[row, column]
            text = "1st" if rank == 1 else "2nd" if rank == 2 else str(rank)
            color = "white" if rank == 1 else INK
            ax.text(column, row, text, ha="center", va="center", fontsize=7.4,
                    color=color, weight="bold" if rank <= 2 else "normal")
    ax.set_xticks(np.arange(len(ALL_DATASETS)), labels=ALL_DATASETS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(PRIMARY_METRICS)), labels=PRIMARY_METRICS)
    ax.tick_params(length=0)
    first = int(np.sum(ranks == 1))
    second = int(np.sum(ranks == 2))
    fig.text(0.08, 0.955, "K-ProtoDiff-J rank profile in the reported main table",
             ha="left", va="top", fontsize=11.5, weight="bold")
    fig.text(0.08, 0.865, f"{first} first-place and {second} second-place cells across 40 comparisons",
             ha="left", va="top", fontsize=7.4, color=MUTED)
    legend = [
        Patch(facecolor=GREEN, edgecolor="none", label="First"),
        Patch(facecolor=GOLD, edgecolor="none", label="Second"),
        Patch(facecolor="#E5E8EC", edgecolor="none", label="Rank 3 or lower"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.99, 0.90),
               ncol=3, frameon=False, handlelength=1.2, columnspacing=1.2)
    save_figure(fig, output, "main_result_rank_profile")


def write_readme(output):
    text = """# Additional manuscript figures

All figures are generated from the archived JSON records or the archived main
table. PNG files are high-resolution previews; PDF and SVG files retain vector
text and geometry for manuscript editing.

## Suggested captions

**Sampling speed–quality trade-off.** Sampling time versus mean segment-DTW at
five denoising budgets. Number labels denote the configured budget, and outlined
markers denote non-dominated operating points. Lower-left is better.

**Multiscale prototype ablation.** Mean rank across seven fidelity metrics for
point, single-scale, two-scale, and full multiscale prototype configurations on
four representative datasets. A small dot marks the best configuration in each
column, and the red outline identifies the complete multiscale design.

**Main-result rank profile.** Dataset-wise ranks of K-ProtoDiff-J for the four
primary metrics in the reported main table. Ranking follows the displayed
three-decimal means; it does not encode statistical significance.

Authoring source: `../../scripts/create_paper_figures.py`.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path,
                        default=PROJECT_ROOT / "paper_experiments")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    archive = args.archive_root.resolve()
    output = (args.output or archive / "figures/analysis").resolve()
    set_style()
    plot_speed_quality(archive, output)
    plot_multiscale_ablation(archive, output)
    plot_main_rank_profile(archive, output)
    write_readme(output)
    print(json.dumps({
        "output": str(output),
        "figures": [
            "sampling_speed_quality_pareto",
            "multiscale_prototype_ablation_heatmap",
            "main_result_rank_profile",
        ],
        "formats": ["pdf", "svg", "png"],
    }, indent=2))


if __name__ == "__main__":
    main()
