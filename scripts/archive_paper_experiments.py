"""Build a compact, auditable archive of every experiment used in the paper."""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TABLES = [
    "MAIN_RESULTS.md",
    "SEGMENT_DTW_RESULTS.md",
    "ABLATION_RESULTS.md",
    "EFFICIENCY_RESULTS.md",
    "TRAINING_EFFICIENCY_RESULTS.md",
    "QUALITATIVE_DOWNSTREAM_RESULTS.md",
    "JOURNAL_CONFIRMATION_RESULTS.md",
    "BASELINE_UNCERTAINTY_STATUS.md",
]

PROTOCOLS = [
    "experiments/TUNED_MAIN_PROTOCOL.md",
    "experiments/JOURNAL_SEARCH.md",
    "experiments/JOURNAL_EXTENDED_SEARCH.md",
    "experiments/JOURNAL_REFINEMENT_SEARCH.md",
    "experiments/QUALITATIVE_DOWNSTREAM_PROTOCOL.md",
    "experiments/FMRI_BASELINES.md",
    "experiments/BASELINE_UNCERTAINTY.md",
    "experiments/BASELINE_CANCELLATION.md",
    "experiments/WEIGHT_RELEASE_AND_CLEANUP.md",
    "experiments/THROUGHPUT_24H.md",
    "experiments/CUDA13_MIGRATION.md",
]

SCRIPTS = [
    "scripts/download_paper_datasets.py",
    "scripts/download_fmri_dataset.py",
    "scripts/setup_venv.sh",
    "scripts/setup_cu130.sh",
    "scripts/prepare_tuned_main.py",
    "scripts/run_tuned_main.py",
    "scripts/promote_tuned_main.py",
    "scripts/run_baseline_reproduction.py",
    "scripts/baseline_reproduction_queue.py",
    "scripts/eval_seeded.py",
    "scripts/build_main_results_table.py",
    "scripts/build_segment_dtw_table.py",
    "scripts/journal_ablation.py",
    "scripts/run_ablation_queue.py",
    "scripts/benchmark_sampling_efficiency.py",
    "scripts/benchmark_training_efficiency.py",
    "scripts/generate_analysis_samples.py",
    "scripts/evaluate_qualitative_downstream.py",
    "scripts/create_paper_figures.py",
    "scripts/published_paper_results.py",
    "scripts/publish_best_weights.py",
]


def copy_file(source, target):
    source = ROOT / source if not isinstance(source, Path) or not source.is_absolute() else source
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source, target, ignore=None):
    source = ROOT / source if not isinstance(source, Path) or not source.is_absolute() else source
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, ignore=ignore)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_ablation(staging):
    source_root = ROOT / "checkpoints/journal_ablation"
    destination_root = staging / "records/ablation"
    allowed = {"record.json", "complete.json", "config.yaml", "status.json"}
    for source in source_root.rglob("*"):
        if source.is_file() and source.name in allowed:
            copy_file(source, destination_root / source.relative_to(source_root))


def compact_qualitative(staging):
    source = ROOT / "OUTPUT/qualitative_downstream"
    destination = staging / "records/qualitative_downstream"
    copy_tree(source / "records", destination / "records")
    copy_file(source / "summary.json", destination / "summary.json")
    for metadata in source.glob("generated/*/k_protodiff_j/seed*/metadata.json"):
        copy_file(metadata, destination / metadata.relative_to(source))
    copy_tree(source / "figures", staging / "figures/qualitative")


def record_counts(staging):
    return {
        "main_result_records": len(list((staging / "records/main_results").glob("*.json"))),
        "tuned_main_records": len(list((staging / "records/tuned_main").glob("*.json"))),
        "ablation_quality_records": len(list((staging / "records/ablation").glob("*/*/*/record.json"))),
        "ablation_completed_variants": len(list((staging / "records/ablation").glob("*/*/complete.json"))),
        "sampling_efficiency_records": len(list((staging / "records/sampling_efficiency").glob("*.json"))),
        "training_efficiency_records": len(list((staging / "records/training_efficiency").glob("*.json"))),
        "qualitative_structure_records": len(list((staging / "records/qualitative_downstream/records").glob("structure*.json"))),
        "downstream_utility_records": len(list((staging / "records/qualitative_downstream/records").glob("downstream*.json"))),
    }


def validate_counts(counts):
    expected = {
        "tuned_main_records": 30,
        "ablation_quality_records": 92,
        "ablation_completed_variants": 44,
        "sampling_efficiency_records": 40,
        "training_efficiency_records": 20,
        "qualitative_structure_records": 36,
        "downstream_utility_records": 45,
    }
    mismatches = {key: (counts[key], value) for key, value in expected.items()
                  if counts[key] != value}
    if mismatches:
        raise RuntimeError(f"Archive completeness check failed: {mismatches}")


def write_readme(staging, counts):
    text = f"""# Paper experiment archive

This directory contains the compact experimental evidence used in the
K-ProtoDiff journal manuscript. It is designed to be versioned in Git and
audited without copying datasets, environments, generated arrays, or model
checkpoints into the repository.

## Completion snapshot

- Final K-ProtoDiff-J runs: 30 three-seed dataset runs.
- Main-result records: {counts['main_result_records']} JSON records.
- Ablation study: {counts['ablation_quality_records']}/92 quality records and {counts['ablation_completed_variants']}/44 completed training variants.
- Sampling-efficiency records: {counts['sampling_efficiency_records']}.
- Training-efficiency records: {counts['training_efficiency_records']}.
- Qualitative temporal-structure records: {counts['qualitative_structure_records']}.
- Downstream-utility records: {counts['downstream_utility_records']}.
- Failed planned ablation jobs: 0.
- Fourteen GT-GAN jobs were explicitly cancelled and are documented; they are
  not presented as completed experiments.

## Directory map

- `tables/`: manuscript-ready result tables.
- `records/`: per-seed and per-condition machine-readable results.
- `figures/`: qualitative and temporal-structure figures.
- `configs/`: final, baseline, tuned, and ablation YAML files.
- `protocols/`: selection, evaluation, cancellation, and environment records.
- `scripts/`: exact launch, evaluation, table-building, and archiving scripts.
- `environment/`: dependency specifications.
- `weights/`: released-weight manifest. The large weight files remain in the
  GitHub release `journal-v1-weights`.
- `MANIFEST.json`: SHA-256 and byte size for every archived payload file.

## Reproduction boundary

Run the scripts from the repository root so they can import the model and data
utilities. Dataset files and checkpoints are intentionally excluded from this
folder. Their acquisition paths, seeds, selected configurations, and weight
checksums are preserved here. All paper tables retain three-decimal formatting;
main results use three seeds, while the predefined ablation and efficiency
studies follow their documented single-seed protocol.
"""
    (staging / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "paper_experiments")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination == ROOT or ROOT not in destination.parents:
        raise ValueError("Destination must be a dedicated directory inside the repository")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in TABLES:
        copy_file(name, staging / "tables" / Path(name).name)
    qualitative_table = staging / "tables/QUALITATIVE_DOWNSTREAM_RESULTS.md"
    qualitative_text = qualitative_table.read_text(encoding="utf-8")
    qualitative_text = qualitative_text.replace(
        "OUTPUT/qualitative_downstream/figures/", "../figures/qualitative/")
    qualitative_text = qualitative_text.replace(
        "experiments/QUALITATIVE_DOWNSTREAM_PROTOCOL.md",
        "../protocols/QUALITATIVE_DOWNSTREAM_PROTOCOL.md")
    qualitative_table.write_text(qualitative_text, encoding="utf-8")

    for name in PROTOCOLS:
        copy_file(name, staging / "protocols" / Path(name).name)
    for name in SCRIPTS:
        copy_file(name, staging / "scripts" / Path(name).name)
    for directory in ["Config/baselines", "Config/journal", "Config/journal_tuned",
                      "Config/journal_ablation"]:
        copy_tree(directory, staging / "configs" / Path(directory).name)

    copy_tree("OUTPUT/main_results_records", staging / "records/main_results")
    copy_tree("OUTPUT/tuned_main_records", staging / "records/tuned_main")
    copy_tree("OUTPUT/efficiency_ablation", staging / "records/sampling_efficiency",
              ignore=shutil.ignore_patterns("*.md"))
    copy_tree("OUTPUT/training_efficiency", staging / "records/training_efficiency",
              ignore=shutil.ignore_patterns("*.md"))
    compact_ablation(staging)
    compact_qualitative(staging)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/create_paper_figures.py"),
        "--archive-root", str(staging),
        "--output", str(staging / "figures/analysis"),
    ], cwd=ROOT, check=True)

    for name in ["requirements-cu130.txt", "requirements-cu130-lock.txt",
                 "requirements-venv.txt", "requirements-tf-gpu.txt"]:
        copy_file(name, staging / "environment" / name)
    copy_file("WEIGHTS_MANIFEST.json", staging / "weights/WEIGHTS_MANIFEST.json")

    counts = record_counts(staging)
    validate_counts(counts)
    write_readme(staging, counts)
    payload = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            payload.append({
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": os.popen(f"git -C '{ROOT}' rev-parse HEAD").read().strip(),
        "record_counts": counts,
        "file_count": len(payload),
        "total_bytes": sum(item["bytes"] for item in payload),
        "files": payload,
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)
    print(json.dumps({
        "destination": str(destination),
        "file_count": manifest["file_count"] + 1,
        "total_bytes": manifest["total_bytes"] + (destination / "MANIFEST.json").stat().st_size,
        "record_counts": counts,
    }, indent=2))


if __name__ == "__main__":
    main()
